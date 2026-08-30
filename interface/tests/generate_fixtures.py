"""Generate fixtures for the C++ ModelRunner test suite.

Invoked as a ctest fixture before any subtest runs. Produces:

    <out_dir>/model_<device>/<name>.pt2        exported AOTI package
    <out_dir>/model_<device>/<name>.meta.json  device + provenance
    <out_dir>/reference_<device>.pt            {input, output} reference
    <out_dir>/alt_<device>.pt                  alternative state_dict
    <out_dir>/reference_alt_<device>.pt        {input, output} with alt weights
    <out_dir>/model_graph_<device>/            three-input graph potential
    <out_dir>/reference_graph_<device>.pt      its inputs + outputs
    <out_dir>/model_graph_fixed_cuda/          four-input fixed-shape variant

`<device>` is `cpu`, plus `cuda` if a GPU is visible. The C++ tests pick
the device based on what's present. `model_graph_fixed_cuda` is emitted only
alongside the CUDA exports: it exists for the CUDA-graph replay case, which has
nothing to run on without a GPU.

This script *requires* a working editable install of MolNex so that
``molix.export.Exporter`` is importable; the test harness is
expected to be run inside the project's dev environment.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn


class TinyLinear(nn.Module):
    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2)
        gen = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.linear.weight.copy_(torch.randn((2, 3), generator=gen))
            self.linear.bias.copy_(torch.randn((2,), generator=gen))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _graph_potential(
    Z: torch.Tensor,
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Energy and forces of a toy graph potential, both in closed form.

    .. math::

        E = \\sum_i Z_i |x_i|^2 + \\sum_e w_e |x_{t(e)} - x_{s(e)}|^2

    with ``edge_index[:, 0] = s`` the source and ``edge_index[:, 1] = t`` the
    target, per the repo Edge Convention. Its gradient is elementary, so the
    forces below are written out ANALYTICALLY rather than obtained from
    ``autograd``:

    .. math::

        -\\frac{\\partial E}{\\partial x_a} = -2 Z_a x_a
            - \\sum_e 2 w_e (x_{t} - x_{s})(\\delta_{a,t} - \\delta_{a,s})

    That is the whole point of this fixture. ``Exporter.export_pretraced`` — the
    path that bakes a real backward pass into the package — refuses CPU targets
    by design (``src/molix/export.py:153``: torch 2.12's CPU inductor
    miscompiles the backward scatter, giving forces ~7 % wrong while the energy
    stays exact; the refusal is locked by
    ``tests/test_molix/test_engine.py::test_export_for_lammps_cpu_refused``). A
    closed-form gradient has no backward pass to miscompile, so this model
    exports through the plain :class:`~molix.export.Exporter` — the established
    fixture path here — and its outputs are exact to the last bit, which is what
    lets the C++ parity case assert at 1e-12 rather than at a tolerance chosen
    to fit the mismatch.

    Args:
        Z: Atomic numbers ``(n,)`` int64.
        pos: Positions ``(n, 3)``.
        edge_index: Source/target pairs ``(e, 2)`` int64.
        edge_weight: Per-edge weight ``(e,)``, in ``pos``'s dtype. 1 for a real
            edge, 0 for a padding row — which is what makes the fixed-shape
            variant agree with the dynamic one edge for edge.

    Returns:
        ``(energy, forces)`` — a 0-dim scalar and ``(n, 3)``.
    """
    source = edge_index[:, 0]
    target = edge_index[:, 1]
    diff = pos.index_select(0, target) - pos.index_select(0, source)  # (e, 3)
    z_f = Z.to(pos.dtype)  # (n,)

    energy = (z_f * pos.pow(2).sum(-1)).sum() + (edge_weight * diff.pow(2).sum(-1)).sum()

    grad = 2.0 * z_f.unsqueeze(-1) * pos  # (n, 3), from the node term
    edge_grad = 2.0 * edge_weight.unsqueeze(-1) * diff  # (e, 3)
    grad = grad.index_add(0, target, edge_grad).index_add(0, source, -edge_grad)
    return energy, -grad


class GraphToy(nn.Module):
    """Three-input graph potential: ``(Z, pos, edge_index) -> (energy, forces)``.

    The signature the ``cutoff-graph`` contract's dynamic execution path
    actually uses, which the single-input ``TinyLinear`` fixture cannot drive.
    Parameter-free: nothing here is being trained or reloaded, and every output
    is a closed-form function of the inputs (see :func:`_graph_potential`).
    """

    def forward(
        self, Z: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = torch.ones(edge_index.shape[0], dtype=pos.dtype, device=pos.device)
        return _graph_potential(Z, pos, edge_index, weight)


class GraphToyMasked(nn.Module):
    """Four-input fixed-shape variant: ``(Z, pos, edge_index, mask)``.

    The signature the ``cuda-graph`` execution path replays. ``mask`` marks the
    real rows of an edge tensor padded out to ``e_max``; padding rows are
    ``(0, 0)`` and weigh 0, so the energy and forces are identical to what
    :class:`GraphToy` returns for the unpadded edge set — which is exactly the
    agreement ``bridge_graphed_exec_matches_dynamic`` asserts.
    """

    def forward(
        self,
        Z: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _graph_potential(Z, pos, edge_index, mask.to(pos.dtype))


def _export_with_runtime_constants(
    model: nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    export_dir: Path,
    device: str,
    name: str = "model",
) -> None:
    """Export via molix.export.Exporter with runtime constant folding.

    Default AOTI compilation folds small constants (like a Linear bias)
    into the package as literals, making them unreloadable via
    update_weights(). `aot_inductor.use_runtime_constant_folding=True`
    keeps every parameter as an updateable constant buffer — required to
    exercise the reload path in this test suite. A dynamic batch dim is
    declared so one package serves the varying input sizes the tests use.
    """
    import json

    from molix.export import Exporter

    export_dir.mkdir(parents=True, exist_ok=True)
    target = model.to(device).eval()
    device_inputs = tuple(
        t.to(device) if isinstance(t, torch.Tensor) else t for t in example_inputs
    )
    batch = torch.export.Dim("batch", min=2, max=4096)
    Exporter(target).export(
        device_inputs,
        export_dir / f"{name}.pt2",
        dynamic_shapes=({0: batch},),
        inductor_configs={"aot_inductor.use_runtime_constant_folding": True},
    )
    meta = {"device": device, "model_class": model.__class__.__name__}
    (export_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=2))


def _emit_for_device(out_dir: Path, device: str, example: torch.Tensor) -> None:
    model_dir = out_dir / f"model_{device}"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True)

    base = TinyLinear(seed=0).to(device).eval()
    with torch.no_grad():
        ref_out = base(example.to(device))
    _export_with_runtime_constants(base, (example.to(device),), model_dir, device)
    torch.save(
        {"input": example.to(device), "output": ref_out},
        out_dir / f"reference_{device}.pt",
    )

    alt = TinyLinear(seed=42).to(device).eval()
    with torch.no_grad():
        alt_out = alt(example.to(device))
    # Save as plain dict (not OrderedDict) so torch::pickle_load in C++ can
    # round-trip it without a custom type resolver.
    torch.save(dict(alt.state_dict()), out_dir / f"alt_{device}.pt")
    torch.save(
        {"input": example.to(device), "output": alt_out},
        out_dir / f"reference_alt_{device}.pt",
    )


# A five-atom configuration, written out rather than sampled so the fixture is
# readable and the C++ side's expectations are inspectable. Edges are the full
# bidirectional chain 0-1-2-3-4, already sorted by (source, target) — the order
# GraphBridge emits and molnex aggregation requires.
_GRAPH_Z = [1, 6, 8, 6, 1]
_GRAPH_POS = [
    [0.00, 0.00, 0.00],
    [1.10, 0.00, 0.00],
    [1.70, 0.95, 0.00],
    [2.90, 0.40, 0.60],
    [3.50, 1.20, -0.30],
]
_GRAPH_EDGES = [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3]]
# Padded capacity for the fixed-shape export. Strictly greater than the edge
# count on purpose: an e_max equal to it would never exercise the padded tail,
# which is the only part of the fixed-shape path the dynamic one lacks. Written
# into the reference file so the C++ case reads it rather than restating it.
_GRAPH_E_MAX = len(_GRAPH_EDGES) + 4


def _emit_graph_fixture(out_dir: Path, device: str) -> None:
    """Export the three-input graph potential and store its reference outputs.

    Uses the bare :class:`~molix.export.Exporter` — the same path
    ``_export_with_runtime_constants`` takes, minus the constant-folding config
    a parameter-free model has no use for — and writes only ``device`` into the
    meta. Deliberately NOT ``molix.engine.export_for_lammps``: its force-export
    path refuses CPU targets (see :func:`_graph_potential`), and hanging a
    ``lammps`` block on this meta would couple a fixture that exists to exercise
    ``Exec`` to the deployment meta schema, which the C++ side does not read
    here.
    """
    import json

    from molix.export import Exporter

    model_dir = out_dir / f"model_graph_{device}"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True)

    Z = torch.tensor(_GRAPH_Z, dtype=torch.int64, device=device)
    pos = torch.tensor(_GRAPH_POS, dtype=torch.float64, device=device)
    edge_index = torch.tensor(_GRAPH_EDGES, dtype=torch.int64, device=device)

    model = GraphToy().to(device).eval()
    with torch.no_grad():
        energy, forces = model(Z, pos, edge_index)

    # One package serves any atom / edge count, exactly as the TinyLinear
    # fixture's dynamic batch dim does — the dynamic path is the one under test,
    # so a statically shaped export would not be evidence about it.
    n_atoms = torch.export.Dim("n_atoms", min=2, max=4096)
    n_edges = torch.export.Dim("n_edges", min=2, max=65536)
    Exporter(model).export(
        (Z, pos, edge_index),
        model_dir / "model.pt2",
        dynamic_shapes=({0: n_atoms}, {0: n_atoms}, {0: n_edges}),
    )
    (model_dir / "model.meta.json").write_text(
        json.dumps({"device": device, "model_class": "GraphToy"}, indent=2)
    )

    torch.save(
        {
            "Z": Z,
            "pos": pos,
            "edge_index": edge_index,
            "energy": energy,
            "forces": forces,
            "e_max": torch.tensor(_GRAPH_E_MAX, dtype=torch.int64),
        },
        out_dir / f"reference_graph_{device}.pt",
    )


def _emit_graph_fixed_fixture(out_dir: Path, device: str) -> None:
    """Export the four-input fixed-shape variant for the CUDA-graph replay case.

    No dynamic dims: a captured CUDA graph binds fixed addresses and therefore
    fixed shapes, so this package is compiled for exactly ``(len(_GRAPH_Z),
    _GRAPH_E_MAX)``. Emitted only for CUDA — a replay case has nothing to run on
    otherwise.
    """
    import json

    from molix.export import Exporter

    model_dir = out_dir / f"model_graph_fixed_{device}"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True)

    Z = torch.tensor(_GRAPH_Z, dtype=torch.int64, device=device)
    pos = torch.tensor(_GRAPH_POS, dtype=torch.float64, device=device)
    edges = torch.zeros((_GRAPH_E_MAX, 2), dtype=torch.int64, device=device)
    edges[: len(_GRAPH_EDGES)] = torch.tensor(_GRAPH_EDGES, dtype=torch.int64, device=device)
    mask = torch.zeros(_GRAPH_E_MAX, dtype=torch.bool, device=device)
    mask[: len(_GRAPH_EDGES)] = True

    model = GraphToyMasked().to(device).eval()
    Exporter(model).export((Z, pos, edges, mask), model_dir / "model.pt2")
    (model_dir / "model.meta.json").write_text(
        json.dumps({"device": device, "model_class": "GraphToyMasked"}, indent=2)
    )


def _emit_lammps_v2_meta(out_dir: Path) -> None:
    """Write a real schema-v2 meta through the production block builder.

    Ties the Python-side ``LAMMPS_META_SCHEMA`` to the C++ expected version at
    test time: ``contract_reads_generated_v2_fixture`` parses this file through
    ``LammpsContract``, so bumping either constant alone turns that case red.
    """
    import json

    from molix.engine.export import _lammps_meta_block

    meta = {
        "device": "cpu",
        "model_class": "MetaOnly",
        "lammps": _lammps_meta_block(
            species=[1, 6, 8],
            cutoff=4.5,
            units="real",
            adapter_name="molnex-tensordict",
            model_dtype="float32",
            supports_pbc=False,
            supports_virial=True,
            cuda_graph=False,
        ),
    }
    (out_dir / "lammps_v2.meta.json").write_text(json.dumps(meta, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    gen = torch.Generator().manual_seed(1234)
    example = torch.randn((4, 3), generator=gen)

    _emit_for_device(args.out_dir, "cpu", example)
    _emit_graph_fixture(args.out_dir, "cpu")
    _emit_lammps_v2_meta(args.out_dir)
    if torch.cuda.is_available():
        _emit_for_device(args.out_dir, "cuda", example)
        _emit_graph_fixture(args.out_dir, "cuda")
        _emit_graph_fixed_fixture(args.out_dir, "cuda")

    return 0


if __name__ == "__main__":
    sys.exit(main())
