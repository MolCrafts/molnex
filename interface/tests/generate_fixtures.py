"""Generate fixtures for the C++ ModelRunner test suite.

Invoked as a ctest fixture before any subtest runs. Produces:

    <out_dir>/model_<device>/<name>.pt2        exported AOTI package
    <out_dir>/model_<device>/<name>.meta.json  device + provenance
    <out_dir>/reference_<device>.pt            {input, output} reference
    <out_dir>/alt_<device>.pt                  alternative state_dict
    <out_dir>/reference_alt_<device>.pt        {input, output} with alt weights
    <out_dir>/model_graph_v3_<device>/         four-input schema-v3 potential
    <out_dir>/reference_graph_v3_<device>.pt   its inputs + outputs
    <out_dir>/model_graph_fixed_v3_cuda/       five-input fixed-shape variant
    <out_dir>/lammps_v3.meta.json              a real schema-v3 meta block

`<device>` is `cpu`, plus `cuda` if a GPU is visible. The C++ tests pick
the device based on what's present. `model_graph_fixed_v3_cuda` is emitted only
alongside the CUDA exports: it exists for the CUDA-graph replay case, which has
nothing to run on without a GPU.

The graph fixtures are schema v3 throughout — `(Z, pos, pair_index, local_mask)
-> (energy, forces, atomic_energy)` on the dynamic path, with `pair_mask`
inserted before `local_mask` on the fixed-shape one. The three-input v2
``GraphToy`` that used to be emitted here is gone: the C++ batch of spec
``lammps-iface-05-pbc`` moved its last consumer (``Exec::run``, which now takes
a local mask) onto the v3 signature, and a fixture no execution path can call is
worse than no fixture at all.

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


def _graph_site_energies(
    Z: torch.Tensor,
    pos: torch.Tensor,
    pair_index: torch.Tensor,
    edge_weight: torch.Tensor,
    local_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Site energies of a toy graph potential, and their masked sum.

    Toy physics, decomposed onto sites. Each edge term is split evenly between
    its two endpoints, which is what makes a site energy depend on its
    neighbours' coordinates — without that coupling the masked-out sites would
    carry no force and the fixture could not exercise the ghost-force path at
    all:

    .. math::

        E_i = Z_i |x_i|^2 + \\tfrac{1}{2}\\sum_{e \\ni i} w_e |d_e|^2, \\qquad
        E = \\sum_i m_i E_i

    with :math:`d_e = x_{t(e)} - x_{s(e)}`, ``pair_index[:, 0] = s`` the source
    and ``pair_index[:, 1] = t`` the target per the repo Edge Convention, and
    :math:`m` the local mask. At an all-ones mask the edge halves regroup
    pairwise and ``E`` reduces to :math:`\\sum_i Z_i |x_i|^2 + \\sum_e w_e
    |d_e|^2`, the undecomposed total.

    Forces are written out ANALYTICALLY rather than obtained from ``autograd``,
    and that is the whole point of this fixture. ``Exporter.export_pretraced`` —
    the path that bakes a real backward pass into the package — refuses CPU
    targets by design (``src/molix/export.py:153``: torch 2.12's CPU inductor
    miscompiles the backward scatter, giving forces ~7 % wrong while the energy
    stays exact). A closed-form gradient has no backward pass to miscompile, so
    this model exports through the plain :class:`~molix.export.Exporter` and its
    outputs are exact to the last bit — which is what lets the C++ parity case
    assert at 1e-12 rather than at a tolerance chosen to fit the mismatch.

    Differentiating the masked sum gives

    .. math::

        -\\frac{\\partial E}{\\partial x_a} = -2 m_a Z_a x_a
            - \\sum_e 2 c_e w_e d_e (\\delta_{a,t} - \\delta_{a,s}),
        \\qquad c_e = \\tfrac{1}{2}(m_{s(e)} + m_{t(e)})

    Note :math:`c_e`: an edge between a local and a ghost site contributes half
    its weight to the partition energy, which is precisely the cross-term a
    masked-out site still feels.

    Args:
        Z: Atomic numbers ``(n,)`` int64.
        pos: Positions ``(n, 3)``.
        pair_index: Source/target pairs ``(e, 2)`` int64.
        edge_weight: Per-edge weight ``(e,)``, in ``pos``'s dtype. 1 for a real
            edge, 0 for a padding row — which is what makes the fixed-shape
            variant agree with the dynamic one edge for edge.
        local_mask: Per-site ownership ``(n,)`` bool — the atoms the rank owns.

    Returns:
        ``(energy, forces, atomic_energy)`` — a 0-dim scalar, ``(n, 3)``, ``(n,)``.
    """
    source = pair_index[:, 0]
    target = pair_index[:, 1]
    diff = pos.index_select(0, target) - pos.index_select(0, source)  # (e, 3)
    z_f = Z.to(pos.dtype)  # (n,)
    mask_f = local_mask.to(pos.dtype)  # (n,)

    edge_energy = edge_weight * diff.pow(2).sum(-1)  # (e,)
    half = 0.5 * edge_energy
    atomic_energy = (z_f * pos.pow(2).sum(-1)).index_add(0, target, half).index_add(0, source, half)
    energy = (atomic_energy * mask_f).sum()

    # c_e = half the mask sum over the edge's endpoints (1 local-local, 1/2
    # local-ghost, 0 ghost-ghost).
    c_e = 0.5 * (mask_f.index_select(0, source) + mask_f.index_select(0, target))
    grad = 2.0 * (mask_f * z_f).unsqueeze(-1) * pos  # (n, 3), node term
    edge_grad = 2.0 * (c_e * edge_weight).unsqueeze(-1) * diff  # (e, 3)
    grad = grad.index_add(0, target, edge_grad).index_add(0, source, -edge_grad)
    return energy, -grad, atomic_energy


class GraphToyV3(nn.Module):
    """Four-input schema-v3 potential.

    ``(Z, pos, pair_index, local_mask) -> (energy, forces, atomic_energy)`` —
    the signature the ``cutoff-graph`` contract's dynamic execution path uses
    from spec ``lammps-iface-05-pbc`` on, where the energy is the site-energy
    sum over locally owned atoms only. Parameter-free and closed-form: nothing
    here is trained or reloaded, and every output is an exact expression of the
    inputs (see :func:`_graph_site_energies`).
    """

    def forward(
        self,
        Z: torch.Tensor,
        pos: torch.Tensor,
        pair_index: torch.Tensor,
        local_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weight = torch.ones(pair_index.shape[0], dtype=pos.dtype, device=pos.device)
        return _graph_site_energies(Z, pos, pair_index, weight, local_mask)


class GraphToyMaskedV3(nn.Module):
    """Five-input fixed-shape variant: ``(Z, pos, pair_index, pair_mask, local_mask)``.

    The signature the ``cuda-graph`` execution path replays under schema v3. The
    two masks are unrelated and differently shaped: ``pair_mask (e_max,)`` marks
    the real rows of an edge tensor padded out to ``e_max``, while
    ``local_mask (n,)`` marks the atoms the rank owns. Padding rows are ``(0, 0)``
    and weigh 0, so the outputs are identical to what :class:`GraphToyV3`
    returns for the unpadded edge set — which is exactly the agreement
    ``bridge_graphed_exec_matches_dynamic`` asserts.
    """

    def forward(
        self,
        Z: torch.Tensor,
        pos: torch.Tensor,
        pair_index: torch.Tensor,
        pair_mask: torch.Tensor,
        local_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _graph_site_energies(Z, pos, pair_index, pair_mask.to(pos.dtype), local_mask)


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
# bidirectional chain 0-1-2-3-4, listed in (source, target) order for
# readability only: molnex aggregation is scatter-class and therefore
# permutation-invariant, and GraphBridge sorts only when a contract declares
# `requires_sorted_edges` (spec lammps-iface-05-pbc, D6). Nothing here may be
# read as a promise that edges arrive ordered.
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


def _emit_graph_v3_fixture(out_dir: Path, device: str) -> None:
    """Export the four-input schema-v3 graph potential and its reference outputs.

    Uses the bare :class:`~molix.export.Exporter` — the same path
    ``_export_with_runtime_constants`` takes, minus the constant-folding config
    a parameter-free model has no use for — and writes only ``device`` into the
    meta. Deliberately NOT ``molix.engine.export_for_lammps``: its force-export
    path refuses CPU targets (see :func:`_graph_site_energies`), and hanging a
    ``lammps`` block on this meta would couple a fixture that exists to exercise
    ``Exec`` to the deployment meta schema, which the C++ side does not read
    here.

    The reference is captured with an **all-ones** mask: at full mask the site
    energies sum to the undecomposed total, so a sign or factor slip in the
    decomposition shows up as a shifted energy rather than hiding inside the
    per-site split.
    """
    import json

    from molix.export import Exporter

    model_dir = out_dir / f"model_graph_v3_{device}"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True)

    Z = torch.tensor(_GRAPH_Z, dtype=torch.int64, device=device)
    pos = torch.tensor(_GRAPH_POS, dtype=torch.float64, device=device)
    pair_index = torch.tensor(_GRAPH_EDGES, dtype=torch.int64, device=device)
    local_mask = torch.ones(len(_GRAPH_Z), dtype=torch.bool, device=device)

    model = GraphToyV3().to(device).eval()
    with torch.no_grad():
        energy, forces, atomic_energy = model(Z, pos, pair_index, local_mask)

    n_atoms = torch.export.Dim("n_atoms", min=2, max=4096)
    n_pairs = torch.export.Dim("n_pairs", min=2, max=65536)
    Exporter(model).export(
        (Z, pos, pair_index, local_mask),
        model_dir / "model.pt2",
        dynamic_shapes=({0: n_atoms}, {0: n_atoms}, {0: n_pairs}, {0: n_atoms}),
    )
    (model_dir / "model.meta.json").write_text(
        json.dumps({"device": device, "model_class": "GraphToyV3"}, indent=2)
    )

    torch.save(
        {
            "Z": Z,
            "pos": pos,
            "pair_index": pair_index,
            "local_mask": local_mask,
            "energy": energy,
            "forces": forces,
            "atomic_energy": atomic_energy,
            "e_max": torch.tensor(_GRAPH_E_MAX, dtype=torch.int64),
        },
        out_dir / f"reference_graph_v3_{device}.pt",
    )


def _emit_graph_fixed_v3_fixture(out_dir: Path, device: str) -> None:
    """Export the five-input fixed-shape variant for the CUDA-graph replay case.

    No dynamic dims: a captured CUDA graph binds fixed addresses and therefore
    fixed shapes, so this package is compiled for exactly ``(len(_GRAPH_Z),
    _GRAPH_E_MAX)``. Emitted only for CUDA — a replay case has nothing to run on
    otherwise.

    The input order is the schema-v3 ``cuda-graph`` one,
    ``(Z, pos, pair_index, pair_mask, local_mask)``, which is also the order
    ``GraphedExec`` binds its persistent buffers in. The two lists must agree:
    the mask lengths differ (``e_max`` against ``n``), so a swap is a shape
    error at replay rather than a silent one, but only because the shapes happen
    to differ.
    """
    import json

    from molix.export import Exporter

    model_dir = out_dir / f"model_graph_fixed_v3_{device}"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True)

    Z = torch.tensor(_GRAPH_Z, dtype=torch.int64, device=device)
    pos = torch.tensor(_GRAPH_POS, dtype=torch.float64, device=device)
    pairs = torch.zeros((_GRAPH_E_MAX, 2), dtype=torch.int64, device=device)
    pairs[: len(_GRAPH_EDGES)] = torch.tensor(_GRAPH_EDGES, dtype=torch.int64, device=device)
    pair_mask = torch.zeros(_GRAPH_E_MAX, dtype=torch.bool, device=device)
    pair_mask[: len(_GRAPH_EDGES)] = True
    local_mask = torch.ones(len(_GRAPH_Z), dtype=torch.bool, device=device)

    model = GraphToyMaskedV3().to(device).eval()
    Exporter(model).export((Z, pos, pairs, pair_mask, local_mask), model_dir / "model.pt2")
    (model_dir / "model.meta.json").write_text(
        json.dumps({"device": device, "model_class": "GraphToyMaskedV3"}, indent=2)
    )


def _emit_lammps_v3_meta(out_dir: Path) -> None:
    """Write a real schema-v3 meta through the production block builder.

    Ties the Python-side ``LAMMPS_META_SCHEMA`` to the C++ expected version at
    test time: ``contract_reads_generated_v3_fixture`` parses this file through
    ``LammpsContract``, so bumping either constant alone turns that case red.

    The filename carries the schema version, which doubles the tie: a C++ side
    still expecting v2 fails to OPEN this file, and a Python side still writing
    v2 leaves the case with nothing to read. Only a matched pair goes green.

    ``receptive_field_layers=2`` is a deliberate choice, not a placeholder: a
    value of 1 would let a fat-ghost check that ignores ``T`` pass anyway,
    making the fixture blind to exactly the bug the field exists to prevent.
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
            receptive_field_layers=2,
            requires_sorted_edges=False,
        ),
    }
    (out_dir / "lammps_v3.meta.json").write_text(json.dumps(meta, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    gen = torch.Generator().manual_seed(1234)
    example = torch.randn((4, 3), generator=gen)

    _emit_for_device(args.out_dir, "cpu", example)
    _emit_graph_v3_fixture(args.out_dir, "cpu")
    _emit_lammps_v3_meta(args.out_dir)
    if torch.cuda.is_available():
        _emit_for_device(args.out_dir, "cuda", example)
        _emit_graph_v3_fixture(args.out_dir, "cuda")
        _emit_graph_fixed_v3_fixture(args.out_dir, "cuda")

    return 0


if __name__ == "__main__":
    sys.exit(main())
