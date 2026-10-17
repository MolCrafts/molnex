"""Generate fixtures for the C++ runtime tests.

Invoked as a ctest fixture before any subtest that needs a package. Produces:

    <out_dir>/model_<device>/model.pt2       TinyLinear AOTI package
    <out_dir>/reference_<device>.pt          {input, output}
    <out_dir>/alt_<device>.pt                alternative state_dict
    <out_dir>/reference_alt_<device>.pt      {input, output} with alt weights
    <out_dir>/model_graph_cpu/model.pt2      site-energy toy potential
    <out_dir>/model_graph_cpu/model.json     its molix.deploy metadata
    <out_dir>/reference_graph_cpu.pt         its inputs and outputs

``<device>`` is ``cpu``, plus ``cuda`` for TinyLinear when a GPU is visible.
The graph package is CPU-only.

Packaging is the deployment path itself: ``molix.export.aoti.package`` with the
site-energy ``DYNAMIC_SHAPES`` of ``molix.deploy.contract``, and the metadata
is written by ``molix.deploy.DeployMetadata`` — so the C++ reader is tested
against the Python writer, not against a copy of its layout. TinyLinear passes
``aot_inductor.use_runtime_constant_folding`` so ``update_weights`` can replace
``linear.weight`` and ``linear.bias`` by state-dict name.
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
    shifts: torch.Tensor,
    edge_weight: torch.Tensor,
    local_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Site energies of a toy graph potential, and their masked sum.

    Each edge term is split evenly between its endpoints, so a site energy
    depends on its neighbours:

        E_i = Z_i |x_i|^2 + (1/2) sum_{e ni i} w_e |d_e|^2
        E = sum_i m_i E_i

    with d_e = x_{t(e)} - x_{s(e)} + shift_e, pair_index[:, 0] = s,
    pair_index[:, 1] = t, and m the local mask. At an all-ones mask the edge halves regroup pairwise
    and E reduces to sum_i Z_i |x_i|^2 + sum_e w_e |d_e|^2.

    Forces are the closed-form gradient of that masked sum, so the C++ parity
    case can assert 1e-12 against an exact expression of the inputs:

        -dE/dx_a = -2 m_a Z_a x_a
            - sum_e 2 c_e w_e d_e (delta_{a,t} - delta_{a,s})
        c_e = (1/2) (m_s + m_t)

    An edge between a local and a ghost site contributes half its weight.

    The virial is the edge part of the strain derivative,
    ``sum_e 2 c_e w_e d_e (x) d_e`` — the contract's definition. (The on-site
    ``Z_i |x_i|^2`` is not translation-invariant, so the toy has no complete
    strain derivative; the fixture only has to pin what the package returns.)

    Args:
        Z: Atomic numbers ``(n,)`` int64.
        pos: Positions ``(n, 3)``.
        pair_index: Source/target pairs ``(e, 2)`` int64.
        shifts: Lattice translation of each target ``(e, 3)``.
        edge_weight: Per-edge weight ``(e,)``, in ``pos``'s dtype.
        local_mask: Per-site ownership ``(n,)`` bool.

    Returns:
        ``(energy, forces, site_energy, virial)`` — a 0-dim scalar,
        ``(n, 3)``, ``(n,)``, ``(3, 3)``.
    """
    source = pair_index[:, 0]
    target = pair_index[:, 1]
    diff = pos.index_select(0, target) - pos.index_select(0, source) + shifts
    z_f = Z.to(pos.dtype)
    mask_f = local_mask.to(pos.dtype)

    edge_energy = edge_weight * diff.pow(2).sum(-1)
    half = 0.5 * edge_energy
    site_energy = (z_f * pos.pow(2).sum(-1)).index_add(0, target, half).index_add(0, source, half)
    energy = (site_energy * mask_f).sum()

    c_e = 0.5 * (mask_f.index_select(0, source) + mask_f.index_select(0, target))
    grad = 2.0 * (mask_f * z_f).unsqueeze(-1) * pos
    edge_grad = 2.0 * (c_e * edge_weight).unsqueeze(-1) * diff
    grad = grad.index_add(0, target, edge_grad).index_add(0, source, -edge_grad)
    virial = diff.transpose(0, 1) @ edge_grad
    return energy, -grad, site_energy, virial


class GraphToy(nn.Module):
    """``(Z, pos, pair_index, shifts, local_mask) -> (energy, forces, site_energy, virial)``.

    Parameter-free. Every output is the closed form in
    :func:`_graph_site_energies`.
    """

    def forward(
        self,
        Z: torch.Tensor,
        pos: torch.Tensor,
        pair_index: torch.Tensor,
        shifts: torch.Tensor,
        local_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        weight = torch.ones(pair_index.shape[0], dtype=pos.dtype, device=pos.device)
        return _graph_site_energies(Z, pos, pair_index, shifts, weight, local_mask)


def _export_tiny(model: nn.Module, example: torch.Tensor, export_dir: Path) -> None:
    """Package TinyLinear with reloadable constants and a dynamic batch dim."""
    from molix.export import aoti

    export_dir.mkdir(parents=True, exist_ok=True)
    batch = torch.export.Dim("batch", min=2, max=4096)
    aoti.package(
        model,
        (example,),
        export_dir / "model.pt2",
        dynamic_shapes=({0: batch},),
        inductor_configs={"aot_inductor.use_runtime_constant_folding": True},
    )


def _emit_for_device(out_dir: Path, device: str, example: torch.Tensor) -> None:
    model_dir = out_dir / f"model_{device}"
    if model_dir.exists():
        shutil.rmtree(model_dir)

    base = TinyLinear(seed=0).to(device).eval()
    moved = example.to(device)
    with torch.no_grad():
        ref_out = base(moved)
    _export_tiny(base, moved, model_dir)
    torch.save({"input": moved, "output": ref_out}, out_dir / f"reference_{device}.pt")

    alt = TinyLinear(seed=42).to(device).eval()
    with torch.no_grad():
        alt_out = alt(moved)
    # A plain dict, so torch::pickle_load in C++ round-trips it.
    torch.save(dict(alt.state_dict()), out_dir / f"alt_{device}.pt")
    torch.save({"input": moved, "output": alt_out}, out_dir / f"reference_alt_{device}.pt")


# Five atoms. Edges are the bidirectional chain 0-1-2-3-4. The builder does
# not sort, so this order is only how the reference is written.
_GRAPH_Z = [1, 6, 8, 6, 1]
_GRAPH_POS = [
    [0.00, 0.00, 0.00],
    [1.10, 0.00, 0.00],
    [1.70, 0.95, 0.00],
    [2.90, 0.40, 0.60],
    [3.50, 1.20, -0.30],
]
_GRAPH_EDGES = [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3]]


#: The toy's cutoff, in angstrom; the metadata declares it.
_GRAPH_CUTOFF = 2.0


def _emit_graph_fixture(out_dir: Path) -> None:
    """Deploy the site-energy toy potential and write its reference outputs.

    The reference uses an all-ones mask, so the site energies sum to the
    undecomposed total, and non-zero shifts on the 2-3 edges, so a runtime that
    dropped or zeroed the shifts would miss it. The package is traced on the
    zero-shift example, as every deployment is.
    """
    from molix.deploy import DeployMetadata
    from molix.deploy.contract import DYNAMIC_SHAPES
    from molix.export import aoti

    device = "cpu"
    model_dir = out_dir / "model_graph_cpu"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True)

    Z = torch.tensor(_GRAPH_Z, dtype=torch.int64, device=device)
    pos = torch.tensor(_GRAPH_POS, dtype=torch.float64, device=device)
    pair_index = torch.tensor(_GRAPH_EDGES, dtype=torch.int64, device=device)
    local_mask = torch.ones(len(_GRAPH_Z), dtype=torch.bool, device=device)
    shifts = torch.zeros(len(_GRAPH_EDGES), 3, dtype=torch.float64, device=device)
    shifts[4] = torch.tensor([0.25, 0.0, -0.5], dtype=torch.float64)
    shifts[5] = -shifts[4]

    model = GraphToy().to(device).eval()
    with torch.no_grad():
        energy, forces, site_energy, virial = model(Z, pos, pair_index, shifts, local_mask)

    written = aoti.package(
        model,
        (Z, pos, pair_index, torch.zeros_like(shifts), local_mask),
        model_dir / "model.pt2",
        dynamic_shapes=DYNAMIC_SHAPES,
    )
    DeployMetadata(
        format="aoti",
        cutoff=_GRAPH_CUTOFF,
        species=tuple(_GRAPH_Z),
        dtype=DeployMetadata.dtype_name(pos.dtype),
        receptive_field_layers=1,
        op_libraries=DeployMetadata.op_libraries_for(written.op_namespaces),
    ).write(model_dir, "model")
    torch.save(
        {
            "Z": Z,
            "pos": pos,
            "pair_index": pair_index,
            "shifts": shifts,
            "local_mask": local_mask,
            "energy": energy,
            "forces": forces,
            "site_energy": site_energy,
            "virial": virial,
        },
        out_dir / "reference_graph_cpu.pt",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gen = torch.Generator().manual_seed(1234)
    example = torch.randn((4, 3), generator=gen)
    _emit_for_device(args.out_dir, "cpu", example)
    _emit_graph_fixture(args.out_dir)
    if torch.cuda.is_available():
        _emit_for_device(args.out_dir, "cuda", example)
    return 0


if __name__ == "__main__":
    sys.exit(main())
