"""Multi-system batch independence — ac-010.

Per-graph energy from a batched ``Sonata`` forward must match the
single-graph energy each constituent graph would yield if forwarded on
its own. The test exercises three systems with different ``N`` and
distinct cell geometries — ``cell=None`` (free), ``cell=8 Å I``, and
``cell=12 Å I`` — to ensure the per-graph dispatch in
``EwaldMultipoleEnergy`` (``cell.dim() == 3 ⇒ index by graph``) is
honoured by Sonata.

Tolerance ≤ 1e-8 (float64). The only place the batched and single-graph
paths can diverge is in the per-graph dispatch and per-graph
accumulation; everything else (encoder, head, Ewald kernels) is per-atom
or per-pair and should give bit-for-bit identical results.
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from molix.config import config
from molpot.composition import Sonata, build_sonata
from molzoo import Allegro

# ---------------------------------------------------------------------------
# Module-local pipeline (so this file is independent of conftest's cache).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sonata_fp64() -> Sonata:
    orig_ftype = config["ftype"]
    config["ftype"] = torch.float64
    try:
        torch.manual_seed(0)
        encoder = Allegro(
            num_elements=10,
            num_scalar_features=16,
            num_tensor_features=4,
            r_max=5.0,
            num_bessel=8,
            num_layers=2,
            l_max=2,
            type_embed_dim=16,
            latent_mlp_depth=1,
            latent_mlp_width=32,
            avg_num_neighbors=12.0,
            expose_tensor_track=True,
        )
        sonata = build_sonata(
            encoder,
            sigma=1.0,
            dl=2.0,
            charge=True,
            dipole=True,
            quadrupole=True,
            constrain_total_charge=True,
            avg_num_neighbors=12.0,
        )
        sonata = sonata.double()
        sonata.eval()
        yield sonata
    finally:
        config["ftype"] = orig_ftype


# ---------------------------------------------------------------------------
# Per-graph constructors.
# ---------------------------------------------------------------------------


def _make_single_graph(
    *,
    pos: torch.Tensor,
    Z: torch.Tensor,
    cell: torch.Tensor | None,
    total_charge: float,
) -> TensorDict:
    n = pos.shape[0]
    edge_index = torch.tensor(
        [[i, j] for i in range(n) for j in range(n) if i != j], dtype=torch.long
    )
    bond_diff = pos[edge_index[:, 1]] - pos[edge_index[:, 0]]
    bond_dist = bond_diff.norm(dim=-1)
    batch_idx = torch.zeros(n, dtype=torch.long)
    num_atoms = torch.tensor([n], dtype=torch.long)

    graphs_kwargs: dict = {
        "num_atoms": num_atoms,
        "total_charge": torch.tensor([total_charge], dtype=torch.float64),
        "batch_size": [1],
    }
    if cell is not None:
        graphs_kwargs["cell"] = cell.unsqueeze(0)  # (1, 3, 3)

    return TensorDict(
        atoms=TensorDict(Z=Z, pos=pos, batch=batch_idx, batch_size=[n]),
        edges=TensorDict(
            edge_index=edge_index,
            bond_diff=bond_diff,
            bond_dist=bond_dist,
            batch_size=[edge_index.shape[0]],
        ),
        graphs=TensorDict(**graphs_kwargs),
        batch_size=[],
    )


def _stack_graphs(graphs: list[TensorDict]) -> TensorDict:
    """Concatenate single-graph ``TensorDict`` instances into one batch.

    ``cell`` is per-graph; for any graph that was constructed without a
    cell, we slot a zero (3, 3) tensor — the EwaldMultipoleEnergy
    forward dispatches on ``det(cell_i).abs() > 1e-6`` per graph, so a
    zero block lands in the realspace path exactly the same way
    ``cell_i = None`` does for the single-graph forward.
    """
    n_graphs = len(graphs)
    pos = torch.cat([g["atoms", "pos"] for g in graphs], dim=0)
    Z = torch.cat([g["atoms", "Z"] for g in graphs], dim=0)
    batch_parts: list[torch.Tensor] = []
    edge_parts: list[torch.Tensor] = []
    offset = 0
    for gi, g in enumerate(graphs):
        n_g = g["atoms", "pos"].shape[0]
        batch_parts.append(torch.full((n_g,), gi, dtype=torch.long))
        edge_parts.append(g["edges", "edge_index"] + offset)
        offset += n_g
    batch_idx = torch.cat(batch_parts, dim=0)
    edge_index = torch.cat(edge_parts, dim=0)
    bond_diff = pos[edge_index[:, 1]] - pos[edge_index[:, 0]]
    bond_dist = bond_diff.norm(dim=-1)

    total_charge = torch.cat([g["graphs", "total_charge"] for g in graphs], dim=0)  # (B,)

    cell_blocks: list[torch.Tensor] = []
    for g in graphs:
        if "cell" in g["graphs"].keys():
            cell_blocks.append(g["graphs", "cell"][0])
        else:
            cell_blocks.append(torch.zeros(3, 3, dtype=torch.float64))
    cell_stacked = torch.stack(cell_blocks, dim=0)  # (B, 3, 3)

    num_atoms = torch.tensor([g["atoms", "pos"].shape[0] for g in graphs], dtype=torch.long)

    return TensorDict(
        atoms=TensorDict(Z=Z, pos=pos, batch=batch_idx, batch_size=[pos.shape[0]]),
        edges=TensorDict(
            edge_index=edge_index,
            bond_diff=bond_diff,
            bond_dist=bond_dist,
            batch_size=[edge_index.shape[0]],
        ),
        graphs=TensorDict(
            num_atoms=num_atoms,
            total_charge=total_charge,
            cell=cell_stacked,
            batch_size=[n_graphs],
        ),
        batch_size=[],
    )


# ---------------------------------------------------------------------------
# 9 — Multi-system batching independence (ac-010)
# ---------------------------------------------------------------------------


def test_batch_independence(sonata_fp64: Sonata) -> None:
    """Three graphs with N ∈ {4, 6, 5}, geometries: free / cubic 8Å / cubic 12Å.

    ``Sonata(batch).energy[i] == Sonata(graph_i).energy`` to ≤ 1e-8.
    """
    torch.manual_seed(31)

    pos0 = torch.tensor(
        [
            [0.10, 0.20, 0.05],
            [1.55, 0.15, 0.10],
            [0.85, 1.30, -0.05],
            [-0.65, 1.25, 0.20],
        ],
        dtype=torch.float64,
    )
    Z0 = torch.tensor([1, 6, 8, 7], dtype=torch.long)

    pos1 = torch.tensor(
        [
            [0.05, 0.05, 0.05],
            [1.40, 0.05, 0.05],
            [2.80, 0.05, 0.05],
            [0.05, 1.40, 0.05],
            [1.40, 1.40, 0.05],
            [2.80, 1.40, 0.05],
        ],
        dtype=torch.float64,
    )
    Z1 = torch.tensor([1, 6, 8, 1, 6, 8], dtype=torch.long)

    pos2 = torch.tensor(
        [
            [0.30, 0.30, 0.30],
            [1.70, 0.30, 0.30],
            [3.10, 0.30, 0.30],
            [0.30, 1.70, 0.30],
            [1.70, 1.70, 0.30],
        ],
        dtype=torch.float64,
    )
    Z2 = torch.tensor([1, 6, 7, 8, 6], dtype=torch.long)

    cell1 = 8.0 * torch.eye(3, dtype=torch.float64)
    cell2 = 12.0 * torch.eye(3, dtype=torch.float64)

    g0 = _make_single_graph(pos=pos0, Z=Z0, cell=None, total_charge=0.0)
    g1 = _make_single_graph(pos=pos1, Z=Z1, cell=cell1, total_charge=0.0)
    g2 = _make_single_graph(pos=pos2, Z=Z2, cell=cell2, total_charge=0.0)

    with torch.no_grad():
        E_single = []
        for g in [g0, g1, g2]:
            E_single.append(float(sonata_fp64(g)["energy"][0]))

        batched = _stack_graphs([g0, g1, g2])
        E_batch = sonata_fp64(batched)["energy"]

    for i, e_s in enumerate(E_single):
        diff = abs(float(E_batch[i]) - e_s)
        assert diff < 1e-8, (
            f"graph {i}: |E_batch - E_single| = {diff:.3e} "
            f"(E_batch={float(E_batch[i]):.6e}, E_single={e_s:.6e})"
        )
