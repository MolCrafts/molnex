"""Graph-aware collation producing nested GraphBatch TensorDicts.

Collates a list of single-molecule sample dicts into a ``GraphBatch``
with per-level batch sizes: atoms (N), edges (E), graphs (B).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from tensordict import TensorDict

from molix.data.types import AtomData, EdgeData, GraphBatch, GraphData


# ---------------------------------------------------------------------------
# Target schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetSchema:
    """Declares how targets are collated.

    ``graph_level`` targets (e.g. energy) are per-molecule scalars → ``(B,)``.
    ``atom_level`` targets (e.g. forces) are per-atom tensors → ``(N_total, ...)``.
    """

    graph_level: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {"energy", "U0", "U", "H", "G", "Cv", "mu", "alpha",
             "homo", "lumo", "gap", "r2", "zpve", "A", "B", "C"}
        )
    )
    atom_level: frozenset[str] = field(
        default_factory=lambda: frozenset({"forces"})
    )


DEFAULT_TARGET_SCHEMA = TargetSchema()


# ---------------------------------------------------------------------------
# Edge normalisation
# ---------------------------------------------------------------------------


def _normalize_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    """Normalize edge_index to canonical ``(E, 2)`` format."""
    if edge_index.ndim != 2:
        raise ValueError(f"edge_index must be 2D, got shape {tuple(edge_index.shape)}")
    if edge_index.shape[1] == 2:
        return edge_index.long()
    if edge_index.shape[0] == 2:
        return edge_index.t().contiguous().long()
    raise ValueError(
        f"edge_index must have shape (E, 2) or (2, E), got {tuple(edge_index.shape)}"
    )


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def collate_molecules(
    samples: list[dict],
    target_schema: TargetSchema = DEFAULT_TARGET_SCHEMA,
) -> GraphBatch:
    """Collate molecule samples into a nested GraphBatch.

    Each sample is a plain dict with at least ``Z`` and ``pos`` keys.
    Optional: ``edge_index``, ``bond_diff``, ``bond_dist``, ``targets``.

    Args:
        samples: List of single-molecule sample dicts.
        target_schema: Declares which targets are graph-level vs atom-level.

    Returns:
        Nested ``GraphBatch`` TensorDict.
    """
    if not samples:
        raise ValueError("Cannot collate an empty sample list")

    z_all: list[torch.Tensor] = []
    pos_all: list[torch.Tensor] = []
    batch_all: list[torch.Tensor] = []
    num_atoms: list[int] = []

    edge_all: list[torch.Tensor] = []
    diff_all: list[torch.Tensor] = []
    dist_all: list[torch.Tensor] = []

    graph_targets: dict[str, list[torch.Tensor]] = {}
    atom_targets: dict[str, list[torch.Tensor]] = {}

    atom_offset = 0

    for graph_idx, sample in enumerate(samples):
        if "Z" not in sample or "pos" not in sample:
            raise KeyError("Each sample must contain 'Z' and 'pos'")

        z = sample["Z"].long()
        pos = sample["pos"].float()
        n_atoms = int(z.shape[0])

        z_all.append(z)
        pos_all.append(pos)
        batch_all.append(
            torch.full((n_atoms,), graph_idx, dtype=torch.long, device=z.device)
        )
        num_atoms.append(n_atoms)

        if "edge_index" in sample and sample["edge_index"] is not None:
            edge_index = _normalize_edge_index(sample["edge_index"])
            edge_all.append(edge_index + atom_offset)

            if "bond_diff" in sample and sample["bond_diff"] is not None:
                diff_all.append(sample["bond_diff"].float())
            if "bond_dist" in sample and sample["bond_dist"] is not None:
                dist_all.append(sample["bond_dist"].float())

        for name, value in sample.get("targets", {}).items():
            value = value if isinstance(value, torch.Tensor) else torch.tensor(value)
            if name in target_schema.atom_level:
                atom_targets.setdefault(name, []).append(value.float())
            else:
                graph_targets.setdefault(name, []).append(value.reshape(-1).float())

        atom_offset += n_atoms

    # --- Build atom-level TensorDict ---
    atoms_dict: dict[str, torch.Tensor] = {
        "Z": torch.cat(z_all, dim=0),
        "pos": torch.cat(pos_all, dim=0),
        "batch": torch.cat(batch_all, dim=0),
    }
    for name, vals in atom_targets.items():
        atoms_dict[name] = torch.cat(vals, dim=0)

    n_total = atoms_dict["Z"].shape[0]
    atoms = AtomData(atoms_dict, batch_size=[n_total])

    # --- Build edge-level TensorDict ---
    if edge_all:
        edges_dict: dict[str, torch.Tensor] = {
            "edge_index": torch.cat(edge_all, dim=0),
        }
        if diff_all:
            edges_dict["bond_diff"] = torch.cat(diff_all, dim=0)
        if dist_all:
            edges_dict["bond_dist"] = torch.cat(dist_all, dim=0)
        e_total = edges_dict["edge_index"].shape[0]
        edges = EdgeData(edges_dict, batch_size=[e_total])
    else:
        # Empty edge data
        edges = EdgeData(
            edge_index=torch.zeros(0, 2, dtype=torch.long),
            bond_diff=torch.zeros(0, 3),
            bond_dist=torch.zeros(0),
            batch_size=[0],
        )

    # --- Build graph-level TensorDict ---
    num_graphs = len(samples)
    graphs_dict: dict[str, torch.Tensor] = {
        "num_atoms": torch.tensor(num_atoms, dtype=torch.long),
    }
    for name, vals in graph_targets.items():
        graphs_dict[name] = torch.cat(vals, dim=0)

    graphs = GraphData(graphs_dict, batch_size=[num_graphs])

    # --- Assemble top-level GraphBatch ---
    return GraphBatch(
        atoms=atoms,
        edges=edges,
        graphs=graphs,
        batch_size=[],
    )
