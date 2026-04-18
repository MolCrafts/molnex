"""Neighbor list computation task."""

from __future__ import annotations

import torch

from molix.data.task import SampleTask
from molix.F.locality import get_neighbor_pairs


def _normalize_to_E2(edge_index: torch.Tensor) -> torch.Tensor:
    """Normalise edge_index to canonical ``[E, 2]``."""
    if edge_index.ndim != 2:
        raise ValueError(f"edge_index must be 2D, got {tuple(edge_index.shape)}")
    if edge_index.shape[1] == 2:
        return edge_index.long()
    if edge_index.shape[0] == 2:
        return edge_index.t().contiguous().long()
    raise ValueError(f"edge_index shape {tuple(edge_index.shape)} is invalid")


class NeighborList(SampleTask):
    """Compute neighbor list for a single sample.

    Wraps the compiled C++ backend ``molix.F.locality.get_neighbor_pairs``,
    which internally enumerates all O(N²) candidate pairs and returns those
    within ``cutoff`` as upper-triangle (half) pairs.

    Args:
        cutoff: Cutoff distance in Angstroms.
        max_num_pairs: Buffer size passed to the C++ kernel; sized for the
            *half*-pair count.  With ``symmetry=True`` the final edge tensor
            holds up to ``2 * max_num_pairs`` rows.
        pbc: Apply periodic boundary conditions.
        check_errors: Raise if the half-pair count exceeds ``max_num_pairs``.
        filter_padding: Strip NaN-padded rows from the C++ output.
        symmetry: If ``True`` (default), add the reverse edge for every pair so
            the output is a **full bidirectional** neighbour list
            (``E = 2 * n_pairs``).  Every atom then sees its complete
            neighbourhood, as required by Allegro, MACE, and all models that
            aggregate to the source node.
            If ``False``, only the upper-triangle pairs are returned
            (``E = n_pairs``, ``edge_index[:, 0] > edge_index[:, 1]``).
    """

    def __init__(
        self,
        cutoff: float = 5.0,
        max_num_pairs: int = 512,
        pbc: bool = False,
        filter_padding: bool = True,
        symmetry: bool = True,
    ) -> None:
        self.cutoff = cutoff
        self.max_num_pairs = max_num_pairs
        self.pbc = pbc
        self.filter_padding = filter_padding
        self.symmetry = symmetry

    @property
    def task_id(self) -> str:
        return (
            f"nlist:cut={self.cutoff}:max={self.max_num_pairs}"
            f":pbc={self.pbc}:sym={self.symmetry}"
        )

    def execute(self, data: dict) -> dict:
        pos = data["pos"]
        box_vectors = data.get("cell") if self.pbc else None

        neighbors, deltas, distances, _ = get_neighbor_pairs(
            positions=pos,
            cutoff=self.cutoff,
            max_num_pairs=self.max_num_pairs,
            box_vectors=box_vectors,
        )

        edge_index = _normalize_to_E2(neighbors)

        if self.filter_padding:
            valid = ~torch.isnan(distances)
            edge_index = edge_index[valid]
            deltas = deltas[valid]
            distances = distances[valid]

        # Convention: bond_diff = pos[target] - pos[source]  (source → target).
        # The C++ kernel returns deltas = pos[rows] - pos[cols] = pos[source] - pos[target],
        # so we negate.  See CLAUDE.md "Edge Convention" for the full spec.
        bond_diff = -deltas

        if self.symmetry:
            # Add reverse edges: for each (src→tgt), append (tgt→src).
            # bond_diff reverses sign: pos[new_tgt] - pos[new_src] = -bond_diff.
            edge_index = torch.cat([edge_index, edge_index[:, [1, 0]]], dim=0)
            bond_diff = torch.cat([bond_diff, -bond_diff], dim=0)
            distances = torch.cat([distances, distances], dim=0)

        return {
            **data,
            "edge_index": edge_index,
            "bond_diff": bond_diff.float(),
            "bond_dist": distances.float(),
        }
