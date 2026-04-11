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

    Wraps the compiled C++ backend ``molix.F.locality.get_neighbor_pairs``.
    """

    def __init__(
        self,
        cutoff: float = 5.0,
        max_num_pairs: int = 512,
        pbc: bool = False,
        check_errors: bool = True,
        filter_padding: bool = True,
    ) -> None:
        self.cutoff = cutoff
        self.max_num_pairs = max_num_pairs
        self.pbc = pbc
        self.check_errors = check_errors
        self.filter_padding = filter_padding

    @property
    def task_id(self) -> str:
        return f"nlist:cut={self.cutoff}:max={self.max_num_pairs}:pbc={self.pbc}"

    def execute(self, data: dict) -> dict:
        pos = data["pos"]
        box_vectors = data.get("cell") if self.pbc else None

        neighbors, deltas, distances, _ = get_neighbor_pairs(
            positions=pos,
            cutoff=self.cutoff,
            max_num_pairs=self.max_num_pairs,
            box_vectors=box_vectors,
            check_errors=self.check_errors,
        )

        edge_index = _normalize_to_E2(neighbors)

        if self.filter_padding:
            valid = ~torch.isnan(distances)
            edge_index = edge_index[valid]
            deltas = deltas[valid]
            distances = distances[valid]

        return {
            **data,
            "edge_index": edge_index,
            "bond_diff": deltas.float(),
            "bond_dist": distances.float(),
        }
