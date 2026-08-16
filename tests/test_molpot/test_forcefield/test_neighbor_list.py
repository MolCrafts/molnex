"""Tests for molpot.forcefield.neighbor_list.NeighborListBuilder."""

import torch

from molpot.forcefield.neighbor_list import NeighborListBuilder


class TestNeighborListBuilder:
    def test_prefers_batch_edge_dist(self) -> None:
        pos = torch.tensor([[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
        cell = torch.eye(3) * 10.0
        edges = {
            "edge_index": torch.tensor([[0, 1]]),
            "edge_dist": torch.tensor([1.0]),
        }
        built = NeighborListBuilder().build(
            pos=pos,
            atom_batch=torch.zeros(2, dtype=torch.long),
            cell=cell,
            cutoff=5.0,
            edges=edges,
        )
        assert built is not None
        _, dist = built
        assert float(dist[0]) == 1.0

    def test_poisoned_edge_dist_is_not_reimaged(self) -> None:
        pos = torch.tensor([[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
        edges = {
            "edge_index": torch.tensor([[0, 1]]),
            "edge_dist": torch.tensor([2.0]),
        }
        built = NeighborListBuilder().build(
            pos=pos,
            atom_batch=torch.zeros(2, dtype=torch.long),
            cell=torch.eye(3) * 10.0,
            cutoff=5.0,
            edges=edges,
        )
        assert built is not None
        assert float(built[1][0]) == 2.0

    def test_no_edges_falls_back_to_open(self) -> None:
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        built = NeighborListBuilder().build(
            pos=pos,
            atom_batch=torch.zeros(2, dtype=torch.long),
            cell=None,
            cutoff=None,
        )
        assert built is not None
        assert float(built[1][0]) == 1.0
