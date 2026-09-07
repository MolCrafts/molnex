"""Half-pair enumeration — mirrors ``molpot/forcefield/neighbor_list.py``.

``composable-a0-02`` makes the builder **capacity-shaped**: ``build`` /
``_from_placed`` return ``(edge_index (C, 2), dist (C,), live_mask (C,))``
instead of compacting the list with ``edge_index[keep]``. Boolean-mask indexing
is a data-dependent shape and therefore a dynamo graph break on the MD force
path; the consumers multiply by ``live_mask`` instead. The open / periodic
constructions enumerate their own pairs and hand back an all-true mask.
"""

from __future__ import annotations

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
        _, dist, _live = built
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

    def test_placed_list_is_returned_at_capacity_with_a_live_mask(self) -> None:
        """Padding survives in place; only ``live_mask`` says which rows count.

        Live is exactly ``source < target`` (the half list), ``0 < r < cutoff``
        and inside ``num_edges``. Row 1 is the reversed duplicate of row 0,
        rows 2-3 are outside the cutoff, rows 4-5 are dead capacity — none may
        be dropped, because dropping them makes the output shape depend on the
        data.
        """
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        edge_index = torch.tensor(
            [[0, 1], [1, 0], [0, 2], [2, 0], [0, 0], [0, 0]], dtype=torch.long
        )
        edges = {
            "edge_index": edge_index,
            "edge_dist": torch.tensor([1.0, 1.0, 3.0, 3.0, 0.0, 0.0]),
            "num_edges": 4,
        }
        built = NeighborListBuilder().build(
            pos=pos,
            atom_batch=torch.zeros(3, dtype=torch.long),
            cell=None,
            cutoff=2.5,
            edges=edges,
        )
        assert built is not None
        returned_index, dist, live_mask = built
        assert returned_index.shape == (6, 2)
        assert dist.shape == (6,)
        assert live_mask.dtype == torch.bool
        assert torch.equal(returned_index, edge_index)
        assert torch.equal(live_mask, torch.tensor([True, False, False, False, False, False]))

    def test_open_enumeration_marks_every_pair_it_produced_live(self) -> None:
        """``_open`` filters while it enumerates, so its mask is all-true."""
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        built = NeighborListBuilder().build(
            pos=pos,
            atom_batch=torch.zeros(3, dtype=torch.long),
            cell=None,
            cutoff=1.5,
        )
        assert built is not None
        edge_index, dist, live_mask = built
        assert live_mask.shape == (edge_index.shape[0],)
        assert dist.shape == (edge_index.shape[0],)
        assert bool(live_mask.all())
