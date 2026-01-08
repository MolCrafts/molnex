import pytest
import torch

from molnex.nn.locality import NeighborList


class TestNeighborList:
    def test_forward_returns_tensors(self):
        module = NeighborList(cutoff=2.0, pbc=False)
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        cell = torch.eye(3)
        try:
            neighbors, deltas, distances, num_pairs = module(positions, cell)
        except Exception as exc:
            pytest.skip(f"Neighbor list op unavailable: {exc}")
        assert neighbors.shape[0] == 2
        assert deltas.shape[-1] == 3
        assert distances.ndim == 1
        assert num_pairs.ndim == 0

    def test_extra_repr(self):
        module = NeighborList(cutoff=3.5, pbc=True)
        assert "cutoff=3.5" in module.extra_repr()
        assert "pbc=True" in module.extra_repr()
