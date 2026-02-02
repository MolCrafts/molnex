import torch

from molix.data import AtomTD
from molix.data.preprocess import NeighborListPreprocessor


def _compute_neighbor_list_naive(positions: torch.Tensor, cutoff: float):
    diff = positions.unsqueeze(0) - positions.unsqueeze(1)
    dist = torch.norm(diff, dim=2)
    mask = (dist < cutoff) & (dist > 0)
    i_idx, j_idx = torch.where(mask)
    valid = i_idx > j_idx
    i_idx = i_idx[valid]
    j_idx = j_idx[valid]
    edge_vec = positions[i_idx] - positions[j_idx]
    edge_dist = dist[i_idx, j_idx]
    return i_idx, j_idx, edge_vec, edge_dist


class TestNeighborListPreprocessor:
    def test_small_system_pairs(self):
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        frame = AtomTD(
            {
                ("atoms", "Z"): torch.tensor([8, 1, 1]),
                ("atoms", "xyz"): positions,
                ("graph", "batch"): torch.zeros(3, dtype=torch.long),
            },
            batch_size=[],
        )
        preprocessor = NeighborListPreprocessor(cutoff=2.0, max_num_pairs=10)
        result = preprocessor.preprocess([frame])[0]

        op_i = result["pairs", "i"]
        op_j = result["pairs", "j"]
        op_vec = result["pairs", "diff"]
        op_dist = result["pairs", "dist"]

        valid_mask = (op_i >= 0) & (~torch.isnan(op_dist))
        op_i = op_i[valid_mask]
        op_j = op_j[valid_mask]
        op_vec = op_vec[valid_mask]
        op_dist = op_dist[valid_mask]

        ref_i, ref_j, ref_vec, ref_dist = _compute_neighbor_list_naive(positions, 2.0)

        assert op_i.shape == ref_i.shape
        assert torch.allclose(op_dist.sort().values, ref_dist.sort().values, atol=1e-5)
        assert torch.allclose(op_vec.norm(dim=1).sort().values, ref_vec.norm(dim=1).sort().values, atol=1e-5)
        assert op_i.numel() == ref_i.numel()

    def test_cutoff_behavior(self):
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        frame = AtomTD(
            {
                ("atoms", "Z"): torch.tensor([1, 1, 1]),
                ("atoms", "xyz"): positions,
                ("graph", "batch"): torch.zeros(3, dtype=torch.long),
            },
            batch_size=[],
        )
        preprocessor = NeighborListPreprocessor(cutoff=2.0, max_num_pairs=10)
        result = preprocessor.preprocess([frame])[0]
        op_dist = result["pairs", "dist"]
        valid_mask = ~torch.isnan(op_dist)
        assert torch.all(op_dist[valid_mask] <= 2.0)
