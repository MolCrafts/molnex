import torch

from molix.data.collate import collate_molecules
from molix.data.types import GraphBatch


def test_collate_basic_fields_and_offsets():
    sample1 = {
        "Z": torch.tensor([1, 8], dtype=torch.long),
        "pos": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "edge_index": torch.tensor([[0, 1]], dtype=torch.long),
        "bond_diff": torch.tensor([[1.0, 0.0, 0.0]]),
        "bond_dist": torch.tensor([1.0]),
        "targets": {"U0": torch.tensor([1.5])},
    }
    sample2 = {
        "Z": torch.tensor([6, 1, 1], dtype=torch.long),
        "pos": torch.tensor([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [1.0, 1.0, 0.0]]),
        "edge_index": torch.tensor([[0, 1], [0, 2]], dtype=torch.long),
        "bond_diff": torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        "bond_dist": torch.tensor([1.0, 1.0]),
        "targets": {"U0": torch.tensor([2.5])},
    }

    batch = collate_molecules([sample1, sample2])

    assert isinstance(batch, GraphBatch)
    assert batch["atoms", "Z"].shape == (5,)
    assert batch["atoms", "pos"].shape == (5, 3)
    assert batch["atoms", "batch"].tolist() == [0, 0, 1, 1, 1]
    assert batch["graphs"].batch_size[0] == 2
    assert batch["graphs", "num_atoms"].tolist() == [2, 3]

    # Canonical [E, 2] format; sample2 indices offset by +2 atoms
    assert batch["edges", "edge_index"].shape == (3, 2)
    assert batch["edges", "edge_index"][1:].tolist() == [[2, 3], [2, 4]]

    assert torch.allclose(batch["graphs", "U0"], torch.tensor([1.5, 2.5]))
