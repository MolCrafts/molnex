import torch
from tensordict import TensorDict

from molix.data.collate import INDEX_KEYS, collate_molecules, rebase


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

    assert isinstance(batch, TensorDict)
    assert batch["atoms", "Z"].shape == (5,)
    assert batch["atoms", "pos"].shape == (5, 3)
    assert batch["atoms", "batch"].tolist() == [0, 0, 1, 1, 1]
    assert batch["graphs"].batch_size[0] == 2
    assert batch["graphs", "num_atoms"].tolist() == [2, 3]

    # Canonical [E, 2] format; sample2 indices offset by +2 atoms
    assert batch["edges", "edge_index"].shape == (3, 2)
    assert batch["edges", "edge_index"][1:].tolist() == [[2, 3], [2, 4]]

    assert torch.allclose(batch["graphs", "U0"], torch.tensor([1.5, 2.5]))


def test_index_keys_registry_contract():
    # Pre-declared registry that sub-specs 02/03 import as a stable contract.
    assert INDEX_KEYS["edge_index"] == (0, 1)
    assert INDEX_KEYS["bond_index"] == (1, 0)
    assert INDEX_KEYS["angle_index"] == (1, 0)
    assert INDEX_KEYS["dihedral_index"] == (1, 0)


def test_rebase_scalar_offset_edge_index():
    # collate_molecules path: scalar atom_offset added to the whole [E, 2] tensor.
    edge_index = torch.tensor([[0, 1], [0, 2]], dtype=torch.long)
    out = rebase(edge_index, 2, "edge_index")
    assert out.tolist() == [[2, 3], [2, 4]]


def test_rebase_vector_offset_edge_index_matches_unsqueeze1():
    # collate_packed path: per-edge segment offset broadcast over both columns.
    edge_index = torch.tensor([[0, 1], [0, 2], [1, 0]], dtype=torch.long)
    seg_offset = torch.tensor([0, 0, 5], dtype=torch.long)
    out = rebase(edge_index, seg_offset, "edge_index")
    assert torch.equal(out, edge_index + seg_offset.unsqueeze(1))
    assert out.tolist() == [[0, 1], [0, 2], [6, 5]]


def test_rebase_generalizes_to_cat_dim1_mock_bond_index():
    # Proves the differing-axis path before phase 03 produces a real bond_index:
    # a [2, N] COO tensor with a per-count (per-bond) offset must broadcast over
    # the rows (index_axis=0), i.e. offset.unsqueeze(0).
    bond_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    per_bond_offset = torch.tensor([10, 10, 20], dtype=torch.long)
    out = rebase(bond_index, per_bond_offset, "bond_index")
    assert torch.equal(out, bond_index + per_bond_offset.unsqueeze(0))
    assert out.tolist() == [[10, 11, 22], [11, 12, 20]]
