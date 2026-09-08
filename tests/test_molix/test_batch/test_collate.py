import torch
from tensordict import TensorDict

from molix.batch.collate import INDEX_KEYS, MoleculeCollator, collate_molecules, rebase
from tests.conftest import frame


def test_collate_basic_fields_and_offsets():
    """Two frames become one batch: atoms concatenate, graphs stack, no edges."""
    batch = collate_molecules(
        [
            frame(Z=[1, 8], pos=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], U0=1.5),
            frame(
                Z=[6, 1, 1],
                pos=[[0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [1.0, 1.0, 0.0]],
                U0=2.5,
            ),
        ]
    )

    assert isinstance(batch, TensorDict)
    assert batch["atoms", "Z"].shape == (5,)
    assert batch["atoms", "pos"].shape == (5, 3)
    assert batch["atoms", "batch"].tolist() == [0, 0, 1, 1, 1]
    assert batch["graphs"].batch_size[0] == 2
    assert batch["graphs", "num_atoms"].tolist() == [2, 3]

    # Frames carry no neighbour graph: NeighborList builds it on the batch.
    assert batch["edges", "edge_index"].shape == (0, 2)

    assert torch.allclose(batch["graphs", "U0"], torch.tensor([1.5, 2.5]))


def test_molecule_collator_is_reusable_callable():
    frames = [
        frame(Z=[1], pos=[[0.0, 0.0, 0.0]], energy=1.0),
        frame(Z=[6, 1], pos=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], energy=2.0),
    ]
    batch = MoleculeCollator()(frames)
    assert batch["atoms", "batch"].tolist() == [0, 1, 1]
    assert batch["graphs", "energy"].tolist() == [1.0, 2.0]


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
    # packed path: per-edge segment offset broadcast over both columns.
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


def test_integer_meta_scalars_stay_integral():
    """A task code must survive collation as an integer.

    ``TaskRouter.route`` refuses a floating task label rather than truncate one
    silently, so casting every ``meta`` scalar to float would break multi-task
    routing for every frame-fed batch.
    """
    batch = collate_molecules(
        [
            frame(Z=[8, 1], pos=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], energy=-1.5, task=3),
            frame(Z=[6, 1], pos=[[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]], energy=-2.5, task=3),
        ]
    )
    assert not batch["graphs", "task"].is_floating_point()
    assert batch["graphs", "task"].tolist() == [3, 3]
    assert batch["graphs", "energy"].is_floating_point()
