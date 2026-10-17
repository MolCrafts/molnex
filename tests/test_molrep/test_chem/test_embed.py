"""Tests for AtomChemEmbedding and BondChemEmbedding."""

from __future__ import annotations

import pytest
import torch

from molrep.chem.embed import (
    AtomChemEmbedding,
    BondChemEmbedding,
    DiscreteChannel,
    DiscreteChannels,
)
from molrep.embedding.node import JointEmbedding

#: cuEquivariance fixes its precision at construction: build at fp64.
DTYPE = torch.float64


class TestAtomChemEmbedding:
    """AtomChemEmbedding maps Z → (N, D_a)."""

    def test_shape_z_only(self, atom_dim: int):
        emb = AtomChemEmbedding(atom_dim=atom_dim, num_elements=20, dtype=DTYPE)
        z = torch.tensor([1, 6, 8, 1], dtype=torch.long)
        out = emb(z)
        assert out.shape == (4, atom_dim)

    def test_reuses_joint_embedding(self, atom_dim: int):
        emb = AtomChemEmbedding(atom_dim=atom_dim, num_elements=20, dtype=DTYPE)
        assert isinstance(emb.joint, JointEmbedding)

    def test_empty_atoms(self, atom_dim: int):
        emb = AtomChemEmbedding(atom_dim=atom_dim, num_elements=20, dtype=DTYPE)
        z = torch.zeros(0, dtype=torch.long)
        out = emb(z)
        assert out.shape == (0, atom_dim)

    def test_empty_atoms_take_the_modules_dtype_not_the_global_config(self, atom_dim: int):
        """The empty answer is this module's, so converting the module moves it.

        It used to be built from a process-wide dtype, which is wrong twice:
        a config attribute read is a graph break on a traced path (that config
        object resolved through a custom ``__getattribute__`` dynamo cannot trace), and
        for a module converted away from the global dtype the caller would
        concatenate a float32 ``(0, D)`` with float64 features.
        """
        emb = AtomChemEmbedding(atom_dim=atom_dim, num_elements=20, dtype=DTYPE).double()
        empty = emb(torch.zeros(0, dtype=torch.long))
        populated = emb(torch.tensor([1, 6], dtype=torch.long))
        assert empty.dtype == populated.dtype == torch.float64


class TestBondChemEmbedding:
    """BondChemEmbedding is endpoint-symmetric: h_ij == h_ji."""

    def test_shape(self, atom_dim: int, bond_dim: int):
        atom_emb = AtomChemEmbedding(atom_dim=atom_dim, num_elements=20, dtype=DTYPE)
        bond_emb = BondChemEmbedding(atom_dim=atom_dim, bond_dim=bond_dim, dtype=DTYPE)
        z = torch.tensor([6, 8, 1], dtype=torch.long)
        h = atom_emb(z)
        atomi = torch.tensor([0, 0], dtype=torch.long)
        atomj = torch.tensor([1, 2], dtype=torch.long)
        out = bond_emb(h, atomi, atomj)
        assert out.shape == (2, bond_dim)

    def test_endpoint_symmetry(self, atom_dim: int, bond_dim: int):
        torch.manual_seed(0)
        atom_emb = AtomChemEmbedding(atom_dim=atom_dim, num_elements=20, dtype=DTYPE)
        bond_emb = BondChemEmbedding(atom_dim=atom_dim, bond_dim=bond_dim, dtype=DTYPE)
        z = torch.tensor([6, 8, 1, 7], dtype=torch.long)
        h = atom_emb(z)
        atomi = torch.tensor([0, 1, 2], dtype=torch.long)
        atomj = torch.tensor([1, 2, 3], dtype=torch.long)
        forward = bond_emb(h, atomi, atomj)
        reverse = bond_emb(h, atomj, atomi)
        assert torch.allclose(forward, reverse, rtol=1e-5, atol=1e-6)

    def test_empty_bonds(self, atom_dim: int, bond_dim: int):
        bond_emb = BondChemEmbedding(atom_dim=atom_dim, bond_dim=bond_dim, dtype=DTYPE)
        h = torch.zeros(3, atom_dim)
        empty = torch.zeros(0, dtype=torch.long)
        out = bond_emb(h, empty, empty)
        assert out.shape == (0, bond_dim)

    def test_explicit_h_bond_none_matches_seed(self, atom_dim: int, bond_dim: int):
        torch.manual_seed(0)
        atom_emb = AtomChemEmbedding(atom_dim=atom_dim, num_elements=20, dtype=DTYPE)
        bond_emb = BondChemEmbedding(atom_dim=atom_dim, bond_dim=bond_dim, dtype=DTYPE)
        h = atom_emb(torch.tensor([6, 8, 1], dtype=torch.long))
        atomi = torch.tensor([0, 0], dtype=torch.long)
        atomj = torch.tensor([1, 2], dtype=torch.long)
        seed = bond_emb(h, atomi, atomj)
        explicit_none = bond_emb(h, atomi, atomj, h_bond=None)
        assert torch.allclose(seed, explicit_none, rtol=1e-5, atol=1e-6)

    def test_h_bond_output_shape(self, atom_dim: int, bond_dim: int):
        torch.manual_seed(0)
        atom_emb = AtomChemEmbedding(atom_dim=atom_dim, num_elements=20, dtype=DTYPE)
        bond_emb = BondChemEmbedding(atom_dim=atom_dim, bond_dim=bond_dim, dtype=DTYPE)
        h = atom_emb(torch.tensor([6, 8, 1], dtype=torch.long))
        atomi = torch.tensor([0, 0], dtype=torch.long)
        atomj = torch.tensor([1, 2], dtype=torch.long)
        prev = bond_emb(h, atomi, atomj)
        out = bond_emb(h, atomi, atomj, h_bond=prev)
        assert out.shape == (2, bond_dim)

    def test_h_bond_endpoint_symmetry(self, atom_dim: int, bond_dim: int):
        torch.manual_seed(0)
        atom_emb = AtomChemEmbedding(atom_dim=atom_dim, num_elements=20, dtype=DTYPE)
        bond_emb = BondChemEmbedding(atom_dim=atom_dim, bond_dim=bond_dim, dtype=DTYPE)
        h = atom_emb(torch.tensor([6, 8, 1, 7], dtype=torch.long))
        atomi = torch.tensor([0, 1, 2], dtype=torch.long)
        atomj = torch.tensor([1, 2, 3], dtype=torch.long)
        prev = bond_emb(h, atomi, atomj)
        forward = bond_emb(h, atomi, atomj, h_bond=prev)
        reverse = bond_emb(h, atomj, atomi, h_bond=prev)
        assert torch.allclose(forward, reverse, rtol=1e-5, atol=1e-6)

    def test_h_bond_changes_output(self, atom_dim: int, bond_dim: int):
        torch.manual_seed(0)
        atom_emb = AtomChemEmbedding(atom_dim=atom_dim, num_elements=20, dtype=DTYPE)
        bond_emb = BondChemEmbedding(atom_dim=atom_dim, bond_dim=bond_dim, dtype=DTYPE)
        h = atom_emb(torch.tensor([6, 8, 1], dtype=torch.long))
        atomi = torch.tensor([0, 0], dtype=torch.long)
        atomj = torch.tensor([1, 2], dtype=torch.long)
        seed = bond_emb(h, atomi, atomj)
        other = torch.randn(seed.shape[0], seed.shape[1], dtype=seed.dtype)
        mixed = bond_emb(h, atomi, atomj, h_bond=other)
        assert not torch.allclose(mixed, seed, rtol=1e-5, atol=1e-6)

    def test_h_bond_empty_bonds(self, atom_dim: int, bond_dim: int):
        bond_emb = BondChemEmbedding(atom_dim=atom_dim, bond_dim=bond_dim, dtype=DTYPE)
        h = torch.zeros(3, atom_dim)
        empty = torch.zeros(0, dtype=torch.long)
        prev = torch.zeros(0, bond_dim)
        out = bond_emb(h, empty, empty, h_bond=prev)
        assert out.shape == (0, bond_dim)


class TestDiscreteChannels:
    """One embedding table per declared integer column, summed into a stream."""

    def test_no_channels_add_nothing(self):
        assert DiscreteChannels((), dim=4, dtype=DTYPE)() is None

    def test_a_per_graph_column_is_broadcast_by_the_batch_index(self):
        torch.manual_seed(0)
        channels = DiscreteChannels(
            (DiscreteChannel(name="total_charge", num_classes=11, offset=5, per="graph"),),
            dim=4,
            dtype=DTYPE,
        )
        total_charge = torch.tensor([-1.0, 2.0])
        batch = torch.tensor([0, 0, 1])
        out = channels(batch=batch, total_charge=total_charge)
        table = channels.tables["total_charge"].weight
        torch.testing.assert_close(out, table[torch.tensor([4, 4, 7])])

    def test_the_offset_maps_the_most_negative_value_to_row_zero(self):
        channels = DiscreteChannels(
            (DiscreteChannel(name="formal_charge", num_classes=5, offset=2),), dim=3, dtype=DTYPE
        )
        out = channels(formal_charge=torch.tensor([-2]))
        torch.testing.assert_close(out[0], channels.tables["formal_charge"].weight[0])

    def test_an_out_of_range_value_clamps_to_the_edge_class(self):
        channels = DiscreteChannels(
            (DiscreteChannel(name="min_ring_size", num_classes=9),), dim=3, dtype=DTYPE
        )
        out = channels(min_ring_size=torch.tensor([12]))
        torch.testing.assert_close(out[0], channels.tables["min_ring_size"].weight[8])

    def test_two_channels_are_summed(self):
        torch.manual_seed(0)
        channels = DiscreteChannels(
            (
                DiscreteChannel(name="a", num_classes=3),
                DiscreteChannel(name="b", num_classes=3),
            ),
            dim=4,
            dtype=DTYPE,
        )
        out = channels(a=torch.tensor([1]), b=torch.tensor([2]))
        want = channels.tables["a"].weight[1] + channels.tables["b"].weight[2]
        torch.testing.assert_close(out[0], want)

    def test_a_duplicate_channel_name_is_refused(self):
        with pytest.raises(ValueError, match="unique"):
            DiscreteChannels(
                (
                    DiscreteChannel(name="a", num_classes=3),
                    DiscreteChannel(name="a", num_classes=3),
                ),
                dim=4,
                dtype=DTYPE,
            )


class TestEmbeddingExtra:
    """The channel sum is added to the seed, on atoms and on bonds alike."""

    def test_atom_extra_is_added_to_the_seed(self, atom_dim: int):
        emb = AtomChemEmbedding(atom_dim=atom_dim, num_elements=20, dtype=DTYPE)
        z = torch.tensor([6, 8], dtype=torch.long)
        extra = torch.randn(2, atom_dim)
        torch.testing.assert_close(emb(z, extra=extra), emb(z) + extra)

    def test_bond_extra_keeps_endpoint_symmetry(self, atom_dim: int, bond_dim: int):
        torch.manual_seed(0)
        atom_emb = AtomChemEmbedding(atom_dim=atom_dim, num_elements=20, dtype=DTYPE)
        bond_emb = BondChemEmbedding(atom_dim=atom_dim, bond_dim=bond_dim, dtype=DTYPE)
        h = atom_emb(torch.tensor([6, 8, 1], dtype=torch.long))
        atomi = torch.tensor([0, 0], dtype=torch.long)
        atomj = torch.tensor([1, 2], dtype=torch.long)
        extra = torch.randn(2, bond_dim)
        forward = bond_emb(h, atomi, atomj, extra=extra)
        reverse = bond_emb(h, atomj, atomi, extra=extra)
        torch.testing.assert_close(forward, reverse)
        torch.testing.assert_close(forward, bond_emb(h, atomi, atomj) + extra)
