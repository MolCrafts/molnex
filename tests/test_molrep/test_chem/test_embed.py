"""Tests for AtomChemEmbedding and BondChemEmbedding."""

from __future__ import annotations

import torch

from molrep.chem.embed import AtomChemEmbedding, BondChemEmbedding
from molrep.embedding.node import JointEmbedding


class TestAtomChemEmbedding:
    """AtomChemEmbedding maps Z → (N, D_a)."""

    def test_shape_z_only(self, atom_dim: int):
        emb = AtomChemEmbedding(atom_dim=atom_dim, num_elements=20)
        z = torch.tensor([1, 6, 8, 1], dtype=torch.long)
        out = emb(z)
        assert out.shape == (4, atom_dim)

    def test_reuses_joint_embedding(self, atom_dim: int):
        emb = AtomChemEmbedding(atom_dim=atom_dim, num_elements=20)
        assert isinstance(emb.joint, JointEmbedding)

    def test_empty_atoms(self, atom_dim: int):
        emb = AtomChemEmbedding(atom_dim=atom_dim, num_elements=20)
        z = torch.zeros(0, dtype=torch.long)
        out = emb(z)
        assert out.shape == (0, atom_dim)


class TestBondChemEmbedding:
    """BondChemEmbedding is endpoint-symmetric: h_ij == h_ji."""

    def test_shape(self, atom_dim: int, bond_dim: int):
        atom_emb = AtomChemEmbedding(atom_dim=atom_dim, num_elements=20)
        bond_emb = BondChemEmbedding(atom_dim=atom_dim, bond_dim=bond_dim)
        z = torch.tensor([6, 8, 1], dtype=torch.long)
        h = atom_emb(z)
        atomi = torch.tensor([0, 0], dtype=torch.long)
        atomj = torch.tensor([1, 2], dtype=torch.long)
        out = bond_emb(h, atomi, atomj)
        assert out.shape == (2, bond_dim)

    def test_endpoint_symmetry(self, atom_dim: int, bond_dim: int):
        torch.manual_seed(0)
        atom_emb = AtomChemEmbedding(atom_dim=atom_dim, num_elements=20)
        bond_emb = BondChemEmbedding(atom_dim=atom_dim, bond_dim=bond_dim)
        z = torch.tensor([6, 8, 1, 7], dtype=torch.long)
        h = atom_emb(z)
        atomi = torch.tensor([0, 1, 2], dtype=torch.long)
        atomj = torch.tensor([1, 2, 3], dtype=torch.long)
        forward = bond_emb(h, atomi, atomj)
        reverse = bond_emb(h, atomj, atomi)
        assert torch.allclose(forward, reverse, rtol=1e-5, atol=1e-6)

    def test_empty_bonds(self, atom_dim: int, bond_dim: int):
        bond_emb = BondChemEmbedding(atom_dim=atom_dim, bond_dim=bond_dim)
        h = torch.zeros(3, atom_dim)
        empty = torch.zeros(0, dtype=torch.long)
        out = bond_emb(h, empty, empty)
        assert out.shape == (0, bond_dim)
