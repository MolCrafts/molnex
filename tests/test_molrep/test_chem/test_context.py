"""Symmetry tests for valence interaction context builders."""

from __future__ import annotations

import torch

from molrep.chem.context import (
    AngleContext,
    BondContext,
    ImproperContext,
    ProperContext,
)
from molrep.chem.embed import AtomChemEmbedding


class TestBondContext:
    def test_delegates_to_bond_embedding_shape(self):
        atom_dim, bond_dim = 8, 6
        atom_emb = AtomChemEmbedding(atom_dim=atom_dim, num_elements=20)
        ctx = BondContext(atom_dim=atom_dim, bond_dim=bond_dim)
        h = atom_emb(torch.tensor([1, 6, 8], dtype=torch.long))
        atomi = torch.tensor([0, 1], dtype=torch.long)
        atomj = torch.tensor([1, 2], dtype=torch.long)
        out = ctx(h, atomi, atomj)
        assert out.shape == (2, bond_dim)

    def test_reverse_symmetry(self):
        torch.manual_seed(1)
        atom_dim, bond_dim = 8, 6
        atom_emb = AtomChemEmbedding(atom_dim=atom_dim, num_elements=20)
        ctx = BondContext(atom_dim=atom_dim, bond_dim=bond_dim)
        h = atom_emb(torch.tensor([1, 6, 8, 7], dtype=torch.long))
        atomi = torch.tensor([0, 1], dtype=torch.long)
        atomj = torch.tensor([2, 3], dtype=torch.long)
        assert torch.allclose(
            ctx(h, atomi, atomj),
            ctx(h, atomj, atomi),
            rtol=1e-5,
            atol=1e-6,
        )


class TestAngleContext:
    def test_shape(self):
        atom_dim, angle_dim = 8, 6
        atom_emb = AtomChemEmbedding(atom_dim=atom_dim, num_elements=20)
        ctx = AngleContext(atom_dim=atom_dim, angle_dim=angle_dim)
        h = atom_emb(torch.tensor([1, 6, 8], dtype=torch.long))
        atomi = torch.tensor([0], dtype=torch.long)
        atomj = torch.tensor([1], dtype=torch.long)
        atomk = torch.tensor([2], dtype=torch.long)
        out = ctx(h, atomi, atomj, atomk)
        assert out.shape == (1, angle_dim)

    def test_reverse_symmetry_ijk_to_kji(self):
        """Angle context invariant under (i,j,k) → (k,j,i)."""
        torch.manual_seed(2)
        atom_dim, angle_dim = 8, 6
        atom_emb = AtomChemEmbedding(atom_dim=atom_dim, num_elements=20)
        ctx = AngleContext(atom_dim=atom_dim, angle_dim=angle_dim)
        h = atom_emb(torch.tensor([1, 6, 8, 7, 16], dtype=torch.long))
        atomi = torch.tensor([0, 1], dtype=torch.long)
        atomj = torch.tensor([2, 2], dtype=torch.long)
        atomk = torch.tensor([3, 4], dtype=torch.long)
        forward = ctx(h, atomi, atomj, atomk)
        reverse = ctx(h, atomk, atomj, atomi)
        assert torch.allclose(forward, reverse, rtol=1e-5, atol=1e-6)


class TestProperContext:
    def test_shape(self):
        atom_dim, proper_dim = 8, 6
        atom_emb = AtomChemEmbedding(atom_dim=atom_dim, num_elements=20)
        ctx = ProperContext(atom_dim=atom_dim, proper_dim=proper_dim)
        h = atom_emb(torch.tensor([1, 6, 8, 7], dtype=torch.long))
        out = ctx(
            h,
            torch.tensor([0], dtype=torch.long),
            torch.tensor([1], dtype=torch.long),
            torch.tensor([2], dtype=torch.long),
            torch.tensor([3], dtype=torch.long),
        )
        assert out.shape == (1, proper_dim)

    def test_reverse_symmetry_ijkl_to_lkji(self):
        """Proper context invariant under (i,j,k,l) → (l,k,j,i)."""
        torch.manual_seed(3)
        atom_dim, proper_dim = 8, 6
        atom_emb = AtomChemEmbedding(atom_dim=atom_dim, num_elements=20)
        ctx = ProperContext(atom_dim=atom_dim, proper_dim=proper_dim)
        h = atom_emb(torch.tensor([1, 6, 8, 7, 16, 15], dtype=torch.long))
        atomi = torch.tensor([0, 1], dtype=torch.long)
        atomj = torch.tensor([1, 2], dtype=torch.long)
        atomk = torch.tensor([2, 3], dtype=torch.long)
        atoml = torch.tensor([3, 4], dtype=torch.long)
        forward = ctx(h, atomi, atomj, atomk, atoml)
        reverse = ctx(h, atoml, atomk, atomj, atomi)
        assert torch.allclose(forward, reverse, rtol=1e-5, atol=1e-6)


class TestImproperContext:
    def test_shape(self):
        atom_dim, improper_dim = 8, 6
        atom_emb = AtomChemEmbedding(atom_dim=atom_dim, num_elements=20)
        ctx = ImproperContext(atom_dim=atom_dim, improper_dim=improper_dim)
        h = atom_emb(torch.tensor([6, 1, 1, 8], dtype=torch.long))
        # center = atomi = 0
        out = ctx(
            h,
            torch.tensor([0], dtype=torch.long),
            torch.tensor([1], dtype=torch.long),
            torch.tensor([2], dtype=torch.long),
            torch.tensor([3], dtype=torch.long),
        )
        assert out.shape == (1, improper_dim)

    def test_outer_swap_center_fixed(self):
        """Improper invariant under outer-leg swap; center (atomi) fixed."""
        torch.manual_seed(4)
        atom_dim, improper_dim = 8, 6
        atom_emb = AtomChemEmbedding(atom_dim=atom_dim, num_elements=20)
        ctx = ImproperContext(atom_dim=atom_dim, improper_dim=improper_dim)
        h = atom_emb(torch.tensor([6, 1, 7, 8, 16], dtype=torch.long))
        center = torch.tensor([0, 0], dtype=torch.long)
        j = torch.tensor([1, 1], dtype=torch.long)
        k = torch.tensor([2, 2], dtype=torch.long)
        l = torch.tensor([3, 4], dtype=torch.long)
        base = ctx(h, center, j, k, l)
        # Swap outer j ↔ k
        swapped_jk = ctx(h, center, k, j, l)
        # Swap outer k ↔ l
        swapped_kl = ctx(h, center, j, l, k)
        # Swap outer j ↔ l
        swapped_jl = ctx(h, center, l, k, j)
        assert torch.allclose(base, swapped_jk, rtol=1e-5, atol=1e-6)
        assert torch.allclose(base, swapped_kl, rtol=1e-5, atol=1e-6)
        assert torch.allclose(base, swapped_jl, rtol=1e-5, atol=1e-6)

    def test_empty(self):
        ctx = ImproperContext(atom_dim=8, improper_dim=6)
        h = torch.zeros(3, 8)
        empty = torch.zeros(0, dtype=torch.long)
        out = ctx(h, empty, empty, empty, empty)
        assert out.shape == (0, 6)
