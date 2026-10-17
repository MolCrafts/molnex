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

    def test_h_bond_output_shape(self):
        torch.manual_seed(1)
        atom_dim, bond_dim = 8, 6
        atom_emb = AtomChemEmbedding(atom_dim=atom_dim, num_elements=20)
        ctx = BondContext(atom_dim=atom_dim, bond_dim=bond_dim)
        h = atom_emb(torch.tensor([1, 6, 8], dtype=torch.long))
        atomi = torch.tensor([0, 1], dtype=torch.long)
        atomj = torch.tensor([1, 2], dtype=torch.long)
        prev = ctx(h, atomi, atomj)
        out = ctx(h, atomi, atomj, h_bond=prev)
        assert out.shape == (2, bond_dim)

    def test_h_bond_reverse_symmetry(self):
        torch.manual_seed(1)
        atom_dim, bond_dim = 8, 6
        atom_emb = AtomChemEmbedding(atom_dim=atom_dim, num_elements=20)
        ctx = BondContext(atom_dim=atom_dim, bond_dim=bond_dim)
        h = atom_emb(torch.tensor([1, 6, 8, 7], dtype=torch.long))
        atomi = torch.tensor([0, 1], dtype=torch.long)
        atomj = torch.tensor([2, 3], dtype=torch.long)
        prev = ctx(h, atomi, atomj)
        assert torch.allclose(
            ctx(h, atomi, atomj, h_bond=prev),
            ctx(h, atomj, atomi, h_bond=prev),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_h_bond_forwards_to_embedding(self):
        torch.manual_seed(1)
        atom_dim, bond_dim = 8, 6
        atom_emb = AtomChemEmbedding(atom_dim=atom_dim, num_elements=20)
        ctx = BondContext(atom_dim=atom_dim, bond_dim=bond_dim)
        h = atom_emb(torch.tensor([1, 6, 8], dtype=torch.long))
        atomi = torch.tensor([0, 1], dtype=torch.long)
        atomj = torch.tensor([1, 2], dtype=torch.long)
        prev = ctx(h, atomi, atomj)
        via_ctx = ctx(h, atomi, atomj, h_bond=prev)
        via_embed = ctx.embed(h, atomi, atomj, h_bond=prev)
        assert torch.equal(via_ctx, via_embed)


class TestAngleContext:
    def test_shape(self):
        atom_dim, bond_dim, angle_dim = 8, 5, 6
        atom_emb = AtomChemEmbedding(atom_dim=atom_dim, num_elements=20)
        ctx = AngleContext(atom_dim=atom_dim, bond_dim=bond_dim, angle_dim=angle_dim)
        h = atom_emb(torch.tensor([1, 6, 8], dtype=torch.long))
        atomi = torch.tensor([0], dtype=torch.long)
        atomj = torch.tensor([1], dtype=torch.long)
        atomk = torch.tensor([2], dtype=torch.long)
        out = ctx(h, atomi, atomj, atomk, torch.randn(1, bond_dim), torch.randn(1, bond_dim))
        assert out.shape == (1, angle_dim)

    def test_reverse_symmetry_ijk_to_kji(self):
        """Angle context invariant under (i,j,k) → (k,j,i) with the bonds swapped along."""
        torch.manual_seed(2)
        atom_dim, bond_dim, angle_dim = 8, 5, 6
        atom_emb = AtomChemEmbedding(atom_dim=atom_dim, num_elements=20)
        ctx = AngleContext(atom_dim=atom_dim, bond_dim=bond_dim, angle_dim=angle_dim)
        h = atom_emb(torch.tensor([1, 6, 8, 7, 16], dtype=torch.long))
        atomi = torch.tensor([0, 1], dtype=torch.long)
        atomj = torch.tensor([2, 2], dtype=torch.long)
        atomk = torch.tensor([3, 4], dtype=torch.long)
        h_ij, h_jk = torch.randn(2, bond_dim), torch.randn(2, bond_dim)
        forward = ctx(h, atomi, atomj, atomk, h_ij, h_jk)
        reverse = ctx(h, atomk, atomj, atomi, h_jk, h_ij)
        assert torch.allclose(forward, reverse, rtol=1e-5, atol=1e-6)

    def test_the_bonds_of_the_angle_are_inputs(self):
        """Same three atoms, a different bond feature: a different angle feature."""
        torch.manual_seed(5)
        atom_dim, bond_dim, angle_dim = 8, 5, 6
        atom_emb = AtomChemEmbedding(atom_dim=atom_dim, num_elements=20)
        ctx = AngleContext(atom_dim=atom_dim, bond_dim=bond_dim, angle_dim=angle_dim)
        h = atom_emb(torch.tensor([1, 6, 8], dtype=torch.long))
        i, j, k = (torch.tensor([n], dtype=torch.long) for n in (0, 1, 2))
        h_jk = torch.randn(1, bond_dim)
        single = ctx(h, i, j, k, torch.zeros(1, bond_dim), h_jk)
        double = ctx(h, i, j, k, torch.ones(1, bond_dim), h_jk)
        assert not torch.allclose(single, double)


class TestProperContext:
    def test_shape(self):
        atom_dim, bond_dim, proper_dim = 8, 5, 6
        atom_emb = AtomChemEmbedding(atom_dim=atom_dim, num_elements=20)
        ctx = ProperContext(atom_dim=atom_dim, bond_dim=bond_dim, proper_dim=proper_dim)
        h = atom_emb(torch.tensor([1, 6, 8, 7], dtype=torch.long))
        out = ctx(
            h,
            torch.tensor([0], dtype=torch.long),
            torch.tensor([1], dtype=torch.long),
            torch.tensor([2], dtype=torch.long),
            torch.tensor([3], dtype=torch.long),
            torch.randn(1, bond_dim),
            torch.randn(1, bond_dim),
            torch.randn(1, bond_dim),
        )
        assert out.shape == (1, proper_dim)

    def test_reverse_symmetry_ijkl_to_lkji(self):
        """Proper context invariant under (i,j,k,l) → (l,k,j,i) with the bonds swapped along."""
        torch.manual_seed(3)
        atom_dim, bond_dim, proper_dim = 8, 5, 6
        atom_emb = AtomChemEmbedding(atom_dim=atom_dim, num_elements=20)
        ctx = ProperContext(atom_dim=atom_dim, bond_dim=bond_dim, proper_dim=proper_dim)
        h = atom_emb(torch.tensor([1, 6, 8, 7, 16, 15], dtype=torch.long))
        atomi = torch.tensor([0, 1], dtype=torch.long)
        atomj = torch.tensor([1, 2], dtype=torch.long)
        atomk = torch.tensor([2, 3], dtype=torch.long)
        atoml = torch.tensor([3, 4], dtype=torch.long)
        h_ij, h_jk, h_kl = (torch.randn(2, bond_dim) for _ in range(3))
        forward = ctx(h, atomi, atomj, atomk, atoml, h_ij, h_jk, h_kl)
        reverse = ctx(h, atoml, atomk, atomj, atomi, h_kl, h_jk, h_ij)
        assert torch.allclose(forward, reverse, rtol=1e-5, atol=1e-6)

    def test_the_central_bond_is_an_input(self):
        """Same four atoms, a different central bond feature: a different torsion feature."""
        torch.manual_seed(6)
        atom_dim, bond_dim, proper_dim = 8, 5, 6
        atom_emb = AtomChemEmbedding(atom_dim=atom_dim, num_elements=20)
        ctx = ProperContext(atom_dim=atom_dim, bond_dim=bond_dim, proper_dim=proper_dim)
        h = atom_emb(torch.tensor([1, 6, 6, 1], dtype=torch.long))
        i, j, k, l = (torch.tensor([n], dtype=torch.long) for n in range(4))
        h_ij, h_kl = torch.randn(1, bond_dim), torch.randn(1, bond_dim)
        single = ctx(h, i, j, k, l, h_ij, torch.zeros(1, bond_dim), h_kl)
        double = ctx(h, i, j, k, l, h_ij, torch.ones(1, bond_dim), h_kl)
        assert not torch.allclose(single, double)


class TestImproperContext:
    def test_shape(self):
        atom_dim, bond_dim, improper_dim = 8, 5, 6
        atom_emb = AtomChemEmbedding(atom_dim=atom_dim, num_elements=20)
        ctx = ImproperContext(atom_dim=atom_dim, bond_dim=bond_dim, improper_dim=improper_dim)
        h = atom_emb(torch.tensor([6, 1, 1, 8], dtype=torch.long))
        # center = atomi = 0
        out = ctx(
            h,
            torch.tensor([0], dtype=torch.long),
            torch.tensor([1], dtype=torch.long),
            torch.tensor([2], dtype=torch.long),
            torch.tensor([3], dtype=torch.long),
            torch.randn(1, bond_dim),
            torch.randn(1, bond_dim),
            torch.randn(1, bond_dim),
        )
        assert out.shape == (1, improper_dim)

    def test_outer_swap_center_fixed(self):
        """Improper invariant under outer-leg swap, each leg carrying its bond; center fixed."""
        torch.manual_seed(4)
        atom_dim, bond_dim, improper_dim = 8, 5, 6
        atom_emb = AtomChemEmbedding(atom_dim=atom_dim, num_elements=20)
        ctx = ImproperContext(atom_dim=atom_dim, bond_dim=bond_dim, improper_dim=improper_dim)
        h = atom_emb(torch.tensor([6, 1, 7, 8, 16], dtype=torch.long))
        center = torch.tensor([0, 0], dtype=torch.long)
        j = torch.tensor([1, 1], dtype=torch.long)
        k = torch.tensor([2, 2], dtype=torch.long)
        l = torch.tensor([3, 4], dtype=torch.long)
        h_cj, h_ck, h_cl = (torch.randn(2, bond_dim) for _ in range(3))
        base = ctx(h, center, j, k, l, h_cj, h_ck, h_cl)
        swapped_jk = ctx(h, center, k, j, l, h_ck, h_cj, h_cl)
        swapped_kl = ctx(h, center, j, l, k, h_cj, h_cl, h_ck)
        swapped_jl = ctx(h, center, l, k, j, h_cl, h_ck, h_cj)
        assert torch.allclose(base, swapped_jk, rtol=1e-5, atol=1e-6)
        assert torch.allclose(base, swapped_kl, rtol=1e-5, atol=1e-6)
        assert torch.allclose(base, swapped_jl, rtol=1e-5, atol=1e-6)

    def test_a_leg_keeps_its_own_bond(self):
        """Moving a bond feature from one leg to another changes the feature: legs are paired."""
        torch.manual_seed(7)
        atom_dim, bond_dim, improper_dim = 8, 5, 6
        atom_emb = AtomChemEmbedding(atom_dim=atom_dim, num_elements=20)
        ctx = ImproperContext(atom_dim=atom_dim, bond_dim=bond_dim, improper_dim=improper_dim)
        h = atom_emb(torch.tensor([6, 1, 7, 8], dtype=torch.long))
        c, j, k, l = (torch.tensor([n], dtype=torch.long) for n in range(4))
        double, single = torch.ones(1, bond_dim), torch.zeros(1, bond_dim)
        assert not torch.allclose(
            ctx(h, c, j, k, l, double, single, single), ctx(h, c, j, k, l, single, double, single)
        )

    def test_empty(self):
        ctx = ImproperContext(atom_dim=8, bond_dim=5, improper_dim=6)
        h = torch.zeros(3, 8)
        empty = torch.zeros(0, dtype=torch.long)
        out = ctx(h, empty, empty, empty, empty, *(torch.zeros(0, 5) for _ in range(3)))
        assert out.shape == (0, 6)
