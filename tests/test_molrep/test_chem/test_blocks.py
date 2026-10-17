"""Unit tests for ChemGraphBlock (one covalent-graph hop)."""

from __future__ import annotations

import inspect

import pytest
import torch

from molrep.chem.blocks import ChemGraphBlock
from molrep.chem.embed import AtomChemEmbedding


class TestChemGraphBlock:
    """ChemGraphBlock updates ``(h_atom, h_bond)`` on the undirected covalent graph."""

    def test_forward_preserves_shapes(self):
        torch.manual_seed(0)
        atom_dim, bond_dim = 8, 6
        block = ChemGraphBlock(atom_dim=atom_dim, bond_dim=bond_dim)
        h_atom = torch.randn(4, atom_dim)
        h_bond = torch.randn(3, bond_dim)
        atomi = torch.tensor([0, 0, 0], dtype=torch.long)
        atomj = torch.tensor([1, 2, 3], dtype=torch.long)
        h_atom_out, h_bond_out = block(h_atom, h_bond, atomi, atomj)
        assert h_atom_out.shape == (4, atom_dim)
        assert h_bond_out.shape == (3, bond_dim)

    def test_reverse_endpoints_bond_features_equal(self):
        torch.manual_seed(1)
        atom_dim, bond_dim = 8, 6
        block = ChemGraphBlock(atom_dim=atom_dim, bond_dim=bond_dim)
        block.eval()
        h_atom = torch.randn(4, atom_dim)
        h_bond = torch.randn(3, bond_dim)
        atomi = torch.tensor([0, 0, 0], dtype=torch.long)
        atomj = torch.tensor([1, 2, 3], dtype=torch.long)
        _, h_fwd = block(h_atom, h_bond, atomi, atomj)
        _, h_rev = block(h_atom, h_bond, atomj, atomi)
        assert torch.allclose(h_fwd, h_rev, rtol=1e-5, atol=1e-6)

    def test_one_block_locality_non_neighbor_unchanged(self):
        """One hop: changing a non-adjacent atom's Z leaves target ``h_i`` unchanged."""
        torch.manual_seed(2)
        atom_dim, bond_dim = 8, 8
        atom_emb = AtomChemEmbedding(atom_dim=atom_dim, num_elements=20)
        block = ChemGraphBlock(atom_dim=atom_dim, bond_dim=bond_dim)
        block.eval()
        # Star 1-0-2 / 0-3: atom 1's only neighbor is 0; atom 2 is not adjacent to 1.
        z = torch.tensor([8, 1, 1, 6], dtype=torch.long)
        h_atom = atom_emb(z)
        h_bond = torch.zeros(3, bond_dim)
        atomi = torch.tensor([0, 0, 0], dtype=torch.long)
        atomj = torch.tensor([1, 2, 3], dtype=torch.long)
        out0, _ = block(h_atom, h_bond, atomi, atomj)
        z_pert = z.clone()
        z_pert[2] = 7
        out1, _ = block(atom_emb(z_pert), h_bond, atomi, atomj)
        assert torch.allclose(out0[1], out1[1], rtol=1e-5, atol=1e-6)

    def test_one_block_neighbor_change_affects_atom(self):
        torch.manual_seed(2)
        atom_dim, bond_dim = 8, 8
        atom_emb = AtomChemEmbedding(atom_dim=atom_dim, num_elements=20)
        block = ChemGraphBlock(atom_dim=atom_dim, bond_dim=bond_dim)
        block.eval()
        z = torch.tensor([8, 1, 1, 6], dtype=torch.long)
        h_bond = torch.zeros(3, bond_dim)
        atomi = torch.tensor([0, 0, 0], dtype=torch.long)
        atomj = torch.tensor([1, 2, 3], dtype=torch.long)
        out0, _ = block(atom_emb(z), h_bond, atomi, atomj)
        z_pert = z.clone()
        z_pert[0] = 7
        out1, _ = block(atom_emb(z_pert), h_bond, atomi, atomj)
        assert not torch.allclose(out0[1], out1[1], rtol=1e-5, atol=1e-6)

    def test_local_attention_normalizes_only_covalent_neighborhood(self):
        """``local_attention=True`` must not softmax over all E bonds."""
        torch.manual_seed(3)
        atom_dim, bond_dim = 8, 8
        atom_emb = AtomChemEmbedding(atom_dim=atom_dim, num_elements=20)
        block = ChemGraphBlock(atom_dim=atom_dim, bond_dim=bond_dim, local_attention=True)
        block.eval()
        # Disjoint bonds 0-1 and 2-3: a global softmax would leak component {2,3} into {0,1}.
        z = torch.tensor([8, 1, 6, 1], dtype=torch.long)
        h_atom = atom_emb(z)
        h_bond = torch.zeros(2, bond_dim)
        atomi = torch.tensor([0, 2], dtype=torch.long)
        atomj = torch.tensor([1, 3], dtype=torch.long)
        out0, _ = block(h_atom, h_bond, atomi, atomj)
        z_pert = z.clone()
        z_pert[2] = 7
        z_pert[3] = 16
        h_bond_pert = h_bond.clone()
        h_bond_pert[1] = h_bond_pert[1] + 1.0
        out1, _ = block(atom_emb(z_pert), h_bond_pert, atomi, atomj)
        assert torch.allclose(out0[0], out1[0], rtol=1e-5, atol=1e-6)
        assert torch.allclose(out0[1], out1[1], rtol=1e-5, atol=1e-6)

    def test_local_attention_defaults_false(self):
        params = inspect.signature(ChemGraphBlock.__init__).parameters
        assert params["local_attention"].default is False

    def test_constructor_has_no_pos_parameter(self):
        params = inspect.signature(ChemGraphBlock.__init__).parameters
        assert "pos" not in params
        with pytest.raises(TypeError):
            ChemGraphBlock(atom_dim=8, bond_dim=8, pos=torch.zeros(4, 3))

    def test_empty_bonds_preserves_shapes(self):
        torch.manual_seed(4)
        atom_dim, bond_dim = 8, 6
        block = ChemGraphBlock(atom_dim=atom_dim, bond_dim=bond_dim)
        h_atom = torch.randn(3, atom_dim)
        h_bond = torch.zeros(0, bond_dim)
        empty = torch.zeros(0, dtype=torch.long)
        h_atom_out, h_bond_out = block(h_atom, h_bond, empty, empty)
        assert h_atom_out.shape == (3, atom_dim)
        assert h_bond_out.shape == (0, bond_dim)

    def test_update_does_not_depend_on_the_scale_of_the_streams(self):
        """Pre-norm: a hop's update is a function of the normalized streams only.

        Scaling ``h_atom`` and ``h_bond`` by 1000 leaves ``out − h_atom`` unchanged,
        so the residual stream cannot compound across hops.
        """
        torch.manual_seed(3)
        block = ChemGraphBlock(atom_dim=8, bond_dim=8).double()
        h_atom = torch.randn(4, 8, dtype=torch.float64)
        h_bond = torch.randn(3, 8, dtype=torch.float64)
        atomi = torch.tensor([0, 0, 0], dtype=torch.long)
        atomj = torch.tensor([1, 2, 3], dtype=torch.long)
        out, bond = block(h_atom, h_bond, atomi, atomj)
        out_big, bond_big = block(1e3 * h_atom, 1e3 * h_bond, atomi, atomj)
        # Not exact: LayerNorm's eps (1e-5) is felt by the unit-scale input only.
        assert torch.allclose(out - h_atom, out_big - 1e3 * h_atom, atol=1e-4)
        assert torch.allclose(bond, bond_big, atol=1e-4)
