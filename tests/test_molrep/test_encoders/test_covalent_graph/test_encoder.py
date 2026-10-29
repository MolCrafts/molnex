"""Unit tests for molrep.encoders.covalent_graph.encoder.CovalentGraphEncoder."""

from __future__ import annotations

import torch

from molrep.encoders.covalent_graph import CovalentGraphEncoder, CovalentGraphEncoderSpec
from tests.test_molrep.test_encoders.test_covalent_graph.conftest import BONDS, DTYPE, TINY, build


def _encoder() -> CovalentGraphEncoder:
    torch.manual_seed(0)
    return CovalentGraphEncoder(CovalentGraphEncoderSpec(**TINY), dtype=DTYPE).eval()


class TestCovalentGraphEncoder:
    def test_one_row_per_atom_and_per_bond(self, acetate):
        out = _encoder()(acetate)
        assert out.atom.shape == (7, TINY["width"]) and out.bond.shape == (6, TINY["edge_mlp"][1])

    def test_resonance_equivalent_atoms_get_equal_features(self, acetate):
        # The two carboxylate O differ in bond order and formal charge as written;
        # their equivalence class makes them equal, and their bonds too.
        out = _encoder()(acetate)
        torch.testing.assert_close(out.atom[2], out.atom[3])
        torch.testing.assert_close(out.bond[1], out.bond[2])

    def test_dead_bond_rows_of_a_padded_batch_leave_the_real_rows_unchanged(self, acetate):
        # As ``MoleculePadder`` pads: two ghost atoms (their own classes), and ten
        # dead bond rows all joining them, ``bond_type`` 0 and ``mask`` False — more
        # than any degree or bond order the embeddings index.
        padded = build(
            bonds=BONDS + [(7, 8)] * 10,
            z=[6, 6, 8, 8, 1, 1, 1, 1, 1],
            equi=[0, 1, 2, 2, 4, 4, 4, 7, 8],
            bond_type=(1, 2, 1, 1, 1, 1) + (0,) * 10,
            charge=(0, 0, 0, -1, 0, 0, 0, 0, 0),
        )
        padded["bonds", "mask"] = torch.tensor([True] * 6 + [False] * 10)
        encoder = _encoder()
        real, out = encoder(acetate), encoder(padded)
        torch.testing.assert_close(out.atom[:7], real.atom)
        torch.testing.assert_close(out.bond[:6], real.bond)

    def test_reversing_a_bond_changes_nothing(self, acetate):
        flipped = build(bonds=[(j, i) for i, j in BONDS])
        encoder = _encoder()
        one, two = encoder(acetate), encoder(flipped)
        torch.testing.assert_close(one.atom, two.atom)
        torch.testing.assert_close(one.bond, two.bond)

    def test_features_follow_a_permutation_of_the_atoms(self, acetate):
        perm = [6, 2, 0, 4, 3, 1, 5]  # new atom k is old atom perm[k]
        inverse = {old: new for new, old in enumerate(perm)}
        z = [acetate["atoms", "Z"][p].item() for p in perm]
        charge = [acetate["atoms", "formal_charge"][p].item() for p in perm]
        equi_old = acetate["atoms", "equi_index"].tolist()
        first = {}
        equi = [first.setdefault(equi_old[p], k) for k, p in enumerate(perm)]
        bonds = [(inverse[i], inverse[j]) for i, j in BONDS]
        permuted = build(bonds=bonds, z=z, equi=equi, charge=charge)
        encoder = _encoder()
        one, two = encoder(acetate), encoder(permuted)
        torch.testing.assert_close(two.atom, one.atom[perm])
        torch.testing.assert_close(two.bond, one.bond)

    def test_positions_are_never_read(self, acetate):
        out = _encoder()(acetate)
        acetate["atoms", "pos"] = torch.randn(7, 3, dtype=DTYPE)
        torch.testing.assert_close(_encoder()(acetate).atom, out.atom)
