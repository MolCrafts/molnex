"""Unit tests for molrep.encoders.mace_conformer.encoder.MaceConformerEncoder."""

from __future__ import annotations

import math

import pytest
import torch
from tensordict import TensorDict

from molix.compile import Compiler
from molrep.encoders.mace_conformer import MaceConformerEncoder
from tests.test_molrep.test_encoders.test_mace_conformer.conftest import (
    TINY,
    conformers,
    encoder,
    place,
)


def _rotation(angle: float) -> torch.Tensor:
    c, s = math.cos(angle), math.sin(angle)
    x = torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=torch.float64)
    z = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    return x @ z


class TestMaceConformerEncoder:
    def test_one_row_per_atom_and_per_bond(self, butane):
        out = encoder()(butane)
        assert out.atom.shape == (14, TINY["width"])
        assert out.bond.shape == (13, TINY["bond_width"])

    def test_features_are_invariant_under_a_rigid_motion(self, butane):
        model = encoder()
        before = model(butane)
        shift = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
        moved = butane["atoms", "pos"] @ _rotation(0.7).T + shift
        after = model(place(butane, moved))
        torch.testing.assert_close(after.atom, before.atom)
        torch.testing.assert_close(after.bond, before.bond)

    def test_equivalent_atoms_and_bonds_get_equal_rows(self, butane):
        out = encoder()(butane)
        classes = butane["atoms", "equi_index"]
        for row in range(14):
            torch.testing.assert_close(out.atom[row], out.atom[classes[row]])
        # The six methyl C-H bonds are one class.
        i, j = butane["bonds", "bond_index"].tolist()
        Z = butane["atoms", "Z"].tolist()
        methyl = [b for b in range(13) if {i[b], j[b]} & {0, 3} and 1 in (Z[i[b]], Z[j[b]])]
        assert len(methyl) == 6
        for b in methyl[1:]:
            torch.testing.assert_close(out.bond[b], out.bond[methyl[0]])

    def test_every_conformer_of_a_profile_reads_the_first_ones_geometry(self):
        model = encoder()
        two = model(conformers([0, 0]))
        first = model(conformers([0]))
        torch.testing.assert_close(two.atom[:14], first.atom)
        torch.testing.assert_close(two.atom[14:], first.atom)
        torch.testing.assert_close(two.bond[13:], two.bond[:13])

    def test_conformers_of_two_profiles_read_their_own_geometry(self):
        out = encoder()(conformers([0, 1]))
        assert not torch.allclose(out.atom[14:], out.atom[:14])

    def test_an_atom_whose_reference_graph_has_another_size_is_its_own_reference(self):
        # The padder's sink: profile 1 is shared by an empty dead slot (graph 1)
        # and the sink graph 2, which holds the batch's last two atoms.
        batch = TensorDict(
            {
                "atoms": TensorDict({"batch": torch.tensor([0] * 12 + [2, 2])}, batch_size=[14]),
                "graphs": TensorDict({"profile": torch.tensor([0, 1, 1])}, batch_size=[3]),
            },
            batch_size=[],
        )
        twin = MaceConformerEncoder.reference_atoms(batch)
        assert twin.tolist() == list(range(14))

    def test_no_gradient_reaches_the_positions(self, butane):
        pos = butane["atoms", "pos"].clone().requires_grad_(True)
        out = encoder()(place(butane, pos))
        (grad,) = torch.autograd.grad(out.atom.sum() + out.bond.sum(), pos, allow_unused=True)
        assert grad is None

    @pytest.mark.slow
    def test_forward_is_one_graph(self):
        """The reference gather, the MACE stack and both readouts trace with no break."""
        model = encoder()
        batch = conformers([0, 0, 1])
        model(batch)  # spends the element-table gate eagerly, as a compiled run does first
        explanation = Compiler.explain(model, batch)
        assert explanation.graph_count == 1, explanation.break_sites
        assert explanation.graph_break_count == 0, explanation.break_sites
