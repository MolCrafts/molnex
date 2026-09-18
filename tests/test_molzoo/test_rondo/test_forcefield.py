"""The Class-I form Rondo fits, and the radius it prices."""

from __future__ import annotations

import pytest

from molpot.forcefield import ClassIForceField
from molzoo.rondo import RondoForceField


class TestRondoForceField:
    def test_declares_the_full_class_i_form(self):
        names = set(RondoForceField().style_names())
        assert {
            "bond:harmonic",
            "angle:harmonic",
            "dihedral:periodic",
            "improper:periodic",
            "pair:lj/cut",
            "pair:coulomb",
        } <= names

    def test_is_a_class_i_field_not_a_vendor_dialect(self):
        # OPLS is a manifold of `dihedral:periodic`, not a separate style: a
        # field born with `dihedral:opls` cannot fit an AMBER torsion.
        ff = RondoForceField()
        assert isinstance(ff, ClassIForceField)
        assert not ff.has_style("dihedral:opls")

    def test_owns_the_radius_it_prices(self):
        # `neighbor_list()` takes no radius: the field states it, so the list
        # a batch carries and the pairs the model charges for are one fact
        # read twice (ledger finding F21).
        assert RondoForceField(cutoff=12.0).neighbor_list().cutoff == pytest.approx(12.0)

    def test_excludes_one_two_and_one_three_and_scales_one_four(self):
        ff = RondoForceField(scale_14_charge=0.5, scale_14_lj=0.5)
        assert list(ff.special_bonds_coul) == pytest.approx([0.0, 0.0, 0.5])
        assert list(ff.special_bonds_lj) == pytest.approx([0.0, 0.0, 0.5])

    def test_each_construction_is_an_independent_field(self):
        # `Parameterizer` fills a copy; two fits must not share tables.
        assert RondoForceField() is not RondoForceField()
