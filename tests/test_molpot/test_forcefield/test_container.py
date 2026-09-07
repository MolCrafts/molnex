"""OPLSAAForceField inherits ForceField; product templates are named subclasses."""

from __future__ import annotations

from molpy import ForceField as MolpyForceField

from molpot.forcefield import ForceField


class TestOPLSAAForceField:
    def test_default_styles_are_oplsaa_set(self):
        from molpot.forcefield import OPLSAAForceField

        ff = OPLSAAForceField()
        assert isinstance(ff, ForceField)
        assert isinstance(ff, MolpyForceField)
        for key in (
            "bond:harmonic",
            "angle:harmonic",
            "dihedral:opls",
            "improper:periodic",
            "pair:lj/cut",
            "pair:coulomb",
        ):
            assert ff.has_style(key), key

    def test_default_nonbonded_scaling_f14_is_half(self):
        from molpot.forcefield import OPLSAAForceField

        scaling = OPLSAAForceField().nonbonded_scaling
        assert scaling.q.s14 == 0.5
        assert scaling.lj.s14 == 0.5
