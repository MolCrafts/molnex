"""ClassIForceField / OPLSForceField inherit ForceField; no classmethod façades."""

from __future__ import annotations

from molpy import ForceField as MolpyForceField

from molpot.forcefield import ClassIForceField, ForceField, OPLSForceField


class TestForceField:
    def test_has_no_class_i_or_opls_classmethod(self):
        assert not hasattr(ForceField, "class_i")
        assert not hasattr(ForceField, "opls")


class TestClassIForceField:
    def test_is_forcefield_with_class_i_styles(self):
        ff = ClassIForceField()
        assert isinstance(ff, ForceField)
        assert isinstance(ff, MolpyForceField)
        assert ff.bond_style() == "harmonic"
        assert ff.angle_style() == "harmonic"
        assert ff.dihedral_style() == "fourier"

    def test_clone_preserves_type_and_extra_styles(self):
        ff = ClassIForceField().def_pairstyle("lj/cut", cutoff=12.0)
        cloned = ff.clone()
        assert type(cloned) is ClassIForceField
        assert cloned.dihedral_style() == "fourier"
        assert cloned.has_style("pair:lj/cut")
        assert cloned.style_options("pair:lj/cut")["cutoff"] == 12.0


class TestOPLSForceField:
    def test_is_class_i_with_opls_dihedral(self):
        ff = OPLSForceField()
        assert isinstance(ff, ClassIForceField)
        assert ff.dihedral_style() == "opls"
        assert ff.bond_style() == "harmonic"

    def test_clone_preserves_type(self):
        assert type(OPLSForceField().clone()) is OPLSForceField
