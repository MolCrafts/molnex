"""Unit tests for TranslationCase + ConventionTable unit goldens.

Spec: learnable-classical-ff-08-ff-export.
Hard-coded goldens only — no live OpenMM.
"""

from __future__ import annotations

import math

import pytest

from molix.ff_export.cases import TranslationCase
from molix.ff_export.conventions import (
    ANGSTROM_TO_NM,
    BOND_K_IR_TO_OPENMM,
    KCAL_PER_MOL_TO_KJ_PER_MOL,
    ConventionTable,
)


class TestTranslationCase:
    def test_four_way_members_exist(self):
        names = {c.name for c in TranslationCase}
        assert names == {
            "DIRECT_UNIT_SCALE",
            "FORM_REPARAMETERIZE",
            "DECOMPOSE",
            "UNSUPPORTED",
        }

    def test_values_are_stable_strings(self):
        assert TranslationCase.DIRECT_UNIT_SCALE.value == "direct_unit_scale"
        assert TranslationCase.FORM_REPARAMETERIZE.value == "form_reparameterize"
        assert TranslationCase.DECOMPOSE.value == "decompose"
        assert TranslationCase.UNSUPPORTED.value == "unsupported"


class TestUnitConstants:
    def test_kcal_to_kj(self):
        assert KCAL_PER_MOL_TO_KJ_PER_MOL == 4.184

    def test_angstrom_to_nm(self):
        assert ANGSTROM_TO_NM == pytest.approx(0.1)

    def test_bond_k_factor_is_418_4(self):
        assert BOND_K_IR_TO_OPENMM == pytest.approx(418.4)


class TestConventionTableScale:
    def test_bond_k_golden_100_to_41840(self):
        """k = 100 kcal mol⁻¹ Å⁻² → 41840 kJ mol⁻¹ nm⁻²."""
        table = ConventionTable.default_openmm()
        assert table.scale("bond_harmonic", "k", 100.0) == 41840.0

    def test_bond_k_zero(self):
        table = ConventionTable.default_openmm()
        assert table.scale("bond_harmonic", "k", 0.0) == 0.0

    def test_torsion_k_unit_scale(self):
        table = ConventionTable.default_openmm()
        assert table.scale("proper_periodic", "k", 1.0) == 4.184

    def test_amber_vn_golden_2_to_4_184(self):
        """AMBER Vn=2 kcal/mol → OpenMM PeriodicTorsion k=4.184 kJ/mol."""
        table = ConventionTable.default_openmm()
        assert table.scale_amber_vn(2.0) == 4.184
        assert table.scale("proper_periodic", "k", 2.0 / 2.0) == 4.184

    def test_energy_and_length_scales(self):
        table = ConventionTable.default_openmm()
        assert table.scale("lj", "epsilon", 1.0) == 4.184
        assert table.scale("lj", "sigma", 10.0) == pytest.approx(1.0)

    def test_angle_k_energy_only(self):
        table = ConventionTable.default_openmm()
        assert table.scale("angle_harmonic", "k", 10.0) == pytest.approx(41.84)

    def test_unknown_field_raises(self):
        row = ConventionTable.default_openmm()["bond_harmonic"]
        with pytest.raises(KeyError, match="no unit factor"):
            row.scale("not_a_field", 1.0)


class TestConventionTable:
    def test_default_table_has_core_terms(self):
        table = ConventionTable.default_openmm()
        for term in (
            "bond_harmonic",
            "angle_harmonic",
            "proper_periodic",
            "lj",
            "charge",
            "improper_harmonic",
        ):
            assert term in table

    def test_bond_is_direct_unit_scale(self):
        row = ConventionTable.default_openmm()["bond_harmonic"]
        assert row.case is TranslationCase.DIRECT_UNIT_SCALE
        assert math.isclose(row.unit_factors["k"], BOND_K_IR_TO_OPENMM)

    def test_proper_is_form_reparameterize(self):
        row = ConventionTable.default_openmm()["proper_periodic"]
        assert row.case is TranslationCase.FORM_REPARAMETERIZE

    def test_improper_harmonic_unsupported(self):
        row = ConventionTable.default_openmm()["improper_harmonic"]
        assert row.case is TranslationCase.UNSUPPORTED

    def test_lookup_unknown_raises(self):
        table = ConventionTable.default_openmm()
        with pytest.raises(KeyError):
            table.row("not_a_term")
