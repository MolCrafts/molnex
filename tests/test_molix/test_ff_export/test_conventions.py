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
    scale_amber_vn,
    scale_angle_k,
    scale_bond_k,
    scale_energy,
    scale_length,
    scale_torsion_k,
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
        assert ANGSTROM_TO_NM == 0.1

    def test_bond_k_factor_is_418_4(self):
        # k_omm = k_ir * 4.184 / (0.1 nm)^2 = k_ir * 418.4
        assert BOND_K_IR_TO_OPENMM == pytest.approx(418.4)


class TestScaleHelpers:
    def test_bond_k_golden_100_to_41840(self):
        """ac-001: k = 100 kcal mol⁻¹ Å⁻² → 41840 kJ mol⁻¹ nm⁻²."""
        assert scale_bond_k(100.0) == 41840.0

    def test_bond_k_zero(self):
        assert scale_bond_k(0.0) == 0.0

    def test_torsion_k_unit_scale(self):
        """IR E=(k/s)[1+cos] prefactor in kcal/mol → OpenMM k in kJ/mol."""
        assert scale_torsion_k(1.0) == 4.184

    def test_amber_vn_golden_2_to_4_184(self):
        """ac-002: AMBER Vn=2 kcal/mol → OpenMM PeriodicTorsion k=4.184 kJ/mol.

        AMBER: E = (Vn/2)[1 + cos(...)]; OpenMM: E = k[1 + cos(...)].
        With Vn=2, k_kcal = 1 and k_omm = 4.184.
        """
        assert scale_amber_vn(2.0) == 4.184
        # Same path as IR coefficient after half-barrier reparameterization.
        assert scale_torsion_k(2.0 / 2.0) == 4.184

    def test_energy_and_length_scales(self):
        assert scale_energy(1.0) == 4.184
        assert scale_length(10.0) == 1.0  # 10 Å = 1 nm

    def test_angle_k_energy_only(self):
        # θ in rad both sides; only energy unit changes.
        assert scale_angle_k(10.0) == pytest.approx(41.84)


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
