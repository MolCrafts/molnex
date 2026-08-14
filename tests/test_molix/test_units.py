"""Tests for molix.units — pint-style molpy.UnitSystem."""

import pytest
from molpy import UnitSystem

from molix.units import DEAD_EDGE_CUTOFF_FACTOR, units


def test_units_is_molpy_registry():
    assert isinstance(units, UnitSystem)


def test_boltzmann_pint():
    kb = (1.0 * units.k_B).to("eV / kelvin")
    assert kb.magnitude == pytest.approx(8.617333262e-5, rel=1e-12)


def test_energy_bridge_pint():
    ev_per_md = (1.0 * units.eV).to("amu * angstrom ** 2 / femtosecond ** 2")
    assert ev_per_md.magnitude == pytest.approx(1.0 / 103.64269652680507, rel=1e-10)
    kcal_per_ev = (1.0 * units.eV).to(units.kilocalorie_per_mole)
    assert kcal_per_ev.magnitude == pytest.approx(23.06054783061903, rel=1e-12)


def test_dead_edge_factor_clears_any_cutoff_envelope():
    assert DEAD_EDGE_CUTOFF_FACTOR > 1.0


def test_quant_uses_pint_kb():
    from molix.quant import EffectiveTemperature

    assert EffectiveTemperature.KB_EV_PER_K == pytest.approx(
        (1.0 * units.k_B).to("eV / kelvin").magnitude
    )
