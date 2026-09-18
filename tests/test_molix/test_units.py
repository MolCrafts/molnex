"""Tests for molix.units — pint-style molpy.UnitSystem."""

import pytest

from molix.units import DEAD_PAIR_CUTOFF_FACTOR, units


def test_boltzmann_pint():
    kb = (1.0 * units.k_B).to("eV / kelvin")
    assert kb.magnitude == pytest.approx(8.617333262e-5, rel=1e-12)


def test_energy_bridge_pint():
    ev_per_md = (1.0 * units.eV).to("amu * angstrom ** 2 / femtosecond ** 2")
    assert ev_per_md.magnitude == pytest.approx(1.0 / 103.64269652680507, rel=1e-10)
    kcal_per_ev = (1.0 * units.eV).to(units.kilocalorie_per_mole)
    assert kcal_per_ev.magnitude == pytest.approx(23.06054783061903, rel=1e-12)


def test_dead_edge_factor_clears_any_cutoff_envelope():
    assert DEAD_PAIR_CUTOFF_FACTOR > 1.0


def test_quant_uses_pint_kb():
    from molix.quant import EffectiveTemperature

    assert EffectiveTemperature.KB_EV_PER_K == pytest.approx(
        (1.0 * units.k_B).to("eV / kelvin").magnitude
    )


class TestPresetEnergyRatio:
    """``molix.units.preset_energy_ratio(source_style, target_style) -> float``.

    Magnitude of one energy unit of LAMMPS ``source_style`` expressed in the
    energy unit of ``target_style`` (``real`` = kcal/mol, ``metal`` = eV).
    Consumed by the schema-v2 ``energy_conv`` table stamped into the LAMMPS
    ``meta.json`` (spec ``lammps-iface-01-meta``).

    Goldens are hand-computed from the 2019 SI *exactly defined* constants,
    **not** read back from the implementation:

        1 cal_th == 4.184 J        (exact, NIST SP 811 §B.8)
        N_A      == 6.02214076e23 /mol   (exact)
        e        == 1.602176634e-19 C    (exact)

        real -> metal : 4184 / (N_A * e)     = 0.043364104241800934
        metal -> real : (N_A * e) / 4184     = 23.06054783061903

    ``preset_energy_ratio`` is imported inside each test so this class is the
    only red one while the function is missing.
    """

    def test_self_ratio_is_exactly_one(self):
        from molix.units import preset_energy_ratio

        assert preset_energy_ratio("real", "real") == 1.0
        assert preset_energy_ratio("metal", "metal") == 1.0

    def test_real_to_metal_matches_si_value(self):
        from molix.units import preset_energy_ratio

        assert preset_energy_ratio("real", "metal") == pytest.approx(0.043364104241800934, rel=1e-6)

    def test_metal_to_real_matches_si_value(self):
        from molix.units import preset_energy_ratio

        # 23.0605 alone is NOT valid at rel=1e-6 (it is off by 2.1e-6 relative);
        # the golden carries enough digits of (N_A * e) / 4184 to be checkable.
        assert preset_energy_ratio("metal", "real") == pytest.approx(23.06054783061903, rel=1e-6)

    def test_round_trip_product_is_unity(self):
        from molix.units import preset_energy_ratio

        product = preset_energy_ratio("real", "metal") * preset_energy_ratio("metal", "real")
        assert product == pytest.approx(1.0, abs=1e-12)

    def test_unknown_style_raises(self):
        from molix.units import preset_energy_ratio

        # "lj" is a real LAMMPS unit style but not a molpy preset; both styles
        # resolve through ``UnitSystem.preset``, which raises ValueError.
        with pytest.raises(ValueError):
            preset_energy_ratio("lj", "metal")
        with pytest.raises(ValueError):
            preset_energy_ratio("real", "bogus")
