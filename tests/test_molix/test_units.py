"""Tests for molix.units — the single source of physical constants."""

from molix.units import DEAD_EDGE_CUTOFF_FACTOR, EV_PER_AMU_A2_FS2, KB_AMU_A_FS, KB_EV_PER_K


def test_boltzmann_constant_is_codata():
    assert KB_EV_PER_K == 8.617333262e-5


def test_kb_amu_a_fs_is_derived_not_restated():
    """The (amu, Å, fs) k_B must be the eV/K value through the unit bridge."""
    assert KB_AMU_A_FS == KB_EV_PER_K / EV_PER_AMU_A2_FS2


def test_dead_edge_factor_clears_any_cutoff_envelope():
    """A dead edge must land strictly outside the cutoff, with margin."""
    assert DEAD_EDGE_CUTOFF_FACTOR > 1.0


def test_quant_shares_the_single_source():
    """molix.quant's class attribute must alias molix.units, not restate it."""
    from molix.quant import EffectiveTemperature

    assert EffectiveTemperature.KB_EV_PER_K == KB_EV_PER_K
