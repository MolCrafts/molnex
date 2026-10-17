"""Tests for molrep.interaction.product (irreps-string helpers)."""

from molrep.interaction.product import irreps_from_l_max, sh_irreps_from_l_max


class TestIrrepsFromLMax:
    def test_uniform_multiplicity_with_natural_parity(self):
        assert irreps_from_l_max(2, 8) == "8x0e+8x1o+8x2e"


class TestShIrrepsFromLMax:
    def test_one_copy_per_degree(self):
        assert sh_irreps_from_l_max(3) == "1x0e+1x1o+1x2e+1x3o"
