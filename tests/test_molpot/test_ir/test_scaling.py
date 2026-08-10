"""NonbondedScaling Class-I defaults (AMBER/GAFF / SMIRNOFF).

Provenance: OpenMM §19 / SMIRNOFF nonbonded section / Cornell 1995 AMBER.
1–2 and 1–3 interactions are fully excluded (scale 0); 1–4 electrostatics
are scaled by 5/6 and 1–4 LJ by 1/2.
"""

import math

from molpot.ir import NonbondedScaling


class TestNonbondedScaling:
    def test_class_i_default_scale_q_12_is_zero(self):
        scaling = NonbondedScaling()
        assert float(scaling.scale_q_12) == 0.0

    def test_class_i_default_scale_q_13_is_zero(self):
        scaling = NonbondedScaling()
        assert float(scaling.scale_q_13) == 0.0

    def test_class_i_default_scale_q_14_is_five_sixths(self):
        scaling = NonbondedScaling()
        assert math.isclose(float(scaling.scale_q_14), 5.0 / 6.0, rel_tol=1e-12, abs_tol=1e-12)

    def test_class_i_default_scale_lj_12_is_zero(self):
        scaling = NonbondedScaling()
        assert float(scaling.scale_lj_12) == 0.0

    def test_class_i_default_scale_lj_13_is_zero(self):
        scaling = NonbondedScaling()
        assert float(scaling.scale_lj_13) == 0.0

    def test_class_i_default_scale_lj_14_is_half(self):
        scaling = NonbondedScaling()
        assert math.isclose(float(scaling.scale_lj_14), 0.5, rel_tol=1e-12, abs_tol=1e-12)
