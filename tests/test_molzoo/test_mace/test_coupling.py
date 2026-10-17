"""Unit tests for :mod:`molzoo.mace.coupling`."""

from __future__ import annotations

import itertools

import cuequivariance as cue
import numpy as np
import pytest

from molzoo.mace.coupling import MaceO3, real_coupling

L_MAX = 4


def _triples():
    for l1, l2 in itertools.product(range(L_MAX + 1), repeat=2):
        for l3 in range(abs(l1 - l2), l1 + l2 + 1):
            yield l1, l2, l3


def _sign_against_cue(l1: int, l2: int, l3: int) -> int:
    p1, p2 = 1, 1
    ours = MaceO3.clebsch_gordan(MaceO3(l=l1, p=p1), MaceO3(l=l2, p=p2), MaceO3(l=l3, p=p1 * p2))
    ref = cue.O3.clebsch_gordan(cue.O3(l1, p1), cue.O3(l2, p2), cue.O3(l3, p1 * p2))
    if np.abs(ours - ref).max() < 1e-12:
        return 1
    if np.abs(ours + ref).max() < 1e-12:
        return -1
    return 0


class TestMaceO3:
    @pytest.mark.parametrize("triple", list(_triples()))
    def test_coupling_is_cue_up_to_a_sign(self, triple):
        """Same real basis as ``cue.O3``; only the phase of a coupling may differ."""
        assert _sign_against_cue(*triple) != 0

    def test_conventions_agree_below_l2_and_split_above(self):
        """Every coupling among ``l <= 1`` matches ``cue.O3``; ``(1, 2, 2)`` is negated.

        That split is why MACE checkpoints whose products stop at ``1o``
        loaded under ``cue.O3`` while MACE-POLAR-1-L, whose product emits
        ``2e``, did not.
        """
        assert all(_sign_against_cue(*t) == 1 for t in _triples() if max(t) <= 1)
        assert _sign_against_cue(1, 2, 2) == -1

    @pytest.mark.parametrize("triple", [t for t in _triples() if t[0] <= t[1] <= t[2]])
    def test_sorted_coupling_has_a_positive_pivot(self, triple):
        """Sorted degrees: the all-``m = 0`` entry, else the first non-zero entry, is positive."""
        l1, l2, l3 = triple
        c = real_coupling(l1, l2, l3)
        pivot = c[l1, l2, l3] if c[l1, l2, l3] != 0 else c.ravel()[np.flatnonzero(c)[0]]
        assert pivot > 0
        assert np.linalg.norm(c) == pytest.approx(1.0)

    def test_swapping_inputs_is_a_transpose_with_the_parity_sign(self):
        """``C(l2, l1, l3)`` is ``C(l1, l2, l3)`` with the input axes swapped, times ``(-1)^Σl``."""
        for l1, l2, l3 in _triples():
            swapped = np.transpose(real_coupling(l2, l1, l3), (1, 0, 2))
            np.testing.assert_allclose(
                swapped, (-1) ** (l1 + l2 + l3) * real_coupling(l1, l2, l3), atol=1e-12
            )
