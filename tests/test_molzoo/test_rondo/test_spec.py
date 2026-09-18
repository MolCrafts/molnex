"""Validation the Rondo configuration performs (not its defaults)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from molzoo.rondo import RondoSpec


class TestRondoSpec:
    def test_rejects_a_torsion_series_with_no_terms(self):
        with pytest.raises(ValidationError):
            RondoSpec(n_torsion_terms=0)

    def test_rejects_a_nonpositive_width(self):
        with pytest.raises(ValidationError):
            RondoSpec(hidden=0)

    def test_rejects_an_eem_capacity_below_one(self):
        with pytest.raises(ValidationError):
            RondoSpec(eem_max_atoms=0)

    def test_rejects_a_nonpositive_cutoff(self):
        # `CoulombOpenPotential` has an unbounded 1/r sum without a radius.
        with pytest.raises(ValidationError):
            RondoSpec(cutoff=0.0)
