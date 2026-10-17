"""Tests for GTOCoulombKernel."""

from __future__ import annotations

import math

import pytest
import torch

from molpot.potentials.gto.kernel import GTOCoulombKernel


class TestGTOCoulombKernel:
    """Official Polar kernel is erf(r/(2σ))/r, not paper Eq.37."""

    def test_matches_official_erf_over_two_sigma(self):
        """Dump σ=1.5 Å, r=1 Å → erf(1/3)."""
        k = GTOCoulombKernel(sigma=1.5)
        r = torch.tensor([1.0], dtype=torch.float64)
        got = k(r)
        expect = math.erf(1.0 / 3.0)
        assert got.item() == pytest.approx(expect, rel=1e-12)

    def test_is_not_paper_eq37(self):
        """Paper Eq.37 erf(r/√2 σ)/r disagrees with the official dump kernel."""
        k = GTOCoulombKernel(sigma=1.5)
        r = torch.tensor([1.0], dtype=torch.float64)
        paper = math.erf(1.0 / (math.sqrt(2.0) * 1.5))
        assert k(r).item() != pytest.approx(paper, rel=1e-6)

    def test_rejects_non_positive_sigma(self):
        """σ is a GTO width in Å."""
        with pytest.raises(ValueError, match="sigma"):
            GTOCoulombKernel(sigma=0.0)
