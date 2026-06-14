"""Parity test: ``EwaldMultipoleEnergy.lr_from_k_sq`` vs the analytic kernel.

The reciprocal-space kernel is ``kfac(k) = exp(-σ²k²/2) / k²``. Verify that
the production-side method reproduces the analytic form to ≤1e-6 over a
range of k² values.
"""

from __future__ import annotations

import math

import pytest
import torch

from molpot.potentials import EwaldMultipoleEnergy


@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("dl", [1.0, 2.0])
def test_lr_from_k_sq_matches_analytic(sigma: float, dl: float) -> None:
    pot = EwaldMultipoleEnergy(sigma=sigma, dl=dl)
    # Sample k² at five points spanning the cutoff range.
    k_sq_max = (2.0 * math.pi / dl) ** 2
    k_sq = torch.linspace(0.05 * k_sq_max, 0.95 * k_sq_max, 5, dtype=torch.float64)

    actual = pot.lr_from_k_sq(k_sq)
    expected = torch.exp(-(sigma**2) * k_sq / 2.0) / k_sq

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
