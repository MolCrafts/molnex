"""Parity test: ``EwaldMultipoleEnergy.from_dist`` vs the brute-force oracle.

Asserts ≤1e-6 (float64) agreement on every kernel tensor
(``f_qq, f_qu, f_uu, f_Qu, f_QQ``) for a fixed-seed random configuration.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from molpot.potentials import EwaldMultipoleEnergy
from tests._oracles.screened_coulomb import make_kernels as oracle_make_kernels


@pytest.fixture(scope="module")
def random_positions() -> np.ndarray:
    """Fixed-seed random positions for the kernel-equality fixtures."""
    rng = np.random.default_rng(0)
    return rng.uniform(-5.0, 5.0, size=(8, 3))


@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
def test_from_dist_matches_oracle(random_positions: np.ndarray, sigma: float) -> None:
    """Production ``from_dist`` must reproduce the oracle's five kernels exactly."""
    prefactor = 90.4756
    norm_const = prefactor / (2.0 * np.pi)
    pot = EwaldMultipoleEnergy(sigma=sigma, prefactor=prefactor)

    r_torch = torch.from_numpy(random_positions).to(torch.float64)
    r_ij_torch = r_torch.unsqueeze(0) - r_torch.unsqueeze(1)
    out = pot.from_dist(r_ij_torch)

    ref = oracle_make_kernels(random_positions, sigma=sigma, norm_const=norm_const)

    for key in ("f_qq", "f_qu", "f_uu", "f_Qu", "f_QQ"):
        actual = out[key].detach().cpu().numpy()
        torch.testing.assert_close(
            torch.from_numpy(actual),
            torch.from_numpy(ref[key]),
            atol=1e-6,
            rtol=1e-6,
            msg=f"{key} mismatch at sigma={sigma}",
        )
