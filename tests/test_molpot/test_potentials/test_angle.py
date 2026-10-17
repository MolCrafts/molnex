"""``angle harmonic`` priced as molpy prices it — ``theta0`` in degrees."""

from __future__ import annotations

import numpy as np
import torch

from molpot.potentials.angle import AngleHarmonic
from tests.test_molpot.test_potentials.conftest import (
    ANGLES,
    chain_batch,
    energy_and_forces,
    oracle,
)

K = [50.0, 60.0, 55.0, 45.0]
THETA0 = [109.5, 115.0, 120.0, 104.5]


class TestAngleHarmonic:
    def test_prices_what_molpy_prices(self):
        energy, forces = energy_and_forces(
            AngleHarmonic(),
            chain_batch(),
            k=torch.tensor(K, dtype=torch.float64),
            theta0=torch.tensor(THETA0, dtype=torch.float64),
        )
        ref_e, ref_f = oracle(("angle", "harmonic", ANGLES, {"k": K, "theta0": THETA0}))
        np.testing.assert_allclose(energy, ref_e, rtol=1e-12)
        np.testing.assert_allclose(forces, ref_f, rtol=1e-9, atol=1e-9)
