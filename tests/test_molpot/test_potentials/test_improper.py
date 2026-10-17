"""Improper styles priced as molpy prices them — the dihedral of the stored order."""

from __future__ import annotations

import numpy as np
import torch

from molpot.potentials.improper import ImproperCvff
from tests.test_molpot.test_potentials.conftest import (
    QUADS,
    chain_batch,
    energy_and_forces,
    oracle,
)


class TestImproperCvff:
    def test_prices_what_molpy_prices(self):
        k, sign, n = [2.0, 1.5], [-1.0, 1.0], [2.0, 2.0]
        energy, forces = energy_and_forces(
            ImproperCvff(),
            chain_batch(),
            k=torch.tensor(k, dtype=torch.float64),
            sign=torch.tensor(sign, dtype=torch.float64),
            periodicity=torch.tensor(n, dtype=torch.float64),
        )
        ref_e, ref_f = oracle(("improper", "cvff", QUADS, {"k": k, "sign": sign, "periodicity": n}))
        np.testing.assert_allclose(energy, ref_e, rtol=1e-12)
        np.testing.assert_allclose(forces, ref_f, rtol=1e-9, atol=1e-9)
