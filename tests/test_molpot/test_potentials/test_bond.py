"""``bond harmonic`` priced as molpy prices it."""

from __future__ import annotations

import molpy as mp
import numpy as np
import torch

from molpot.potentials.bond import BondHarmonic
from tests.test_molpot.test_potentials.conftest import (
    BONDS,
    chain_batch,
    energy_and_forces,
    oracle,
)

K = [300.0, 280.0, 310.0, 300.0, 280.0, 310.0]
R0 = [1.40, 1.45, 1.50, 1.40, 1.45, 1.50]


class TestBondHarmonic:
    def test_prices_what_molpy_prices(self):
        energy, forces = energy_and_forces(
            BondHarmonic(),
            chain_batch(),
            k=torch.tensor(K, dtype=torch.float64),
            r0=torch.tensor(R0, dtype=torch.float64),
        )
        ref_e, ref_f = oracle(("bond", "harmonic", BONDS, {"k": K, "r0": R0}))
        np.testing.assert_allclose(energy, ref_e, rtol=1e-12)
        np.testing.assert_allclose(forces, ref_f, rtol=1e-10, atol=1e-10)

    def test_a_style_with_types_carries_their_parameters(self):
        ff = mp.ff.forcefield.ForceField(name="t", units="real")
        c = ff.def_style("atom", "full").def_type("c", mass=12.0)
        style = ff.def_style("bond", "harmonic")
        style.def_type("short", c, c, k=300.0, r0=1.40)
        style.def_type("long", c, c, k=250.0, r0=1.55)
        batch = chain_batch()
        type_id = torch.tensor([0, 1, 0, 1, 0, 1])
        batch["bonds", "bond_types"] = type_id
        carried = BondHarmonic.from_style(style)(batch)
        passed = BondHarmonic()(
            batch,
            k=torch.tensor([300.0, 250.0], dtype=torch.float64)[type_id],
            r0=torch.tensor([1.40, 1.55], dtype=torch.float64)[type_id],
        )
        torch.testing.assert_close(carried, passed)

    def test_dead_rows_contribute_nothing(self):
        batch = chain_batch()
        batch["bonds", "mask"] = torch.tensor([True, True, True, False, False, False])
        energy = BondHarmonic()(
            batch, k=torch.tensor(K, dtype=torch.float64), r0=torch.tensor(R0, dtype=torch.float64)
        )
        assert energy[1] == 0.0 and energy[0] > 0.0
