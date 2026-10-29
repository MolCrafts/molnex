"""Improper styles priced as molpy prices them — the dihedral of the stored order."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from molpot.potentials.improper import ImproperCvff, ImproperPeriodic
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


class TestImproperPeriodic:
    def test_literal_phase_matches_molpy_energy_and_force(self):
        k, phase, n = [2.0, 1.5], [180.00007714362235, 23.0], [2.0, 3.0]
        energy, forces = energy_and_forces(
            ImproperPeriodic(),
            chain_batch(),
            k=torch.tensor(k, dtype=torch.float64),
            phase=torch.tensor(phase, dtype=torch.float64),
            periodicity=torch.tensor(n, dtype=torch.float64),
        )
        ref_e, ref_f = oracle(
            ("improper", "periodic", QUADS, {"k": k, "phase": phase, "periodicity": n})
        )
        np.testing.assert_allclose(energy, ref_e, rtol=1e-12)
        np.testing.assert_allclose(forces, ref_f, rtol=1e-9, atol=1e-9)

    @pytest.mark.slow
    def test_periodic_improper_is_one_graph(self):
        from molix.compile import Compiler

        kernel = ImproperPeriodic()
        parameters = {
            "k": torch.tensor([1.2, 2.0], dtype=torch.float64),
            "periodicity": torch.tensor([2.0, 3.0], dtype=torch.float64),
            "phase": torch.tensor([180.0, 23.0], dtype=torch.float64),
        }
        explanation = Compiler.explain(kernel, chain_batch(), **parameters)
        assert explanation.graph_count == 1
        assert explanation.graph_break_count == 0, explanation.break_sites
