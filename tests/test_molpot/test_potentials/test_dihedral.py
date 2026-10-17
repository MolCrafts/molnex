"""``dihedral periodic`` priced as molpy prices it — a padded series, phases in degrees."""

from __future__ import annotations

import molpy as mp
import numpy as np
import torch

from molpot.potentials.dihedral import DihedralOpls, DihedralPeriodic
from tests.test_molpot.test_potentials.conftest import (
    QUADS,
    chain_batch,
    energy_and_forces,
    oracle,
)

K = np.array([[1.3, 0.4, 0.0], [0.8, -0.5, 0.2]])
N = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
PHASE = np.array([[0.0, 180.0, 0.0], [0.0, 0.0, 180.0]])


class TestDihedralPeriodic:
    def test_prices_what_molpy_prices(self):
        energy, forces = energy_and_forces(
            DihedralPeriodic(),
            chain_batch(),
            k=torch.tensor(K, dtype=torch.float64),
            periodicity=torch.tensor(N, dtype=torch.float64),
            phase=torch.tensor(PHASE, dtype=torch.float64),
        )
        series = {
            f"{name}{m + 1}": table[:, m].tolist()
            for m in range(K.shape[1])
            for name, table in (("k", K), ("periodicity", N), ("phase", PHASE))
        }
        ref_e, ref_f = oracle(("dihedral", "periodic", QUADS, series))
        np.testing.assert_allclose(energy, ref_e, rtol=1e-12)
        np.testing.assert_allclose(forces, ref_f, rtol=1e-9, atol=1e-9)

    def test_types_of_different_lengths_share_one_padded_series(self):
        ff = mp.ff.forcefield.ForceField(name="t", units="real")
        c = ff.def_style("atom", "full").def_type("c", mass=12.0)
        style = ff.def_style("dihedral", "periodic")
        style.def_type(
            "two",
            c,
            c,
            c,
            c,
            k1=1.3,
            periodicity1=1,
            phase1=0.0,
            k2=0.4,
            periodicity2=2,
            phase2=180.0,
        )
        style.def_type("one", c, c, c, c, k=0.8, periodicity=3, phase=0.0)
        potential = DihedralPeriodic.from_style(style)
        torch.testing.assert_close(
            potential.table_k, torch.tensor([[1.3, 0.4], [0.8, 0.0]], dtype=torch.float64)
        )
        batch = chain_batch()
        batch["propers", "type"] = torch.tensor([0, 1])
        energy, _ = energy_and_forces(potential, batch)
        ref_e, _ = oracle(
            (
                "dihedral",
                "periodic",
                QUADS,
                {
                    "k1": [1.3, 0.8],
                    "periodicity1": [1.0, 3.0],
                    "phase1": [0.0, 0.0],
                    "k2": [0.4, 0.0],
                    "periodicity2": [2.0, 1.0],
                    "phase2": [180.0, 0.0],
                },
            )
        )
        np.testing.assert_allclose(energy, ref_e, rtol=1e-12)


class TestDihedralOpls:
    def test_is_the_periodic_series_at_half_k_and_alternating_phases(self):
        batch = chain_batch()
        ks = [
            torch.tensor(v, dtype=torch.float64)
            for v in ([1.3, 0.8], [-0.4, 0.5], [0.9, 0.2], [0.1, -0.3])
        ]
        opls = DihedralOpls()(batch, k1=ks[0], k2=ks[1], k3=ks[2], k4=ks[3])
        k = 0.5 * torch.stack(ks, dim=1)
        periodicity = torch.tensor([[1.0, 2.0, 3.0, 4.0]] * 2, dtype=torch.float64)
        phase = torch.tensor([[0.0, 180.0, 0.0, 180.0]] * 2, dtype=torch.float64)
        periodic = DihedralPeriodic()(batch, k=k, periodicity=periodicity, phase=phase)
        torch.testing.assert_close(opls, periodic)
