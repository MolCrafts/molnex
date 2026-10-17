"""Periodic-cell tests for Sonata — cell consistency and the force / virial derivatives.

* cell-periodic invariance under one-atom lattice translation (≤ 1e-7)
  using minimum-image bond geometry;
* ``F = -∂E/∂pos`` against a fourth-order central difference of ``energy``;
* ``W = ∂E/∂ε`` against a fourth-order central difference of the strained
  energy — positions and cell mapped by ``v ↦ v (I + ε_sym)ᵀ``, the edge
  geometry rebuilt from the strained positions, exactly what the derivative
  op strains. Cauchy convention: no doubled shear, so ``σ = W / V``.

Every finite difference runs on the module's bound pipeline: the reciprocal
grid is frozen on the unperturbed cell, so no k-vector flips across the
``k² ≤ k_max²`` cutoff between two displaced evaluations (the canonical PME
stress-FD technique, LAMMPS ``fix_numdiff_virial``). The virial is
single-graph — one strain leaf is one lattice.
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from molix.schema import ENERGY_KEY, FORCES_KEY, VIRIAL_KEY
from molpot.composition import Sonata
from tests.test_molpot.test_composition.conftest import (
    CELL_LENGTH,
    full_bidirectional_edges,
    make_batch,
    sonata_fp64,
)

#: Finite-difference step (Å for positions, dimensionless for strain) and the
#: gate of the fourth-order stencil in float64.
FD_STEP = 1e-3
FD_ATOL = 1e-8
FD_RTOL = 1e-6

#: Atom/axis pairs probed by the force finite-difference test.
FD_PROBES = ((0, 0), (2, 1), (5, 2), (7, 0))

#: Strain components probed by the virial finite-difference test.
STRAIN_PROBES = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2))


def _four_point(values: tuple[float, float, float, float], step: float) -> float:
    """``f'(x)`` from ``f(x ± step)``, ``f(x ± 2 step)`` — the fourth-order central stencil."""
    plus2, plus1, minus1, minus2 = values
    return (-plus2 + 8.0 * plus1 - 8.0 * minus1 + minus2) / (12.0 * step)


def _rebuilt(batch: TensorDict, pos: torch.Tensor, cell: torch.Tensor) -> TensorDict:
    """*batch*'s system on new positions and cell, the edge geometry recomputed."""
    return make_batch(
        pos=pos,
        Z=batch["atoms", "Z"],
        pair_index=batch["pairs", "pair_index"],
        batch_idx=batch["atoms", "batch"],
        total_charge=batch["graphs", "total_charge"],
        cell=cell,
    )


@pytest.fixture(scope="module")
def single_graph_periodic() -> tuple[Sonata, TensorDict]:
    """One 4-atom graph in a 10 Å cubic cell and a bound float64 Sonata deriving the virial."""
    sonata = sonata_fp64(wrt=("atoms.pos", "graphs.strain"))
    cell = CELL_LENGTH * torch.eye(3, dtype=torch.float64)
    sonata.bind_cell(cell)
    pos = torch.tensor(
        [
            [0.10, 0.20, 0.05],
            [1.55, 0.15, 0.10],
            [0.85, 1.30, -0.05],
            [-0.65, 1.25, 0.20],
        ],
        dtype=torch.float64,
    )
    batch = make_batch(
        pos=pos,
        Z=torch.tensor([1, 6, 8, 7], dtype=torch.long),
        pair_index=full_bidirectional_edges([(0, 4)]),
        batch_idx=torch.zeros(4, dtype=torch.long),
        total_charge=torch.zeros(1, dtype=torch.float64),
        cell=cell.unsqueeze(0),
    )
    return sonata, batch


class TestCellPeriodicConsistency:
    """Translating one atom by a lattice vector keeps the energy invariant
    when bond geometry is computed under the minimum-image convention.

    For an orthorhombic cell, MI is the standard Δr → Δr − L·round(Δr/L)
    wrap. This test exercises the contract that the Sonata pipeline,
    fed PBC-correct edge inputs, produces a periodic-correct energy
    surface — which is the production setup any real periodic training
    uses (the user-side ``NeighborList`` does PBC).
    """

    def test_cell_periodic_atom0_shift(
        self, sonata_pipeline: Sonata, sample_neutral_batch_periodic: TensorDict
    ) -> None:
        batch = sample_neutral_batch_periodic
        pos = batch["atoms", "pos"]
        cell = batch["graphs", "cell"]

        def energy(positions: torch.Tensor) -> torch.Tensor:
            imaged = make_batch(
                pos=positions,
                Z=batch["atoms", "Z"],
                pair_index=batch["pairs", "pair_index"],
                batch_idx=batch["atoms", "batch"],
                total_charge=batch["graphs", "total_charge"],
                cell=cell,
                minimum_image=True,
            )
            with torch.no_grad():
                return sonata_pipeline.energy(imaged)[ENERGY_KEY]

        reference = energy(pos)
        # Atom 0 lives in graph 0; pull graph 0's lattice vectors.
        for shift in (cell[0, 0], -cell[0, 0], cell[0, 1], cell[0, 2]):
            shifted = pos.clone()
            shifted[0] = shifted[0] + shift
            rel = (energy(shifted) - reference).abs() / reference.abs().clamp(min=1e-12)
            assert rel.max() < 1e-7, f"shift {shift.tolist()}: rel.max()={rel.max():.2e}"


class TestForceFD:
    def test_forces_match_central_differences(
        self, sonata_pipeline: Sonata, sample_neutral_batch_periodic: TensorDict
    ) -> None:
        """``F = -dE/dx`` against a fourth-order central difference of ``energy``."""
        batch = sample_neutral_batch_periodic
        with torch.no_grad():
            forces = sonata_pipeline(batch.clone())[FORCES_KEY]
        pos = batch["atoms", "pos"]
        cell = batch["graphs", "cell"]

        def total(atom: int, axis: int, steps: float) -> float:
            shifted = pos.clone()
            shifted[atom, axis] += steps * FD_STEP
            with torch.no_grad():
                return float(
                    sonata_pipeline.energy(_rebuilt(batch, shifted, cell))[ENERGY_KEY].sum()
                )

        for atom, axis in FD_PROBES:
            numerical = -_four_point(
                tuple(total(atom, axis, steps) for steps in (2.0, 1.0, -1.0, -2.0)), FD_STEP
            )
            assert float(forces[atom, axis]) == pytest.approx(numerical, abs=FD_ATOL, rel=FD_RTOL)


class TestVirialFD:
    def test_virial_matches_central_differences_of_the_strained_energy(
        self, single_graph_periodic: tuple[Sonata, TensorDict]
    ) -> None:
        """``W = ∂E/∂ε`` against a fourth-order central difference of the strained energy."""
        sonata, batch = single_graph_periodic
        with torch.no_grad():
            virial = sonata(batch.clone())[VIRIAL_KEY][0]
        pos = batch["atoms", "pos"]
        cell = batch["graphs", "cell"][0]

        def strained_energy(i: int, j: int, steps: float) -> float:
            delta = torch.zeros(3, 3, dtype=torch.float64)
            delta[i, j] = steps * FD_STEP
            transform = (torch.eye(3, dtype=torch.float64) + 0.5 * (delta + delta.T)).T
            strained = _rebuilt(batch, pos @ transform, (cell @ transform).unsqueeze(0))
            with torch.no_grad():
                return float(sonata.energy(strained)[ENERGY_KEY].sum())

        for i, j in STRAIN_PROBES:
            numerical = _four_point(
                tuple(strained_energy(i, j, steps) for steps in (2.0, 1.0, -1.0, -2.0)), FD_STEP
            )
            assert float(virial[i, j]) == pytest.approx(numerical, abs=FD_ATOL, rel=FD_RTOL)
