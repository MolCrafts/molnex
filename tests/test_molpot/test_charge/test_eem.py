"""Tests for ElectronegativityEqualization (EEM/QEq-family).

The solver only: who *owns* a charge model is
``tests/test_molpot/test_charge/test_assignment.py``, and what a
``Parameterizer`` does with one is
``tests/test_molpot/test_forcefield/test_parameterizer.py``.
"""

from __future__ import annotations

import math

import torch

from molpot.charge import (
    DiagonalHardnessKernel,
    ElectronegativityEqualization,
    ShieldedCoulombKernel,
)
from molpot.charge.problem import ChargeEquilibrationProblem


class TestElectronegativityEqualizationDiagonal:
    def test_heterogeneous_packed_batch_padding_is_nonsingular(self):
        atom_batch = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
        problem = ChargeEquilibrationProblem(
            atom_batch=atom_batch,
            num_graphs=2,
            total_charge=torch.zeros(2, dtype=torch.float64),
        )
        chi = torch.tensor([1.0, 2.0, 1.0, 2.0, 3.0], dtype=torch.float64)
        eta = torch.ones(5, dtype=torch.float64)
        q = ElectronegativityEqualization(DiagonalHardnessKernel(), jitter=0.0).solve(
            chi=chi, eta=eta, problem=problem
        )
        torch.testing.assert_close(q[atom_batch == 0].sum(), torch.tensor(0.0, dtype=q.dtype))
        torch.testing.assert_close(q[atom_batch == 1].sum(), torch.tensor(0.0, dtype=q.dtype))

    def test_two_atom_analytic(self):
        """Diagonal A: q1 = (χ2 − χ1)/(η1+η2), q2 = −q1 for Q=0."""
        chi = torch.tensor([2.0, 4.0], dtype=torch.float64)
        eta = torch.tensor([1.0, 3.0], dtype=torch.float64)
        batch = torch.zeros(2, dtype=torch.long)
        problem = ChargeEquilibrationProblem(
            atom_batch=batch,
            num_graphs=1,
            total_charge=torch.zeros(1, dtype=torch.float64),
            pos=torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float64),
        )
        eq = ElectronegativityEqualization(DiagonalHardnessKernel(), jitter=0.0)
        q = eq.solve(chi=chi, eta=eta, problem=problem)
        q1 = (4.0 - 2.0) / (1.0 + 3.0)
        assert math.isclose(float(q[0]), q1, abs_tol=1e-10)
        assert math.isclose(float(q[1]), -q1, abs_tol=1e-10)
        assert math.isclose(float(q.sum()), 0.0, abs_tol=1e-10)

    def test_total_charge_constraint(self):
        chi = torch.tensor([1.0, 1.0, 2.0], dtype=torch.float64)
        eta = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64)
        batch = torch.zeros(3, dtype=torch.long)
        problem = ChargeEquilibrationProblem(
            atom_batch=batch,
            num_graphs=1,
            total_charge=torch.tensor([1.0], dtype=torch.float64),
        )
        eq = ElectronegativityEqualization(DiagonalHardnessKernel(), jitter=1e-12)
        q = eq.solve(chi=chi, eta=eta, problem=problem)
        assert math.isclose(float(q.sum()), 1.0, abs_tol=1e-8)

    def test_batched_two_molecules(self):
        chi = torch.tensor([2.0, 4.0, 1.0, 3.0], dtype=torch.float64)
        eta = torch.tensor([1.0, 1.0, 2.0, 2.0], dtype=torch.float64)
        batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        problem = ChargeEquilibrationProblem(
            atom_batch=batch,
            num_graphs=2,
            total_charge=torch.zeros(2, dtype=torch.float64),
        )
        q = ElectronegativityEqualization(DiagonalHardnessKernel(), jitter=0.0).solve(
            chi=chi, eta=eta, problem=problem
        )
        assert math.isclose(float(q[:2].sum()), 0.0, abs_tol=1e-10)
        assert math.isclose(float(q[2:].sum()), 0.0, abs_tol=1e-10)


class TestShieldedCoulomb:
    def test_two_atom_neutral_sum(self):
        chi = torch.tensor([2.0, 5.0], dtype=torch.float64)
        eta = torch.tensor([1.5, 2.5], dtype=torch.float64)
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=torch.float64)
        problem = ChargeEquilibrationProblem(
            atom_batch=torch.zeros(2, dtype=torch.long),
            num_graphs=1,
            total_charge=torch.zeros(1, dtype=torch.float64),
            pos=pos,
        )
        q = ElectronegativityEqualization(ShieldedCoulombKernel(), jitter=1e-12).solve(
            chi=chi, eta=eta, problem=problem
        )
        assert math.isclose(float(q.sum()), 0.0, abs_tol=1e-8)
        # more electronegative atom 1 should be more negative
        assert float(q[1]) < float(q[0])
