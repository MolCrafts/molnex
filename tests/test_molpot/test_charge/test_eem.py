"""Tests for ElectronegativityEqualization (EEM/QEq-family).

The solver only: who *owns* a charge model is
``tests/test_molpot/test_charge/test_assignment.py``, and what a model
does with one is ``tests/test_molzoo/test_rondo/test_model.py``.
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

    def test_a_declared_capacity_reproduces_the_batch_sized_solve(self):
        """``max_atoms`` fixes the dense layout; the charges do not move.

        The padded solve is the same linear system whether ``n_max`` is read
        off the batch or declared up front, so a capacity above every molecule
        gives the batch-sized answer to round-off — padding rows are pinned to
        the identity and never reach a real atom.
        """
        atom_batch = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
        problem = ChargeEquilibrationProblem(
            atom_batch=atom_batch,
            num_graphs=2,
            total_charge=torch.tensor([0.0, 1.0], dtype=torch.float64),
        )
        chi = torch.tensor([1.0, 2.0, 1.0, 2.0, 3.0], dtype=torch.float64)
        eta = torch.tensor([1.0, 2.0, 1.0, 3.0, 2.0], dtype=torch.float64)
        sized = ElectronegativityEqualization(DiagonalHardnessKernel(), jitter=0.0)
        capped = ElectronegativityEqualization(DiagonalHardnessKernel(), jitter=0.0, max_atoms=8)
        torch.testing.assert_close(
            capped.solve(chi=chi, eta=eta, problem=problem),
            sized.solve(chi=chi, eta=eta, problem=problem),
            atol=1e-12,
            rtol=0.0,
        )

    def test_trailing_ghosts_do_not_oob_or_wipe_live_charges(self):
        """MoleculePadder ghosts (batch=0, mask=False) stay out of the solve.

        Ghosts sit at the end of the packed atom axis. Their global index is
        not a per-graph slot; gathering with it used to CUDA-assert, and
        assigning 0 onto the clamped slot used to wipe graph-0's last atom.
        """
        chi = torch.tensor([2.0, 4.0, 99.0, 99.0], dtype=torch.float64)
        eta = torch.tensor([1.0, 3.0, 1.0, 1.0], dtype=torch.float64)
        problem = ChargeEquilibrationProblem(
            atom_batch=torch.tensor([0, 0, 0, 0], dtype=torch.long),
            num_graphs=1,
            total_charge=torch.zeros(1, dtype=torch.float64),
            atom_mask=torch.tensor([True, True, False, False]),
        )
        q1 = (4.0 - 2.0) / (1.0 + 3.0)
        for max_atoms in (None, 8):
            q = ElectronegativityEqualization(
                DiagonalHardnessKernel(), jitter=0.0, max_atoms=max_atoms
            ).solve(chi=chi, eta=eta, problem=problem)
            assert math.isclose(float(q[0]), q1, abs_tol=1e-10)
            assert math.isclose(float(q[1]), -q1, abs_tol=1e-10)
            assert math.isclose(float(q[2]), 0.0, abs_tol=1e-10)
            assert math.isclose(float(q[3]), 0.0, abs_tol=1e-10)

    def test_infinite_ghost_chi_does_not_poison_live_charges(self):
        """Ghost ``chi = inf`` must not ``inf * 0 → nan`` into graph-0's last slot."""
        chi = torch.tensor([2.0, 4.0, float("inf"), float("inf")], dtype=torch.float64)
        eta = torch.tensor([1.0, 3.0, float("inf"), float("inf")], dtype=torch.float64)
        problem = ChargeEquilibrationProblem(
            atom_batch=torch.tensor([0, 0, 0, 0], dtype=torch.long),
            num_graphs=1,
            total_charge=torch.zeros(1, dtype=torch.float64),
            atom_mask=torch.tensor([True, True, False, False]),
        )
        q1 = (4.0 - 2.0) / (1.0 + 3.0)
        q = ElectronegativityEqualization(DiagonalHardnessKernel(), jitter=0.0, max_atoms=8).solve(
            chi=chi, eta=eta, problem=problem
        )
        assert math.isclose(float(q[0]), q1, abs_tol=1e-10)
        assert math.isclose(float(q[1]), -q1, abs_tol=1e-10)
        assert math.isfinite(float(q[0]))
        assert math.isclose(float(q[2]), 0.0, abs_tol=1e-10)

    def test_over_capacity_is_refused_eagerly_naming_the_capacity(self):
        """A molecule larger than ``max_atoms`` raises, it is never truncated."""
        import pytest

        problem = ChargeEquilibrationProblem(
            atom_batch=torch.tensor([0, 0, 0], dtype=torch.long),
            num_graphs=1,
            total_charge=torch.zeros(1, dtype=torch.float64),
        )
        eq = ElectronegativityEqualization(DiagonalHardnessKernel(), jitter=0.0, max_atoms=2)
        with pytest.raises(ValueError, match="max_atoms=2"):
            eq.solve(
                chi=torch.ones(3, dtype=torch.float64),
                eta=torch.ones(3, dtype=torch.float64),
                problem=problem,
            )

    def test_an_empty_molecule_keeps_the_batch_nonsingular(self):
        """A graph with no atoms is pinned to the identity, the others solve.

        Selected by mask rather than by a Python branch on ``counts == 0``, so
        the same code traces under ``torch.compile`` without a break.
        """
        atom_batch = torch.tensor([0, 0, 2, 2], dtype=torch.long)
        problem = ChargeEquilibrationProblem(
            atom_batch=atom_batch,
            num_graphs=3,
            total_charge=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        )
        chi = torch.tensor([2.0, 4.0, 1.0, 3.0], dtype=torch.float64)
        eta = torch.tensor([1.0, 3.0, 2.0, 2.0], dtype=torch.float64)
        q = ElectronegativityEqualization(DiagonalHardnessKernel(), jitter=0.0).solve(
            chi=chi, eta=eta, problem=problem
        )
        assert bool(torch.isfinite(q).all())
        assert math.isclose(float(q[:2].sum()), 0.0, abs_tol=1e-10)
        assert math.isclose(float(q[2:].sum()), 1.0, abs_tol=1e-10)
        assert math.isclose(float(q[0]), (4.0 - 2.0) / (1.0 + 3.0), abs_tol=1e-10)


class TestDiagonalClosedForm:
    """The closed form must be the dense solve's answer, not an approximation."""

    @staticmethod
    def _dense_reference(solver, *, chi, eta, problem):
        """Charges via the generic bordered factorization, bypassing the shortcut."""
        dense_chi, dense_eta, dense_pos, atom_mask, counts = solver._pack(chi, eta, problem)
        a = solver.kernel.assemble(
            eta=eta, dense_eta=dense_eta, dense_pos=dense_pos, atom_mask=atom_mask
        )
        bsz, n_max, _ = a.shape
        eye = torch.eye(n_max, device=a.device, dtype=a.dtype).expand(bsz, n_max, n_max)
        a = a + solver.jitter * eye * atom_mask.unsqueeze(1).to(dtype=a.dtype)
        q_dense = solver._solve_bordered(a, dense_chi, problem.total_charge, atom_mask, counts)
        return solver._unpack(q_dense, problem.atom_batch, counts, None)

    def test_closed_form_matches_the_factorized_solve(self):
        """Same charges as ``linalg.solve`` — this is algebra, not a heuristic.

        The generic path factorizes one ``(n_max+1)²`` matrix per molecule; at
        the Class-I operating point that batched LU was 22 % of device time in
        the compiled step, three times the next operator.
        """
        torch.manual_seed(2666)
        atom_batch = torch.tensor([0, 0, 0, 1, 1, 2, 3, 3, 3, 3], dtype=torch.long)
        problem = ChargeEquilibrationProblem(
            atom_batch=atom_batch,
            num_graphs=4,
            total_charge=torch.tensor([0.0, -1.0, 2.0, 0.5], dtype=torch.float64),
        )
        chi = torch.randn(10, dtype=torch.float64)
        eta = torch.rand(10, dtype=torch.float64) + 0.5
        solver = ElectronegativityEqualization(DiagonalHardnessKernel(), max_atoms=8, jitter=1e-4)

        closed = solver.solve(chi=chi, eta=eta, problem=problem)
        dense = self._dense_reference(solver, chi=chi, eta=eta, problem=problem)

        torch.testing.assert_close(closed, dense, atol=1e-10, rtol=1e-10)

    def test_closed_form_holds_the_declared_total_charge(self):
        atom_batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
        totals = torch.tensor([-1.0, 2.0], dtype=torch.float64)
        problem = ChargeEquilibrationProblem(
            atom_batch=atom_batch, num_graphs=2, total_charge=totals
        )
        chi = torch.tensor([0.3, -1.2, 0.7, 2.0, -0.4], dtype=torch.float64)
        eta = torch.tensor([1.0, 2.0, 0.7, 1.3, 3.0], dtype=torch.float64)
        q = ElectronegativityEqualization(DiagonalHardnessKernel(), jitter=1e-4).solve(
            chi=chi, eta=eta, problem=problem
        )
        for graph, total in enumerate(totals):
            torch.testing.assert_close(q[atom_batch == graph].sum(), total, atol=1e-12, rtol=0)

    def test_a_molecule_with_no_atoms_stays_finite(self):
        """An empty graph has no equation; it must not divide by zero."""
        atom_batch = torch.tensor([0, 0, 2], dtype=torch.long)
        problem = ChargeEquilibrationProblem(
            atom_batch=atom_batch,
            num_graphs=3,
            total_charge=torch.zeros(3, dtype=torch.float64),
        )
        chi = torch.tensor([1.0, -1.0, 0.5], dtype=torch.float64)
        eta = torch.ones(3, dtype=torch.float64)
        q = ElectronegativityEqualization(DiagonalHardnessKernel(), jitter=1e-4).solve(
            chi=chi, eta=eta, problem=problem
        )
        assert bool(torch.isfinite(q).all())

    def test_the_gradient_reaches_chi_and_eta(self):
        """The charge model is trained through this solve."""
        atom_batch = torch.tensor([0, 0, 0], dtype=torch.long)
        problem = ChargeEquilibrationProblem(
            atom_batch=atom_batch,
            num_graphs=1,
            total_charge=torch.zeros(1, dtype=torch.float64),
        )
        chi = torch.tensor([0.4, -0.2, 0.9], dtype=torch.float64, requires_grad=True)
        eta = torch.tensor([1.1, 0.8, 1.7], dtype=torch.float64, requires_grad=True)
        q = ElectronegativityEqualization(DiagonalHardnessKernel(), jitter=1e-4).solve(
            chi=chi, eta=eta, problem=problem
        )
        q.square().sum().backward()
        assert chi.grad is not None and bool(torch.isfinite(chi.grad).all())
        assert eta.grad is not None and bool(torch.isfinite(eta.grad).all())


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
