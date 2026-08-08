"""Unit tests for the shared batch-level force passes in molpot.derivation.kernels.

Mirrors ``src/molpot/derivation/kernels.py``. Each test targets exactly one of
the two kernels on the analytic toy potential ``E = s · Σ‖r‖²`` (conftest), for
which ``F = -2 s r`` is known in closed form — no reference implementation and
no third-party oracle is involved.

Units: positions Å, energy eV, forces eV/Å. All assertions run in float64 at
``atol=1e-12, rtol=0`` (the "exact" energy/force band for a closed-form
gradient), except the finite-difference domain check, which uses the numerical
band ``1e-6 eV/Å``.
"""

from __future__ import annotations

import pytest
import torch
from molpot.derivation.kernels import func_force_pass, grad_force_pass
from tensordict import TensorDict

from molpot.derivation.protocol import ENERGY_KEY, FORCES_KEY, POS_KEY
from tests.test_molpot.test_derivation.conftest import (
    KernelToyPotential,
    central_difference_forces,
    kernel_toy_batch,
)

#: ``F = -2 s r`` for ``s = 1.0`` and ``pos = arange(12).reshape(4, 3) * 0.1`` Å,
#: i.e. ``-0.2 * arange(12)`` in eV/Å. Hard-coded, not recomputed from ``pos``.
EXPECTED_FORCES: list[list[float]] = [
    [-0.0, -0.2, -0.4],
    [-0.6, -0.8, -1.0],
    [-1.2, -1.4, -1.6],
    [-1.8, -2.0, -2.2],
]

EXACT_FORCE_ATOL: float = 1e-12
FD_FORCE_ATOL: float = 1e-6


def _expected(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.tensor(EXPECTED_FORCES, dtype=dtype)


def _with_position_leaf(batch: TensorDict) -> torch.Tensor:
    """Replace ``atoms.pos`` with a live ``requires_grad`` leaf; return it."""
    leaf = batch[POS_KEY].detach().clone().requires_grad_(True)
    batch[POS_KEY] = leaf
    return leaf


class TestGradForcePass:
    """``grad_force_pass`` — one energy forward + ``torch.autograd.grad``."""

    def test_forces_match_hardcoded_analytic_gradient(self) -> None:
        toy = KernelToyPotential()
        batch = grad_force_pass(toy, kernel_toy_batch())
        torch.testing.assert_close(batch[FORCES_KEY], _expected(), atol=EXACT_FORCE_ATOL, rtol=0.0)

    def test_energy_core_none_reuses_materialised_energy(self) -> None:
        toy = KernelToyPotential()
        batch = kernel_toy_batch()
        _with_position_leaf(batch)
        with torch.enable_grad():
            batch = toy(batch)
        toy.n_forward = 0  # only count what the kernel does

        batch = grad_force_pass(None, batch, detach_energy=False)

        assert toy.n_forward == 0
        torch.testing.assert_close(batch[FORCES_KEY], _expected(), atol=EXACT_FORCE_ATOL, rtol=0.0)

    def test_energy_core_none_without_position_leaf_raises(self) -> None:
        batch = kernel_toy_batch()  # atoms.pos does not require grad
        with pytest.raises(RuntimeError):
            grad_force_pass(None, batch)

    def test_energy_core_none_with_non_leaf_position_raises(self) -> None:
        toy = KernelToyPotential()
        batch = kernel_toy_batch()
        leaf = batch[POS_KEY].detach().clone().requires_grad_(True)
        batch[POS_KEY] = leaf * 1.0  # requires_grad, but not a leaf
        with torch.enable_grad():
            batch = toy(batch)
        with pytest.raises(RuntimeError):
            grad_force_pass(None, batch)

    def test_energy_core_without_graphs_energy_raises(self) -> None:
        toy = KernelToyPotential(write_energy_key=False)
        with pytest.raises(RuntimeError, match=r"graphs\.energy"):
            grad_force_pass(toy, kernel_toy_batch())

    def test_detach_energy_false_keeps_energy_attached(self) -> None:
        toy = KernelToyPotential()
        batch = grad_force_pass(toy, kernel_toy_batch(), detach_energy=False)
        assert batch[ENERGY_KEY].requires_grad is True

    def test_detach_energy_true_detaches_energy(self) -> None:
        toy = KernelToyPotential()
        batch = grad_force_pass(toy, kernel_toy_batch(), detach_energy=True)
        assert batch[ENERGY_KEY].requires_grad is False

    def test_detach_energy_none_detaches_when_kernel_creates_leaf(self) -> None:
        toy = KernelToyPotential()
        batch = grad_force_pass(toy, kernel_toy_batch(), detach_energy=None)
        assert batch[ENERGY_KEY].requires_grad is False

    def test_detach_energy_none_keeps_energy_when_caller_supplies_leaf(self) -> None:
        toy = KernelToyPotential()
        batch = kernel_toy_batch()
        leaf = _with_position_leaf(batch)

        batch = grad_force_pass(toy, batch, detach_energy=None)

        assert batch[POS_KEY] is leaf  # kernel must not create a second leaf
        assert batch[ENERGY_KEY].requires_grad is True

    def test_create_graph_true_force_loss_reaches_parameters(self) -> None:
        toy = KernelToyPotential()
        batch = grad_force_pass(toy, kernel_toy_batch(), create_graph=True, detach_energy=False)
        batch[FORCES_KEY].pow(2).mean().backward()
        assert toy.scale.grad is not None
        assert toy.scale.grad.abs().item() > 0.0

    def test_create_graph_false_disconnects_forces(self) -> None:
        toy = KernelToyPotential()
        batch = grad_force_pass(toy, kernel_toy_batch(), create_graph=False, detach_energy=False)
        assert batch[FORCES_KEY].requires_grad is False

    def test_no_grad_context_still_returns_forces(self) -> None:
        toy = KernelToyPotential()
        with torch.no_grad():
            batch = grad_force_pass(toy, kernel_toy_batch())
        torch.testing.assert_close(batch[FORCES_KEY], _expected(), atol=EXACT_FORCE_ATOL, rtol=0.0)

    def test_forces_match_central_finite_differences(self) -> None:
        toy = KernelToyPotential()
        batch = kernel_toy_batch()
        pos = batch[POS_KEY].detach().clone()

        batch = grad_force_pass(toy, batch)
        numerical = central_difference_forces(toy.total_energy, pos, h=1e-5)

        deviation = (batch[FORCES_KEY].detach() - numerical).abs().max().item()
        assert deviation < FD_FORCE_ATOL


class TestFuncForcePass:
    """``func_force_pass`` — single ``torch.func.grad(..., has_aux=True)`` pass."""

    def test_forces_match_hardcoded_analytic_gradient(self) -> None:
        toy = KernelToyPotential()
        batch = func_force_pass(toy, kernel_toy_batch())
        torch.testing.assert_close(batch[FORCES_KEY], _expected(), atol=EXACT_FORCE_ATOL, rtol=0.0)

    def test_agrees_with_grad_force_pass(self) -> None:
        func_batch = func_force_pass(KernelToyPotential(), kernel_toy_batch())
        grad_batch = grad_force_pass(KernelToyPotential(), kernel_toy_batch())
        torch.testing.assert_close(
            func_batch[FORCES_KEY].detach(),
            grad_batch[FORCES_KEY].detach(),
            atol=EXACT_FORCE_ATOL,
            rtol=0.0,
        )

    def test_runs_exactly_one_energy_forward(self) -> None:
        toy = KernelToyPotential()
        func_force_pass(toy, kernel_toy_batch())
        assert toy.n_forward == 1

    def test_atomic_energy_written_back_when_core_provides_it(self) -> None:
        toy = KernelToyPotential(write_atomic=True)
        batch = func_force_pass(toy, kernel_toy_batch())
        assert "energy" in batch["atoms"].keys()
        assert batch["atoms", "energy"].shape == (4,)

    def test_core_without_atomic_energy_is_accepted(self) -> None:
        toy = KernelToyPotential(write_atomic=False)
        batch = func_force_pass(toy, kernel_toy_batch())
        assert "energy" not in batch["atoms"].keys()
        torch.testing.assert_close(batch[FORCES_KEY], _expected(), atol=EXACT_FORCE_ATOL, rtol=0.0)

    def test_force_loss_reaches_parameters_in_one_backward(self) -> None:
        toy = KernelToyPotential()
        batch = func_force_pass(toy, kernel_toy_batch())
        batch[FORCES_KEY].pow(2).mean().backward()
        assert toy.scale.grad is not None
        assert toy.scale.grad.abs().item() > 0.0

    def test_forces_match_central_finite_differences(self) -> None:
        toy = KernelToyPotential()
        batch = kernel_toy_batch()
        pos = batch[POS_KEY].detach().clone()

        batch = func_force_pass(toy, batch)
        numerical = central_difference_forces(toy.total_energy, pos, h=1e-5)

        deviation = (batch[FORCES_KEY].detach() - numerical).abs().max().item()
        assert deviation < FD_FORCE_ATOL
