"""Unit tests for the tensor-level autograd primitives in molpot.derivation.force.

Mirrors ``src/molpot/derivation/force.py``: one backend, ``torch.autograd.grad``.
"""

from __future__ import annotations

import torch

from molpot.derivation.force import (
    ForceDerivation,
    autograd_forces,
    autograd_forces_from_energy,
)


class _CountedQuadratic(torch.autograd.Function):
    """Quadratic energy node recording physical forward/backward executions."""

    forward_calls = 0
    backward_calls = 0

    @staticmethod
    def forward(ctx, pos: torch.Tensor) -> torch.Tensor:
        _CountedQuadratic.forward_calls += 1
        ctx.save_for_backward(pos)
        return 0.5 * pos.square().sum()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        _CountedQuadratic.backward_calls += 1
        (pos,) = ctx.saved_tensors
        return (grad_output * pos,)

    @classmethod
    def reset(cls) -> None:
        cls.forward_calls = 0
        cls.backward_calls = 0


def test_autograd_forces_from_energy_matches_autograd_forward():
    pos = torch.randn(4, 3, requires_grad=True)
    energy = (pos.pow(2).sum(dim=-1) * torch.tensor([1.0, 2.0, 0.5, 1.5])).sum()
    f_from_e = autograd_forces_from_energy(energy, pos, create_graph=False)

    def energy_fn(p: torch.Tensor) -> torch.Tensor:
        return (p.pow(2).sum(dim=-1) * torch.tensor([1.0, 2.0, 0.5, 1.5])).sum()

    f_fwd = ForceDerivation()(energy_fn, pos.detach())
    torch.testing.assert_close(f_from_e, f_fwd, atol=1e-5, rtol=1e-5)


def test_autograd_force_evaluates_the_energy_and_its_backward_once():
    """An arbitrary energy graph is traversed exactly once per force request."""
    _CountedQuadratic.reset()

    forces = ForceDerivation()(
        lambda pos: _CountedQuadratic.apply(pos),
        torch.randn(4, 3),
    )

    assert tuple(forces.shape) == (4, 3)
    assert _CountedQuadratic.forward_calls == 1
    assert _CountedQuadratic.backward_calls == 1


def test_autograd_forces_follow_the_ambient_grad_mode():
    """Detached under ``no_grad``; attached to the parameters with grad enabled."""
    weight = torch.nn.Parameter(torch.tensor(1.7, dtype=torch.float64))
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.5, 0.0]], dtype=torch.float64)

    def energy_fn(p: torch.Tensor) -> torch.Tensor:
        return (weight * p * p).sum()

    with torch.no_grad():
        detached = autograd_forces(energy_fn, pos)
    assert not detached.requires_grad
    assert torch.allclose(detached, -2.0 * 1.7 * pos)

    attached = autograd_forces(energy_fn, pos)
    assert attached.requires_grad
    attached.pow(2).sum().backward()
    assert weight.grad is not None and float(weight.grad.abs()) > 0.0
