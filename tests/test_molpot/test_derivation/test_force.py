"""Unit tests for molpot.derivation.force.ForceDerivation backends.

Each backend is pure: functorch helpers never run under method=autograd and
autograd helpers never run under method=functorch.
"""

from __future__ import annotations

import torch

from molpot.derivation.force import (
    ForceDerivation,
    autograd_forces_from_energy,
)


def test_functorch_and_autograd_agree_on_energy_fn():
    pos = torch.randn(5, 3)

    def energy_fn(p: torch.Tensor) -> torch.Tensor:
        return (p.pow(2).sum(dim=-1) * torch.tensor([1.0, 2.0, 0.5, 1.5, 0.25])).sum()

    f_ft = ForceDerivation(method="functorch")(energy_fn, pos)
    f_ag = ForceDerivation(method="autograd")(energy_fn, pos)
    torch.testing.assert_close(f_ft, f_ag, atol=1e-5, rtol=1e-5)


def test_autograd_forces_from_energy_matches_autograd_forward():
    pos = torch.randn(4, 3, requires_grad=True)
    energy = (pos.pow(2).sum(dim=-1) * torch.tensor([1.0, 2.0, 0.5, 1.5])).sum()
    f_from_e = autograd_forces_from_energy(energy, pos, create_graph=False)

    def energy_fn(p: torch.Tensor) -> torch.Tensor:
        return (p.pow(2).sum(dim=-1) * torch.tensor([1.0, 2.0, 0.5, 1.5])).sum()

    f_fwd = ForceDerivation(method="autograd")(energy_fn, pos.detach())
    torch.testing.assert_close(f_from_e, f_fwd, atol=1e-5, rtol=1e-5)


def test_forces_from_energy_rejected_on_functorch_instance():
    deriv = ForceDerivation(method="functorch")
    pos = torch.zeros(2, 3, requires_grad=True)
    try:
        deriv.forces_from_energy(pos.pow(2).sum(), pos)
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "autograd-only" in str(e)


def test_has_aux_rejected_on_autograd_instance():
    deriv = ForceDerivation(method="autograd")

    def energy_fn_aux(p: torch.Tensor):
        return p.pow(2).sum(), p

    try:
        deriv(energy_fn_aux, torch.zeros(2, 3), has_aux=True)
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "functorch-only" in str(e)


def test_invalid_method_raises():
    try:
        ForceDerivation(method="magic")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
