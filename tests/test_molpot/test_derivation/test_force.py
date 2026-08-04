"""Unit tests for molpot.derivation.force.ForceDerivation backends."""

from __future__ import annotations

import torch

from molpot.derivation.force import ForceDerivation


def test_functorch_and_autograd_agree():
    pos = torch.randn(5, 3)

    def energy_fn(p: torch.Tensor) -> torch.Tensor:
        return (p.pow(2).sum(dim=-1) * torch.tensor([1.0, 2.0, 0.5, 1.5, 0.25])).sum()

    f_ft = ForceDerivation(method="functorch")(energy_fn, pos)
    f_ag = ForceDerivation(method="autograd")(energy_fn, pos)
    torch.testing.assert_close(f_ft, f_ag, atol=1e-5, rtol=1e-5)


def test_invalid_method_raises():
    try:
        ForceDerivation(method="magic")  # type: ignore[arg-type]
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
