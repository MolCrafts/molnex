"""Unit tests for :mod:`molpot.derivation.stress`."""

from __future__ import annotations

import torch

from molpot.derivation.stress import StressDerivation, symmetrize_strain_grad


class TestStressDerivation:
    """``σ = W / V`` with ``W = ∂E/∂ε`` over the symmetric strain."""

    def test_shear_is_the_cauchy_value_not_the_engineering_double(self) -> None:
        """A harmonic bond has ``W_αβ = k d_α d_β`` — no factor 2 on shear."""
        k = 3.0
        d0 = torch.tensor([0.7, -1.3, 0.4], dtype=torch.float64)
        cell = torch.eye(3, dtype=torch.float64) * 2.0

        def energy(strain: torch.Tensor) -> torch.Tensor:
            d = d0 @ (torch.eye(3, dtype=torch.float64) + strain).T
            return 0.5 * k * d.dot(d)

        stress = StressDerivation()(energy, torch.zeros_like(cell), cell)

        volume = torch.linalg.det(cell).abs()
        expected = k * torch.outer(d0, d0) / volume
        assert stress.shape == (3, 3)
        torch.testing.assert_close(stress, expected)

    def test_asymmetric_raw_gradient_is_symmetrized(self) -> None:
        """Raw ``∂E/∂ε`` need not be symmetric; the reported stress is."""
        m = torch.tensor([[1.0, 2.0, 0.0], [0.0, 1.0, -3.0], [0.5, 0.0, 1.0]], dtype=torch.float64)
        cell = torch.eye(3, dtype=torch.float64)

        stress = StressDerivation()(lambda s: (m * s).sum(), torch.zeros_like(cell), cell)

        torch.testing.assert_close(stress, 0.5 * (m + m.T))


def test_symmetrize_strain_grad_batches_over_leading_dims() -> None:
    """The protocol helper transposes the last two axes only."""
    grad = torch.randn(4, 3, 3, dtype=torch.float64)
    out = symmetrize_strain_grad(grad)
    torch.testing.assert_close(out, out.transpose(-1, -2))
    torch.testing.assert_close(out.sum(), grad.sum())
