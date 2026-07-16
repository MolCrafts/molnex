import pytest
import torch

from molpot.derivation import ForceDerivation, autograd_forces, functorch_forces


class _LegacySquare(torch.autograd.Function):
    """A legacy ``autograd.Function`` WITHOUT ``setup_context`` — stands in for a
    fused cuEquivariance kernel. ``torch.func.grad`` refuses such ops, but they
    support ordinary ``autograd.grad`` (incl. double backward)."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * x

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return 2.0 * x * g  # differentiable in x -> double-backward works


class TestForceDerivation:
    def test_default_is_autograd(self):
        assert ForceDerivation().method == "autograd"

    def test_invalid_method_rejected(self):
        with pytest.raises(ValueError, match="method must be one of"):
            ForceDerivation(method="nope")

    @pytest.mark.parametrize("method", ["functorch", "autograd"])
    def test_forces_correct_on_pure_torch(self, method):
        head = ForceDerivation(method=method)
        x = torch.randn(2, 3)
        # E = Σ x²  ->  F = -∂E/∂x = -2x
        forces = head(lambda p: p.pow(2).sum(), x)
        assert forces.shape == x.shape
        assert torch.allclose(forces, -2.0 * x, atol=1e-6)

    def test_functorch_raises_on_incompatible_op(self):
        """No fallback: the functorch backend must surface the setup_context
        error on a fused-kernel-like op rather than silently switching backend."""
        x = torch.randn(4, 3)
        energy = lambda p: _LegacySquare.apply(p).sum()  # noqa: E731
        head = ForceDerivation(method="functorch")
        with pytest.raises(RuntimeError, match="setup_context"):
            head(energy, x)

    def test_autograd_handles_incompatible_op(self):
        """The autograd backend computes correct forces on the fused-kernel-like
        op that functorch rejects."""
        x = torch.randn(4, 3)
        energy = lambda p: _LegacySquare.apply(p).sum()  # noqa: E731
        forces = ForceDerivation(method="autograd")(energy, x)
        assert torch.allclose(forces, -2.0 * x, atol=1e-6)

    def test_autograd_supports_double_backward_for_force_training(self):
        """The autograd backend keeps the force connected to parameters so a
        force loss backprops into them (the point of force supervision)."""
        w = torch.nn.Parameter(torch.tensor(1.7))
        x = torch.randn(5, 3)

        def energy(p):
            return _LegacySquare.apply(w * p).sum()  # E = Σ (w p)²

        forces = ForceDerivation(method="autograd")(energy, x)  # F = -2 w² x
        assert torch.allclose(forces, -2.0 * w**2 * x, atol=1e-5)

        forces.pow(2).sum().backward()  # double backward -> w.grad
        assert w.grad is not None and w.grad.abs() > 0

    def test_module_functions_match(self):
        """The standalone helpers equal the module backends."""
        x = torch.randn(3, 3)
        e = lambda p: p.pow(2).sum()  # noqa: E731
        assert torch.allclose(functorch_forces(e, x), -2.0 * x, atol=1e-6)
        assert torch.allclose(autograd_forces(e, x), -2.0 * x, atol=1e-6)
