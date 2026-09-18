import torch

from molpot.derivation import ForceDerivation, autograd_forces


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
    def test_forces_correct_on_pure_torch(self):
        head = ForceDerivation()
        x = torch.randn(2, 3)
        # E = Σ x²  ->  F = -∂E/∂x = -2x
        forces = head(lambda p: p.pow(2).sum(), x)
        assert forces.shape == x.shape
        assert torch.allclose(forces, -2.0 * x, atol=1e-6)

    def test_autograd_handles_a_legacy_autograd_function(self):
        """The one backend computes correct forces on the fused-kernel-like op
        that ``torch.func`` rejects."""
        x = torch.randn(4, 3)
        energy = lambda p: _LegacySquare.apply(p).sum()  # noqa: E731
        forces = ForceDerivation()(energy, x)
        assert torch.allclose(forces, -2.0 * x, atol=1e-6)

    def test_autograd_supports_double_backward_for_force_training(self):
        """The force stays connected to parameters so a force loss backprops
        into them (the point of force supervision)."""
        w = torch.nn.Parameter(torch.tensor(1.7))
        x = torch.randn(5, 3)

        def energy(p):
            return _LegacySquare.apply(w * p).sum()  # E = Σ (w p)²

        forces = ForceDerivation()(energy, x)  # F = -2 w² x
        assert torch.allclose(forces, -2.0 * w**2 * x, atol=1e-5)

        forces.pow(2).sum().backward()  # double backward -> w.grad
        assert w.grad is not None and w.grad.abs() > 0

    def test_module_function_matches(self):
        """The standalone helper equals the module backend."""
        x = torch.randn(3, 3)
        e = lambda p: p.pow(2).sum()  # noqa: E731
        assert torch.allclose(autograd_forces(e, x), -2.0 * x, atol=1e-6)
