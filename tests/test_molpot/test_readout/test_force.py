import pytest
import torch

from molpot.derivation import ForceDerivation


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


def _functorch_rejects(fn, x) -> bool:
    try:
        torch.func.grad(fn)(x)
        return False
    except RuntimeError as e:
        return "setup_context" in str(e)


class TestForceDerivation:
    def test_forward_forces(self):
        head = ForceDerivation()
        atoms_x = torch.randn(2, 3)
        # E = Σ x²  ->  F = -∂E/∂x = -2x
        forces = head(lambda p: p.pow(2).sum(), atoms_x)
        assert forces.shape == atoms_x.shape
        assert torch.allclose(forces, -2.0 * atoms_x, atol=1e-6)

    def test_functorch_incompatible_op_does_not_crash_and_is_correct(self):
        """Eager must NOT crash when the energy graph contains a
        functorch-incompatible op — it falls back to ``torch.autograd.grad``."""
        x = torch.randn(4, 3)
        energy = lambda p: _LegacySquare.apply(p).sum()  # noqa: E731

        # Precondition: functorch genuinely rejects this op (else the test is moot).
        assert _functorch_rejects(energy, x)

        head = ForceDerivation()
        with pytest.warns(RuntimeWarning, match="functorch-incompatible"):
            forces = head(energy, x)  # would raise without the fallback
        assert torch.allclose(forces, -2.0 * x, atol=1e-6)
        assert head._functorch_unsupported is True  # sticky after first fallback

    def test_fallback_supports_double_backward_for_force_training(self):
        """The autograd fallback must keep the force connected to parameters so a
        force loss backprops into them (the whole point of force supervision)."""
        w = torch.nn.Parameter(torch.tensor(1.7))
        x = torch.randn(5, 3)

        def energy(p):
            return _LegacySquare.apply(w * p).sum()  # E = Σ (w p)²

        head = ForceDerivation()
        with pytest.warns(RuntimeWarning):
            forces = head(energy, x)  # F = -2 w² x
        assert torch.allclose(forces, -2.0 * w**2 * x, atol=1e-5)

        (forces.pow(2).sum()).backward()  # double backward -> w.grad
        assert w.grad is not None and w.grad.abs() > 0

    def test_functorch_path_unaffected_for_compatible_ops(self):
        """Pure-torch energies still go through functorch (no spurious fallback),
        preserving the compile-friendly single-backward path."""
        head = ForceDerivation()
        x = torch.randn(3, 3)
        forces = head(lambda p: p.pow(2).sum(), x)
        assert torch.allclose(forces, -2.0 * x, atol=1e-6)
        assert head._functorch_unsupported is False
