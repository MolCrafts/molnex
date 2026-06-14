import torch

from molpot.derivation import ForceDerivation


class TestForceDerivation:
    def test_forward_forces(self):
        head = ForceDerivation()
        atoms_x = torch.randn(2, 3)
        # E = Σ x²  ->  F = -∂E/∂x = -2x
        forces = head(lambda p: p.pow(2).sum(), atoms_x)
        assert forces.shape == atoms_x.shape
        assert torch.allclose(forces, -2.0 * atoms_x, atol=1e-6)
