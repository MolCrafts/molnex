import torch
from molpot.derivation import ForceDerivation


class TestForceDerivation:
    def test_forward_forces(self):
        head = ForceDerivation()
        atoms_x = torch.randn(2, 3, requires_grad=True)
        energy = atoms_x.pow(2).sum().unsqueeze(0)
        out = head(energy, atoms_x)
        assert out.shape == atoms_x.shape
