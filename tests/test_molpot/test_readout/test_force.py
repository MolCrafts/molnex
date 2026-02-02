import torch
from molpot.heads.heads import ForceHead


class TestForceHead:
    def test_requires_grad(self):
        head = ForceHead()
        atoms_x = torch.zeros(2, 3, requires_grad=False)
        energy = torch.zeros(1)
        try:
            head(energy, atoms_x)
        except RuntimeError as exc:
            assert "requires_grad" in str(exc)
        else:
            raise AssertionError("Expected RuntimeError when atoms.x has no gradients")

    def test_forward_forces(self):
        head = ForceHead()
        atoms_x = torch.randn(2, 3, requires_grad=True)
        energy = atoms_x.pow(2).sum().unsqueeze(0)
        out = head(energy, atoms_x)
        assert out.shape == atoms_x.shape
