import pytest
import torch
from molpot.derivation import ForceDerivation

from tests.utils import assert_compile_compatible


class TestForceDerivation:
    def test_forward_forces(self):
        head = ForceDerivation()
        atoms_x = torch.randn(2, 3, requires_grad=True)
        energy = atoms_x.pow(2).sum().unsqueeze(0)
        out = head(energy, atoms_x)
        assert out.shape == atoms_x.shape

    @pytest.mark.xfail(reason="ForceDerivation uses torch.autograd.grad which breaks torch.compile", strict=False)
    def test_compile(self):
        head = ForceDerivation()
        atoms_x = torch.randn(2, 3, requires_grad=True)
        energy = atoms_x.pow(2).sum().unsqueeze(0)
        assert_compile_compatible(head, energy, atoms_x, strict=False)
