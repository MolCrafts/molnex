import pytest
import torch
from molpot.heads.heads import EnergyHead

from tests.utils import assert_compile_compatible


class TestEnergyHead:
    def test_forward_energy(self):
        head = EnergyHead(hidden_dim=4)
        h = torch.ones(3, 4)
        batch = torch.tensor([0, 0, 1])
        out = head(h, batch)
        assert out.shape == torch.Size([2])

    @pytest.mark.xfail(reason="EnergyHead uses scatter pooling which can break graph", strict=False)
    def test_compile(self):
        head = EnergyHead(hidden_dim=4)
        h = torch.ones(3, 4)
        batch = torch.tensor([0, 0, 1])
        assert_compile_compatible(head, h, batch, strict=False)
