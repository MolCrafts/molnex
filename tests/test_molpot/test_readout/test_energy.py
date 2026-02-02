import torch
from molpot.heads.heads import EnergyHead


class TestEnergyHead:
    def test_forward_energy(self):
        head = EnergyHead(hidden_dim=4)
        h = torch.ones(3, 4)
        batch = torch.tensor([0, 0, 1])
        out = head(h, batch)
        assert out.shape == torch.Size([2])
