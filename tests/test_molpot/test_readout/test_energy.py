import torch
from tensordict import TensorDict

from molpot.heads.heads import EnergyHead


class TestEnergyHead:
    def test_forward_energy(self):
        head = EnergyHead(hidden_dim=4)
        td = TensorDict(
            {
                ("atoms", "h"): torch.ones(3, 4),
                ("graph", "batch"): torch.tensor([0, 0, 1]),
            },
            batch_size=[],
        )
        out = head(td)
        assert ("target", "energy") in out.keys(include_nested=True)
        assert out["target", "energy"].shape == torch.Size([2])
