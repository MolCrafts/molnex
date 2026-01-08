import torch
from tensordict import TensorDict

from molrep.readout.heads import EnergyHead


class TestEnergyHead:
    def test_forward_mean_pooling(self):
        head = EnergyHead(d_model=8, pooling="mean")
        td = TensorDict(
            {
                ("rep", "h"): torch.ones(2, 3, 8),
                ("atoms", "mask"): torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool),
            },
            batch_size=[],
        )
        out = head(td)
        assert ("pred", "energy") in out.keys(include_nested=True)
        assert out["pred", "energy"].shape == torch.Size([2, 1])

    def test_invalid_pooling(self):
        try:
            EnergyHead(d_model=8, pooling="max")
        except ValueError as exc:
            assert "pooling" in str(exc)
        else:
            raise AssertionError("Expected ValueError for invalid pooling")
