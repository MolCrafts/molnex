import torch

from molpot.readout.pooling import MeanPooling


class TestMeanPooling:
    def test_mean_pooling(self):
        pooling = MeanPooling()
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0]])
        batch = torch.tensor([0, 0, 1])
        out = pooling(x, batch)
        assert torch.allclose(out, torch.tensor([[2.0, 3.0], [10.0, 20.0]]))
