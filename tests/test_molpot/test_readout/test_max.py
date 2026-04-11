import torch

from molpot.pooling import MaxPooling


class TestMaxPooling:
    def test_max_pooling(self):
        pooling = MaxPooling()
        x = torch.tensor([[1.0, 2.0], [3.0, 1.0], [10.0, 20.0]])
        batch = torch.tensor([0, 0, 1])
        out = pooling(x, batch)
        assert torch.allclose(out, torch.tensor([[3.0, 2.0], [10.0, 20.0]]))
