import torch

from molrep.readout.pooling import masked_sum_pooling, masked_mean_pooling


class TestMaskedSumPooling:
    def test_sum_pooling_masks_padding(self):
        features = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [10.0, 10.0]]])
        mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
        pooled = masked_sum_pooling(features, mask)
        assert pooled.shape == torch.Size([1, 2])
        assert torch.allclose(pooled, torch.tensor([[4.0, 6.0]]))


class TestMaskedMeanPooling:
    def test_mean_pooling_masks_padding(self):
        features = torch.tensor([[[2.0, 4.0], [6.0, 8.0], [10.0, 10.0]]])
        mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
        pooled = masked_mean_pooling(features, mask)
        assert pooled.shape == torch.Size([1, 2])
        assert torch.allclose(pooled, torch.tensor([[4.0, 6.0]]))
