import torch
from molpot.heads.heads import TypeHead


class TestTypeHead:
    def test_forward_logits(self):
        head = TypeHead(hidden_dim=4, num_types=5)
        h = torch.ones(3, 4)
        out = head(h)
        assert out.shape == torch.Size([3, 5])
