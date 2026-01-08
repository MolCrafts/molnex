import torch
from tensordict import TensorDict

from molpot.heads.heads import TypeHead


class TestTypeHead:
    def test_forward_logits(self):
        head = TypeHead(hidden_dim=4, num_types=5)
        td = TensorDict(
            {
                ("atoms", "h"): torch.ones(3, 4),
            },
            batch_size=[],
        )
        out = head(td)
        assert out["atoms", "type_logits"].shape == torch.Size([3, 5])
