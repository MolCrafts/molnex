# MolRep Quickstart

```python
import torch
from molrep.embedding.node import DiscreteEmbeddingSpec, JointEmbedding

embed = JointEmbedding(
    embedding_specs=[DiscreteEmbeddingSpec(input_key="Z", num_classes=119, emb_dim=32)],
    out_dim=64,
)

Z = torch.tensor([6, 1, 1, 1, 1])
h = embed(Z=Z)
print(h.shape)  # (5, 64)
```

In a full model, `h` is typically passed into interaction and readout modules.
