# Build an Encoder

This tutorial builds the first stage of a molecular encoder: atom features from
atomic numbers. Full reference encoders such as MACE and Allegro live in
`molzoo`.

## 1. Create an Atom Embedding

```python
import torch
from molrep.embedding.node import DiscreteEmbeddingSpec, JointEmbedding

embed = JointEmbedding(
    embedding_specs=[
        DiscreteEmbeddingSpec(input_key="Z", num_classes=119, emb_dim=32),
    ],
    out_dim=64,
)
```

## 2. Embed Atomic Numbers

```python
Z = torch.tensor([6, 1, 1, 1, 1])
h = embed(Z=Z)

print(h.shape)  # torch.Size([5, 64])
```

The output is a per-atom feature tensor. Interaction blocks and readout layers
can consume this tensor directly or combine it with radial and angular edge
features.

## 3. Add Radial Features

```python
from molrep.embedding.cutoff import CosineCutoff
from molrep.embedding.radial import BesselRBF

bond_dist = torch.rand(12) * 5.0

rbf = BesselRBF(r_cut=5.0, num_radial=8)
cutoff = CosineCutoff(r_cut=5.0)

edge_features = rbf(bond_dist) * cutoff(bond_dist).unsqueeze(-1)
print(edge_features.shape)  # torch.Size([12, 8])
```

## Next Steps

- [Encoders](../user-guide/encoders.md)
- [Embeddings](../user-guide/embeddings.md)
- [MolZoo reference models](../../molzoo/index.md)
