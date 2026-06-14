# Embeddings

Embeddings convert discrete and continuous molecular inputs into learned feature
vectors.

## Discrete Atom Embeddings

```python
import torch
from molrep.embedding.node import DiscreteEmbeddingSpec, JointEmbedding

embed = JointEmbedding(
    embedding_specs=[
        DiscreteEmbeddingSpec(input_key="Z", num_classes=119, emb_dim=64),
    ],
    out_dim=128,
)

Z = torch.tensor([1, 6, 8])
vectors = embed(Z=Z)
print(vectors.shape)  # torch.Size([3, 128])
```

## Continuous Inputs

`JointEmbedding` can combine multiple specs. Continuous specs use a learned
linear projection before concatenation.

```python
from molrep.embedding.node import ContinuousEmbeddingSpec

embed = JointEmbedding(
    embedding_specs=[
        DiscreteEmbeddingSpec(input_key="Z", num_classes=119, emb_dim=64),
        ContinuousEmbeddingSpec(input_key="charge", in_dim=1, emb_dim=16),
    ],
    out_dim=128,
)
```

## Edge Features

Distances and directions are usually encoded separately:

```python
from molrep.embedding.angular import SphericalHarmonics
from molrep.embedding.cutoff import CosineCutoff
from molrep.embedding.radial import BesselRBF

rbf = BesselRBF(r_cut=5.0, num_radial=8)
cutoff = CosineCutoff(r_cut=5.0)
sh = SphericalHarmonics(l_max=2)
```

These modules are used inside reference encoders such as MACE and Allegro.
