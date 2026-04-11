# Atom Embeddings

## What is an Embedding?

Before a neural network can process a "Carbon" atom, it must be converted into a vector of numbers. The embedding module maps atomic numbers (Z) to learnable vectors.

## Usage

```python
import torch
from molrep.embedding.node import DiscreteEmbeddingSpec, JointEmbedding

embed = JointEmbedding(
    embedding_specs=[
        DiscreteEmbeddingSpec(input_key="Z", num_classes=119, emb_dim=64),
    ],
    out_dim=128,
)

Z = torch.tensor([1, 6, 8])  # H, C, O
vectors = embed(Z=Z)  # (3, 128)
```

## JointEmbedding

`JointEmbedding` supports combining multiple discrete and continuous embeddings into a single output vector. Each embedding spec defines an input key, and the results are concatenated and projected to the output dimension.

## Integration

The embedding layer is used as the first stage in encoders like MACE and Allegro (see `molzoo`).
