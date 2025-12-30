# Atom Embeddings

## What is an Embedding?

Before a neural network can process a "Carbon" atom, it must be converted into a vector of numbers. The `AtomEmbedding` module acts as a lookup table, mapping atomic numbers (Z) to learnable vectors.

## Usage

```python
from molrep.encoder.embedding import AtomEmbedding

# Create embedding for elements H(1) to Zn(30)
embed = AtomEmbedding(
    num_embeddings=31,  # 0 is padding, 1-30 are elements
    embedding_dim=64
)

# Input: Batch of atomic numbers
z = torch.tensor([1, 6, 8]) # H, C, O
vectors = embed(z) # [3, 64]
```

## Initialization Principles

Can we do better than random initialization? Yes. `molrep` initializes embeddings using random orthogonal matrices or simplified versions of electronic properties (like period and group) to give the model a "head start" in understanding chemistry.

## Integration

The `AtomEmbedding` is automatically included when you use `MolRepModel`, but you can also use it standalone if you are building a custom architecture.
