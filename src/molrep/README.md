# molrep

Generic molecular representation components (embeddings, irreps helpers, chemistry perception), built on pure PyTorch modules. A block that serves one model lives in that model's `molzoo` package.

## Input Conventions

Components consume the following tensors (or subsets):

- `Z`: Atomic numbers `(N,)`
- `pos`: Coordinates `(N, 3)`
- `edge_index`: Edge indices `(E, 2)`
- `edge_diff`: Edge vectors `(E, 3)`
- `edge_dist`: Edge distances `(E,)`

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

Z = torch.tensor([6, 1, 1, 1, 1])
h = embed(Z=Z)
print(h.shape)  # (5, 128)
```

## Modules

- `embedding/`: Node, radial, angular, and cutoff embeddings
- `interaction/`: Irreps-string helpers (`product.py`)
- `heads/`: Scalar / type heads over node features
