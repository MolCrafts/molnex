# molzoo

Molecular encoder zoo. Provides encoder-only architectures (MACE, Allegro) without built-in energy/force readout. Downstream potential composition is handled by `molpot.composition`.

## Input Conventions

Both encoders accept keyword tensors:

- `Z`: Atomic numbers `(N,)`
- `bond_dist`: Edge distances `(E,)`
- `bond_diff`: Edge vectors `(E, 3)`
- `edge_index`: Edge indices `(E, 2)`

Output: `(N, num_layers, feature_dim)` — per-atom, per-layer features.

## Usage

```python
import torch
from molzoo import MACE, MACESpec
from molrep.embedding.node import DiscreteEmbeddingSpec
from molpot import LayerPooling, PotentialComposer, LJParameterHead, LJ126

encoder = MACE(MACESpec(
    node_attr_specs=[DiscreteEmbeddingSpec(input_key="Z", num_classes=119, emb_dim=64)],
    num_elements=119,
    num_features=64,
    r_max=5.0,
))

Z = torch.randint(0, 10, (20,))
features = encoder(
    Z=Z,
    bond_dist=torch.rand(80),
    bond_diff=torch.randn(80, 3),
    edge_index=torch.randint(0, 20, (80, 2)),
)

pool = LayerPooling("mean")
node_features = pool(features)  # (20, 64)
```
