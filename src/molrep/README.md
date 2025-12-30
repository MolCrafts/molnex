# molrep: Molecular Typing and Embedding

`molrep` is a subpackage in `molix/src/molrep/` for molecule/atom feature prediction.

## Features

- Uses **AtomicTD** from molix as the batch container

Provides TensorDictModule components for representation learning:
- **Initializers**: Atom embeddings, spherical harmonics
- **Interactions**: Message passing blocks (topology, geometry)
- **Encoders**: Complete encoder architectures

## Example

```python
from molix.data.atomic_td import AtomicTD
from molrep import MolRepModel, ProxyLabeler
import torch

# Create batch using AtomicTD
batch = AtomicTD.create(
    z=torch.tensor([6, 1, 1, 1, 1]),
    x=torch.randn(5, 3),
    batch=torch.tensor([0, 0, 0, 0, 0]),
)

# Create model with transformer encoder
model = MolRepModel.from_config(
    encoder_type='geometry',
    hidden_dim=64,
    num_heads=4,
    num_types=11,
)

# Predict types
types = model.predict_types(batch)  # [5]

# Or extract features only
features = model.extract_features(batch)  # [5, 64]
```

## Architecture

- TopologyEncoder: Uses `torch.nn.TransformerEncoder` with per-molecule attention mask
- GeometryEncoder: Equivariant transformer with edge-conditioned attention

## File Structure

```
molix/src/molrep/
├── encoder/
│   ├── topology.py   # Transformer-based topology encoder
│   └── geometry.py   # SO(3)-equivariant geometry encoder
├── head/
│   ├── type_head.py  # Classification head
│   └── labeler.py    # Labeler protocol + ProxyLabeler
├── export/
│   └── exporter.py   # Type export utilities
├── utils/
│   └── geometry.py   # NeighborGraphBuilder, SphericalBasis, etc.
└── model.py          # MolRepModel composition
```
