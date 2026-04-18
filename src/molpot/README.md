# molpot

ML potential toolkit: classical potential functions, prediction heads, derivation operators, and a modular composition layer (pooling -> parameter heads -> potential terms -> aggregation).

## Modules

- `potentials/`: LJ126, BondHarmonic, AngleHarmonic, DihedralHarmonic, and more
- `heads/`: AtomicEnergyMLP, EnergyHead, TypeHead
- `derivation/`: EnergyAggregation, ForceDerivation, StressDerivation
- `pooling/`: LayerPooling, EdgeToNodePooling, SumPooling, MeanPooling, MaxPooling
- `composition/`: LJParameterHead, MultiHead, PotentialComposer

## Usage

```python
import torch
from molpot.composition import (
    LayerPooling,
    LJParameterHead,
    PotentialComposer,
)
from molpot.potentials import LJ126

# Encoder output: (n_nodes, n_layers, feat_dim)
encoder_features = torch.randn(12, 3, 64)

pool = LayerPooling("mean")
node_features = pool(encoder_features)

composer = PotentialComposer(
    head=LJParameterHead(feature_dim=64, hidden_dim=64),
    terms={"lj126": LJ126()},
)

data = {
    "edge_index": torch.randint(0, 12, (40, 2)),
    "bond_dist": torch.rand(40) + 0.2,
    "batch": torch.zeros(12, dtype=torch.long),
    "pos": torch.randn(12, 3, requires_grad=True),
}
out = composer(node_features=node_features, data=data, compute_forces=True)

print(out["energy"].shape)   # (num_graphs,)
print(out["forces"].shape)   # (n_nodes, 3)
```

## Training Integration

Use `molix.Trainer` for training and `molix.core.losses` for loss functions.
