# molpot

ML potential toolkit: classical potential functions, prediction heads, derivation operators, and a modular composition layer (pooling -> parameter heads -> potential terms -> aggregation).

## Modules

- `potentials/`: LJ126, BondHarmonic, AngleHarmonic, DihedralHarmonic, and more
- `heads/`: AtomicEnergyMLP, EnergyHead, TypeHead
- `derivation/`: EnergyAggregation, HessianDerivation (`H = d²E/dpos²`); batch-level `F` / `W` are `molix.autograd.Derivative`
- `pooling/`: LayerPooling, EdgeToNodePooling, SumPooling, MeanPooling, MaxPooling
- `composition/`: LJParameterHead, MultiHead, PotentialComposer, Sonata

## Usage

`PotentialComposer` is a `molix.autograd.Potential`: `energy(frame)` runs
head -> mixing -> terms in place on a post-collate batch, and `forward` is
`Derivative(self.energy, wrt)` built once in `__init__`.

```python
import torch
from tensordict import TensorDict

from molpot.composition import LJParameterHead, PotentialComposer
from molpot.pooling import LayerPooling
from molpot.potentials import LJ126

# Encoder output (n_atoms, n_layers, feat_dim) -> atoms.node_features (n_atoms, feat_dim)
encoder_features = torch.randn(12, 3, 64, dtype=torch.float64)
pair_index = torch.stack([torch.arange(11), torch.arange(1, 12)], dim=1)  # a chain, (E, 2)
frame = TensorDict(
    atoms=TensorDict(
        Z=torch.full((12,), 6),
        pos=torch.arange(12, dtype=torch.float64).unsqueeze(1) * torch.tensor([[1.5, 0.0, 0.0]], dtype=torch.float64),
        batch=torch.zeros(12, dtype=torch.long),
        node_features=LayerPooling("mean")(encoder_features),
        batch_size=[12],
    ),
    pairs=TensorDict(pair_index=pair_index, batch_size=[11]),
    graphs=TensorDict(batch_size=[1]),
    batch_size=[],
)

composer = PotentialComposer(
    head=LJParameterHead(feature_dim=64, hidden_dim=64, dtype=torch.float64),
    potentials={"lj126": LJ126()},
    wrt=("atoms.pos",),
)
frame = composer(frame)

print(frame["graphs", "energy"].shape)  # (num_graphs,)
print(frame["graphs", "energy_lj126"].shape)  # (num_graphs,), one column per term
print(frame["atoms", "forces"].shape)  # (n_atoms, 3)
```

`composer.energy(frame)` is the same pipeline without the derivative;
`wrt=("atoms.pos", "graphs.strain")` adds `graphs.virial` on a one-graph
periodic batch.

## Training Integration

Use `molix.Trainer` for training and `molix.core.losses` for loss functions.
