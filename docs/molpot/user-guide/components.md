# Components

MolPot provides potential terms, parameter heads, pooling layers, derivation
operators, and composition utilities.

## Potential Terms

Potential terms are PyTorch modules that compute energy from distances and
per-pair parameters.

```python
from molpot.potentials import AngleHarmonic, BondHarmonic, LJ126

lj = LJ126()
bond = BondHarmonic()
angle = AngleHarmonic()
```

`LJ126` expects `distance`, `epsilon_ij`, and `sigma_ij`. Its default mixing
function converts per-atom `epsilon` and `sigma` values to per-pair values.

## Parameter Heads

Heads predict per-atom parameters from node features.

```python
from molpot.composition import LJParameterHead

head = LJParameterHead(feature_dim=128, hidden_dim=64)
params = head(node_features)
```

## Pooling

Pooling layers reshape encoder output before potential composition.

```python
from molpot.pooling import LayerPooling, SumPooling

layer_pool = LayerPooling("mean")
sum_pool = SumPooling()
```

## Composition

```python
from molpot.composition import PotentialComposer
from molpot.potentials import LJ126

composer = PotentialComposer(
    head=LJParameterHead(feature_dim=64),
    potentials={"lj126": LJ126()},
)
```

The composer returns:

- `energy`
- `term_energies`
- `parameters`
- `forces`, when `compute_forces=True`
