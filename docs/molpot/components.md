# MolPot Components

Building a machine learning potential is like assembling Lego blocks. `molpot` provides a suite of atomic components - from pooling operators to physical readout heads - that can be tied together to build custom architectures.

## Featurization

Neural networks struggle to learn from raw distances directly. To expand distances into a richer representation, use radial basis functions from `molrep`:

```python
from molrep.embedding.radial import BesselRBF

# Bessel basis expansion
rbf = BesselRBF(num_rbf=8, cutoff=5.0)
features = rbf(distances)  # (E, 8)
```

## Cutoff Functions

A cutoff function ensures that all interactions decay smoothly to zero at the boundary, avoiding force discontinuities:

```python
from molrep.embedding.cutoff import CosineCutoff

cutoff = CosineCutoff(cutoff=5.0)
envelope = cutoff(distances)
# envelope is 1.0 at r=0 and smoothly goes to 0.0 at r=5.0
```

## Classical Potentials

`molpot` provides classical potential forms:

```python
from molpot.potentials import LJ126, BondHarmonic, AngleHarmonic

# Lennard-Jones 12-6
lj = LJ126()

# Harmonic bond spring
bond_pot = BondHarmonic()
```

## Readout Heads

The final stage of any potential produces physical observables:

- **EnergyHead**: Predicts a scalar energy from atomic features
- **ForceDerivation**: Derives forces via $F = -\nabla E$ using autograd

```python
from molpot import EnergyHead, SumPooling

# Pool per-atom energies into a total system energy
pooling = SumPooling()

# Predict scalar energy from features
head = EnergyHead(in_dim=128)
```

## Composition

`PotentialComposer` chains the full pipeline: pooling -> parameter heads -> potential terms -> aggregation.

```python
from molpot.composition import PotentialComposer, LJParameterHead
from molpot.potentials import LJ126

composer = PotentialComposer(
    head=LJParameterHead(feature_dim=64, hidden_dim=64),
    terms={"lj126": LJ126()},
)
```
