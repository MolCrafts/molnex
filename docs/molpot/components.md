# MolPot Components

Building a machine learning potential is like assembling Lego blocks. `molpot` provides a suite of atomic components—from feature expansions to physical readout heads—that can be tied together to build custom architectures.

## Featurization

Neural networks struggle to learn from raw distances directly. A distance of $1.5$Å vs $1.6$Å might mean the difference between a stable bond and a broken one, but to a linear layer, it's just a small numerical difference.

To fix this, we project distances into a higher-dimensional space using **Radial Basis Functions (RBFs)**.

```python
from molpot import GaussianRBF

# Creates a "fingerprint" of the local environment
# 50 filters spread between 0 and 5.0 Angstroms
rbf = GaussianRBF(num_rbf=50, cutoff=5.0)

distances = torch.tensor([1.0, 1.5, 2.0])
features = rbf(distances) # Output shape: [3, 50]
```

This transforms a single scalar into a rich vector of activitis, allowing the network to distinguish specific bonding patterns easily.

## Cutoff Functions

In molecular simulations, atoms are constantly moving in and out of each other's "neighborhood". If an interaction abruptly turns on or off when an atom crosses a $5.0$Å threshold, it creates an infinite force spike (derivative of a step function). This crashes simulations instantly.

A **Cutoff Function** ensures that all interactions decay smoothly to zero at the boundary.

```python
from molpot import CosineCutoff

cutoff = CosineCutoff(cutoff=5.0)
envelope = cutoff(distances) 
# envelope is 1.0 at r=0 and smoothly goes to 0.0 at r=5.0
```

## Interaction Blocks

Once we have geometric features, we need to process them. In standard potentials, this is often done with a simple Multi-Layer Perceptron (MLP) that updates the atomic representations.

However, `molpot` also supports classical potential forms. For example, the **Harmonic Bond** potential models bonds as simple springs.

```python
from molpot.potentials.bond_harmonic import BondHarmonic

# k=spring constant, r0=equilibrium length
bond_pot = BondHarmonic(k=..., r0=...)
```

## Readout Heads

The final stage of any potential is the "Readout". This component takes the high-dimensional latent vectors of atoms and produces physical observables.

*   **EnergyHead**: Predicts a scalar energy value. Usually, this involves a weighted sum of atomic energies.
*   **ForceHead**: Derives forces by taking the negative gradient of the energy with respect to atomic positions ($F = -\nabla E$). 

Because `molpot` is built on PyTorch's autograd, you rarely need to implement forces manually. You define the energy model, and `ForceHead` handles the derivatives automatically.

```python
from molpot import EnergyHead, SumPooling

# 1. Pool per-atom energies into a total system energy
pooling = SumPooling()

# 2. Predict scalar energy from features
head = EnergyHead(in_dim=128)
```
