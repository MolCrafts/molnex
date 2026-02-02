# MolPot Quickstart

**Goal**: Define a simple potential and calculate the energy of a bond.

In this guide, we won't train a neural network. Instead, we'll use a classical **Harmonic Bond** potential to demonstrate how `molpot` calculates energy from geometry.

## 1. Define the Potential

We want to model a bond between two atoms that acts like a spring:
$E = 0.5 \cdot k \cdot (r - r_0)^2$

Let's say our spring constant $k=100.0$ and equilibrium length $r_0=1.0$.

```python
import torch
from molpot.potentials.bond_harmonic import BondHarmonic

# Define parameters for 1 bond type (type 0)
k = torch.tensor([100.0])   # Spring constant
r0 = torch.tensor([1.0])    # Equilibrium length

potential = BondHarmonic(k=k, r0=r0)
```

## 2. Create the System

We create two atoms positioned $1.5$Å apart. Since $r_0=1.0$, the bond is stretched by $0.5$Å.

```python
from molix.data.atomic_td import AtomTD

# Two atoms at (0,0,0) and (1.5, 0, 0)
batch = AtomTD.create(
    z=torch.tensor([1, 1]),
    x=torch.tensor([[0., 0., 0.], [1.5, 0., 0.]]),
    batch=torch.tensor([0, 0]),
)

# We must manually define the topology (bond) between them
# Bond from atom 0 -> atom 1, type 0
batch["bonds", "i"] = torch.tensor([[0], [1]]) 
batch["bonds", "type"] = torch.tensor([0])
```

## 3. Calculate Energy

Run the potential.

```python
energy = potential(batch)
print(f"Bond Energy: {energy.item()}")
```

**Expected Result**:
$E = 0.5 \cdot 100 \cdot (1.5 - 1.0)^2 = 0.5 \cdot 100 \cdot 0.25 = 12.5$

## Next Steps

*   [**Components**](components.md): Learn how to build neural network potentials.
*   [**Automatic Gradients**](gradients.md): How to get Forces from Energy for free.
