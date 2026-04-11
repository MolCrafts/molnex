# MolPot Quickstart

```python
import torch
from molpot.potentials import LJ126

# Create a Lennard-Jones potential
potential = LJ126()

# Compute energy from pairwise distances
bond_dist = torch.rand(40) + 0.5
energy = potential(bond_dist=bond_dist, sigma=3.0, epsilon=0.1)

print(energy.shape)
```

For composing potentials with ML encoder features, see [Components](components.md).
