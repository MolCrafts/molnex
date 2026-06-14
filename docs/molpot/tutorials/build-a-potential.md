# Build a Potential

This tutorial calls a Lennard-Jones potential directly, then shows how it fits
into `PotentialComposer`.

## 1. Call a Potential Term

```python
import torch
from molpot.potentials import LJ126

potential = LJ126()

distance = torch.rand(40) + 0.5
epsilon_ij = torch.full((40,), 0.1)
sigma_ij = torch.full((40,), 3.0)

energy = potential(
    distance=distance,
    epsilon_ij=epsilon_ij,
    sigma_ij=sigma_ij,
)

print(energy.shape)  # scalar tensor
```

## 2. Compose Learned Parameters

`PotentialComposer` predicts per-atom parameters, mixes them into per-pair
parameters, and evaluates one or more potential terms.

```python
from molpot.composition import LJParameterHead, PotentialComposer

composer = PotentialComposer(
    head=LJParameterHead(feature_dim=64, hidden_dim=64),
    potentials={"lj126": LJ126()},
)

node_features = torch.randn(12, 64)
data = {
    "edge_index": torch.randint(0, 12, (40, 2)),
    "bond_dist": torch.rand(40) + 0.5,
    "batch": torch.zeros(12, dtype=torch.long),
}

outputs = composer(node_features=node_features, data=data)
print(outputs["energy"].shape)
```

## Next Steps

- [Components](../user-guide/components.md)
- [Gradients and Forces](../user-guide/gradients.md)
- [Potential Composition](../explanation/potential-composition.md)
