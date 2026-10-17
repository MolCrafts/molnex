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
    num_graphs=1,
)

print(energy.shape)  # torch.Size([]) — summed; pass pair_batch=... for one energy per graph
```

## 2. Compose Learned Parameters

`PotentialComposer` predicts per-atom parameters, mixes them into per-pair
parameters, and evaluates one or more potential terms. It is a
`molix.autograd.Potential`: `energy(frame)` writes the energy in place on a
post-collate batch, and `forward` adds the derivatives `wrt` names.

```python
from tensordict import TensorDict

from molpot.composition import LJParameterHead, PotentialComposer

composer = PotentialComposer(
    head=LJParameterHead(feature_dim=64, hidden_dim=64, dtype=torch.float64),
    potentials={"lj126": LJ126()},
    wrt=("atoms.pos",),
)

pair_index = torch.stack([torch.arange(11), torch.arange(1, 12)], dim=1)  # a chain, (E, 2)
frame = TensorDict(
    atoms=TensorDict(
        Z=torch.full((12,), 6),
        pos=torch.arange(12, dtype=torch.float64).unsqueeze(1) * torch.tensor([[1.5, 0.0, 0.0]], dtype=torch.float64),
        batch=torch.zeros(12, dtype=torch.long),
        node_features=torch.randn(12, 64, dtype=torch.float64),
        batch_size=[12],
    ),
    pairs=TensorDict(pair_index=pair_index, batch_size=[11]),
    graphs=TensorDict(batch_size=[1]),
    batch_size=[],
)

frame = composer(frame)
print(frame["graphs", "energy"].shape)        # (1,)
print(frame["graphs", "energy_lj126"].shape)  # (1,), one column per term
print(frame["atoms", "forces"].shape)         # (12, 3)
```

## Next Steps

- [Components](../user-guide/components.md)
- [Gradients and Forces](../user-guide/gradients.md)
- [Potential Composition](../explanation/potential-composition.md)
