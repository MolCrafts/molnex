# Gradients and Forces

MolPot derives forces from energy with functorch (`torch.func.grad`):

```text
forces = -dE / dpos
```

The energy is differentiated as a **pure function of positions**, so the force
is traced into the forward graph — `energy → force → loss` needs only one
ordinary backward and composes with `torch.compile(fullgraph=True)` (no
double-backward barrier).

## Usage

```python
import torch
from molpot.derivation import ForceDerivation

pos = torch.randn(10, 3)  # no requires_grad needed — torch.func.grad tracks it

force_deriv = ForceDerivation()
forces = force_deriv(lambda p: (p**2).sum(), pos)  # energy_fn(pos) -> scalar

print(forces.shape)  # torch.Size([10, 3])
```

`energy_fn` must recompute any position-derived geometry (edge vectors,
distances) inside itself so the gradient flows `pos → geometry → energy`.

## With PotentialComposer

`PotentialComposer` derives forces when positions are present in `data`:

```python
data = {
    "edge_index": edge_index,
    "batch": batch,
    "pos": pos,
}

outputs = composer(
    node_features=node_features,
    data=data,
    compute_forces=True,
)

forces = outputs["forces"]
```

The composer recomputes edge distances from `pos` inside its force closure, so
the force gradient flows correctly. A force loss backpropagates to parameters
with a single ordinary backward.
