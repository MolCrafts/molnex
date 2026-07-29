# Gradients and Forces

MolPot derives forces as the negative energy gradient w.r.t. positions:

```text
forces = -dE / dpos
```

The single public entry point is `molpot.derivation.ForceDerivation`. It has
**two explicit backends** (no auto-detect, no silent fallback):

| `method` | Implementation | When to use |
|----------|----------------|-------------|
| `"autograd"` (**default**) | `torch.autograd.grad` | Any model, including cuEquivariance fused kernels (MACE / Allegro). Always correct. |
| `"functorch"` | `torch.func.grad` | Pure-PyTorch energy graphs only (e.g. PiNet). Traces force into the forward graph so `energy → force → loss` is a single backward and can use `torch.compile(fullgraph=True)`. |

cuEq fused ops register a legacy `autograd.Function` without `setup_context`,
which functorch rejects (pytorch#170834) — pick `"autograd"` for those models.

`BasePotential.calc_forces` always uses the autograd backend so the molpy
`PotentialProtocol` path works for every potential.

## Usage

```python
import torch
from molpot.derivation import ForceDerivation

pos = torch.randn(10, 3)

# Default: safe for every model (including cuEq / MACE)
force_deriv = ForceDerivation()  # method="autograd"
forces = force_deriv(lambda p: (p**2).sum(), pos)

# Pure-torch models that want fullgraph compile (e.g. PiNet):
# force_deriv = ForceDerivation(method="functorch")
# pos needs no requires_grad_ — torch.func.grad tracks it.

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
the force gradient flows correctly. With the autograd backend, a force loss
keeps parameters connected via double backward; with functorch, force is part
of the forward graph and a single ordinary backward reaches the parameters.
