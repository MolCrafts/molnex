# Gradients and Forces

In MolPot, forces are computed via autograd:

$$
F = -\frac{\partial E}{\partial x}
$$

## Usage

```python
import torch
from molpot.derivation import ForceDerivation

pos = torch.randn(10, 3, requires_grad=True)
energy = (pos ** 2).sum().reshape(1)

force_deriv = ForceDerivation()
forces = force_deriv(energy, pos)
print(forces.shape)  # (10, 3)
```

Prerequisite: `pos.requires_grad = True`.

## How It Works

`ForceDerivation` uses `torch.autograd.grad` to compute the negative gradient of the scalar energy with respect to atomic positions. This is the standard approach in ML potentials, ensuring exact forces consistent with the energy surface.
