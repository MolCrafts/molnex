# Automatic Gradients

One of the killer features of `molpot` is that you rarely need to write code to calculate forces. Because we build on PyTorch, we use **Automatic Differentiation** to compute forces as the negative gradient of the energy.

$$ \vec{F}_i = -\nabla_{\vec{r}_i} E $$

## Enabling Gradients

When you create input data, simply enable gradient tracking on positions.

```python
batch = AtomTD.create(...)
batch["atoms", "x"].requires_grad_(True)
```

## computing Forces

You can compute forces manually using `torch.autograd.grad`.

```python
# 1. Forward Pass
energy = model(batch)

# 2. Backward Pass (Compute derivatives)
forces = -torch.autograd.grad(
    energy, 
    batch["atoms", "x"], 
    create_graph=True,  # Keep graph if you need Hessians later
    retain_graph=True
)[0]
```

## The ForceHead

In practice, you should use `molpot.heads.ForceHead`. This component wraps the logic above and handles edge cases (like batching) for you.

```python
from molpot.heads import ForceHead

model = ForceHead(energy_model)
outputs = model(batch)
# outputs["target", "force"] now contains the computed forces
```

## Why is this better?
1.  **Consistency**: Guaranteed conservation of energy (forces are conservative).
2.  **Simplicity**: You only design the Energy function.
3.  **Higher Order**: You can easily compute Hessians (vibration) or stress tensors.
