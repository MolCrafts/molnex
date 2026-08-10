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

## Batch-level force-pass kernels (批级力学传递内核)

`ForceDerivation` / `force.py` is the **tensor-level** layer: you hand it
`energy_fn(pos) -> scalar` and get `forces (N, 3)` back. A potential, though,
works on a post-collate batch — it has to own the position leaf, write
`graphs.energy` / `atoms.forces` in place, and decide whether the returned
energy stays attached. That is the **batch-level** layer,
`molpot.derivation.kernels`:

| Layer | Module | Signature | Owns |
|-------|--------|-----------|------|
| tensor | `molpot.derivation.force` | `energy_fn(pos) -> scalar` | the only `torch.autograd.grad` / `torch.func.grad` calls |
| batch | `molpot.derivation.kernels` | `energy_core(batch) -> batch` | position-leaf ownership, key writes, `detach_energy` |

```python
from molpot.derivation import func_force_pass, grad_force_pass

# one energy forward + torch.autograd.grad on the position leaf
batch = grad_force_pass(self._write_energy, batch, detach_energy=False)

# single torch.func.grad(..., has_aux=True) pass — fullgraph-compilable
batch = func_force_pass(self._write_energy, batch)
```

The kernels **compose** `force.py` — they never re-derive a gradient — so the
backend boundary of the table above still holds: `grad_force_pass` for cuEq
fused kernels (MACE-shaped), `func_force_pass` for pure-PyTorch graphs (PiNet)
that want `torch.compile(fullgraph=True)`. `PiNetPotential`, `GradMode` and
`FuncMode` all bind these; a potential must not hand-roll a third pass body.

`grad_force_pass(energy_core=None, ...)` skips the forward and differentiates
an energy already materialised on a live position leaf — that is what keeps
`EnergyReadout(backward=True)` + `ForceReadout` at a single model forward.

### `detach_energy` (three states)

Who owns the position leaf decides whether the returned `graphs.energy` can
still be part of a loss:

| `detach_energy` | Behaviour | Caller |
|-----------------|-----------|--------|
| `False` | never detach — energy stays attached for an energy loss | `PiNetPotential`, `GradMode` |
| `True` | always detach — energy is a reported quantity only | inference / logging |
| `None` (default) | detach **iff the kernel created the position leaf**, i.e. the caller had no graph to lose | MACE-style `get_outputs` |

`func_force_pass` has no such knob: its only callers want the energy attached,
and the fused kernels that need leaf-detaching cannot use functorch anyway
(pytorch#170834).

Units throughout: positions Å, energies eV, forces eV/Å. The kernels only
differentiate whatever the energy core wrote, so a core on another unit system
produces mismatched forces silently.

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
