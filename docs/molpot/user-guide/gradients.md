# Gradients, Forces and the Virial

MolNex derives forces as the negative energy gradient w.r.t. positions, and the
virial as the derivative w.r.t. a symmetric strain of the whole geometry:

```text
forces = -dE / dpos
virial =  dE / dε        (Cauchy stress σ = W / V; a shear is not doubled)
```

There is **one** derivative op for the whole project,
`molix.autograd.Derivative`, and one backend behind it, `torch.autograd.grad`.
It works on every model, including cuEquivariance fused kernels (MACE /
Allegro), and it traces in-graph under `torch.compile`
(`molix.compile.Compiler` turns on `trace_autograd_ops`). There is no
`method=` switch, no functorch route, no `compute_forces` flag and no
`compile_core`. A caller states *which quantity with respect to which
variable*, as TensorFrame keys.

## The op

```python
from molix.autograd import Derivative
from molix.compile import Compiler
from molzoo.mace import MaceMatpes

model = Derivative(MaceMatpes(...).energy, wrt=("atoms.pos",))                 # E, F = -dE/dpos
npt   = Derivative(pots.energy,            wrt=("atoms.pos", "graphs.strain"))  # + W = dE/dε
frame = model(frame)      # graphs.energy, atoms.forces[, graphs.virial] written on the frame
model = Compiler(fullgraph=True)(model)                                  # the only compile spelling
```

`Derivative(energy, *, of="graphs.energy", wrt=(), outputs=None)`:

- `energy(frame) -> frame` writes `of` (per-graph `(B,)` or a scalar) and,
  optionally, `atoms.energy`, reading only the frame it is handed. An
  `nn.Module` is registered as the submodule `energy` (its parameters flow
  through the op under an `energy.` prefix); a plain callable such as a bound
  method is held as is. An object exposing an `energy(frame)` method — every
  molnex potential — is differentiated **through that method**, whatever its
  own `wrt` is.
- `wrt` is the ordered tuple of variables, spelled as dotted strings or tuples
  (`"atoms.pos"` and `("atoms", "pos")` are the same key). `wrt=()` means
  "energy only": the op runs `energy` and returns the frame.
- `outputs` renames where each derivative lands, one entry per `wrt` key
  (`None` keeps the table). The sign is the variable's, not the key's.

| variable | output | scale | what the leaf perturbs |
|---|---|---|---|
| `atoms.pos` | `atoms.forces` | −1 | the column itself |
| `graphs.strain` | `graphs.virial` `(1, 3, 3)` | +1 | `atoms.pos`, `graphs.cell` and the lattice remainder, by `v ↦ v (I + ε_sym)ᵀ`; the gradient is symmetrised (Cauchy) |
| any other column | `<ns>.<name>_grad` | +1 | the column itself |

`graphs.strain` must follow `atoms.pos` in `wrt`, needs a one-graph batch with
`graphs.cell`, and is a `(1, 3, 3)` zero column the op creates when absent and
leaves on the frame as the variable.

Who owns the leaf: **the op makes a leaf unless the column already is one.**
A plain column — no `requires_grad`, or attached to some history — is detached
into a fresh `requires_grad` leaf inside `forward` and never left in the frame:
the caller's `atoms.pos`, `graphs.strain`, `graphs.cell` and `edges.*` tensor
objects come back exactly as handed in, with only the outputs added. A column
that already *is* a live leaf (`requires_grad`, no history) is the variable
itself: the op differentiates against it as is and leaves it in the frame,
because it is the caller's. `Derivative.mark_leaves(frame)` is the eager
prologue that makes such leaves — one fresh leaf per variable per call
(`graphs.strain` created as a `(1, 3, 3)` zero column when absent), so a
`.grad` never accumulates across steps; harmless under `torch.no_grad()`,
where nothing needs it. Cached edge geometry
(`pair_diff`, `pair_dist`, `shifts`) is rebuilt once from the perturbed
positions and lattice remainder for the energy call, so `dE/dpos` and `dE/dε`
flow through the geometry the energy actually reads. The remainder is
`edges.shifts` when the list stored it, otherwise `pair_diff − (pos[t] −
pos[s])`, a constant either way — which is why the classical container
refuses a virial on a placed list without `edges.shifts` (finding F13).

`create_graph` follows the ambient grad mode:

| mode | `graphs.energy`, `atoms.energy` | outputs |
|------|---------------------------------|---------|
| `torch.no_grad()` (MD, evaluation) | detached | detached |
| grad enabled (training) | attached | attached to the parameters — a force loss reaches them |

The op is the compile boundary: `Compiler(...)(op)` compiles leaves, strain,
geometry, energy and `autograd.grad` as one region. In inference that region
is one dynamo graph with no break. In training the outputs come back attached,
and a leaf made *inside* the region leaks through them: stock torch 2.15
breaks once, at `leaf` in `molix/autograd/variables.py`
(pytorch/pytorch#196653 lifts it). A leaf made *before* the region is a graph
input, which dynamo accepts — so a caller that wants the whole training step
as one graph on stock torch marks the columns before the traced region:

```python
op = Derivative(pot.energy, wrt=("atoms.pos", "graphs.strain"))
compiled = Compiler(fullgraph=True)(op)
frame = compiled(op.mark_leaves(frame))      # 1 graph, 0 breaks, outputs attached
```

The default training and evaluation steps do exactly this in `forward_loss`,
so `Trainer.compile()` — `cuda_graphs=True` included — needs nothing from the
caller. What an energy needs done eagerly before a trace (MACE's element-table
check, Polar's k-space grid sizing, the classical topology bind) is the
potential's precondition: run one eager forward, then compile. For the
classical container that precondition is *per frame*: `Potentials` memoises
its typed view and its geometry gate on the batch object, so the compiled op
is one graph on a frame it has evaluated once before (MD's bound batch) and
breaks at the typed-view wrap on a frame it has never seen — a fresh training
batch is such a frame; see the notes topic `compile-no-graph-breaks`.

## Potentials

Every potential exposes `energy(frame) -> frame` and takes `wrt=()` at
construction; its `forward` is `Derivative(self.energy, wrt=wrt)` built once
in `__init__`, so these are one code path:

```python
pot = MaceMatpes(..., wrt=("atoms.pos",))       # pot(frame) writes E and F
pot = Derivative(MaceMatpes(...).energy, wrt=("atoms.pos",))   # the same op, composed outside
pot = PiNetPotential(...)                        # wrt=() — energy only

pots = ff.to_potentials(wrt=("atoms.pos",))      # the classical container, forces on
npt = Derivative(ff.to_potentials().energy, wrt=("atoms.pos", "graphs.strain"))
```

`Potentials.calc_energy(frame)` is the total-energy tensor (`(B,)`, `()` for
one molecule); `Potentials.energy(frame)` stores it at `graphs.energy`; there
is no `calc_forces`, `calc_energy_forces` or `calc_energy_forces_virial`.
`molix.md.MD` composes `Derivative(pots.energy, wrt=("atoms.pos"[, "graphs.strain"]))`
once per run and applies `set_compiler`'s compiler to it after one eager
evaluation (see the MD user guide).

## Tensor level

`molpot.derivation.force` holds the raw-tensor primitives for callers that
hold an `energy_fn(pos) -> scalar` rather than a batch:

| primitive | in | out |
|-----------|----|-----|
| `autograd_forces_from_energy(E, leaf)` | `E` already on a leaf | `F` |
| `autograd_forces(energy_fn, pos)` | `energy_fn(pos) -> E` | `F` |

`ForceDerivation()` is the stateless module form of `autograd_forces`
(`BasePotential.calc_forces`, `PotentialComposer`, `Sonata`);
`HessianDerivation` differentiates the first derivative a second time on a
leaf of its own (`molix.autograd.leaf`):

```python
import torch
from molpot.derivation import ForceDerivation

pos = torch.randn(10, 3)
forces = ForceDerivation()(lambda p: (p**2).sum(), pos)
print(forces.shape)  # torch.Size([10, 3])
```

`energy_fn` must recompute any position-derived geometry (edge vectors,
distances) inside itself so the gradient flows `pos → geometry → energy`.

Units throughout: positions Å, energies eV, forces eV/Å on the MLIP side; the
ForceField's own convention on the classical side. The op only differentiates
whatever the energy writes, so a core on another unit system produces
mismatched forces silently.

## With PotentialComposer

`PotentialComposer` derives forces when positions are present in `data`:

```python
data = {
    "pair_index": pair_index,
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
the force gradient flows correctly; a force loss keeps the parameters connected
through the double backward.
