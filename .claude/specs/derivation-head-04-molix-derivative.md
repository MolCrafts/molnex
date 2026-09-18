---
title: molix.autograd.Derivative — the one derivative op; Potential.forward = energy; compile via Compiler()
status: active (operator decisions 2026-09-11: D-a both, D-b frame keys only, D-c delete, D-d accept, D-e molix.autograd)
created: 2026-09-11
chain: derivation-head
depends: derivation-head-01-unify
supersedes: DerivativeHead (molpot.derivation.head), compile_core (everywhere)
---

## Summary

Operator directives (2026-09-11): (1) lift the derivative head to a molix-internal
op — the user only states *which quantity w.r.t. which variable*; (2) redesign the
Potential forward API on top of it (review required); (3) remove `compile_core`,
compile with `molix.Compiler()` like any module. One usage for the whole project:

```python
from molix.autograd import Derivative

model = Derivative(MaceMatpes(...), wrt=("atoms.pos",))                  # E, F = -∂E/∂pos
npt   = Derivative(pots,            wrt=("atoms.pos", "graphs.strain"))  # + W = ∂E/∂ε
frame = model(frame)          # graphs.energy, atoms.forces[, graphs.virial] written on the frame
model = Compiler(fullgraph=True)(model)                                   # the only compile spelling

pot = MaceMatpes(..., wrt=("atoms.pos",))   # same op built in __init__; pot(frame) writes E and F
```

Everything is addressed by TensorFrame keys — nested tuples or dotted strings,
normalised to tuples; no `Positions` / `Strain` classes in the public API. The
variable → output conventions live in `molix.autograd` as a key table:
`atoms.pos → atoms.forces` (scale −1), `graphs.strain → graphs.virial`
(symmetrised strain gradient, `(1, 3, 3)`), any other column → `<ns>.<name>_grad`
(scale +1); `outputs=` overrides the table when a caller needs another key.

## Design

### `molix.autograd` (new module, molix base layer — no molpot import)

`Derivative(energy, *, of=ENERGY_KEY, wrt=(), outputs=None)` — `nn.Module`.
`energy` is either an `nn.Module` (registered as submodule `energy`, so
parameters / `state_dict` flow through the op with an `energy.` prefix) or a
plain callable such as a bound method (not registered — this is how a potential
builds the op over its own `energy` method in `__init__` without a module cycle).
Contract of `energy`: `energy(frame) -> frame`, writes `of` (per-graph `(B,)`
or scalar) and optionally `atoms.energy`, reads only the frame it is handed.
`wrt=()` means "energy only": the op just runs `energy` and returns the frame,
so every potential forward is the same code path.

`forward(frame) -> frame`:

1. `create_graph = torch.is_grad_enabled()` at entry.
2. For each key in `wrt`, in order, the op makes a fresh `requires_grad` leaf it
   owns and writes the perturbed columns: `atoms.pos` → the leaf replaces the
   column; `graphs.strain` → a `(1, 3, 3)` zero column (created if absent, left
   on the frame as the variable) whose leaf `ε` maps `atoms.pos`, `graphs.cell`
   and the lattice remainder by `v ↦ v (I + ε_sym)ᵀ`; any other column → the
   leaf replaces it. Order matters only for `graphs.strain`, which must follow
   `atoms.pos` (the op enforces it).
3. Edge geometry is rebuilt once from the perturbed columns:
   `edges.pair_diff = pos[t] − pos[s] + shifts`, `edges.pair_dist`, with `shifts`
   = `edges.shifts` or `edge_diff_buf − Δpos` (detached).
4. `frame = energy(frame)`; `E = frame[of]`; one
   `torch.autograd.grad(E.sum(), leaves, create_graph=create_graph)`.
5. Outputs written per the key table (or `outputs=`); the caller's `atoms.pos`,
   `graphs.cell`, `edges.shifts/pair_diff/pair_dist` objects put back; `of` and
   `atoms.energy` detached iff not `create_graph`. Straight-line code, no
   tensor-valued branches, no `try/finally`.

The per-key behaviours (`leaf`, `apply`, `write`) are private handlers inside
`molix.autograd`, selected by key; adding a variable kind (a field column for
dipoles) is a new private handler plus a table row, no public class.
`symmetrize_strain_grad` and the leaf helper move here from molpot (pure math);
`molix.schema` gains `STRAIN_KEY = ("graphs", "strain")`; `molix.engine`
adapters use the op (F10 closed).

### Potential forward API

A potential exposes `energy(frame) -> frame` (writes `graphs.energy`, and
`atoms.energy` when it has one) and takes `wrt=()` at construction: `forward`
is `Derivative(self.energy, wrt=wrt)` built once in `__init__`, so
`pot(frame)` writes the energy alone (`wrt=()`, the default) or energy plus the
requested derivatives — the same op a caller composes outside with
`Derivative(pot, wrt=...)`. No `compute_forces`, no `energy_forces`, no head,
no `compile_core`, no second kernel.

- `MacePotential` / `MacePolar` / `PiNetPotential` / `Allegro`: `forward` unpacks
  the frame into the raw kernel (`energy_core(pos, Z, pair_index, batch, B,
  pair_diff, cell)` stays as the model's own raw-tensor seam, used by export)
  and writes the energy. `_write_energy` → `forward`; `energy_forces`, `head`,
  `compile_core`, `compute_forces` deleted.
- `Potentials` (classical container, optional mlip): `forward(frame)` = today's
  `calc_energy`; `calc_forces`, `calc_energy_forces`, `calc_energy_forces_virial`,
  `compile_core`, the `pos=` overrides and the `Mapping` input path go. The
  analytic classical force core `_classical_core` is deleted (D-c): forces come
  from the op through the same energy.
- `EnergyReadout` / `ForceReadout` deleted (`pot(frame)` / `Derivative(pot, wrt=("atoms.pos",))(frame)`).
- MD: `_TensorDictForce.bind_work` composes `Derivative(pots, wrt=(POS_KEY,
  *((STRAIN_KEY,) if barostat else ())))` once; `MD.set_compiler(c)` applies `c` to
  that op; `_eval_*` variants collapse to one call that reads
  `work[FORCES_KEY]`, `work[VIRIAL_KEY]`, `work[ENERGY_KEY]`.
- Trainer: `Trainer(Derivative(model, wrt=("atoms.pos",)), loss_fn, ...)` or
  `Trainer(MaceMatpes(..., wrt=("atoms.pos",)), ...)`;
  `smoke` / `train` unchanged; `Trainer.compile` = `Compiler(...)(self.model)`.

### compile_core → `Compiler()`

Deleted on `DerivativeHead` (gone), `MacePotential`, `Potentials`. The eager
warm-up precondition (element table, Polar k-grid) is documented on the
potential: run one eager forward, then compile the composed op.

## Decisions (operator, 2026-09-11)

- **D-a** both: the op composes outside a potential, and a potential builds the
  same op in `__init__` from `wrt=` — one code path, no second kernel.
- **D-b** no variable classes; everything is a TensorFrame key
  (`"atoms.pos"`, `"graphs.strain"`, `("graphs", "energy")`).
- **D-c** delete the classical analytic force core.
- **D-d** the `energy.` state_dict prefix is accepted.
- **D-e** `molix.autograd`.

## Files

Create `src/molix/autograd/{__init__,derivative,variables}.py`, tests
`tests/test_molix/test_autograd/`. Delete `src/molpot/derivation/{head,force_readout,energy_readout}.py`,
`stress.py` (moved), the raw adapters. Modify `src/molzoo/mace/{potential,polar}.py`,
`src/molzoo/pinet/potential.py`, `src/molzoo/allegro.py`, `src/molpot/forcefield/potentials.py`,
`src/molix/md/driver.py`, `src/molix/core/trainer.py`, `src/molix/engine/{adapter,static}.py`,
`src/molpot/derivation/{__init__,force,hessian}.py`, docs (`gradients.md`, `md.md`,
`throughput-and-compilation.md`), CLAUDE.md "Force derivation (one head)" paragraph.

## Testing

`test_autograd/test_derivative.py`: forces vs analytic + central differences;
virial vs central-difference strain with explicit shifts; frame restored exactly;
detach policy; `create_graph=True` loss reaches parameters; variable order
(`graphs.strain` after `atoms.pos`, enforced); a generic column variable via `outputs=`; slow: `Compiler.explain`
1 graph / 0 breaks in inference for `wrt=("atoms.pos",)` and `("atoms.pos", "graphs.strain")`,
training-mode break only at the leaf. Mirrored updates for every migrated file;
MD driver test asserts one composed op per run and NPT reads `graphs.virial`.

## Out of scope

Static shapes (spec 02) and step compile (spec 03) — both now depend on this spec.
