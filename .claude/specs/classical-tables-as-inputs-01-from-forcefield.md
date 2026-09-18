---
title: Predicted Class-I tables enter the compiled region as inputs — Parameterizer.from_forcefield
status: draft
created: 2026-09-11
chain: classical-tables-as-inputs
depends: derivation-head-04-molix-derivative
---

## Summary

Operator directive (2026-09-11): 把参数表改成纯输入进编译区，按 `from_forcefield`
那个形状做.

The learnable Class-I training step cannot be compiled today. Measured on
GH200 (`.scratch/classical_train_compile/`, torch 2.15.0.dev, with and without
the local dynamo patches):

| shape | result |
|---|---|
| parameterize + assemble **inside** the traced region | 23 graphs / 22 breaks — `ForceField.__new__`, PME option branches, four host syncs, one dynamic shape. Every one of these is a molpy object or an eager precondition, i.e. exactly what this spec removes from the hot path — none is physics. |
| parameterize **outside**, trace the op (today) | `Unsupported: autograd.grad with external grad_fn` under `fullgraph=True` |

Both are the same fact: the predicted parameter tables are produced by the
encoder **outside** the region and reach the energy as attributes of per-batch
term modules, carrying a `grad_fn` dynamo classifies as external. Inference is
unaffected and is already 1 graph / 0 breaks, forces and virial alike.

**Handing the tables in as frame columns does not help** — measured, not
assumed (`.scratch/classical_train_compile/where_tables_enter.py`): the same
energy with `k` as a module attribute and with `k` as a column of the argument
frame both give 3 graphs / 2 breaks and the identical
`autograd.grad with external grad_fn`. What dynamo objects to is the
**grad_fn**, not the entry point; a table computed outside is external however
it arrives. (A history-free buffer clears that error and hits the leak check
instead.)

So the fix is not to move the boundary but to remove the seam: one long-lived
module whose `energy(frame)` *computes* the tables inside the region, so they
are ordinary intermediates of it, and which never constructs a molpy object on
the hot path.

## Design

1. **`Parameterizer.from_forcefield(template, *, encoder, **heads)`** builds one
   `nn.Module` and reads the template **once**: declared styles, resolved pair
   options, nonbonded scaling, units, cutoffs. Nothing of molpy survives into
   `forward`. `Parameterizer.forward` keeps returning a filled `ForceField` for
   the export path (`ff.with_params(**model.tables(batch))` → LAMMPS / OpenMM /
   manifest); the training path stops using it.

2. **Bonded terms take their parameters per call**, so the physics can be built
   once and fed tables that were computed inside the region — as the pair terms
   already do (`LJCutPotential.energy(*, epsilon, sigma, ...)`,
   `CoulombOpenPotential.energy(*, charges, ...)`). `BondHarmonic`,
   `AngleHarmonic`, `ProperTorsionPeriodic` and the two improper terms register
   `k` / `r0` / … as buffers at construction; they gain a keyword path
   (`energy(*, k=..., r0=..., ...)`) that overrides the buffer. The buffer stays
   for a fixed, non-learned field — that is what an exported force field is.

3. **`ClassIEnergy`** — the physics container, built once, holding **no** table:
   `energy(frame, tables) -> frame` walks the terms it assembled from the
   template and passes each its slice of `tables`. This is `Potentials` minus
   the ForceField reads (8 sites, 6 tables: `bonds`, `angles`, `propers`,
   `impropers`, `atoms.charge`, `atoms.{epsilon,sigma}`).

4. **The Class-I model** is `encoder → heads → tables → ClassIEnergy`, all in
   one `energy(frame)` — this is the load-bearing part, per the measurement
   above: the encoder and the heads must run **inside** the traced region, or
   the tables carry an external `grad_fn` wherever they are put. Wrapped once by
   `molix.autograd.Derivative` and compiled once with `Compiler(fullgraph=True)` — `fullgraph` is load-bearing, not a
   preference: a break at the autograd leaf leaves the `autograd.grad` outside
   the region and the outer backward becomes a double backward, which
   AOTAutograd refuses (measured: `aot_eager` and `inductor` both raise,
   `+fullgraph` both pass with parameter gradients equal to eager at 0.0).

5. **Eager preconditions stay eager and stay once.** The topology bind, the
   net-charge report, the edge-orientation refusal, the unwrapped-bond gate,
   the exclusion table and (periodic) the k-space grid bind run at bind time,
   not per energy call — the contract `molix.autograd.Derivative` already
   states.

## Files

`src/molpot/forcefield/parameterizer.py`, `src/molpot/forcefield/potentials.py`,
`src/molpot/potentials/{bonds,angles,dihedrals,impropers}/*.py`,
`scripts/mm_param_learning/class_i_fit.py` (`ForceAwareModel` composes the new
model), mirrored tests.

## Testing

- Bonded terms: the keyword path and the buffer path give the same energy and
  the same forces; the keyword path is differentiable w.r.t. the passed table.
- `ClassIEnergy` holds no parameter table (attribute scan) and reproduces
  `Potentials.energy` on the butane fixture to 1e-12.
- slow: `Compiler.explain` of the composed op on a real packed batch →
  `graph_count == 1 and graph_break_count == 0` in inference, and in training
  under the local patch; a force-supervised step through
  `Compiler(fullgraph=True)` matches eager parameter gradients to 1e-12.
- The export path still produces a filled `ForceField` whose tables equal the
  ones the energy used.

## Out of scope

Static shapes / CUDA graphs (spec `derivation-head-02`); compiling the
optimizer; the `MultiTaskTrainStep` router host sync (spec `derivation-head-03`).
