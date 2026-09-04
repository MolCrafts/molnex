# Molecular Dynamics

`molix.md` is the in-process MD engine. **`MD` is a composer**, not a façade:
each component is constructed on its own and attached.

## Quick start

```python
import torch
from molix.md import (
    MD,
    MaxwellBoltzmann,
    NeighborList,
    TrajectoryHook,
    VelocityInitHook,
    VelocityVerletIntegrator,
)
from molpot.forcefield import ForceField

ff = ForceField("water", units="metal").def_pairstyle(potential)   # MLIP or a
                                                                  # classical style
nl = NeighborList(cell=cell, cutoff=r_cut, positions=pos, skin=1.0)
md = (
    MD(torch.float64)
    .set_forcefield(ff)
    .set_neighbors(nl)                                  # PBC is the list's job
    .set_integrator(VelocityVerletIntegrator(0.5))
    .set_hooks([
        VelocityInitHook(MaxwellBoltzmann(300, seed=0)),
        TrajectoryHook("traj.pt", stride=10, numbers=Z),
    ])
)
final = md.run(system, n_steps=100_000)
```

`set_forcefield` takes a `molpot.forcefield.ForceField` (anything satisfying
`MDForceField`) — the container, not a wrapper around `ff.to_potentials()`,
which `MD` calls itself on every setter. `molix.md.ForceField` re-exports
`molpy.ForceField` for readers who want the base type.

Units are (amu, Å, fs); energy in amu·Å²/fs². **There is no `energy_scale`
keyword**: the factor is derived from the ForceField's own unit style by
`molix.units.preset_energy_to_md(ff.units)` — `units="metal"` for an eV/Å
potential, `units="real"` for a Class-I / LAMMPS kcal/mol one. Spell a
conversion out by hand only outside this path, with pint-style
`molix.units.units`::

    from molix.units import units
    ev_to_md = (1.0 * units.eV).to("amu * angstrom ** 2 / femtosecond ** 2").magnitude

## Two precisions, deliberately independent

`MD(dtype)` governs the **MD side only**: trajectory state and the
integrator's step constants. The potential's precision is the force field's:

```python
force.to(dtype=torch.float32)                        # inference
md = (
    MD(torch.float64)                                # fp64 trajectory
    .set_forcefield(force)
    .set_integrator(VelocityVerletIntegrator(0.5))
)
```

The integrator casts force output back into the state dtype at the component
boundary, so the two precisions never silently promote mid-step. Mixed
precision inside the model is the force field's job (wrap the callable /
`torch.autocast` there), not an `MD` keyword.

## Choosing a force field

There is one force-field class — `molpot.forcefield.ForceField` — and the
styles you declare on it decide what it evaluates:

| You have | Use |
|---|---|
| An MLIP (MACE, Polar, PiNet, …) | `ForceField(name, units="metal").def_pairstyle(model)` + `NeighborList` |
| A bulk Lennard-Jones system (periodic, truncated-shifted, LAMMPS `lj/cut`) | `ForceField(name).def_pairstyle("lj/cut", cutoff=...)` + `NeighborList` |
| A classical Class-I system | `ForceField(name).def_bondstyle(...)`/`def_anglestyle(...)` + `.with_params(...)` |

The **list owns the rebuild cadence**: `NeighborList(skin=, every=, delay=,
check=)` carries the LAMMPS `neigh_modify` policy, and `Integrator.eval_force`
merely asks once per force evaluation, *at the positions being evaluated* —
the list decides, into fixed-capacity buffers whose shapes never change, which
is what lets the force path stay inside a CUDA graph across the whole
trajectory. `skin=0, every=1, delay=0, check=True` is the accurate no-skin
limit (rebuild whenever anything moved); `skin > 0` is the production setting.
A frozen list for a fully compiled rollout is constructed on the integrator:
`VelocityVerletIntegrator(dt, rebuild=False)`.

Migration: `MD(rebuild_every=)` and `NeighborListHook` are removed —
the cadence lives on the list now.

A pure-GPU compiled bulk run composes the primitives directly — compile the
potential's hot core, keep the rebuild eager:

```python
from molix.compile import Compiler
from molix.md import (
    MD,
    MaxwellBoltzmann,
    NeighborList,
    VelocityInitHook,
    VelocityVerletIntegrator,
)
from molpot.forcefield import ForceField

nl = NeighborList(cell=cell, cutoff=2.5 * sigma, positions=pos, skin=1.02)
ff = (
    ForceField("ljcut", units="metal")
    .def_pairstyle("lj/cut", cutoff=2.5 * sigma)
    .with_params(atoms={"epsilon": epsilon, "sigma": sigma})
)
md = (
    MD(torch.float64)
    .set_forcefield(ff)
    .set_neighbors(nl)
    .set_integrator(VelocityVerletIntegrator(4.0))
    .set_hooks([VelocityInitHook(MaxwellBoltzmann(172.0, seed=1))])
    .to("cuda")
)
state = md.run(system, n_steps=25_000)
```

### Compiling an MLIP core

A neural potential mounted with `def_pairstyle(model)` compiles through one
primitive, `compile_core`, and the compiled build is installed **on the model**
— not on the `Potentials` container, which `MD` rebuilds on every setter call.
The sequence is the caller's to compose, and its order is load-bearing:

```python
pots = ff.to_potentials()
pots.bind_cell(cell)        # size the reciprocal grids + capacity check
pots.calc_energy(work)      # one eager forward: element table + grids
pots.compile_core(Compiler(dynamic=True))
md = (
    MD(torch.float64)
    .set_forcefield(ff)     # every setter rebuilds Potentials; the core survives
    .set_neighbors(nl)
    .set_integrator(VelocityVerletIntegrator(0.5))
)
```

`bind_cell` and the eager forward come first because both do work a traced call
cannot: sizing a k-space grid and validating the element table each read a
tensor back to the host. `compile_core` refuses with a `RuntimeError` if that
eager forward has not happened.

Use `dynamic=True` on the MD path. The neighbour list exposes only its live
edges each step, so the edge count `E` changes with every rebuild; under static
shapes dynamo would recompile on each new `E`, while a dynamic shape traces `E`
symbolically and keeps one graph. (Fixed-capacity static edges plus whole-step
CUDA-graph capture is a different design, not this one.)

The compiled region is the potential's energy core only. Forces still come from
one `torch.autograd.grad` **outside** it — a deliberate, named seam in
`molpot.derivation.force` — so a whole NPT step is "compiled core + that one
break", not zero breaks. Force-supervised *training* needs a second backward,
which is not compilable at all; keep training on the eager core.

See `benchmarks/verify_md_ljcut_nve.py` for the full melt-benchmark version
with energy-conservation checks.

A batch `TensorDict` can carry the live neighbour buffers directly —
`build(batch)` binds `edges.edge_index` / `edges.shifts` **by reference**, so
every in-place rebuild is visible through the batch with no re-binding:

```python
nl = NeighborList(cell=cell, cutoff=r_cut, positions=pos, skin=1.0)
nl.build(batch)                 # bind + first build; returns the same batch
out = potential(nl.build(batch))   # composes with forward(td) -> td

for _ in range(n_steps):
    ...
    nl.update(batch)            # per step: policy-gated rebuild
```

Note the two-statement idiom: `nl.build(batch).update(batch)` would call
**`TensorDict.update`** (a silent self-merge), not the neighbour policy —
`build` returns the batch for pipeline composition, so the policy call goes
through the list.

For large condensed-phase systems, `bin=` selects a pure-torch binned
(cell-list) O(N) build instead of the O(N²) kernel: `bin=0.0` picks the
automatic `r_build / 2` size (LAMMPS `nbin_standard`), a positive float is an
explicit perpendicular bin thickness in Å. The bin size is a cost knob, never
a physics knob — both backends produce the identical edge set. Measured at
N=4096 (CPU, `OMP_NUM_THREADS=4`, 2026-08-09): binned 0.037 s vs kernel
0.450 s per rebuild; at 48 threads the tiny-op OpenMP overhead inverts the
ratio, so pin the thread count when profiling. No timing threshold is
asserted anywhere.

## Observing a run: MD hooks

`MDRunner` drives a small, MD-specific hook protocol (`MDHook`) — these are
*not* `Trainer` hooks. Built-ins:

- `TrajectoryHook` — strided frames to a `.pt` (+ optional extended-XYZ),
  sharded to disk so host memory stays bounded.
- `MDCheckpointHook` — restartable `(pos, vel, step)` every N steps, written
  atomically; doubles as a heartbeat line in the log.

A custom hook overrides any of `on_run_start` / `on_step_start` /
`on_step_end(runner, step, obs)` / `on_run_end`; `obs` is a typed
`MDObservables` (pos, vel, forces, potential, kinetic, total, temperature).
A hook that acts every N steps declares `cadence = N` so `run(chunk=...)`
can refuse a chunking that would silently skip it.

## Custom integrators

`MD.set_integrator(...)` accepts any constructed `Integrator` subclass.
NVE is `VelocityVerletIntegrator(dt)`. Langevin is
`LangevinVerletIntegrator(dt, gamma=..., kbt=...)` with `gamma > 0`.
Mass is `system["atoms", "mass"]` (`molix.schema.MASS_KEY`), bound at run.
A conforming subclass implements `advance` (one eager step) and `rollout`
(compile-friendly loop) and inherits `initial` / `advance_n`; override the
`removed_dof` property if your scheme thermostats all 3N degrees of freedom
(the temperature estimator reads it — Langevin reports 0, NVE 3).
