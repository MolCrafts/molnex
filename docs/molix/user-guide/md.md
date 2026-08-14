# Molecular Dynamics

`molix.md` is the in-process MD engine. **`MD` is a composer**, not a façade:
each component is constructed on its own and attached.

## Quick start

```python
import torch
from molix.md import (
    MD,
    MaxwellBoltzmann,
    PotentialForceField,
    TrajectoryHook,
    VelocityVerletIntegrator,
)

force = PotentialForceField(potential, template)   # bind a potential to a system
# system["atoms"] holds pos, vel, mass (amu)
md = (
    MD(torch.float64)                              # MD-side dtype only
    .set_forcefield(force)                         # force field owns its dtype
    .set_integrator(VelocityVerletIntegrator(0.5))
    .set_hooks([TrajectoryHook("traj.pt", stride=10, numbers=Z)])
)

MaxwellBoltzmann().sample(system, 300.0, seed=0)   # reads atoms.mass, writes atoms.vel
final = md.run(system, n_steps=100_000)            # -> MDState; writes atoms.pos / vel
```

Units are (amu, Å, fs); energy in amu·Å²/fs². Convert with pint-style
`molix.units.units`::

    from molix.units import units
    ev_to_md = (1.0 * units.eV).to("amu * angstrom ** 2 / femtosecond ** 2").magnitude
    kcal_to_md = (1.0 * units.kilocalorie_per_mole).to(
        "amu * angstrom ** 2 / femtosecond ** 2"
    ).magnitude

Drive an eV/Å potential with `energy_scale=ev_to_md`. Drive a Class-I /
LAMMPS `units real` (kcal/mol) potential with `energy_scale=kcal_to_md`.

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

| You have | Use |
|---|---|
| A molpot potential + collated template (open system) | `PotentialForceField` |
| A TensorDict potential reading `edges.shifts` + a periodic cell | `PeriodicPotentialForceField` (owns a rebuilding `NeighborList`) |
| A bulk Lennard-Jones system (periodic, truncated-shifted, LAMMPS `lj/cut`) | `LennardJonesCutForceField` over a `NeighborList` |
| Any `pos -> (energy, forces)` callable (AOTI `.pt2`, compiled closure, external engine) | `CallableForceField` |
| An analytic test PES | `HarmonicForceField`, `LennardJonesForceField` |

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
force field, keep the rebuild eager:

```python
from molix import Compiler
from molix.md import (
    MD,
    LennardJonesCutForceField,
    MaxwellBoltzmann,
    NeighborList,
    VelocityVerletIntegrator,
)

nl = NeighborList(cell=cell, cutoff=2.5 * sigma, positions=pos, skin=1.02)
ff = LennardJonesCutForceField(epsilon=eps, sigma=sigma, neighbors=nl).to("cuda", torch.float64)
ff = Compiler(cuda_graphs=True)(ff)          # or Compiler(fullgraph=True)
md = (
    MD(torch.float64)
    .set_forcefield(ff)
    .set_integrator(VelocityVerletIntegrator(4.0))
    .to("cuda")
)
MaxwellBoltzmann().sample(system, 172.0, seed=1)
state = md.run(system, n_steps=25_000)
```

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
