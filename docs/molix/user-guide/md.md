# Molecular Dynamics

`molix.md` is the in-process MD engine: BAOAB Langevin velocity-Verlet over
any trained potential, compilable end to end. **`MD` is the entry point** —
the lower layers (`ForceField`, `Integrator`, `MDRunner`) are the primitives
it composes, and you reach for them directly only when you need a custom loop.

## Quick start

```python
import torch
from molix.md import MD, MaxwellBoltzmann, PotentialForceField, TrajectoryHook

force = PotentialForceField(potential, template)   # bind a potential to a system
md = MD(force, mass=masses, dt=0.5, gamma=0.1, temperature=300.0,
        dtype=torch.float64,
        hooks=[TrajectoryHook("traj.pt", stride=10, numbers=Z)])

vel = MaxwellBoltzmann(masses).sample(300.0, seed=0)
final = md.run(pos, vel, n_steps=100_000)          # -> MDState(pos, vel, forces, energy)
```

Units are (amu, Å, fs); energy in amu·Å²/fs². Drive an eV/Å potential with
`energy_scale=1 / molix.units.EV_PER_AMU_A2_FS2` on the force field.

## Two precisions, deliberately independent

`MD(dtype=)` governs the **MD side only**: trajectory state, integrator step
constants, mass. The potential's precision is a separate axis:

```python
md = MD(force, mass=m, dt=0.5, dtype=torch.float64)  # fp64 trajectory ...
md.set_potential_dtype(torch.float32)                # ... over fp32 inference
```

This is what lets you study the MD process and the inference process
separately — an fp64 trajectory over a quantized/fp32 model is a supported,
meaningful configuration. At the component boundary the integrator casts the
force field's output back into the state dtype, so the two precisions never
silently promote mid-step. For mixed precision inside the model only, use
`MD(autocast_dtype=torch.bfloat16)`.

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
A frozen list for a fully compiled rollout goes through the integrator seam:
`MD(ff, ..., integrator=LangevinVerletIntegrator(ff, ..., rebuild=False))`.

Migration: `MD(rebuild_every=)` and `NeighborListHook` are removed —
the cadence lives on the list now.

A pure-GPU compiled bulk run composes the primitives directly — compile the
force field, keep the rebuild eager:

```python
from molix import Compiler
from molix.md import MD, LennardJonesCutForceField, MaxwellBoltzmann, NeighborList

nl = NeighborList(cell=cell, cutoff=2.5 * sigma, positions=pos, skin=1.02)
ff = LennardJonesCutForceField(epsilon=eps, sigma=sigma, neighbors=nl).to("cuda", torch.float64)
ff = Compiler(cuda_graphs=True)(ff)          # or Compiler(fullgraph=True)
md = MD(ff, mass=39.95, dt=4.0, gamma=0.0,   # γ=0 → NVE; the list owns the cadence
        dtype=torch.float64, device="cuda")
vel = MaxwellBoltzmann(39.95, n_atoms=len(pos)).sample(172.0, seed=1)
state = md.run(pos, vel, n_steps=25_000)
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

`MD(integrator=...)` accepts any constructed `Integrator` subclass. A
conforming subclass implements `advance` (one eager step) and `rollout`
(compile-friendly loop) and inherits `initial` / `advance_n`; override the
`removed_dof` property if your scheme thermostats all 3N degrees of freedom
(the temperature estimator reads it — Langevin reports 0, NVE 3).
