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

Two things are `MD`'s own policy rather than a component's, because both have to
be re-applied on every one of those rebuilds: `set_compiler(compiler)` compiles
the potential's hot core, and `MD(dtype, potential_dtype=…)` evaluates the
potential at a second precision while the trajectory stays at `dtype`. Both are
described below.

Units are (amu, Å, fs); energy in amu·Å²/fs². **There is no `energy_scale`
keyword**: the factor is derived from the ForceField's own unit style by
`molix.units.preset_energy_to_md(ff.units)` — `units="metal"` for an eV/Å
potential, `units="real"` for a Class-I / LAMMPS kcal/mol one. Spell a
conversion out by hand only outside this path, with pint-style
`molix.units.units`::

    from molix.units import units
    ev_to_md = (1.0 * units.eV).to("amu * angstrom ** 2 / femtosecond ** 2").magnitude

## Two precisions, deliberately independent

`MD(dtype)` governs the **MD side**: trajectory state (positions, velocities,
masses) and the integrator's step constants. `potential_dtype` governs the
force field's numbers. Leave it out and there is one precision everywhere and
no cast on the force path at all; pass it and the whole evaluation moves:

```python
md = (
    MD(torch.float64, potential_dtype=torch.float32)  # fp64 trajectory,
    .set_forcefield(ff)                               # fp32 potential
    .set_integrator(VelocityVerletIntegrator(0.5))
)
```

A field's numbers live in three places, so `MD` moves all three: the container's
parameter tables (the per-atom ε / σ / q a `Potentials` reads live off the
ForceField on every call, restated by `ForceField.to_dtype`), the assembled
module's own buffers and any MLIP weights (`.to(potential_dtype)`), and the
geometry handed over per step — positions, `graphs.cell`, the edge columns —
cast on the way in, with energy, forces and virial cast back to `dtype` on the
way out. Partial casts are not an option: molpot's Coulomb kernels compare the
dtype of the charges and pair distances against the positions' and raise rather
than promote. The neighbour list stays on the MD dtype throughout; it owns
`edges` and validates every update against its own build precision.

Why the split exists at all: the quantization-as-thermal-noise study
(`molix.quant`) runs an fp64 integrator over an fp32 potential, because
lowering the *potential's* precision is the force perturbation under study and
its effect on the trajectory can only be read against an integrator that is not
itself a second noise source. Running the whole loop at fp32 confounds the two.
`benchmarks/verify_md_ljcut_nve.py --potential-precision fp32` is that arm.

Below `MD`, the integrator casts force output back into the state dtype at the
component boundary as well, so the two precisions never silently promote
mid-step. Mixed precision *inside* a model (`torch.autocast`) remains the force
field's business, not an `MD` keyword.

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
potential's hot core with `set_compiler`, keep the rebuild eager:

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
    .set_compiler(Compiler())            # or Compiler(cuda_graphs=True)
    .to("cuda")
)
state = md.run(system, n_steps=25_000)
```

`set_compiler` is a setter like the others, and it has to be: every `set_*`
call re-runs `ff.to_potentials()`, so a compiled core installed by hand on the
container would survive exactly until the next setter and the run would go
eager with nothing said. Passing the compiler to `MD` instead re-applies it
after each rebuild, which is why the order of the setters above does not
matter. The compiler is structural — any `callable -> callable`;
`molix.compile.Compiler` is molnex's, with `Compiler(cuda_graphs=True)` the
fixed-shape CUDA-graph preset the list's fixed capacity was built for.

What gets compiled is `Potentials`' classical `(system, pos) -> (energy,
forces)` core. A `lj/cut` + bonded field traces as one graph with zero breaks;
a field that also declares `pair:coulomb` shows exactly one, at the
`torch.autograd.grad` seam in `molpot/derivation/force.py` — Coulomb has no
closed-form force — and that break is named and owned there rather than
appearing unexplained mid-kernel. The once-per-topology bind (exclusion table,
net-charge / edge-orientation / unwrapped-molecule checks) happens *outside*
the compiled region on the first call, so the core is one graph from the start.

`benchmarks/verify_md_ljcut_nve.py` drives all of this from the command line:
`--compile` / `--no-compile`, `--cuda-graphs`, and `--potential-precision fp32`
for the split-precision arm above.

### Compiling an MLIP core

A neural potential mounted with `def_pairstyle(model)` compiles through the same
primitive, `compile_core`, but the compiled build is installed **on the model** —
not on the `Potentials` container, which `MD` rebuilds on every setter call. The
model is the one object that survives those rebuilds, so an MLIP core installed
by hand does survive them, unlike a classical one:

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

`md.set_compiler(Compiler(dynamic=True))` reaches the same seam — it forwards to
`pots.compile_core` after each rebuild — and is the shorter spelling once the
eager forward above has run. Attach it to a wired `MD` (force field *and*
integrator already set) before that forward has happened and it raises the same
`RuntimeError`, which is the intended failure: an MLIP whose element table was
never validated cannot be traced, and a silent eager fallback would be reported
as a compiled run.

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
- `EnergyDriftHook` — the NVE (constant atom number, volume and total energy)
  energy-conservation gate: it records `(t, E_tot)` on every observation and
  fits a straight line through the series.

Why a fit and not a tolerance on `E_tot`: velocity-Verlet is *symplectic*, so
it conserves a Hamiltonian slightly displaced from the real one and the true
total energy oscillates with a bounded amplitude set by `dt`. That amplitude is
not an error. An error is a *secular* term underneath it — energy leaking in or
out step after step, which comes from the model rather than the integrator (a
force truncated discontinuously at a cutoff, a neighbour list rebuilt too late,
a force that is not the gradient of the energy beside it). The line separates
the two, and the run is judged from the `EnergyDrift` record the hook publishes
on its `.result` attribute at `on_run_end`:

```python
from molix.md import EnergyDriftHook

drift = EnergyDriftHook()
md.set_hooks([VelocityInitHook(MaxwellBoltzmann(300, seed=0)), drift])
md.run(system, n_steps=200, chunk=10)

assert drift.result.rel_drift < 1e-3      # secular drift, dimensionless
print(drift.result.rms_rel)               # bounded fluctuation, dimensionless
```

`rel_drift` is `|slope · duration_fs| / |e0|`: the fitted `dE_tot/dt`
(amu·Å²/fs³) extrapolated over the sampled window `duration_fs = t_last −
t_first` (fs) and divided by the first sampled total energy `e0`
(amu·Å²/fs²). `rms_rel` is the root-mean-square deviation of `E_tot` from its
own mean over the same window, divided by `|e0|` as well, so both numbers are
dimensionless and comparable across system sizes. The hook declares
no `cadence`, so the sample spacing is `run(chunk=...)` and the first sample
lands at `chunk · dt` — a run shorter than two observations has no slope, so
`on_run_end` leaves `result` at `None` (ending such a run is not an error);
`drift()` is what raises `ValueError`, for a caller that asks for a fit the
samples cannot support. `on_run_start` drops the previous run's samples, so a
hook reused for a warm-up and then a timed run reports the timed run only.
The `< 1e-3` figure is the gate `benchmarks/verify_md_ljcut_nve.py` applies to
bulk lj/cut argon.

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
