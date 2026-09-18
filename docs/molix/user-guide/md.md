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
survive every one of those rebuilds: `set_compiler(compiler)` compiles the
derivative op `MD` composes over the potential for each run, and `MD(dtype,
potential_dtype=…)` evaluates the potential at a second precision while the
trajectory stays at `dtype`. Both are described below.

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
geometry handed over — positions and `graphs.cell` cast per step, the edge
columns written at `potential_dtype` by the neighbour list itself
(`NeighborList.geometry_dtype`, which `set_neighbors` sets; its cast shifts
are refreshed in place on each rebuild, never per step) — with energy, forces
and virial cast back to `dtype` on the way out. Partial casts are not an
option: molpot's Coulomb kernels compare the dtype of the charges and pair
distances against the positions' and raise rather than promote. The neighbour
list's own buffers stay on the MD dtype throughout; it owns `edges` and
validates every update against its own build precision.

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
force op with `set_compiler`, keep the neighbour rebuild eager:

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
    .set_compiler(Compiler(cuda_graphs=True))   # the MD build of choice
    .to("cuda")
)
state = md.run(system, n_steps=25_000)
```

`set_compiler` is a setter like the others, and it has to be: every `set_*`
call re-runs `ff.to_potentials()`, so a build installed by hand on one
container would survive exactly until the next setter and the run would go
eager with nothing said. Holding the compiler on `MD` instead applies it to
the op of every run, which is why the order of the setters above does not
matter. The compiler is structural — any `callable -> callable`;
`molix.compile.Compiler` is molnex's, with `Compiler(cuda_graphs=True)` the
fixed-shape CUDA-graph preset the list's fixed capacity was built for — and
the recommended MD compiler, because **static shapes are the MD default**:
the op is handed the list's whole capacity buffer on every step (dead rows
included — `(0, 0)` self-loops displaced by the dead-edge shift, which price
exactly nothing), so `E` never changes between rebuilds and the whole op can
be recorded once and replayed. `MD` applies the replay contract for you (a
`cudagraph_mark_step_begin()` before each call, outputs cloned out of the
graph's static pool) whenever the installed compiler reports `cuda_graphs`;
the neighbour-list rebuild itself stays eager, outside the captured region.

What `MD.run` compiles is one `molix.autograd.Derivative` composed over the
assembled potential at `bind_work` — `wrt=("atoms.pos",)` for NVE,
`("atoms.pos", "graphs.strain")` under a barostat — the same op you would
write by hand:

```python
from molix.autograd import Derivative

pots = ff.to_potentials()
nve = Derivative(pots, wrt=("atoms.pos",))                 # graphs.energy, atoms.forces
npt = Derivative(pots, wrt=("atoms.pos", "graphs.strain"))  # + graphs.virial (1, 3, 3)
```

The op owns the position and strain leaves, rebuilds the edge geometry from
them, differentiates once and puts the batch back as it found it; it is the
compile boundary, and there is no barostat-specific core on `Potentials`. An
`lj/cut` + bonded field and a periodic Coulomb field each trace as one graph
with zero breaks in inference, forces alone and forces with the virial. The
compile sequence is always the same three steps, and `MD` does them for you:
`bind_cell` (through `rebind_cell`, once per run and after every barostat
update), **one eager evaluation** of the op on the working batch (the
once-per-topology bind — exclusion table, net-charge / edge-orientation /
unwrapped-molecule checks — an mlip's element-table check and any k-space grid
sizing are host-side work a trace cannot do), then `compiler(op)`.

`benchmarks/verify_md_ljcut_nve.py` drives all of this from the command line:
`--compile` / `--no-compile`, `--cuda-graphs`, and `--potential-precision fp32`
for the split-precision arm above.

### Compiling an MLIP field

A neural potential mounted with `def_pairstyle(model)` compiles through the
same op: `MD` composes `Derivative(pots, wrt=...)` over the container whose
`energy` hands the batch to the model's own `energy`, evaluates it once eagerly
and compiles it:

```python
md = (
    MD(torch.float64)
    .set_forcefield(ff)             # def_pairstyle(model), units="metal"
    .set_neighbors(nl)
    .set_integrator(VelocityVerletIntegrator(0.5))
    .set_compiler(Compiler(cuda_graphs=True))
)
```

Outside `MD` the same three steps are spelled by hand — bind the cell, run
one eager forward, then compile the composed op:

```python
pots = ff.to_potentials()
pots.bind_cell(cell)                                   # size the reciprocal grids + capacity check
op = Derivative(pots, wrt=("atoms.pos", "graphs.strain"))
op(work)                                               # one eager forward: element table + grids
op = Compiler(cuda_graphs=True)(op)
```

`bind_cell` and the eager forward come first because both do work a traced
call cannot: sizing a k-space grid and validating the element table each read
a tensor back to the host, and a traced call skips the element gate rather
than performing it.

Do **not** reach for `dynamic=True` on the MD path. The neighbour list binds
its full fixed-capacity buffer, so `E` is the same on every step and every
rebuild; a static trace is the one that captures, and `Compiler(cuda_graphs=True)`
is what the fixed capacity was built for (the E+F step of a small periodic
system is launch-bound, and a replayed graph is the only thing that removes
launches — see `docs/molix/explanation/throughput-and-compilation.md` for the
measured numbers). `dynamic=True` is for **ragged training** batches that are
not padded to fixed shapes — nowhere else.

The compiled region is the whole op — leaves, geometry, energy and the one
`torch.autograd.grad`; AOTAutograd supplies the energy's backward inside it.
In inference (`MD.run` is under `torch.no_grad()`) it is one dynamo graph with
no break; force-supervised *training* returns the forces attached to the
parameters and stock torch breaks once at the leaf (pytorch/pytorch#196653).

See `benchmarks/verify_md_ljcut_nve.py` for the full melt-benchmark version
with energy-conservation checks.

A batch `TensorDict` can carry the live neighbour buffers directly —
`build(batch)` binds `edges.pair_index` / `edges.shifts` **by reference**, so
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
