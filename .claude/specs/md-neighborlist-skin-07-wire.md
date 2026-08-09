---
title: Wire the list-owned rebuild policy through integrator, driver and consumers (md-neighborlist-skin link 07)
status: approved
created: 2026-08-09
grilled: true
chain: md-neighborlist-skin
---

# Wire the list-owned rebuild policy through integrator, driver and consumers (md-neighborlist-skin link 07)

## Summary

The rebuild cadence still lives in three places: `Integrator.rebuild_every` /
`_force_eval_count` (a modulo counter on force evaluations), `MD(rebuild_every=)`
(a driver kwarg that poked that counter), and `NeighborListHook` (a legacy
step-start rebuild with a measured ~30× NVE drift penalty). Link 04 gave
`molix.md.NeighborList` a real policy — `update(positions) -> bool` under the
LAMMPS `every` / `delay` / `check` gate over a Verlet skin — so those three are
now duplicate, weaker owners of a decision the list already makes correctly.
This link deletes all three and leaves exactly one owner: the integrator asks
once per force evaluation, at the positions being evaluated, and the list
decides. What the integrator keeps is a *static* on/off — a construction-time
Python bool derived from the force field — so a frozen-list integrator still
`torch.compile(fullgraph=True)`s while a live-list integrator drives the policy
eagerly. `MD(rebuild_every=)` is removed outright (breaking, pre-1.0, hard
removal per repo norm), every in-repo consumer moves to `NeighborList(skin=,
every=, delay=, check=)`, and the GH200 melt benchmark is rerun at `skin > 0` as
the scientific close-out of the chain.

## Domain basis

Units: `skin` / `cutoff` / `r_build` / displacements in Å; `every` / `delay` /
`ago` in MD steps; `dt` in fs; energy in amu·Å²/fs² (`molix.units.EV_PER_AMU_A2_FS2`
converts to eV); `rebuild_count` / `ndanger` dimensionless counters; relative
energy drift dimensionless.

**Why the policy must be evaluated at the force-evaluation positions.** BAOAB /
velocity-Verlet evaluates `F` at the *end-of-step* positions (`Integrator.step`
/ `step_nve` call `eval_force(pos)` after both half drifts). Refreshing the list
at step *start* — what `NeighborListHook` does — leaves the connectivity one
displacement behind the positions entering `F = -∇E`, so the forces are not the
gradient of the energy surface the list defines. That is a systematic, one-signed
leak, not a symmetric discretisation error: it was measured ~30× worse at
`every=5` than at `every=1` on MACE-MatPES water (recorded in the hook's own
warning and in `src/molzoo/specs/mace_matpes.md` §7.4). `Integrator.eval_force`
is therefore the only admissible seam, and this link makes it the only one that
exists.

**Half-skin criterion (from link 04, not re-derived here).** A list built at
`r_build = cutoff + skin` contains every pair within `cutoff` while
`max_i ‖x_i − x_i(t_build)‖ ≤ skin/2` (worst case: the two largest displacements
sit on the two atoms of the offending pair). `update()` tests exactly that, with
the raw (non-minimum-image) displacement, under the conjunctive
`ago ≥ delay ∧ ago mod every == 0` gate.

**Why a missed pair shows up as energy drift (basis of invariant (c)).**
`LennardJonesCutForceField` shifts the pair energy by `E_lj(r_cut)` so the energy
is continuous at the cutoff, but the *force* is not: `F(r_cut⁻) ≠ 0`. A pair that
is inside `r_cut` and absent from the list therefore contributes an `O(1)` force
error for as long as it is missing, which integrates into a one-signed energy
leak. Conversely a *superset* list (the skin region) changes nothing: those pairs
are masked to exactly zero by `r² < cutoff_sq`. Hence the binding statement of
this link: a skin-gated run must conserve energy as well as a rebuild-every-step
run to within a small factor, and must not merely "look similar" — the
rebuild-every-step baseline drift is itself nonzero (velocity-Verlet at finite
`dt`), so a ratio test with a nonzero-baseline assertion is the falsifiable form.

**Conservation metrics.** Unit test: `max_t |E_tot(t) − E_tot(0)| / |E_tot(0)|`
over a 100-step float64 CPU trajectory. Benchmark: `|slope·duration| / |E_tot(0)|`
from a least-squares fit of the sampled `E_tot(t)` series, PASS bound `1e-3`
(unchanged from the current script).

**Benchmark state point.** LAMMPS `melt`: FCC argon at reduced density
ρ\* = 0.8442, `T0*` = 1.44 (⇒ `T0` = `T0*·ε/k_B` ≈ 172 K), `r_c = 2.5 σ`, with
ε = 0.0103 eV, σ = 3.4 Å, m = 39.95 amu, `dt` = 4 fs. New default skin
`1.02 Å = 0.3 σ`: half-skin 0.51 Å against a per-atom ballistic displacement of
`v_rms·dt ≈ sqrt(k_B T/m)·√3·dt ≈ 0.013 Å` per step at 172 K (and less once the
crystal melts and T falls), i.e. a rebuild roughly every 20–40 steps, which is
the `rebuild_count ≪ n_steps` this link claims. (0.3 σ is also the
`neighbor 0.3 bin` setting distributed with the LAMMPS `melt` example — cited for
orientation, **not** re-verified offline for this spec.)

**References.**

- Leimkuhler & Matthews, "Rational Construction of Stochastic Numerical Methods
  for Molecular Sampling", *Appl. Math. Res. Express* 2013 —
  https://doi.org/10.1093/amrx/abs010 (BAOAB; `F` at end-of-step positions).
- Allen & Tildesley, *Computer Simulation of Liquids*, 2nd ed. (2017), §5.2 —
  truncated-and-shifted LJ: energy continuous at `r_cut`, force discontinuous.
- LAMMPS `neigh_modify` / `neighbor` / "Run output (Dangerous builds)" —
  https://docs.lammps.org/neigh_modify.html
- K. Nordlund, *Introduction to molecular dynamics simulations*, lecture 3 —
  https://www.mv.helsinki.fi/home/knordlun/moldyn/lecture03.pdf (two-atom /
  half-skin criterion, open and directly verifiable).
- L. Verlet, *Phys. Rev.* **159**, 98 (1967) —
  https://doi.org/10.1103/PhysRev.159.98; B. Quentrec & C. Brot,
  *J. Comput. Phys.* **13**, 430 (1973) —
  https://doi.org/10.1016/0021-9991(73)90046-6. **Caveat inherited from link 04:**
  both are paywalled and were not re-verified; attribution only.

## Design

Preconditions (assumed landed): link 03 renamed the MD-side list to
`molix.md.NeighborList`; link 04 gave it `skin` / `every` / `delay` / `check`,
`update(positions) -> bool`, `rebuild(positions)`, `ago`, `rebuild_count`,
`ndanger`, `r_build`, and widened `NeighborStrategy` with `cutoff` / `skin` /
`update`; link 05 moved edge ownership into `build(td)`. This link changes **no**
policy semantics — it only moves the *ownership of the decision to ask* and
deletes the competing owners. It is agnostic to the build backend, so link 06
(binned build) may land before or after it.

**1. Force field — the seam grows a capability flag, `rebuild_neighbors` shifts
meaning.**

```python
class ForceField(nn.Module):
    @property
    def rebuilds_neighbors(self) -> bool:      # base: no list, nothing to refresh
        return False

    def rebuild_neighbors(self, pos: torch.Tensor) -> None:  # base: no-op
        ...
```

`rebuilds_neighbors` mirrors the existing `Integrator.removed_dof` pattern — a
read-only property with a documented default that subclasses override, replacing
what would otherwise be a `getattr(force, "neighbors", None)` duck-read (forbidden
by the repo rules, and link 04 already deleted the analogous duck-read in
`LennardJonesCutForceField.__init__`). Overrides:

| type | `rebuilds_neighbors` | `rebuild_neighbors(pos)` |
|---|---|---|
| `ForceField`, `PotentialForceField`, `HarmonicForceField`, `LennardJonesForceField` | `False` | no-op (inherited) |
| `PeriodicPotentialForceField`, `LennardJonesCutForceField` | `True` | `self.neighbors.update(pos)` |
| `CallableForceField` | `self.neighbors is not None` | `update(pos)` when bound, else no-op |
| `driver._AutocastForceField` | `self.inner.rebuilds_neighbors` | delegates (unchanged) |

The **semantic shift** is the load-bearing part: `rebuild_neighbors` no longer
means "rebuild now", it means "run the neighbour policy at these positions" —
the list may decline. The name is kept deliberately (renaming would churn every
subclass, `_AutocastForceField`, and the docs for zero behavioural gain); the
docstring states the shift and names the escape hatch: **a forced rebuild is
`neighbors.rebuild(pos)`**, still available and still the primitive `update`
delegates to. Return type stays `None` — callers that need counts read
`neighbors.rebuild_count` / `ndanger`, which is where they already live.

**2. Integrator — a static bool, not a counter.**

`rebuild_every` and `_force_eval_count` are deleted. `Integrator.__init__` gains
a keyword-only `rebuild: bool | None = None` and sets

```python
self.rebuild: bool = bool(force.rebuilds_neighbors) if rebuild is None else bool(rebuild)
```

and `eval_force` guards a single call:

```python
def eval_force(self, pos):
    if self.rebuild:
        self.force.rebuild_neighbors(pos)   # the LIST decides (skin/every/delay/check)
    out = self.force(pos)
    return ForceOutput(out.energy.to(pos.dtype), out.forces.to(pos.dtype))
```

`self.rebuild` is a plain Python bool (**never** a buffer or tensor): dynamo
specialises the branch at trace time, so `rebuild=False` leaves the body dead and
`torch.compile(ig.rollout, fullgraph=True)` still traces to one graph — the
compiled-path invariant. `rebuild=True` is the eager production path: the policy
runs between compiled force calls, exactly as `rebuild_every` did, and the
fixed-capacity buffers keep every shape static so a compiled / CUDA-graph-captured
force field survives. `Integrator.rebuild` answers *whether this integrator asks*;
`ForceField.rebuilds_neighbors` answers *whether the force field can* — two
different questions, hence two names. The positional `force` argument is
unchanged, so the conforming-subclass seam (`_MidpointEuler(Integrator)` in the
tests, `LangevinVerletIntegrator`) keeps working with `super().__init__(force)`.

**3. Driver — derives, never pokes.**

`MD(rebuild_every=)` and the `integrator.rebuild_every = …` / `_force_eval_count = 0`
poke are deleted, along with `MD.rebuild_every`. `MD` builds the default
integrator and lets it derive its own flag from the force field it holds
(the autocast wrapper is applied *before* the integrator is constructed, and
delegates the property, so wrapping order stays irrelevant). No new `MD` kwarg:
the frozen/compiled-rollout configuration is reached through the existing
`integrator=` seam —
`MD(ff, mass=…, integrator=LangevinVerletIntegrator(ff, dt=…, gamma=0.0, kbt=0.0, mass=…, rebuild=False))`
— which is composition at the caller, per the repo's "no all-in-one façade" rule.
Docstrings rewritten: the module docstring's "owns … the **neighbour-list
cadence**" claim becomes integrator + MD-side precision only; the
`rebuild_every` argument block is replaced by a paragraph pointing at
`NeighborList(skin=, every=, delay=, check=)`; `MD.run`'s `chunk` note drops
"when `rebuild_every` is set"; the "`rebuild_every=1` is the accurate NVE setting
without a Verlet skin" line becomes "`skin=0, every=1, delay=0, check=True` is
the accurate no-skin limit (rebuild whenever anything moved); `skin > 0` is the
production setting".

**4. `NeighborListHook` is deleted**, not deprecated — from
`src/molix/md/runner.py`, the `molix.md` import and `__all__`, the
`MDRunner.run` cadence example, the `forcefield.py` module docstring, the
`molix.md` package docstring, `TestNeighborListHook`, and the `test_runner.py`
docstring that cites it. Stage is `experimental`; the repo norm is hard removal
with a CHANGELOG entry and a migration line. `on_step_start` itself stays (a
lifecycle callback with other users); only the wrong-seam hook goes.

**5. Accepted cost.** With `rebuild=True` every force evaluation now runs one
max-displacement reduction plus a `float()` host sync inside `update()`. That is
strictly cheaper than what it replaces (`rebuild_every=1` ran the full neighbour
kernel every force evaluation, and also synced), but on GPU it is a per-step
device→host sync that a fully captured loop would not have. It is the accepted
price of a correct, list-owned policy; the melt benchmark prints `steps/s` so the
cost stays measured, and ac-008 is the guard.

**Reuse decision.** No `librarian_report` was supplied with this task — the
caller should note the missing blueprint advisory. Candidates below come from a
direct read of `src/molix/md/`, `benchmarks/`, `scripts/`, `docs/`.

- `molix.md.NeighborList.update()` / `.rebuild()` / `.rebuild_count` / `.ndanger`
  (link 04) — **reuse**. This link adds no policy state and no second gate; every
  cadence question is answered by the list.
- `NeighborStrategy` (link 04, declares `cutoff` / `skin` / `update`) — **reuse**
  unchanged. The new capability flag is asked of the *force field*, not the list,
  so the protocol is untouched.
- `ForceField.rebuild_neighbors` — **generalize**: the existing seam is promoted
  from "unconditional rebuild" to "run the list's policy", serving both the
  integrator's per-force-eval call and the frozen/no-list case, with
  `neighbors.rebuild(pos)` documented as the forced-build route (spawns task 4,
  not a parallel "implement" task).
- `Integrator.eval_force` — **reuse** as the single call site; this link removes
  code from it rather than adding a new hook point.
- `driver._AutocastForceField` — **reuse**; one added delegating property.
- `molix.md.MD` / `MDRunner` / `MDHook` / `MaxwellBoltzmann` /
  `LennardJonesCutForceField` / `molix.units` constants — **reuse** as the test
  and regression harness. No new MD driver, no hand-rolled velocity-Verlet loop
  in `tests/` or `regressions/`.
- `molix.compile.Compiler` — **reuse** in the benchmark and the doc example
  (compile the force field, keep the loop and the policy eager).
- `molix.md.runner.NeighborListHook` — **deleted, not reused** (wrong seam;
  see Domain basis).
- `tests/…/test_forcefield.py::_cubic_lattice` and
  `tests/…/test_neighbors.py::_lattice` — **generalize**: byte-identical
  duplicates; promoted once into `tests/test_molix/test_md/conftest.py` as
  `make_cubic_lattice(n_side=3, spacing=3.0)` and imported by package path (the
  repo's shared-test-code rule), now that a third caller (`test_driver.py`) needs it.
- `ForceField.rebuilds_neighbors` (property) and `Integrator.rebuild`
  (construction-time bool + kwarg) — **new**: no existing symbol answers "is a
  neighbour policy live on this run". Naming and shape follow the closest
  in-tree pattern, `Integrator.removed_dof` (read-only capability property with a
  documented default, overridden by subclasses) and the existing
  `ForceField.rebuild_neighbors` no-op default; construction-time
  `ValueError`-style validation is not needed (a bool has no invalid value).

## Files to create or modify

- `src/molix/md/forcefield.py` — `rebuilds_neighbors` property on the base and
  four subclasses; `rebuild_neighbors` delegates to `neighbors.update(pos)`;
  module + method docstrings (semantic shift, forced `neighbors.rebuild(pos)`,
  `NeighborListHook` references removed).
- `src/molix/md/integrators.py` — delete `rebuild_every` / `_force_eval_count`;
  add keyword-only `rebuild: bool | None = None` and `self.rebuild`; single
  guarded call in `eval_force`; docstrings (compile invariant, list-owned cadence).
- `src/molix/md/driver.py` — remove the `rebuild_every` kwarg, its validation,
  the attribute poke and `MD.rebuild_every`; `_AutocastForceField.rebuilds_neighbors`
  delegation; module / class / `run` docstring rewrite.
- `src/molix/md/runner.py` — delete `NeighborListHook`; drop it from the
  `MDRunner.run` cadence example.
- `src/molix/md/__init__.py` — drop the `NeighborListHook` import and `__all__`
  entry (`__all__` stays alphabetised); package docstring updated to the
  list-owned policy.
- `tests/test_molix/test_md/conftest.py` — `make_cubic_lattice(n_side, spacing)`
  promoted from the two duplicate test-local helpers.
- `tests/test_molix/test_md/test_forcefield.py` — `TestForceFieldNeighborSeam`
  (capability flag per subclass, `update` delegation, forced `rebuild`); stub
  neighbour list grows `update`; local `_cubic_lattice` replaced by the conftest import.
- `tests/test_molix/test_md/test_neighbors.py` — local `_lattice` replaced by the
  conftest import (no test-body changes).
- `tests/test_molix/test_md/test_integrators.py` — replace
  `test_rebuild_every_fires_at_force_evaluation_positions` with the static-switch
  tests (derivation, override, one call per force eval at the evaluated positions,
  absence of the deleted attributes).
- `tests/test_molix/test_md/test_compile.py` — existing lj/cut and rollout tests
  construct with `rebuild=False`; new `rebuild=False` fullgraph / `rebuild=True`
  eager pair.
- `tests/test_molix/test_md/test_driver.py` — replace the two `rebuild_every`
  tests with the removed-kwarg guard, the derived-bool tests (incl. autocast- and
  `torch.compile`-wrapped force fields), invariant (c) drift ratio, invariant (f)
  observables.
- `tests/test_molix/test_md/test_runner.py` — delete `TestNeighborListHook`, its
  import, and the docstring reference at `test_step_start_precedes_each_advance`.
- `benchmarks/verify_md_ljcut_nve.py` — `--skin` (default 1.02) / `--every` /
  `--delay` / `--no-check` replace `--rebuild-every`; print policy settings,
  `rebuild_count`, `rebuild_count/n_steps`, `ndanger`; fold `ndanger == 0` into
  the PASS condition for `skin > 0`; docstring rewrite.
- `scripts/matpes_port/run_nve.py` — same flag migration; drop
  `MD(rebuild_every=)` and the `expected_rebuilds` heuristic; report actual
  `rebuild_count` / `ndanger`; fix the stale "frozen neighbour list" wording in
  `--compile`'s help.
- `docs/molix/user-guide/md.md` — rewrite the rebuild paragraph, the pure-GPU
  example, and the hook list; add the migration line.
- `CHANGELOG.md` — `[Unreleased] / ### Changed` bullet recording both breaking
  removals and the migration.
- `regressions/md-neighborlist-skin-07-wire.py` (new) — public-API wiring +
  rebuild-accounting scenario with hard-coded goldens.

## Tasks

- [ ] Write failing unit tests for the force-field neighbour-policy seam in `tests/test_molix/test_md/test_forcefield.py`, promoting the duplicated cubic-lattice helper into `tests/test_molix/test_md/conftest.py` and re-pointing `tests/test_molix/test_md/test_neighbors.py` at it
- [ ] Write failing unit tests for the integrator's static rebuild switch in `tests/test_molix/test_md/test_integrators.py` and migrate/extend the compiled-path tests in `tests/test_molix/test_md/test_compile.py` (`rebuild=False` fullgraph, `rebuild=True` eager)
- [ ] Write failing unit tests for the driver path in `tests/test_molix/test_md/test_driver.py` (removed `rebuild_every` kwarg, derived bool through autocast and compiled force fields, invariant (c) drift ratio, invariant (f) rebuild-count/energy observables)
- [ ] Generalize `ForceField.rebuild_neighbors` to the list-owned policy and add the `rebuilds_neighbors` property (base + four subclasses) in `src/molix/md/forcefield.py`, with google-style docstrings naming the semantic shift and the forced `neighbors.rebuild(pos)` route
- [ ] Replace `rebuild_every` / `_force_eval_count` with the construction-time `rebuild` bool and its single guarded call in `src/molix/md/integrators.py`
- [ ] Remove the `rebuild_every` kwarg, its validation and the attribute poke from `src/molix/md/driver.py`, add the `_AutocastForceField.rebuilds_neighbors` delegation, and rewrite the cadence docstrings to the list-owned policy
- [ ] Delete `NeighborListHook` and its export, docstring references and `TestNeighborListHook` from `src/molix/md/runner.py`, `src/molix/md/__init__.py` and `tests/test_molix/test_md/test_runner.py`
- [ ] Migrate the consumers and user-facing docs off `MD(rebuild_every=)`: `benchmarks/verify_md_ljcut_nve.py`, `scripts/matpes_port/run_nve.py`, `docs/molix/user-guide/md.md`, `CHANGELOG.md`
- [ ] Add regression example `regressions/md-neighborlist-skin-07-wire.py` (public API only; hard-coded goldens, no third-party runtime)
- [ ] Run full check + test suite

## Testing strategy

Unit tests only, mirroring source paths
(`src/molix/md/<mod>.py` → `tests/test_molix/test_md/test_<mod>.py`), each
targeting one function/method. Everything is deterministic CPU float64 and
finishes in seconds. Shared fixture: `make_cubic_lattice` in the package
`conftest.py` (64 atoms at `n_side=4, spacing=3.0` → 12 Å cube, minimum
perpendicular half-width 6.0 Å), imported by package path.

**`test_forcefield.py` — `TestForceFieldNeighborSeam` (mirrors `forcefield.py`).**

- `rebuilds_neighbors` is `False` for `ForceField`, `HarmonicForceField`,
  `LennardJonesForceField`, `PotentialForceField` and
  `CallableForceField(neighbors=None)`; `True` for `LennardJonesCutForceField`,
  `PeriodicPotentialForceField` and `CallableForceField(neighbors=<list>)`.
- A stub strategy recording both calls sees `rebuild_neighbors(pos)` land on
  `update(pos)` **only** — never on `rebuild(pos)` — and the existing
  `test_without_neighbors_rebuild_is_a_noop` still passes.
- Forced build: `nl.rebuild(pos)` raises `rebuild_count` by exactly 1 and resets
  `ago` (the escape hatch the docstring promises).
- Existing `test_rebuild_is_visible_through_the_bound_buffers` keeps passing
  under the default policy (`skin=0, every=1, delay=0, check=True`): the
  compression is a nonzero displacement, so `update` rebuilds.

**`test_integrators.py` (mirrors `integrators.py`).**

- `Integrator(force).rebuild` is derived from `force.rebuilds_neighbors`;
  `rebuild=False` / `rebuild=True` override the derivation.
- With `rebuild=True`, `eval_force` calls the seam exactly once per force
  evaluation and **with the positions being evaluated** — `initial()` at `pos0`,
  `step_nve()` at the new end-of-step positions (the surviving assertion of the
  deleted `test_rebuild_every_fires_at_force_evaluation_positions`).
- With `rebuild=False` over a list-backed force field the seam is never called.
- `not hasattr(ig, "rebuild_every")` and `not hasattr(ig, "_force_eval_count")`.
- `_MidpointEuler(Integrator)` (the conforming-subclass guard) still constructs
  via `super().__init__(force)`.

**`test_compile.py` (compile invariants).**

- Existing tests stay green; `test_ljcut_step_fullgraph_compiles_and_matches_eager`
  now constructs `LangevinVerletIntegrator(ff, …, rebuild=False)`.
- New: a `rebuild=False` integrator over `LennardJonesCutForceField` compiles
  **both** `step` and `rollout` with `fullgraph=True` (backend `aot_eager`),
  matches eager to `1e-12`, and leaves `nl.rebuild_count == 0`.
- New: the same force field with `rebuild=True` advances eagerly through
  `advance_n` and raises `nl.rebuild_count` above 0.

**`test_driver.py` (mirrors `driver.py`).**

- `MD(…, rebuild_every=1)` raises `TypeError`; `md.runner.hooks == []`.
- `md.integrator.rebuild` follows the force field: `False` for
  `HarmonicForceField`, `True` for `LennardJonesCutForceField`, `True` through
  `autocast_dtype=torch.bfloat16` (wrapper delegation) and through a
  `torch.compile(ff, backend="eager")`-wrapped force field (the `OptimizedModule`
  attribute-forwarding assumption the GPU benchmark depends on).
- **Invariant (c) — drift ratio (domain validation).** Harness: 64-atom lattice,
  `LennardJonesCutForceField(epsilon=0.0103/EV_PER_AMU_A2_FS2, sigma=2.5,
  neighbors=NeighborList(cell, cutoff=3.5, positions=pos, skin=s, every=1,
  delay=0, check=True))`, `MD(mass=39.95, dt=4.0, gamma=0.0, dtype=torch.float64)`,
  velocities `MaxwellBoltzmann(39.95, n_atoms=64).sample(300.0, seed=0)`,
  100 steps, `chunk=1`, an `MDHook` sampling `obs.total`. Drift =
  `max_t |E−E0|/|E0|`. Assert `drift(skin=1.0) <= 3 · drift(skin=0.0)` and
  `drift(skin=0.0) > 0` (the baseline is nonzero by construction — finite-`dt`
  velocity-Verlet — so the ratio is not vacuous).
- **Invariant (f) — observables through the public driver path.** Same harness at
  `skin ∈ {0.0, 0.5, 1.0}`: `rebuild_count` non-increasing in `skin` with
  `rebuild_count(1.0) < rebuild_count(0.0)`; final `MDState.energy` agrees across
  arms to `atol=1e-10, rtol=0`; hard-coded counters —
  `rebuild_count(skin=0.0) == 100` (one policy call per force evaluation over 100
  steps; the entry evaluation sits at the build positions, `max_d2 == 0`, strict
  `>`, so it does not rebuild) and, per link 04's documented degenerate limit,
  `ndanger(skin=0.0) == rebuild_count(skin=0.0) == 100` while
  `ndanger(skin=0.5) == ndanger(skin=1.0) == 0`. The `rebuild_count == 100`
  literal is the anti-vacuity assertion: a wiring that never calls the policy
  passes every equality test but fails this one.

**`test_runner.py`.** `TestNeighborListHook` deleted; the remaining lifecycle
tests unchanged except the docstring reference.

**Not unit-tested here (by design).** `benchmarks/` and `scripts/` are
out-of-suite; they are covered by ac-007 (`--help` + a short CPU run), the
regression example, and the GPU rerun (ac-008).

**Regression example.** `regressions/md-neighborlist-skin-07-wire.py` — a
standalone public-API script (not collected by pytest;
`PYTHONPATH=src python regressions/md-neighborlist-skin-07-wire.py`, prints `OK`,
exits 0/1) that drives the same 64-atom argon lattice through `MD` and checks:

1. **Wiring.** `md.integrator.rebuild is True` for the lj/cut force field;
   a twin built with `MD(ff, …, integrator=LangevinVerletIntegrator(ff, …,
   rebuild=False))` runs 100 steps with `rebuild_count == 0`.
2. **Rebuild accounting (hand-derived integers).** `skin=0.0` over 100 steps ⇒
   `rebuild_count == 100`, `ndanger == 100`; `skin=1.0` ⇒
   `rebuild_count == <literal captured at implementation>` with
   `0 < rebuild_count < 100` and `ndanger == 0`.
3. **Physics unchanged.** `|E_tot,final(skin=1.0) − E_tot,final(skin=0.0)| <= 1e-10`
   (amu·Å²/fs²), and `E_tot(0)` equals its recorded float64 literal to `rtol=1e-9`
   (a pure function of the lattice, the LJ parameters and the seeded velocities).
4. **Migration.** `MD(…, rebuild_every=1)` raises `TypeError`, and
   `hasattr(molix.md, "NeighborListHook")` is `False`.

Goldens are literals produced by this repo's own code, captured once offline; no
third-party software is imported or subprocessed at runtime. The header records
the capture command, commit sha, torch version, date and device/precision per
`regressions/README.md`.

## Out of scope

- **Changing the policy itself.** `skin` / `every` / `delay` / `check`,
  `ndanger`, `r_build` sizing, the unwrapped-positions guard and their tests are
  link 04's contract and are not touched (`tests/test_molix/test_md/test_neighbors.py`
  changes only its lattice import).
- **The binned / cell-list build (link 06).** This link is backend-agnostic; it
  never names the kernel.
- **Renaming `ForceField.rebuild_neighbors` → `update_neighbors`.** Considered
  and rejected: it would churn four subclasses, the autocast wrapper, three test
  modules and the docs for no behavioural change; the docstring carries the
  semantic shift and the forced-build route instead.
- **A `MD(rebuild=)` kwarg or any per-run cadence knob on the driver.** Rejected:
  it would recreate the second owner this link exists to delete. Frozen runs go
  through the existing `MD(integrator=…)` seam.
- **Compiling the integrator loop with a live policy.** `update()` needs a
  max-displacement reduction and a host sync, so a fullgraph rollout with
  `rebuild=True` is not attempted; the production path compiles the force field
  and keeps the loop eager.
- **`benchmarks/bench_mace_matpes.py`** — checked: it builds a `NeighborList`
  directly and times the model, with **no** `MD(...)` construction site and no
  `rebuild_every`, so it needs no change here (its list construction was already
  handled by links 03/04). Likewise `benchmarks/verify_md_lj_nve.py`, whose
  all-pairs `LennardJonesForceField` has no list.
- **`benchmarks/run_gh200_ljcut_nve.sbatch`** — unchanged; it forwards `"$@"`, so
  the new defaults apply without an edit.
- **Historical run-log rows** in `src/molzoo/specs/mace_matpes.md` §7.4 that
  record `rebuild_every=5` — append-only history, left byte-identical; the
  grep criterion excludes them together with `CHANGELOG.md` and `.claude/specs/`.
- **Restoring or deprecating a step-start rebuild path** (`NeighborListHook`) in
  any form, and any back-compat shim for `MD(rebuild_every=)`: hard removal per
  the repo's `stage: experimental` norm.
- **Auto-tuning `skin` from `ndanger` or a target rebuild rate**, multi-GPU /
  domain decomposition, and any benchmark artifact work beyond the stdout numbers
  ac-008 reads.
