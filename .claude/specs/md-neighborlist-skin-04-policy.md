---
title: Verlet skin + neigh_modify rebuild policy for molix.md.NeighborList
status: approved
created: 2026-08-09
revised: 2026-08-09
grilled: true
chain: md-neighborlist-skin
---

# Verlet skin + neigh_modify rebuild policy for molix.md.NeighborList

## Summary

`molix.md.NeighborList` currently rebuilds on a blind step cadence
(`MD(rebuild_every=)`), which is either wasteful (`rebuild_every=1` rebuilds
every force evaluation whether or not anything moved) or silently wrong
(`rebuild_every=5` can miss a pair that walked into the cutoff on step 2). This
link adds the LAMMPS Verlet-skin machinery that removes that dilemma: the list
is built at an enlarged radius `r_build = cutoff + skin`, and a once-per-step
`update(positions)` decides — from the actual maximum atomic displacement since
the last build, under the exact `neigh_modify every/delay/check` gate — whether
a rebuild is needed. The user-visible outcome is a neighbour list that is
*provably complete* between rebuilds for a stated skin, rebuilds an order of
magnitude less often than `rebuild_every=1`, and reports a `ndanger` counter
that flags any run where a rebuild may have come too late. The policy lands
usable through the existing `rebuild(positions)` primitive and the existing
`ForceField.rebuild_neighbors` seam, so this link is independently testable
before the TensorDict surface (link 05), the binned build (link 06), or the
integrator rewiring (link 07) exist.

## Domain basis

Units throughout: lengths in Å (`cutoff`, `skin`, `r_build`, positions,
displacements), `ago` / `every` / `delay` in MD steps, `rebuild_count` /
`ndanger` dimensionless counters.

**Two-atom completeness (the skin criterion).** For atoms *i*, *j* let
`d_i = ‖x_i(t) − x_i(t₀)‖` be the displacement since the last build at `t₀`.
The triangle inequality gives

```
r_ij(t) ≥ r_ij(t₀) − d_i − d_j
⇒  r_ij(t) ≤ r_cut   implies   r_ij(t₀) ≤ r_cut + d_i + d_j
```

so a list built at `r_build = r_cut + s` at `t₀` contains **every** pair that is
within `r_cut` at time `t`, provided `d_(1) + d_(2) ≤ s` for the two largest
displacements. Since the two largest displacements are not known to be on
different atoms of the offending pair, the conservative sufficient condition is
the **half-skin** criterion

```
max_i d_i ≤ s/2        (worst case d_(1) = d_(2) = s/2 — this is why it is HALF)
```

**LAMMPS-exact gating** (verified against `lammps/lammps` develop
`src/neighbor.cpp:2408-2424` (`Neighbor::decide`), `2438-2490`
(`Neighbor::check_distance`), `:344`):

```
ago += 1
permitted = (ago >= delay) and (ago % every == 0)          # conjunctive
if permitted:
    rebuild if (not check) or max_i ‖x_i − x_hold_i‖² > (s/2)²   # strict >, RAW difference
```

`ago` resets to 0 at every build; `x_hold` is a copy of the positions at that
build. The displacement uses the **raw** difference — **no minimum image**.
This repo's MD drifts positions unwrapped (verified across
`molix.md.integrators` / `molix.md.runner`: nothing wraps), and min-imaging the
displacement would clamp genuine `> L/2` excursions and therefore *suppress*
rebuilds that are actually needed. LAMMPS does the same thing
(`src/verlet.cpp:277` wraps only on reneighbour steps, before `xhold` is
stored).

**`ndanger` (dangerous builds).** LAMMPS increments `ndanger` when the distance
check fires at `ago == MAX(every, delay)` — the *first permitted opportunity*
(`neighbor.cpp:2488`). A nonzero count means the rebuild may already have been
overdue on an earlier, non-permitted step, i.e. pairs may have been missed. It
is the cheapest correctness alarm available and is exposed as a public
attribute.

**Correctness trade of `check=False` / a coarse `delay`.** A pair that is
inside `r_cut` but absent from the list contributes exactly zero. When the next
build finally inserts it, the PES changes discontinuously, so the total energy
takes an `O(1)`, one-signed injection per missed event — **not** the `O(dt²)`
error of a discretisation artefact. Repeated events integrate into a systematic
NVE energy leak. `check=False` (and any `delay` large enough to skip a needed
rebuild) buys speed by accepting that leak; it must be documented as such, never
as a free optimisation.

**Capacity and kernel sizing.** Both the fixed buffer capacity
(`capacity = ceil(capacity_factor · E_initial)`) and the C++ kernel's
`max_num_pairs` must be sized from `r_build`, not `cutoff`: the live edge count
scales as `(1 + s/r_cut)³` (2.74× at `r_cut = 5 Å`, `s = 2 Å`). The kernel is
invoked at `r_build`; the model's cutoff envelope masks the skin-region pairs to
zero (`LennardJonesCutForceField` already masks `r² < cutoff_sq`). The dead-edge
shift `DEAD_EDGE_CUTOFF_FACTOR × cutoff = 10 × cutoff` must stay beyond
`r_build`, hence the construction assertion
`skin < (DEAD_EDGE_CUTOFF_FACTOR − 1) × cutoff`.

**Half-cell guard, re-derived on `r_build`.** From link 01: the minimum-image
convention requires `r_build ≤ min_i w_i / 2` with perpendicular widths
`w_i = V / ‖a_j × a_k‖`. The guard must now be evaluated at `r_build`, not
`cutoff` — a legal `cutoff` with a generous `skin` can cross the half-width and
silently miss periodic images.

**Unwrapped-positions invariant (load-bearing).** Frozen shift vectors plus a
raw displacement test are correct **only** while positions stay unwrapped
mid-run. This is promoted to a documented invariant with a runtime guard inside
the displacement check: if `max_i d_i ≥ min_i w_i / 2`, raise `RuntimeError` —
which catches mid-run wrapping, a changed cell, and trajectory blow-up alike.
Fail loud; never continue.

**References.**

- LAMMPS documentation: `neigh_modify`, `neighbor`, and the "Run output"
  (`Dangerous builds`) page — https://docs.lammps.org/neigh_modify.html
- `lammps/lammps` develop, `src/neighbor.cpp` (`decide`, `check_distance`) and
  `src/verlet.cpp` at the lines cited above.
- K. Nordlund, *Introduction to molecular dynamics simulations*, lecture 3 —
  https://www.mv.helsinki.fi/home/knordlun/moldyn/lecture03.pdf (the open,
  directly re-verified source for the two-atom criterion).
- L. Verlet, *Phys. Rev.* **159**, 98 (1967), https://doi.org/10.1103/PhysRev.159.98
  — original neighbour-list construction.
- B. Quentrec & C. Brot, *J. Comput. Phys.* **13**, 430 (1973),
  https://doi.org/10.1016/0021-9991(73)90046-6 — cell/skin refinement.

  **Caveat:** the Verlet 1967 and Quentrec & Brot 1973 texts are paywalled and
  were **not** re-verified for this spec; they are cited for attribution only.
  Every equation above is verified against the Nordlund lecture notes and the
  LAMMPS source.

## Design

Preconditions from earlier links in the chain: link 01 fixed the
perpendicular-width guard (`w_i = V/‖a_j × a_k‖`, replacing the
`min ‖a_i‖ / 2` approximation), link 03 renamed the class to
`molix.md.NeighborList` and aliased the kernel task import
(`from molix.data.tasks.neighbor import NeighborList as ...`) so the two names
cannot shadow each other. This link assumes both have landed; the alias is
load-bearing and must not be dropped.

**Constructor (keyword-only, extends the existing signature):**

```python
NeighborList(*, cell, cutoff, positions, skin=0.0, every=1, delay=0,
             check=True, capacity_factor=1.35, device=None)
```

`cutoff` remains the **interaction** cutoff — the number every consumer
(`LennardJonesCutForceField`, TensorDict potentials, link 07's driver) means by
"cutoff". `r_build = cutoff + skin` is derived, exposed read-only as
`NeighborList.r_build` (link 07 logs it; the sizing tests assert on it) and never
settable.

**Construction-time validation**, in this order, all `ValueError` with the
offending numbers in the message:

1. `skin >= 0`, `every >= 1`, `delay >= 0`, and `delay % every == 0`
   (LAMMPS `Neighbor::init` parity — `delay = 0` is always legal since
   `0 % every == 0`; a non-multiple `delay` would make the danger threshold
   `MAX(every, delay)` unreachable and silently disable the alarm, which is
   exactly why LAMMPS rejects it too).
2. `r_build <= min_i w_i / 2` — the link-01 half-width guard, re-evaluated at
   `r_build` and naming `r_build`, `cutoff`, `skin` and the width in the message.
3. `skin < (DEAD_EDGE_CUTOFF_FACTOR - 1) * cutoff` — otherwise the dead padding
   edges would fall inside the build radius.

**Sizing.** The kernel task is constructed with `cutoff=r_build` (not `cutoff`)
and an explicit `max_num_pairs = max(1, N*(N-1)//2)` so the enlarged radius can
never be truncated by the task's stale `512` default. `capacity` is then
`ceil(capacity_factor * E_initial(r_build))`, i.e. the `(1 + s/r_cut)³` growth is
absorbed by the same default factor rather than by asking the user to inflate it.

**New state.** Public: `skin`, `ago` (steps since the last build),
`rebuild_count` (already exists — semantics unchanged: the constructor's initial
build is *not* counted, so a fresh list reports `0`), `ndanger`. Private:
`_x_hold` — a `(N, 3)` buffer allocated once at construction and filled with
`copy_()` on every build (in-place, so no per-rebuild allocation and no shape
churn), `_half_skin_sq = (skin/2)**2`, `_wrap_guard_sq = (min_i w_i / 2)**2`,
`_danger_ago` (below).

**`rebuild(positions) -> None`** stays the forced-build primitive: recompute at
`r_build`, write into the fixed-capacity buffers, `rebuild_count += 1`, then
`ago = 0` and `_x_hold.copy_(positions.detach())`. Existing callers
(`ForceField.rebuild_neighbors`, `MD(rebuild_every=)`, the benchmarks) keep
working unchanged.

**`update(positions) -> bool`** is the once-per-step policy entry — the only new
public method:

```python
self.ago += 1
if self.ago < self.delay or self.ago % self.every:
    return False                      # not a permitted opportunity
if not self.check:
    self.rebuild(positions); return True
max_d2 = float(((positions.detach() - self._x_hold) ** 2).sum(-1).max())
self._assert_unwrapped(max_d2)        # RuntimeError if max_d2 >= _wrap_guard_sq
if max_d2 > self._half_skin_sq:       # strict >, raw difference, no min-image
    if self.ago == self._danger_ago:
        self.ndanger += 1
    self.rebuild(positions); return True
return False
```

It is eager Python control flow (like `rebuild`) and is called *between* force
evaluations, never inside a compiled graph. It returns whether it rebuilt so
link 07's integrator can count rebuilds and tests can assert the decision
sequence directly.

**Degenerate limit `skin=0.0, every=1, delay=0, check=True`.**
`_half_skin_sq = 0` and the comparison is strict `>`, so *any* nonzero motion
triggers a rebuild — that is the correct degenerate limit, and it reproduces
today's rebuild-every-ask behaviour edge-for-edge. The one difference is
bookkeeping, not physics: if nothing moved at all (`max_d2 == 0`) no rebuild
fires, because the list is already exactly right. In this mode
`_danger_ago == 1`, so *every* triggered rebuild counts as dangerous and
`ndanger ≈ n_steps`; the docstring must say so, or the alarm reads as a false
alarm.

**`_danger_ago` — strict LAMMPS parity (operator decision, 2026-08-09 grill).**
LAMMPS uses `ago == MAX(every, delay)` and rejects a `delay` that is not a
multiple of `every` in `Neighbor::init`; we mirror **both** halves. The
construction guard above (`delay % every == 0`) makes the literal threshold
always reachable, so:

```
_danger_ago = max(every, delay)     # LAMMPS neighbor.cpp:2488, verbatim
```

The alternative — accepting any `every`/`delay` pair and generalising the
threshold to `every * ceil(delay / every)` — was drafted and explicitly
rejected at the audit grill: behavioural isomorphism with LAMMPS is worth more
here than one extra degree of configuration freedom nobody asked for. The
docstring records the parity claim and cites `Neighbor::init`.

**Unwrapped-positions guard.** `_assert_unwrapped(max_d2)` raises `RuntimeError`
naming the measured displacement, the half-width, and the three causes (mid-run
wrapping, a changed cell, a blown-up trajectory). Note the guard lives *inside*
the displacement branch, so `check=False` disables it along with the criterion —
that is stated explicitly in the docstring as part of what `check=False` buys and
costs.

**Protocol.** `NeighborStrategy` grows three declared members: `cutoff: float`,
`skin: float`, `update(positions) -> bool`. That kills the `getattr(neighbors,
"cutoff", None)` duck-read in `src/molix/md/forcefield.py:386-397`: the
`cutoff is None` / "exposes no `.cutoff`" branch is deleted and the ctor reads
`neighbors.cutoff` directly. `LennardJonesCutForceField`'s check keeps comparing
the requested cutoff against the list's **interaction** cutoff, not `r_build` —
correct as-is: the skin-region pairs exist in the buffers but are only guaranteed
*complete* out to `cutoff` between rebuilds, so allowing an interaction cutoff up
to `r_build` would be exactly the silent truncation that check exists to prevent.
The `_Recorder` stub in `tests/test_molix/test_md/test_forcefield.py:86` is never
`isinstance`-checked against the protocol and only feeds
`CallableForceField.rebuild_neighbors`, so widening the protocol does not touch it.

No changes to `src/molix/md/__init__.py`: no new exported symbol (link 03 already
exports `NeighborList`).

**Shape check.** Every new symbol is a method or attribute on the owning type
(`NeighborList`); no factory function, no context blob, no façade. The gate
decision stays a private `_decide`-style branch inside `update` (single in-tree
call site — not extracted), and is tested through `update`'s observable
return value plus `ago` / `rebuild_count` / `ndanger`.

**Reuse decision.** No `librarian_report` was supplied with this task; the
candidates below come from a direct read of `src/molix/md/` and
`src/molix/data/tasks/`. The caller should note the missing blueprint advisory.

- `molix.data.tasks.neighbor.NeighborList` (kernel-output normalisation: pbc,
  NaN-padding strip, symmetry expansion, edge-sign convention) — **reuse**,
  reconstructed at `cutoff=r_build` with an explicit `max_num_pairs`.
- Link 01's perpendicular-width guard in `neighbors.py` — **reuse**, re-evaluated
  at `r_build`. If link 01 left the width computation inline in `__init__`, this
  link's task 2 extracts it to a module-private `_min_perpendicular_width(cell)`
  — the second real use, per the repo's "inline until the second use" rule.
- `molix.units.DEAD_EDGE_CUTOFF_FACTOR` — **reuse** for the dead-edge assertion;
  do not restate `10.0` anywhere.
- `molix.md.LennardJonesCutForceField`, `molix.md.MD(rebuild_every=)`,
  `molix.md.MDHook`, `molix.md.MaxwellBoltzmann` — **reuse** as the test harness
  for the trajectory falsification (see Testing strategy); no new MD driver, no
  hand-rolled velocity-Verlet loop in `tests/`.
- `molix.md.runner.NeighborListHook` — **new — not used**: it rebuilds at
  step-start positions, the documented one-step-lag energy-leak path;
  `Integrator.eval_force`'s `rebuild_every` seam is the correct hook point and
  already exists.
- `NeighborList.update` / `NeighborList.r_build` / `_x_hold` — **new**: no
  existing symbol carries rebuild-policy state; naming, keyword-only
  construction, and `ValueError`-at-construction / `RuntimeError`-at-runtime
  error handling follow the closest pattern, the existing `NeighborList`
  constructor guard and `_write` overflow check.

## Files to create or modify

- `src/molix/md/neighbors.py` — policy state, `update()`, `r_build` sizing of
  capacity + kernel `max_num_pairs`, construction guards, unwrapped-positions
  runtime guard, `to()` extended to `_x_hold`, `NeighborStrategy` protocol
  members, module-docstring rewrite.
- `src/molix/md/forcefield.py` — drop the `getattr(neighbors, "cutoff", None)`
  duck-read in `LennardJonesCutForceField.__init__` now that the protocol
  declares `cutoff`.
- `tests/test_molix/test_md/test_neighbors.py` — new `TestNeighborListPolicy`
  (construction contract, gating arithmetic, guards, trajectory falsification).
- `regressions/md-neighborlist-skin-04-policy.py` (new) — public-API rebuild-decision
  scenario with hand-derived hard-coded goldens.

## Tasks

- [ ] Write failing unit tests for the `r_build` construction contract in `tests/test_molix/test_md/test_neighbors.py` (`TestNeighborListPolicy`: edge counts at `r_build` vs `cutoff`, capacity sizing, half-width guard on `r_build`, dead-edge assertion, `to()` moving `_x_hold`)
- [ ] Implement `r_build` derivation, `r_build`-based capacity + kernel `max_num_pairs` sizing, and the three construction guards in `src/molix/md/neighbors.py`
- [ ] Write failing unit tests for the LAMMPS gating arithmetic, the `skin=0` degenerate limit, `ndanger`, and the unwrapped-positions guard in `tests/test_molix/test_md/test_neighbors.py`
- [ ] Write failing trajectory falsification tests over an NVE LJ-lattice run in `tests/test_molix/test_md/test_neighbors.py` (O(N²) subset completeness, skin-vs-no-skin parity, `ndanger == 0`, `rebuild_count` monotonicity in `skin`)
- [ ] Implement `skin` / `ago` / `ndanger` state and `NeighborList.update()` with the LAMMPS-exact gate and the generalised `_danger_ago` in `src/molix/md/neighbors.py`
- [ ] Implement the unwrapped-positions invariant guard inside the displacement check in `src/molix/md/neighbors.py`
- [ ] Declare `cutoff` / `skin` / `update` on `NeighborStrategy` and delete the `getattr` duck-read in `src/molix/md/forcefield.py`
- [ ] Add docstrings per google style with units for the new constructor arguments, `update`, `r_build`, and rewrite the `src/molix/md/neighbors.py` module docstring (half-skin derivation, `ndanger`, the `check=False`/`delay` correctness trade, the unwrapped invariant, the LAMMPS + Nordlund + Verlet/Quentrec-Brot references and the paywall caveat)
- [ ] Add regression example `regressions/md-neighborlist-skin-04-policy.py` (public API only; hard-coded goldens, no third-party runtime)
- [ ] Run full check + test suite

## Testing strategy

Unit tests only, mirroring the source path
(`src/molix/md/neighbors.py` → `tests/test_molix/test_md/test_neighbors.py`),
in a new `TestNeighborListPolicy` alongside the existing class-level tests.
Everything is deterministic CPU float64. The shared fixture is the existing
module helper `_lattice(n_side=4, spacing=3.0)`: a 64-atom simple-cubic lattice
in a 12 Å cube (perpendicular half-width 6.0 Å), whose neighbour counts are exact
integers from crystallography.

**Happy path / construction (hard-coded domain values).**

- `skin=1.5, cutoff=3.5` ⇒ `r_build == 5.0`; `num_edges == 1152`
  (18 neighbours × 64 atoms: 6 at 3.0 Å plus 12 at 3·√2 = 4.2426 Å,
  bidirectional), versus `num_edges == 384` at `skin=0.0` — the direct
  measurement of the `(1 + s/r_cut)³` growth.
- `capacity >= ceil(1.35 * 1152)` and buffer shapes remain `(capacity, 2)` /
  `(capacity, 3)`.
- A fresh list reports `ago == 0`, `rebuild_count == 0`, `ndanger == 0`.
- `nl.to(torch.float32)` casts `_x_hold` along with `shifts` / `cell`
  (a missed `_x_hold` would compare mixed dtypes on the next `update`).

**Edge cases / construction errors.**

- `cutoff=3.5, skin=3.0` ⇒ `r_build = 6.5 > 6.0` raises `ValueError` naming
  `r_build` (the link-01 guard re-derived; note `cutoff` alone would pass).
- `skin >= 9 * cutoff` raises `ValueError` referencing the dead-edge factor.
- `skin < 0`, `every < 1`, `delay < 0` each raise `ValueError`.
- `every=4, delay=10` raises `ValueError` matching "multiple" (the
  `delay % every == 0` LAMMPS-parity guard); `every=4, delay=8` and any
  `delay=0` construct cleanly.

**Gating arithmetic (pure policy, no dynamics — positions displaced by hand).**

- `every=4, delay=8`: `update()` returns `False` for `ago = 1..7` even with a
  displacement far beyond the half-skin, and the first permitted opportunity is
  `ago == 8` (conjunction of both arms, with a LAMMPS-legal `delay`).
- Each arm alone: `every=4, delay=0` first permits `ago == 4`;
  `every=1, delay=10` first permits `ago == 10`.
- A forced `rebuild(positions)` sets `ago == 0` and re-phases the schedule — the
  next permitted opportunity is counted from the build, not from an absolute step
  index.
- Strict `>`: a displacement of exactly `skin/2` does **not** trigger; `skin/2 + 1e-12` does.
- `skin=0.0, every=1, delay=0`: an unchanged position returns `False`, any
  nonzero displacement returns `True` (the documented degenerate limit).
- `check=False`: rebuilds on every permitted opportunity regardless of
  displacement, and does not evaluate the displacement guard.

**`ndanger`.**

- `every=1, delay=0, skin=1.0`: a displacement of 1.0 Å in one `update`
  (`> skin/2 = 0.5`) fires at `ago == 1 == _danger_ago` ⇒ `ndanger == 1`;
  repeating gives `ndanger == 2`.
- `every=2, delay=6, skin=1.0`: crossing the half-skin only after the first
  permitted opportunity (`ago == 6`) rebuilds at `ago == 8` with `ndanger == 0`.
- `every=2, delay=6, skin=1.0` with a jump beyond the half-skin already at the
  first permitted opportunity: the rebuild lands at `ago == 6 == max(every,
  delay) == _danger_ago` and increments `ndanger` — pinning the verbatim LAMMPS
  threshold on a LAMMPS-legal configuration.

**Unwrapped-positions guard.**

- Artificially wrap one atom by a full cell vector (12 Å ≥ half-width 6.0 Å)
  between builds ⇒ `update()` raises `RuntimeError` matching "unwrapped".
- The same jump under `check=False` does **not** raise (documented consequence,
  pinned so the trade stays visible).

**Domain validation — trajectory falsification (the binding physics tests).**
Driven entirely through public API: `LennardJonesCutForceField` (argon in
(amu, Å, fs): `ε = 0.0103 eV / EV_PER_AMU_A2_FS2`, `σ = 2.5 Å`, `cutoff = 3.5 Å`)
over the 64-atom lattice, `mass = 39.95 amu`, velocities from
`MaxwellBoltzmann(39.95, n_atoms=64).sample(300.0, seed=0)`, `MD(..., gamma=0.0,
dtype=torch.float64, rebuild_every=1)`, 100 steps at `dt = 4 fs`, `chunk=1`. A
test-local `ForceField` subclass overrides `rebuild_neighbors` to call
`neighbors.update(pos)` — a three-line preview of link 07's wiring — so the
policy is driven once per force evaluation *at the evaluated positions*, and an
`MDHook.on_step_end` observes `obs.pos` (the configuration those forces were
computed at).

- **(d) PRIMARY falsification — completeness.** At every step, recompute the
  exact minimum-image O(N²) pair set within `cutoff` from `obs.pos` and assert it
  is a **subset** of the live list `edge_index[:num_edges]`; additionally assert
  that for each such pair the list-reconstructed distance
  `‖pos[t] − pos[s] + shift‖` equals the reference distance to `1e-9 Å` (a stale
  *shift* on a surviving index pair is the frozen-shift failure mode, and an index
  subset check alone would not see it). Fails on the first missed pair.
- **(a) Policy equivalence.** `skin=0.5, every=1, delay=0, check=True` versus
  `skin=0.0` with `rebuild()` on every force evaluation: per-step total energy
  and forces agree to `atol=1e-10, rtol=0` in float64 on CPU. Deliberately not
  bitwise — the masked-zero skin edges reorder the `index_add_` accumulation.
- **(e) `ndanger == 0`** over that standard run (`skin=0.5` ⇒ half-skin 0.25 Å,
  ~0.01 Å of motion per step, so no rebuild is ever overdue).
- **(f) Monotonicity.** Over the identical trajectory with
  `skin ∈ {0.0, 0.25, 0.5, 1.0}`, `rebuild_count` is non-increasing in `skin`,
  and `rebuild_count(skin=1.0) < rebuild_count(skin=0.0)` — catches an inverted
  or dead gate that a non-strict monotonicity assertion alone would pass.

**Protocol.** `isinstance(nl, NeighborStrategy)` still holds with the widened
protocol; `LennardJonesCutForceField(..., cutoff=r_build)` over a skinned list
still raises `ValueError` (the interaction cutoff, not `r_build`, is the bar).

**Regression example.** `regressions/md-neighborlist-skin-04-policy.py` — a
standalone public-API script (not collected by pytest) that constructs the
64-atom lattice list at `cutoff=3.5, skin=1.5` and drives `update()` through
three scripted displacement schedules, checking **hand-derived integer goldens**
that need no oracle of any kind:

1. `every=2, delay=4, check=True`, one atom translated by 0.1 Å per update:
   half-skin is 0.75 Å, permitted `ago ∈ {4, 6, 8, …}`, displacements
   `{0.4, 0.6, 0.8}` ⇒ rebuild at every 8th update; over 40 updates the
   `update()` return sequence is `True` exactly at updates `{8, 16, 24, 32, 40}`,
   `rebuild_count == 5`, `ndanger == 0`.
2. `every=1, delay=0, check=True`, 1.0 Å per update ⇒ every rebuild lands at
   `ago == 1 == _danger_ago`: over 5 updates `rebuild_count == 5`,
   `ndanger == 5` (the alarm firing as designed).
3. `every=5, delay=0, check=False`, zero motion ⇒ rebuilds purely on cadence at
   updates `{5, 10, 15, 20}`, `rebuild_count == 4`, `ndanger == 0`.

Plus the crystallographic goldens `num_edges == 1152` at `r_build = 5.0 Å` and
`384` edges within `cutoff = 3.5 Å`. The header records the capture command,
commit sha, torch version, date, and device/precision per `regressions/README.md`.
No third-party software is imported or subprocessed.

## Out of scope

- The TensorDict build/update surface (link 05) — this link exposes the policy
  only through `rebuild(positions)` / `update(positions)` on raw `(N, 3)` tensors.
- The binned / cell-list build (link 06); the O(N²) kernel stays the build
  backend here, just invoked at `r_build`.
- Integrator and driver rewiring (link 07): `MD(rebuild_every=)` keeps calling
  `rebuild`, and the `driver.py:139-143` docstring text ("the accurate NVE
  setting without a Verlet skin" → fallback description) is edited there, not
  here. This link deliberately leaves `update()` unwired by default.
- LAMMPS features with no consumer here: `neigh_modify one/page/binsize`,
  `build_once`, `exclude`, multi-cutoff / hybrid / half-vs-full list variants,
  domain decomposition and ghost-atom communication.
- Auto-tuning `skin` from a target rebuild rate or from `ndanger`; the skin is
  the user's declared parameter and stays so.
- Wrapping positions mid-run (rejected: it would break both the frozen shifts and
  the raw displacement test — the guard exists precisely to forbid it).
- Triclinic-specific validation beyond the perpendicular-width guard inherited
  from link 01.
