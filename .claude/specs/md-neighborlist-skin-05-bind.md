---
title: TensorDict bind surface for molix.md.NeighborList (md-neighborlist-skin link 05)
status: approved
created: 2026-08-09
grilled: true
chain: md-neighborlist-skin
---

# TensorDict bind surface for molix.md.NeighborList (md-neighborlist-skin link 05)

## Summary

`molix.md.NeighborList` owns fixed-capacity `edge_index` / `shifts` buffers, but the
wiring that makes a collated batch *carry* those buffers lives somewhere else — in
`PeriodicPotentialForceField._bind_neighbors`, a private method the list knows nothing
about, duplicated in that class's `_apply` because a `.to()` silently severs the tie.
This link moves the wiring onto the list itself: `nl.build(batch)` refreshes the buffers
at `batch["atoms", "pos"]`, validates that the batch's positions (and its optional
`("graphs", "cell")`) actually describe the system the list was constructed for, writes
`batch["edges"]` as a `TensorDict` holding the *live* buffers **by reference**, and
returns the same batch so it composes with the repo-wide `forward(td) -> td` pipeline.
`nl.update(...)` becomes one method that accepts either a batch `TensorDict` or the raw
`(N, 3)` MD hot-path tensor from link 04, with an `isinstance` dispatch at the top — not
a pair of `update_td` / `update_pos` twins. The user-visible outcome is that a periodic
system can be driven as `nl.build(batch)` once, then `nl.update(batch)` per step, with
every tensor shape constant and the potential seeing the current neighbour set without
any re-binding; and that `PeriodicPotentialForceField` stops owning a copy of that
knowledge. No collate code changes: the two optional batch keys this path relies on
(`("edges", "shifts")`, `("graphs", "cell")`) are already live in-tree and are merely
promoted from de-facto to documented.

## Design

**Chain preconditions.** Link 03 renamed the class to `molix.md.NeighborList` and aliased
the kernel task to `NeighborListTask`; link 04 added `skin` / `every` / `delay` / `check`,
`update(positions) -> bool`, `r_build`, `ago`, `ndanger`, and widened `NeighborStrategy`
with `cutoff` / `skin` / `update`. That surface is **fixed** and is not redesigned here —
in particular link 04's keyword parameter name `positions` on `update` is preserved, so
`nl.update(positions=pos)` keeps working.

**No new physics.** This link declares no equations and introduces no reference values.
The minimum-image contract (`r_ij = pos[target] − pos[source] + shift`, lengths in Å), the
dead-edge padding convention, and the half-skin rebuild criterion are inherited from links
01/04 unchanged; every physical quantity crossing the new surface is one of those, in Å.
There is therefore deliberately no Domain-basis section (same call as link 03).

### The surface

```python
def build(self, batch: TensorDict) -> TensorDict: ...
def update(self, positions: TensorDict | torch.Tensor) -> bool: ...
```

Two entries, two audiences, one owner:

| path | build/bind | per-step policy |
|---|---|---|
| raw tensor (MD hot path, link 04, link 07) | `rebuild(pos)` | `update(pos)` |
| batch `TensorDict` (this link) | `build(batch)` | `update(batch)` |

There is deliberately **no** `rebuild(batch)`: `build(batch)` *is* the TensorDict-side
forced build (it rebuilds and binds), so a third name for the same operation would be
noise.

**`build(batch)` semantics.**

1. Validate, in this order, every failure a `ValueError` naming the offending values:
   - `("atoms", "pos")` missing → message names the key and the batch's nested keys;
   - `pos.shape != (N, 3)` with `N` the atom count the list was constructed with →
     message names both counts (a different `N` would break `_x_hold.copy_` and the
     capacity sizing);
   - `pos.device != self._device` or `pos.dtype != self._dtype` → message names **both**
     sides and points at `NeighborList.to(...)`. No silent cast: the owner casts.
   - if `("graphs", "cell")` is present: accept `(3, 3)` or `(1, 3, 3)`; a leading batch
     dim `B > 1` is refused (this list is single-system, one cell); values must match the
     constructor cell within `rtol=0, atol=1e-8` Å — loose enough to survive an fp32
     template round-trip, far below any physically meaningful cell difference. **The
     constructor cell stays the owner**; the batch's copy is checked, never adopted.
2. Rebuild the buffers at `pos` via the shared private `_build_at(positions)` (below).
3. Write the binding, replacing whatever was there:
   ```python
   batch["edges"] = TensorDict(
       {"edge_index": self.edge_index, "shifts": self.shifts},
       batch_size=[self.capacity],
   )
   ```
   This is byte-for-byte what `PeriodicPotentialForceField._bind_neighbors` does today, so
   the batch schema seen by `_ShiftAwarePairPotential` and the MACE-family potentials
   (`src/molzoo/mace/potential.py:695` reads `("edges", "shifts")`) is unchanged.
4. `return batch` — the same object, not a copy.

**`build` replaces the `edges` namespace wholesale.** Any pre-existing `edges` entry —
including a stale `edge_diff` / `edge_dist` pair, exactly what
`PotentialForceField._STALE_EDGE_KEYS` strips at construction — is dropped. Documented as
"the list owns `edges` once bound", and strictly stronger than today's stripping.

**`build` does not increment `rebuild_count`.** Link 04 fixed the counter's meaning as
"rebuilds driven during the run; the constructor's initial build is not counted". `build`
is a binding operation of the same kind (it also runs on every `.to()` re-sync), so
counting it would make `.to(dtype)` look like physics and would break
`TestPeriodicPotentialForceField::test_rebuild_is_visible_through_the_bound_buffers`'s
`rebuild_count == 1`. It *does* reset `ago = 0` and refresh `_x_hold` — the buffers are
fresh, so the policy clock starts at the bind.

This forces one small refactor inside `neighbors.py`: the recompute-write-reset body is
extracted to a private `_build_at(positions)`; `rebuild()` becomes `_build_at` plus the
counter increment, and `build()` calls `_build_at` without it. Second real call site, so
extraction is exactly the repo's "inline until the second use" rule — the same move link
04 makes for `_min_perpendicular_width`. Validation is likewise extracted once, to
`_positions_from(batch) -> Tensor`, shared by `build` and `update`.

**`update` dispatch.** One method, `isinstance` at the top:

```python
if isinstance(positions, TensorDictBase):
    positions = self._positions_from(positions)
```

against `tensordict.TensorDictBase` (not `TensorDict`) so lazy / stacked batches dispatch
too. Everything below is link 04's policy verbatim; the return value, `ago`,
`rebuild_count` and `ndanger` semantics are identical on both input types. The validation
in `_positions_from` is metadata-only (shape / device / dtype attribute compares, no host
sync), so the batch path costs the hot loop nothing measurable, and a batch whose `pos`
was silently re-cast fails loud instead of promoting against `_x_hold`.

Because the in-place rebuild never changes a shape or rebinds a buffer, the by-reference
tie survives every `update` / `rebuild` — that is the whole point, and it is what the
identity assertion in the tests pins.

**Name-collision hazard (must be documented).** `build` returns the batch, so
`nl.build(batch).update(batch)` parses — but `.update` there is **`TensorDict.update`**,
not `NeighborList.update`: it merges the batch into itself and never touches the policy.
The returned batch exists to compose with the repo's `forward(td) -> td` convention
(`potential(nl.build(batch))`), not to chain the policy call. The `build` docstring, the
`update` docstring and the user-guide example all state the correct two-statement idiom:

```python
nl.build(batch)          # once
...
nl.update(batch)         # per step — NeighborList.update, not batch.update
```

**`to()` severs the tie; the owner re-binds.** `to(device/dtype)` rebinds
`self.edge_index` / `self.shifts` to new tensors, so a batch bound before the move keeps
pointing at the old ones. Auto-re-binding from inside `to()` is **rejected**: it would
require the list to hold a reference to a batch it does not own (lifetime coupling, and a
"context blob" the shape check forbids). Instead `to()`'s docstring states the
consequence, and the single owner re-binds. A unit test pins both halves — the FF path
stays live across a cast, and a bare `nl.to(...)` on a hand-bound batch does not.

**Ownership move in `forcefield.py`.** `PeriodicPotentialForceField._bind_neighbors` is
deleted. The constructor calls `self.neighbors.build(self._work)`; `_apply` keeps its
`getattr(self, "neighbors", None)` guard, then `neighbors.to(ref.device, ref.dtype)`
followed by `self.neighbors.build(self._work)` (the parent's `TensorDict.apply` produced
new leaves, so both the cast and the re-bind are still required — only the *knowledge of
how to bind* moves). `rebuild_neighbors(pos)` still calls `self.neighbors.rebuild(pos)`:
wiring `update` into the cadence is link 07. `LennardJonesCutForceField` is untouched —
it reads `neighbors.edge_index` / `.shifts` directly, the raw-buffer hot path, and gains
nothing from a batch it does not have.

**Protocol.** `NeighborStrategy` gains `build(batch: TensorDict) -> TensorDict` (the FF
calls it through that annotation, so `ty` needs it declared) and its `update` annotation
widens to `TensorDict | torch.Tensor`. The `_Recorder` stub at
`tests/test_molix/test_md/test_forcefield.py:86` is never `isinstance`-checked and only
feeds `CallableForceField.rebuild_neighbors`, so it needs no change — as link 04 already
recorded.

**Documentation-only schema additions.** CLAUDE.md's *Post-collate batch schema* block
gains two keys already produced and consumed in-tree:

- `("edges", "shifts")` `(E, 3)` `[optional]` — periodic remainder such that
  `pos[target] − pos[source] + shift` is the minimum-image vector, in Å. Written by this
  bind path (and today by `forcefield.py:210`); read by `src/molzoo/mace/potential.py:695`
  and `tests/conftest.py`.
- `("graphs", "cell")` `(B, 3, 3)` `[optional]` — cell vectors in Å; read by
  `src/molpot/composition/sonata.py:372,484` on the stress path, and validated (never
  adopted) by `build`.

Plus one sentence: on the MD bind path `edges.batch_size == [capacity]` with live edges in
`[0, num_edges)` and the tail carrying dead padding edges. `collate_molecules` and
`INDEX_KEYS` are **not** touched — this documents live keys, it does not change the
contract, which is why it does not fall under "what must never change casually".

**Shape check.** `build` / `update` are methods on the owning type (`NeighborList` — the
very example CLAUDE.md's design preferences cite). No factory function, no ambient context
object, no all-in-one façade: `build` does one named thing (bind + refresh), `update` does
one named thing (policy step), and composition (`nl.build(batch)`, then `potential(batch)`)
stays the caller's job. `_build_at` / `_positions_from` are private and extracted only at
their second real call site.

### Reuse decision

No `librarian_report` was supplied with this task — the caller should note the missing
blueprint advisory. Candidates resolved by direct read of `src/molix/md/`:

- `PeriodicPotentialForceField._bind_neighbors` (`src/molix/md/forcefield.py:207-212`) —
  **generalize**: its body is promoted onto `NeighborList.build` so it serves both the
  force field and direct TensorDict callers; the private method is then removed, not left
  as a wrapper.
- `NeighborList.rebuild` / `_write` / `_compute` (link 04 state) — **reuse** through the
  extracted `_build_at`; no second build path is written.
- `molix.data.tasks.neighbor.NeighborList` (via link 03's `NeighborListTask` alias) —
  **reuse**, untouched; kernel-output normalisation stays its job.
- `molix.units.DEAD_EDGE_CUTOFF_FACTOR` — **reuse**, untouched.
- `tensordict.TensorDictBase` — **reuse** as the `isinstance` target rather than a
  hand-rolled `hasattr(x, "keys")` duck-test.
- `molix.data.collate.collate_molecules` / `INDEX_KEYS` — **new — not used**: the batch
  this list binds into is an MD working batch built by the force field, never a collated
  DataLoader batch, and `edges.batch_size == [capacity]` is a fixed-capacity buffer view
  that the offset-rebasing collate has no business producing.
- `NeighborList.build` / `_positions_from` / `_build_at` — **new**: nothing in tree binds
  buffers into a batch except the method being generalized. Naming (`build` beside
  `rebuild` / `update`), keyword-free positional single argument, `ValueError` at bind
  time and `RuntimeError` at run time follow the closest pattern — the existing
  `NeighborList` constructor guard and `_write` overflow check.

## Files to create or modify

- `src/molix/md/neighbors.py` — `build(batch)`, `_positions_from`, `_build_at` extraction,
  `update` dispatch, `NeighborStrategy` members, `to()` and module-docstring additions.
- `src/molix/md/forcefield.py` — delete `_bind_neighbors`; constructor and `_apply` call
  `self.neighbors.build(self._work)`; docstring update on the ownership.
- `tests/test_molix/test_md/test_neighbors.py` — new `TestNeighborListBind`.
- `tests/test_molix/test_md/test_forcefield.py` — new cast-liveness test in
  `TestPeriodicPotentialForceField`; existing two tests unmodified in behaviour.
- `regressions/md-neighborlist-skin-05-bind.py` (new) — public-API bind + drive scenario.
- `CLAUDE.md` — two optional keys in the *Post-collate batch schema* block (free-form
  region, outside the `mol:bootstrap:managed` markers).
- `docs/molix/user-guide/md.md` — build/update example and the `TensorDict.update` warning.
- `CHANGELOG.md` — one `## [Unreleased]` → `### Added` bullet for the new public method
  (repo norm for a public-surface change; link 03 set the precedent).

## Tasks

- [ ] Write failing unit tests for the bind surface in `tests/test_molix/test_md/test_neighbors.py` (`TestNeighborListBind`: `build(batch) is batch`, `edges` leaf identity with `batch_size=[capacity]`, wholesale `edges` replacement, `rebuild_count` unchanged / `ago` reset, liveness of the tie through an in-place rebuild)
- [ ] Write failing unit tests for `build()` validation and `update()` dispatch in `tests/test_molix/test_md/test_neighbors.py` (missing `atoms.pos`, atom-count mismatch, dtype mismatch, meta-device mismatch, cell `(3,3)`/`(1,3,3)` accept vs mismatch vs `B > 1`, and batch-vs-tensor decision/state equivalence)
- [ ] Write a failing cast-liveness test in `tests/test_molix/test_md/test_forcefield.py` (`TestPeriodicPotentialForceField`: after `ff.to(torch.float32)` a `rebuild_neighbors` still changes the energy, i.e. the re-bind survived the cast)
- [ ] Generalize `PeriodicPotentialForceField._bind_neighbors` into `NeighborList.build(batch)` in `src/molix/md/neighbors.py` (extract `_build_at`, add `_positions_from` validation incl. the cell check, write `edges` by reference, return the same batch, leave `rebuild_count` alone)
- [ ] Implement the `TensorDictBase`-vs-tensor dispatch in `NeighborList.update` and declare `build` plus the widened `update` annotation on `NeighborStrategy` in `src/molix/md/neighbors.py`
- [ ] Replace `_bind_neighbors` and its `_apply` re-bind with `self.neighbors.build(self._work)` in `src/molix/md/forcefield.py`, keeping `TestPeriodicPotentialForceField`'s two existing tests green and `LennardJonesCutForceField`'s raw-buffer path unchanged
- [ ] Add docstrings per google style with units (Å) for `build`, the `update` dispatch, the `to()`-severs-the-tie consequence and the `TensorDict.update` name-collision warning, and extend the `src/molix/md/neighbors.py` module docstring with the "the list owns `edges` once bound" ownership statement
- [ ] Document the optional `("graphs", "cell")` `(B, 3, 3)` and `("edges", "shifts")` `(E, 3)` keys in the CLAUDE.md post-collate schema block (documentation only, no collate change), add the build/update example to `docs/molix/user-guide/md.md`, and record `NeighborList.build` in `CHANGELOG.md`
- [ ] Add regression example `regressions/md-neighborlist-skin-05-bind.py` (public API only; hard-coded goldens, no third-party runtime)
- [ ] Run full check + test suite

## Testing strategy

Unit tests only, mirroring the source paths
(`src/molix/md/neighbors.py` → `tests/test_molix/test_md/test_neighbors.py`;
`src/molix/md/forcefield.py` → `tests/test_molix/test_md/test_forcefield.py`), one
behaviour per test, deterministic CPU float64. New class `TestNeighborListBind` sits
alongside `TestNeighborList` (link 03) and `TestNeighborListPolicy` (link 04). The fixture
is the module's existing `_lattice()` helper plus a small local `_batch(pos, cell)` builder
producing `{"atoms": {pos, Z, batch}, "graphs": {cell}}` with root `batch_size=[]` —
the same shape `_periodic_template` builds in `test_forcefield.py`.

**Happy path — bind.**

- `nl.build(batch) is batch` (the returned object is the argument, so
  `potential(nl.build(batch))` composes).
- `batch["edges", "edge_index"] is nl.edge_index` and
  `batch["edges", "shifts"] is nl.shifts` — **identity**, the load-bearing property; a
  TensorDict that ever copied on assignment would silently produce a frozen PES, and this
  assertion is the alarm.
- `batch["edges"].batch_size == torch.Size([nl.capacity])` and
  `set(batch["edges"].keys()) == {"edge_index", "shifts"}`.
- `rebuild_count` is unchanged by `build` and `ago == 0` after it.
- `build` at positions different from the constructor's (a dilated lattice) leaves
  `num_edges` equal to the count at *those* positions.

**Liveness.**

- After `nl.rebuild(compressed)` (and again after `nl.update(compressed_batch)`), the
  identity above still holds, `num_edges` has changed, and the values read through the
  batch equal the list's buffers (`torch.equal` on both leaves) — index identity plus a
  value read, so a stale `shifts` on a surviving index pair cannot hide.
- A bare `nl.to(torch.float32)` after a hand-made `build` does **not** keep the tie
  (documented consequence, pinned so the trade stays visible), while
  `PeriodicPotentialForceField.to(torch.float32)` does — the second test lives in
  `test_forcefield.py` and asserts through the public energy, not through `_work`.

**Edges replacement.**

- A batch whose `edges` already carries `edge_diff` / `edge_dist` / a shorter
  `edge_index` comes back with exactly `{"edge_index", "shifts"}` at `batch_size=[capacity]`.

**Validation (all `ValueError`, message-matched).**

- missing `("atoms", "pos")` → matches `atoms` and `pos`;
- 8-atom batch against a 27-atom list → message contains both `8` and `27`;
- float32 `pos` against a float64 list → message names both dtypes (no silent cast);
- `pos` on the `meta` device against a CPU list → message names both devices (validation
  precedes any kernel call, so this needs no second real device and no CUDA in CI);
- `("graphs", "cell")` equal to the constructor cell as `(3, 3)` and as `(1, 3, 3)` both
  build cleanly; a cell perturbed by `0.01 Å` raises, matching `cell`; a `(2, 3, 3)` cell
  raises naming the batch size (single-system list).

**Dispatch equivalence.** Two lists constructed identically, driven over the same scripted
displacement schedule — one through `update(pos)`, one through `update(batch)` — produce
identical `update` return sequences and identical `ago` / `rebuild_count` / `ndanger`, and
`torch.equal` on `edge_index` and `shifts` with equal `num_edges`. A `update(batch)` whose
`pos` was re-cast to float32 raises the same `ValueError` as `build` (shared
`_positions_from`).

**Protocol.** `isinstance(nl, NeighborStrategy)` still holds with `build` declared.

**Force-field ownership (`test_forcefield.py`).** The two existing
`TestPeriodicPotentialForceField` tests (`test_dead_edge_padding_is_invisible`,
`test_rebuild_is_visible_through_the_bound_buffers`) must pass **unmodified in behaviour**
— any assertion that needs editing is evidence the ownership move changed semantics: stop
and report rather than adjust it. One new test covers the cast path described above.

**Domain validation.** None is re-derived here: this link declares no physics, and the
completeness / equivalence / `ndanger` falsification battery over an NVE trajectory belongs
to link 04, which owns the policy. The physical content exercised here is the shift
reconstruction, and it is checked through the batch in the regression example below.

**Regression example.** `regressions/md-neighborlist-skin-05-bind.py` — a standalone
public-API script (not collected by pytest) that drives the surface end to end on a 4×4×4
simple-cubic lattice, spacing 3.0 Å, cubic cell 12.0 Å (minimum perpendicular half-width
6.0 Å), float64, CPU, `NeighborList(cutoff=3.5, skin=1.5, every=1, delay=0, check=True)`.
All goldens are hand-derived from crystallography and integer arithmetic — no oracle, no
network, no subprocess, no third-party import:

- `r_build == 5.0` Å; `num_edges == 1152` (64 sites × 18 neighbours: 6 at 3.0 Å plus 12 at
  3√2 = 4.2426 Å; the 8 at 5.196 Å are outside `r_build`); `capacity == 1556`
  (`ceil(1.35 × 1152)`).
- `nl.build(batch) is batch`; `batch["edges", "edge_index"] is nl.edge_index`;
  `batch["edges"].batch_size == [1556]`; `nl.rebuild_count == 0` after the build.
- Driving 20 `update(batch)` calls with a rigid translation of +0.2 Å along x per call
  (half-skin 0.75 Å, so the criterion first fires at a 0.8 Å accumulated displacement):
  `True` exactly at updates `{4, 8, 12, 16, 20}`, `rebuild_count == 5`, `ndanger == 0`
  (each build lands at `ago == 4 ≠ _danger_ago == 1`).
- A rigid translation preserves every minimum-image displacement exactly, so after the run
  `num_edges == 1152` still, and reading **through the batch** —
  `‖pos[t] − pos[s] + batch["edges", "shifts"]‖` over `batch["edges", "edge_index"]
  [:num_edges]` — gives max 5.0 Å and exactly `384` pairs within the interaction cutoff
  3.5 Å (64 × 6).
- A second list driven with raw `update(pos)` over the same schedule ends at the same
  `rebuild_count == 5` and `torch.equal` buffers — the dispatch equivalence, at the
  public-API level.

The script prints `OK` and exits 0; it exits 1 on drift. The header records the capture
command, commit sha, torch version, date, and device/precision per `regressions/README.md`.
A runtime disagreement with any literal above is a defect in the neighbour path (or link-04
drift), not a stale golden — report it, do not edit the literal.

## Out of scope

- **Wiring `update` into the cadence** (link 07): `PeriodicPotentialForceField.
  rebuild_neighbors` still calls `rebuild(pos)`, `MD(rebuild_every=)` is untouched, and
  `NeighborListHook` keeps its documented one-step lag.
- **The binned / cell-list build** (link 06); the O(N²) kernel at `r_build` stays the
  backend.
- **Any collate change.** `collate_molecules`, `INDEX_KEYS`, `collate_packed` and the
  `PackedCache` layout are untouched; `("edges", "shifts")` / `("graphs", "cell")` are
  *documented*, not newly produced, and no dataset starts emitting them here.
- **`rebuild(batch)`**, `update_td` / `update_pos` twins, or any second name for the
  TensorDict-side forced build — rejected by design above.
- **Auto re-binding inside `to()`** — rejected: it would make the list hold a reference to
  a batch it does not own. The owner re-binds; the consequence is documented and tested.
- **`LennardJonesCutForceField` migrating to the batch surface** — it has no batch and the
  raw-buffer path is the hot path; unchanged.
- **`PotentialForceField`** (open, non-periodic, frozen list) — unchanged, including its
  `_STALE_EDGE_KEYS` stripping.
- **Multi-system batched MD** (`("graphs", "cell")` with `B > 1`, or a batch root with a
  non-empty `batch_size`) — refused at `build` with a named `ValueError`, not supported.
  A single-cell MD list has no meaningful multi-system semantics; adding one is a separate
  spec, not a widened guard here.
- **New exported symbols** in `src/molix/md/__init__.py` — `build` is a method on an
  already-exported class, so `__all__` (alphabetized, per the 2026-08-09 note) is untouched.
- **The edge convention, dead-edge padding, capacity policy and half-skin criterion** —
  inherited unchanged from links 01/04.
