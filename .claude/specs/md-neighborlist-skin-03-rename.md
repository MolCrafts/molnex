---
title: Hard-rename PeriodicNeighborList to NeighborList (md-neighborlist-skin link 03)
status: approved
created: 2026-08-09
grilled: true
chain: md-neighborlist-skin
---

# Hard-rename PeriodicNeighborList to NeighborList (md-neighborlist-skin link 03)

## Summary

The MD-side rebuilding neighbour list is renamed from `PeriodicNeighborList` to
`NeighborList` across the whole repository. This link is strictly
behaviour-neutral: no logic, no signature, no default, and no numerical result
changes — only identifiers, docstrings, `__all__` entries, prose and one new
CHANGELOG entry. There is no back-compat alias (`stage: experimental`, repo
norm), so every call site moves in the same commit. One import inside
`src/molix/md/neighbors.py` must be aliased at the same time: the module
currently pulls the pipeline task `molix.data.tasks.neighbor.NeighborList` into
its own namespace, which the renamed class would otherwise hard-shadow — the
class would silently reference itself at construction time.

**Chain precondition.** This is link 3 of 7 in the `md-neighborlist-skin` chain
and assumes link 02 has landed, having freed the bare name `NeighborList` on
`molix.md`'s public surface. Verified current state of that surface:
`molix.md.__all__` exports `NeighborListHook` and `NeighborStrategy` but **not**
`NeighborList`; the only in-package occupant of the bare name is the kernel-task
import at `src/molix/md/neighbors.py:36`, which this link aliases. The rename
itself is mechanically independent of link 02's content — it neither reads nor
writes any skin/Verlet state, and links 04+ (skin buffer, half-skin rebuild
trigger) build on the renamed symbol.

## Design

### Naming decision: two deliberate same-name types in different layers

After this link the repository contains two classes named `NeighborList`, and
this is intentional, not an accident to be resolved later:

| Symbol | Layer | Role | Lifecycle |
|---|---|---|---|
| `molix.md.NeighborList` (`molix.md.neighbors`) | MD engine | Stateful, fixed-capacity **buffer owner** — holds `edge_index (capacity, 2)`, `shifts (capacity, 3)`, `num_edges`, `rebuild_count`; rebuilt in place so shapes never change and the force path stays CUDA-graph capturable | One instance per run, held by a `ForceField`, deliberately outside the `nn.Module` tree / `state_dict` |
| `molix.data.tasks.neighbor.NeighborList` (`molix.data.tasks`) | Data pipeline | Stateless **`SampleTask`** — maps one flat sample dict to `edge_index` / `edge_diff` / `edge_dist`, contributes to `task_id` for cache keying | Constructed once per pipeline definition, no per-call state |

They are not variants of one concept: one is a per-run mutable buffer with an
overflow policy, the other is a pure pipeline transform. Each is the natural,
shortest name in its own layer, and the two layers never appear in the same
import block in normal code. The MD module docstring
(`src/molix/md/neighbors.py`) **must** state this distinction explicitly, name
both fully-qualified paths, and explain the alias below — otherwise the next
reader hits the shadow and "fixes" it by renaming the wrong thing.

### The shadow and its alias

`src/molix/md/neighbors.py:36` currently reads:

```python
from molix.data.tasks.neighbor import NeighborList
```

and uses it at line 119 (`self._nl = NeighborList(cutoff=..., pbc=True, symmetry=True)`).
Renaming the class in this module without touching that import makes the
module-level name `NeighborList` ambiguous — the class definition at line 75
would rebind the name after the import, so line 119 would call the class's own
constructor recursively. The import becomes:

```python
from molix.data.tasks.neighbor import NeighborList as NeighborListTask
```

with the use site updated to `NeighborListTask(...)`. The existing comment above
the import (why the kernel-output normalisation is not reimplemented here) stays
and is extended with one sentence naming the alias reason. The `_nl` attribute
name is unchanged — it is private and already unambiguous.

### What does not move

- `NeighborStrategy` (the protocol in the same module) keeps its name. It is the
  type used in annotations across `forcefield.py`, so nothing in `src/` has a
  `PeriodicNeighborList` *annotation* to update — every remaining `src/`
  occurrence is a docstring cross-reference. This is what makes the link
  behaviour-neutral by construction.
- The pipeline task keeps its name and its module. It is referenced by
  `qm9.py`, `revmd17.py`, `data/pipeline.py`, `profiler/task.py`, and named in
  four molzoo spec contracts as `molix.data.tasks.NeighborList` — all out of
  scope and untouched.
- `NeighborListHook` (`molix.md.runner`) keeps its name; it is a hook, not a
  list, and does not collide.

### `__all__` ordering

Per the captured rule *"`__all__` stays alphabetized"* (`.claude/notes/notes.md`,
2026-08-09), `src/molix/md/__init__.py`'s `__all__` must be re-sorted after the
substitution: `"NeighborList"` moves **above** `"NeighborListHook"` (it currently
sits at the `"P"` block). The parenthesised `from molix.md import (...)` lists in
`tests/test_molix/test_md/test_forcefield.py` and `test_compile.py` are ruff-isort
sorted; `NeighborList` happens to sort into the same slot `PeriodicNeighborList`
occupied (after `LennardJonesForceField`), so those blocks need no reordering —
but `ruff check` is the arbiter, not this note.

### CHANGELOG handling

`CHANGELOG.md:90` (`PeriodicNeighborList.edge_index` is now `(capacity, 2)`…) is
an immutable record of a prior change and is **never** rewritten, even though it
sits under `## [Unreleased]`. The rename is recorded as a **new** bullet in the
same `## [Unreleased]` → `### Changed` block, naming both the old and new symbol
and the no-alias decision. Two successive entries about the same class reading in
chronological order is the correct outcome.

### Reuse decision

No `librarian_report` was supplied with this task; the reuse surface was resolved
directly from the tree, and is trivial for a rename:

- `reuse molix.data.tasks.neighbor.NeighborList` — the kernel-output
  normalisation (pbc handling, NaN-padding strip, symmetry expansion, edge-sign
  convention) stays the single owner, consumed through the `NeighborListTask`
  alias. Not duplicated, not generalized, not touched.
- `reuse molix.units.DEAD_EDGE_CUTOFF_FACTOR` — the dead-edge padding constant is
  shared with `molix.engine.static.StaticForward`; only the docstring naming the
  MD class beside it changes (`src/molix/units.py:23`).
- **No new symbols are designed in this link.** A rename adds nothing to the
  public surface: the class count, method set, and constructor signature are
  identical before and after. The "closest pattern" question is therefore moot —
  the pattern *is* the existing file.

## Files to create or modify

**Source (`src/`)**

- `src/molix/md/neighbors.py` — class definition (75), self-referencing return
  annotation (176), module docstring (22) + new two-layer naming paragraph;
  aliased kernel-task import (36) and its use (119).
- `src/molix/md/__init__.py` — module docstring (32), import (50), `__all__`
  entry re-sorted (82).
- `src/molix/md/forcefield.py` — docstring cross-references at 23, 67, 176, 362.
  (Line 341 references `NeighborStrategy`, **not** the renamed class — leave it.)
- `src/molix/units.py` — docstring cross-reference at 23.

**Tests (`tests/`)**

- `tests/test_molix/test_md/test_neighbors.py` — import (6), fixture (20), class
  `TestPeriodicNeighborList` → `TestNeighborList` (23), construction sites
  (101, 106); plus the new guard tests.
- `tests/test_molix/test_md/test_forcefield.py` — import (13) and construction
  sites (182, 200, 228, 239, 251, 276, 287, 309).
- `tests/test_molix/test_md/test_compile.py` — import (18), construction (42).
- `tests/test_molzoo/test_mace/test_variants.py` — comment (460), local imports
  (468, 495), construction sites (477, 501).

**Executable out-of-suite scripts**

- `benchmarks/bench_mace_matpes.py` — module docstring (15), import (32),
  docstring (52), construction (85), inline comment (122).
- `benchmarks/verify_md_ljcut_nve.py` — module docstring (5), import (42),
  construction (128).
- `scripts/matpes_port/run_nve.py` — import (36), parameter annotation (99),
  comment (265), construction (267).
- `regressions/mace-subpackage-restructure-07-cleanup.py` — docstring mention
  only (202); no executable change, goldens untouched.
- `regressions/md-neighborlist-skin-03-rename.py` **(new)** — this spec's
  regression example.

**Prose**

- `docs/molix/user-guide/md.md` — force-field table (48, 49), example import and
  construction (65, 67).
- `docs/molix/explanation/throughput-and-compilation.md` — static-shapes
  paragraph (224).
- `CHANGELOG.md` — **add** one bullet under `## [Unreleased]` → `### Changed`;
  line 90 stays byte-identical.

## Tasks

- [ ] Write failing guard tests for the renamed MD symbol in `tests/test_molix/test_md/test_neighbors.py` (rename `TestPeriodicNeighborList` → `TestNeighborList`; assert `from molix.md import NeighborList` resolves, `PeriodicNeighborList` is absent from `molix.md` and its `__all__`, and `molix.md.NeighborList is not molix.data.tasks.neighbor.NeighborList`)
- [ ] Rename `PeriodicNeighborList` → `NeighborList` in `src/molix/md/neighbors.py`, alias the kernel-task import as `NeighborListTask` (line 36) with its use at line 119, and extend the module docstring with the MD-buffer-owner vs pipeline-task distinction
- [ ] Update the remaining molix references in `src/molix/md/__init__.py` (docstring, import, re-sorted `__all__`), `src/molix/md/forcefield.py` (4 docstring refs), and `src/molix/units.py:23`
- [ ] Update the test-suite call sites in `tests/test_molix/test_md/test_forcefield.py`, `tests/test_molix/test_md/test_compile.py`, and `tests/test_molzoo/test_mace/test_variants.py`, changing identifiers only
- [ ] Update the executable out-of-suite call sites in `benchmarks/bench_mace_matpes.py`, `benchmarks/verify_md_ljcut_nve.py`, `scripts/matpes_port/run_nve.py`, and the docstring mention in `regressions/mace-subpackage-restructure-07-cleanup.py:202`
- [ ] Update the prose in `docs/molix/user-guide/md.md` and `docs/molix/explanation/throughput-and-compilation.md`, and add a new `### Changed` bullet to `CHANGELOG.md` recording the rename (leave line 90 byte-identical)
- [ ] Add regression example `regressions/md-neighborlist-skin-03-rename.py` (public API only; hard-coded goldens; header records capture command, sha, torch version, date, device/precision)
- [ ] Verify zero residual references with `grep -rn "PeriodicNeighborList" src/ tests/ benchmarks/ scripts/ docs/ regressions/` returning nothing
- [ ] Run full check + test suite

## Testing strategy

The rename is proven by the **existing** suites passing with only identifier
substitutions in their sources — no assertion, tolerance, golden, or expected
value is edited anywhere. Any test that needs a *changed* assertion to pass is
evidence the rename was not behaviour-neutral: stop and report rather than adjust
the assertion.

Unit tests (mirror layout, one behaviour per test), all in
`tests/test_molix/test_md/test_neighbors.py` under `TestNeighborList` (type
mirror of the renamed class):

- Happy path — the eleven existing tests in `TestNeighborList` (protocol
  conformance, capacity headroom, buffer shapes, shape-invariant rebuild, changed
  neighbour set, dead-edge self-loops, dead-edge shift length, minimum-image
  shift reconstruction, `rebuild_count`, positional-dtype `to`,
  cutoff > L/2 refusal, overflow refusal) pass **unmodified in behaviour**.
- Guard 1 (new) — `from molix.md import NeighborList` succeeds and
  `hasattr(molix.md, "PeriodicNeighborList")` is `False` and
  `"PeriodicNeighborList" not in molix.md.__all__`.
- Guard 2 (new, anti-shadow) — `molix.md.NeighborList` is **not**
  `molix.data.tasks.neighbor.NeighborList`, and
  `molix.md.neighbors.NeighborListTask` **is** the pipeline task. This is the
  single test that would have caught the recursive-shadow failure mode.

Domain validation is inherited, not re-derived: this link declares no physics, so
no new hard-coded physical reference values are introduced. The minimum-image
contract and the `r_cut ≤ L/2` guard are exercised by the untouched existing
tests.

**Regression example** — `regressions/md-neighborlist-skin-03-rename.py`: a
minimal public-API script (`from molix.md import NeighborList`, nothing private)
on a 3×3×3 simple-cubic lattice, spacing 3.0 Å, cubic cell 9.0 Å, `cutoff=3.5`,
float64, CPU, default `capacity_factor=1.35`. Goldens are **analytically
derived** (no third-party oracle, no network, no subprocess), which is why they
are safe to hard-code:

- `num_edges == 162` — each site's only in-cutoff neighbours are the 6 at 3.0 Å
  (next shell √2·3.0 = 4.243 Å > 3.5 Å); 27 sites × 6 = 162 directed edges under
  the full-bidirectional (`symmetry=True`) convention.
- `capacity == 219` — `ceil(1.35 × 162) = ceil(218.7)`.
- `edge_index.shape == (219, 2)` and `shifts.shape == (219, 3)`.
- dead-edge shift norm `== 35.0` Å — `DEAD_EDGE_CUTOFF_FACTOR (10.0) × 3.5`.
- after `rebuild(pos + torch.tensor([1.234, 0.0, 0.0]))` (rigid translation, which
  preserves every minimum-image displacement exactly): `num_edges == 162`,
  shapes unchanged, `rebuild_count == 1`.

The script prints `OK` and exits 0; it exits 1 on drift. If the runtime disagrees
with a golden above, that is a defect in the neighbour path, not a stale golden —
report it, do not edit the literal.

## Out of scope

- **Any behaviour change.** Skin/Verlet buffering, rebuild triggers, capacity
  policy and half-skin displacement checks belong to links 04+ of the
  `md-neighborlist-skin` chain. Zero logic edits land here.
- **Back-compat alias.** No `PeriodicNeighborList = NeighborList` shim, no
  deprecation warning, no `__getattr__` fallback — `stage: experimental`, repo
  norm, and a shim would defeat the guard test.
- **Renaming the pipeline task** `molix.data.tasks.neighbor.NeighborList`, or its
  call sites in `qm9.py`, `revmd17.py`, `data/pipeline.py`, `profiler/task.py`.
- **`NeighborStrategy`** (protocol) and **`NeighborListHook`** (runner hook) keep
  their names; `src/molix/md/forcefield.py:341` refers to the protocol and is
  deliberately left alone.
- **`CHANGELOG.md:90`** and the `src/molzoo/specs/*.md` §7.4 run-log rows —
  append-only history, never rewritten.
- **`src/molix/md/driver.py`, `src/molix/md/runner.py`,
  `tests/test_molix/test_md/test_driver.py`, `tests/test_molix/test_md/test_runner.py`** —
  grep-verified to contain **zero** `PeriodicNeighborList` occurrences; they need
  no edit and appear here only to record that they were checked, not skipped.
- **`benchmarks/` lint coverage.** `mol_project.build.check` deliberately excludes
  `benchmarks/` (note: *build.check scope*, 2026-08-09); the benchmark edits are
  verified by grep and a manual import smoke, not by the gate. Do not widen the
  gate in this link.
- Alternatives considered and rejected: (a) keeping `PeriodicNeighborList` and
  naming the skinned variant separately — rejected, the skin is not a different
  concept and the chain would carry two names for one buffer owner; (b) renaming
  the *pipeline* task instead to avoid the collision — rejected, it is the
  broader public contract, named in four molzoo spec contracts and every dataset
  pipeline.
