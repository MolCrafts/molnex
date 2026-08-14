---
title: Dissolve molix.data — training-side transforms out of data/ and datasets/
status: approved
slug: molix-data-dissolution-03-transforms
created: 2026-08-14
---

# Dissolve molix.data — training-side transforms out of data/ and datasets/

## Summary

Three modules filed under `molix/data/tasks/` and `molix/datasets/` are not
ETL at all — they have live molnex consumers and encode molnex contracts, so
they cannot leave with the acquisition layer without inverting the repository
dependency direction. This phase rehomes them: the neighbour-list transform
(consumed by `molix.md.NeighborList` and pinned by an in-repo regression) and
the connectivity column↔COO adapters go to `src/molix/batch/`; the extended-XYZ
codec (written by the MD runner, read by two numerical regression suites) goes
to `src/molix/io/`. It also decouples `TaskProfiler` from the pipeline DSL so
`molix.profiler` stops depending on symbols destined for molhub.

## Design

No new physics is introduced. One invariant is carried across verbatim and
pinned by hard-coded goldens: `edge_index[:, 0]` is the source,
`edge_index[:, 1]` the target, `edge_diff = pos[target] − pos[source]` (Å),
`edge_dist = ‖edge_diff‖`, full-bidirectional by default (`symmetry=True`,
`E = 2 × n_pairs`), the C++ kernel's `pos[rows] − pos[cols]` delta negated
exactly once. That is CLAUDE.md's Edge Convention, unchanged.

**`molix.batch.neighbor.NeighborList`** ← `molix/data/tasks/neighbor.py`. Body
moved verbatim (kernel call, buffer sizing, overflow raise, NaN-padding strip,
sign negation, symmetry expansion, `task_id` string). One change: it no longer
inherits `SampleTask`. The ETL base leaves the repo, and this class does not
need it — it keeps `task_id` and `execute`, which is exactly the structural
`Runnable` contract, so a molhub pipeline can still wrap it. Its home is
`molix.batch` because it is the sole owner of the edge convention that the
collate schema depends on; `molix.md.neighbors` keeps its
`NeighborList as NeighborListTask` alias for the same shadowing reason
documented there today, retargeted at `molix.batch.neighbor`.

*Cross-repo note (advisory, implemented in phase 05):* molhub's pipeline
decides cacheability by `isinstance(task, (SampleTask, DatasetTask))`. To keep
neighbour-list nodes cacheable, molhub declares a thin `SampleTask` subclass
that delegates to `molix.batch.NeighborList` — the physics is never duplicated.

**`molix.batch.valence_columns`** ← `molix/datasets/_valence_columns.py`. Its
own docstring says "kernel-local stack helpers … not part of the collate /
cache schema"; it stacks post-collate namespace columns
(`angles.atomi/atomj/atomk`, propers `+atoml`, impropers center-first) into COO
`[arity, N]` at kernel call sites. That is batch-side, and it was only ever
misfiled under `datasets/`. Moved verbatim; the underscore prefix is dropped
because the module is now a public in-repo surface.

**`molix.batch.bond_adapter`** ← `molix/datasets/_bond_adapter.py`.
`bond_index_from_columns` is the canonical constructor of the `(2, N)`
`bond_index` that the `bonds` namespace and `BondHarmonic`'s anti-alias guard
require. It stays in molnex (and is importable by molhub, which builds samples).
Moved verbatim, underscore dropped.

**`molix.io.extxyz`** ← `molix/datasets/_extxyz.py`. `write_extxyz_frames` is
called by `molix.md.runner` on `MDHook.on_run_end`; `parse_extxyz_frames` is
called by `tests/regression/test_values_{ewald,dipole}.py`,
`tests/test_molpot/test_elec/calculators/test_padding.py` and
`scripts/matpes_port/neighbor_graph_audit.py`. It is a file-format codec, not a
dataset. `molix.io` already owns training-run file IO, so the codec joins it and
is exported from `molix/io/__init__.py` (`ExtxyzFrame`, `parse_extxyz_frames`,
`write_extxyz_frames`, `__all__` re-alphabetized).

**`TaskProfiler` decoupling.** It imports `molix.data.pipeline.Node` (only to
build a one-node wrapper so it can call `.apply`) and
`molix.data.task.DatasetTask` (only for an `isinstance` fit check). Both are
replaced by structural dispatch: call `task.execute(sample)` when present else
`task(sample)`; fit when `fit` is callable. Behaviour is identical for every
existing input, and the profiler stops depending on the departing DSL.
`molix/profiler/{__init__,mock,dataset}.py` docstrings are repointed.

**Stale docstring references found while surveying — fixed here.**
`src/molpot/heads/edge.py:55` cites `molix.data.tasks.AvgNumNeighborsStat`,
which **does not exist anywhere in the repo**; it is repointed at the dataset
`avg_num_neighbors` cached property that actually supplies the value.
`src/molpot/heads/multipole.py:177` and `heads/charge_bond.py:82` are rewritten
so they do not name symbols that are leaving (`ConstantLabel`) or that changed
address (`NeighborList`).

Rewrite map:

| old | new |
|---|---|
| `molix.data.tasks.neighbor` / `molix.data.tasks.NeighborList` | `molix.batch.neighbor` / `molix.batch.NeighborList` |
| `molix.datasets._valence_columns` | `molix.batch.valence_columns` |
| `molix.datasets._bond_adapter` | `molix.batch.bond_adapter` |
| `molix.datasets._extxyz` | `molix.io.extxyz` |

### Reuse decision

No `librarian_report` supplied; resolved from a direct importer sweep.

- `NeighborList` (pipeline task) — **generalize**: promoted out of the ETL task
  tree so one implementation serves `molix.md`, the in-repo regression and a
  molhub-side pipeline wrapper. Explicitly **not** reimplemented: forking the
  kernel-output normalisation (pbc, NaN strip, symmetry, sign) is the exact
  failure mode `molix/md/neighbors.py:258-263` already warns against.
- `parse_extxyz_frames` / `write_extxyz_frames` / `ExtxyzFrame` — **reuse**,
  moved verbatim. CLAUDE.md pins extxyz IO to this in-tree parser (no ASE).
- `stack_angle_index` / `stack_proper_index` / `stack_improper_index` /
  `bond_index_from_columns` — **reuse**, moved verbatim.
- `Node` / `DatasetTask` in `TaskProfiler` — **new — not adopted**: replaced by
  structural dispatch rather than reproducing a DAG wrapper molnex does not own.
- Closest pattern: `molix.md.neighbors`' documented aliasing convention and
  `molix.io`'s existing export block.

## Files to create or modify

- `src/molix/batch/neighbor.py` (new — content of `src/molix/data/tasks/neighbor.py`)
- `src/molix/batch/valence_columns.py` (new — content of `src/molix/datasets/_valence_columns.py`)
- `src/molix/batch/bond_adapter.py` (new — content of `src/molix/datasets/_bond_adapter.py`)
- `src/molix/io/extxyz.py` (new — content of `src/molix/datasets/_extxyz.py`)
- `src/molix/data/tasks/neighbor.py` (delete)
- `src/molix/datasets/_valence_columns.py` (delete)
- `src/molix/datasets/_bond_adapter.py` (delete)
- `src/molix/datasets/_extxyz.py` (delete)
- `src/molix/batch/__init__.py`
- `src/molix/io/__init__.py`
- `src/molix/data/tasks/__init__.py`
- `src/molix/md/neighbors.py`
- `src/molix/md/runner.py`
- `src/molix/datasets/water_les.py`
- `src/molix/profiler/task.py`
- `src/molix/profiler/__init__.py`
- `src/molix/profiler/mock.py`
- `src/molix/profiler/dataset.py`
- `src/molpot/heads/edge.py`
- `src/molpot/heads/multipole.py`
- `src/molpot/heads/charge_bond.py`
- `tests/test_molix/test_batch/test_neighbor.py` (new — from `tests/test_molix/test_data/test_neighbor_sign.py`)
- `tests/test_molix/test_batch/test_valence_columns.py` (new — from `tests/test_molix/test_datasets/test_valence_columns.py`)
- `tests/test_molix/test_io/test_extxyz.py` (new — from `tests/test_molix/test_datasets/test_extxyz.py`)
- `tests/test_molix/test_profiler/test_task.py` (new)
- `tests/test_molix/test_data/test_neighbor_sign.py` (delete)
- `tests/test_molix/test_datasets/test_valence_columns.py` (delete)
- `tests/test_molix/test_datasets/test_extxyz.py` (delete)
- `tests/test_molix/test_data/test_transform.py`
- `tests/test_molix/test_batch/test_bond_collate.py`
- `tests/test_molix/test_md/test_neighbors.py`
- `tests/test_molpot/test_elec/calculators/test_padding.py`
- `tests/regression/test_values_ewald.py`
- `tests/regression/test_values_dipole.py`
- `regressions/md-neighborlist-skin-02-prune.py`
- `regressions/learnable-classical-ff-02-valence-topology.py`
- `regressions/molix-data-dissolution-03-transforms.py` (new)
- `scripts/matpes_port/neighbor_graph_audit.py`
- `scripts/check_test_mirror.py`
- `docs/molix/user-guide/profiling.md`
- `docs/molzoo/index.md`
- `docs/molzoo/user-guide/allegro.md`
- `docs/molzoo/specs/allegro.md`
- `docs/molzoo/specs/mace_omol.md`

## Tasks

- [ ] Write failing tests for the rehomed transforms: move `test_neighbor_sign.py` → `tests/test_molix/test_batch/test_neighbor.py`, `test_valence_columns.py` → `tests/test_molix/test_batch/`, `test_extxyz.py` → `tests/test_molix/test_io/`, all on the new import paths (RED)
- [ ] Write failing `tests/test_molix/test_profiler/test_task.py` covering `TaskProfiler` dispatch over an `execute`-only object and over a plain callable, plus the fit path (RED — closes an existing mirror gap)
- [ ] Move `tasks/neighbor.py` to `src/molix/batch/neighbor.py` as a standalone class (`task_id` + `execute`, no `SampleTask` base) and export `NeighborList` from `molix/batch/__init__.py`
- [ ] Repoint `src/molix/md/neighbors.py`'s `NeighborListTask` alias and its module docstring at `molix.batch.neighbor`
- [ ] Move `_valence_columns.py` and `_bond_adapter.py` into `src/molix/batch/` as `valence_columns.py` / `bond_adapter.py` and export their functions
- [ ] Move `_extxyz.py` to `src/molix/io/extxyz.py`, export it from `molix/io/__init__.py`, and repoint `molix/md/runner.py`, `molix/datasets/water_les.py`, both `tests/regression/test_values_*.py`, `tests/test_molpot/test_elec/calculators/test_padding.py` and `scripts/matpes_port/neighbor_graph_audit.py`
- [ ] Implement structural dispatch in `src/molix/profiler/task.py` (drop the `Node` and `DatasetTask` imports) and repoint the profiler docstrings
- [ ] Fix the stale `molpot` docstring references (`heads/edge.py` `AvgNumNeighborsStat` does not exist; `heads/multipole.py`, `heads/charge_bond.py`) and update `scripts/check_test_mirror.py` ALLOW_MISSING entries
- [ ] Rewrite the remaining old-path references in `tests/`, `regressions/md-neighborlist-skin-02-prune.py`, `regressions/learnable-classical-ff-02-valence-topology.py` and `docs/`
- [ ] Add regression example `regressions/molix-data-dissolution-03-transforms.py` (public API only; hard-coded goldens, no third-party runtime)
- [ ] Run full check + test suite

## Testing strategy

Unit tests only under `tests/`, mirroring source; moved tests keep their
assertions verbatim.

- `tests/test_molix/test_batch/test_neighbor.py` → `TestNeighborList`: one
  method per behaviour — edge-sign convention (`edge_diff = pos[target] −
  pos[source]`), `symmetry=True` doubling, `symmetry=False` half-pairs,
  `task_id` string stability (cache identity), buffer auto-sizing, overflow
  raise, NaN-padding strip, PBC path.
- `tests/test_molix/test_md/test_neighbors.py` keeps its identity assertions,
  retargeted: `molix.md.NeighborList is not molix.batch.neighbor.NeighborList`
  and `molix.md.neighbors.NeighborListTask is molix.batch.neighbor.NeighborList`
  — the anti-shadowing guard stays live.
- `tests/test_molix/test_batch/test_valence_columns.py` → per-function tests of
  the three stackers, including the improper center-first (`atomi` = center)
  ordering and the ragged-length `ValueError`.
- `tests/test_molix/test_io/test_extxyz.py` → parse/write round-trip, lattice
  and property-string parsing, malformed-header raise.
- `tests/test_molix/test_profiler/test_task.py` (new) → dispatch over an
  `execute`-only object, over a plain callable, and the `fit` branch.
- Domain validation with hard-coded expected values: the two-atom chain
  `pos = [[0,0,0],[1,0,0]]` at `cutoff = 5.0` must yield
  `edge_index = [[0,1],[1,0]]`, `edge_diff = [[1,0,0],[-1,0,0]]`,
  `edge_dist = [1.0, 1.0]` — literals, no oracle.

Regression example: `regressions/molix-data-dissolution-03-transforms.py` —
public API only (`from molix.batch import NeighborList`,
`from molix.batch.valence_columns import stack_angle_index`,
`from molix.io.extxyz import parse_extxyz_frames, write_extxyz_frames`). It runs
the hard-coded 3-atom chain through `NeighborList` and pins
`edge_index`/`edge_diff`/`edge_dist` literals and the `task_id` string; stacks a
hard-coded angles namespace to `[3, N]` COO and pins it; writes and re-parses a
two-frame extxyz buffer and pins the recovered lattice and positions; and
asserts `molix.data.tasks.neighbor` and `molix.datasets._extxyz` are no longer
importable. No third-party software runs.

## Out of scope

- Deleting `src/molix/data/` and `src/molix/datasets/` — phase 04.
- Any change to the neighbour-list algorithm, cutoff inequality or kernel.
- The molhub-side `SampleTask` wrapper that keeps neighbour nodes cacheable — phase 05.
- Refreshing `.claude/notes/architecture.md` — `/mol:map` after phase 04.
