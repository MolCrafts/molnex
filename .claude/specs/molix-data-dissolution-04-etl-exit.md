---
title: Dissolve molix.data — delete the ETL layer from molnex
status: approved
slug: molix-data-dissolution-04-etl-exit
created: 2026-08-14
---

# Dissolve molix.data — delete the ETL layer from molnex

## Summary

Final molnex phase. Everything training-side has left `molix/data/` and
`molix/datasets/`; what remains is data acquisition — the `DataSource`
protocol, the `PipelineSpec` DAG, the `Task` hierarchy and its remaining
transforms, and the five concrete dataset adapters — which under the operator's
boundary is molhub's responsibility. This phase deletes them outright, with no
deprecation shim and no alias (stage: experimental), deletes the tests and
benchmarks that exist only to exercise them, and updates the documentation and
CLAUDE.md so the repository describes what it actually contains. It ends with a
zero-reference gate: no `molix.data` or `molix.datasets` string survives
anywhere in the tree, and neither package directory exists.

## Design

No physics is introduced or changed. What is deleted computes nothing a model
consumes at training time — it produces sample dicts and packed caches, and the
packed-cache reader/writer itself already lives in `molix.store`.

**Deleted from `src/molix/data/`:** `source.py` (`DataSource`, `Sample`,
`InMemorySource`, `SubsetSource`), `pipeline.py` (`Node`, `Edge`, `DAGCache`,
`PipelineSpec`, `Pipeline`), `task.py` (`Runnable`, `Task`, `SampleTask`,
`DatasetTask`), `tasks/` (`AtomicDress`, `ConstantLabel`, `UnitConvert` and the
package `__init__`), `README.md`, `__init__.py` — i.e. the whole directory.

**Deleted from `src/molix/datasets/`:** `qm9.py`, `revmd17.py`, `threebpa.py`,
`water_les.py`, `molrec.py`, `_data_acquisition.md`, `__init__.py` — the whole
directory. Verified precondition: after phase 03 nothing under `src/` imports
either package.

**`src/molix/data/README.md` disposition.** Its two-tier data-contract table is
already stated in CLAUDE.md (§ Architecture) and its module list describes
modules that no longer exist here; it is deleted rather than relocated. Its one
non-duplicated claim, `DataModule.from_cached_pipeline` as "the one-shot
factory", was already **false** before this chain started — that method does not
exist in `datamodule.py`. Recorded here so nobody hunts for a lost feature.

**Deleted tests:** `tests/test_molix/test_data/` (remaining: `test_pipeline.py`,
`test_transform.py`, `test_constant_label.py`, `__init__.py`) and
`tests/test_molix/test_datasets/` (`test_qm9.py`, `test_molrec.py`,
`test_water_les.py`, `test_units_contract.py`, `conftest.py`, `__init__.py`).
These are the coverage that travels with the code; phase 05 re-establishes it in
molhub against molhub's own dataset protocol.

**Deleted benchmarks:** `benchmarks/train_pinet_ef_padded.py`,
`train_pinet_ef_compare.py`, `bench_dataloader_datasets.py`,
`bench_pinet_ef_full.py`. Each is a `Pipeline` + `QM9Source`/`RevMD17Source`
driver; none can run without the acquisition layer. Rewriting them against
`molix.store` alone would mean shipping a fixture dataset molnex has no business
owning, and leaving them broken is silent debt. They are deleted; git history is
the reference, and phase 05 records the SHA. `benchmarks/bench_mace_matpes.py`
and the MD verification benchmarks are untouched.

**Boundary tests.** `tests/test_repository_boundaries.py` already asserts that
molnex library code never imports `molhub`. Two assertions are added in the same
file (it is the repo's declared home for dependency-boundary checks): the
`molix.data` and `molix.datasets` packages must not resolve, and no file under
`src/`, `tests/`, `scripts/`, `regressions/` or `benchmarks/` may mention them.
This is what makes the dissolution enforceable rather than a one-off cleanup.

**Documentation.** `docs/api/molix.md` drops the `molix.data*` and
`molix.datasets` mkdocstrings blocks and gains `molix.batch` / `molix.store`
sections (phases 01–02 already added their own entries; this phase removes the
stragglers `molix.data`, `molix.data.pipeline`, `molix.data.task`,
`molix.data.tasks`, `molix.datasets`). `docs/molix/user-guide/data.md` is a
pipeline-centric page describing departed code — deleted, with its
`zensical.toml` nav entry ("Data Pipeline"); the surviving Data Loading / Data
Modules pages gain a short pointer that dataset acquisition lives in MolHub.
`docs/molzoo/{index,tutorials/index,user-guide/allegro,specs/allegro,specs/mace_omol}.md`
have their `molix.data.NeighborList` / `molix.data pipeline` mentions retargeted
at `molix.batch.NeighborList` and MolHub respectively.

**CLAUDE.md.** The managed § Architecture prose is updated: the module
dependency graph replaces `molix.data (Dataset, collate, preprocess)` and
`molix.datasets (QM9, RevMD17, …)` with `molix.batch` / `molix.store` and a
one-line statement that dataset acquisition is MolHub's; the two-tier data
contract keeps both shapes but names `molix.batch.collate.collate_molecules` as
the conversion point; the cache-format section points at
`src/molix/store/cache.py`; the Extxyz sentence points at `molix.io.extxyz`.

### Reuse decision

No `librarian_report` supplied; resolved from a direct importer sweep.

- `DataSource` / `InMemorySource` / `SubsetSource` — **new — not adopted**:
  deleted, not reproduced. molhub owns acquisition; molnex reads packed caches.
- `PipelineSpec` / `Pipeline` / `Node` / `Edge` / `DAGCache` — **new — not
  adopted**: deleted. Their materialisation role is scheduled for replacement by
  `streaming-data-tiers-03-conveyor` (molhub); reproducing them in molnex would
  fork a design already superseded.
- `Task` / `SampleTask` / `DatasetTask` / `Runnable` — **new — not adopted**:
  deleted; the only training-side member (`BatchTask`) was promoted in phase 02.
- `AtomicDress` / `UnitConvert` / `ConstantLabel` — **new — not adopted** here:
  deleted from molnex; their semantics and goldens are re-expressed in molhub
  (phase 05), not re-implemented in molnex.
- `QM9Source` / `RevMD17Source` / `ThreeBPASource` / `WaterLESSource` /
  `MolRecSource` — **new — not adopted** here: deleted; molhub's approved
  `python-native-dataset-protocol` chain already owns QM9/RevMD17/ThreeBPA, and
  phase 05 covers the remainder.
- Closest pattern: `tests/test_repository_boundaries.py`'s existing AST-walking
  import check — the new assertions are written in the same style, in that file.

## Files to create or modify

- `src/molix/data/` (delete: `__init__.py`, `source.py`, `pipeline.py`, `task.py`, `tasks/__init__.py`, `tasks/constant.py`, `tasks/dress.py`, `tasks/unit.py`, `README.md`)
- `src/molix/datasets/` (delete: `__init__.py`, `qm9.py`, `revmd17.py`, `threebpa.py`, `water_les.py`, `molrec.py`, `_data_acquisition.md`)
- `tests/test_molix/test_data/` (delete)
- `tests/test_molix/test_datasets/` (delete)
- `benchmarks/train_pinet_ef_padded.py` (delete)
- `benchmarks/train_pinet_ef_compare.py` (delete)
- `benchmarks/bench_dataloader_datasets.py` (delete)
- `benchmarks/bench_pinet_ef_full.py` (delete)
- `tests/test_repository_boundaries.py`
- `regressions/molix-data-dissolution-04-etl-exit.py` (new)
- `scripts/matpes_port/run_nve.py`
- `scripts/check_test_mirror.py`
- `docs/api/molix.md`
- `docs/molix/user-guide/data.md` (delete)
- `docs/molix/user-guide/data-loading.md`
- `docs/molix/user-guide/data-modules.md`
- `docs/molix/user-guide/profiling.md`
- `docs/molzoo/index.md`
- `docs/molzoo/tutorials/index.md`
- `docs/molzoo/user-guide/allegro.md`
- `docs/molzoo/specs/allegro.md`
- `docs/molzoo/specs/mace_omol.md`
- `zensical.toml`
- `CLAUDE.md`

## Tasks

- [ ] Write failing boundary tests in `tests/test_repository_boundaries.py`: `molix.data` and `molix.datasets` must not resolve, and no file under `src/`, `tests/`, `scripts/`, `regressions/`, `benchmarks/` may reference them (RED while the packages still exist)
- [ ] Delete `src/molix/data/` in full (`source.py`, `pipeline.py`, `task.py`, `tasks/`, `README.md`, `__init__.py`)
- [ ] Delete `src/molix/datasets/` in full (`qm9.py`, `revmd17.py`, `threebpa.py`, `water_les.py`, `molrec.py`, `_data_acquisition.md`, `__init__.py`)
- [ ] Delete `tests/test_molix/test_data/`, `tests/test_molix/test_datasets/` and the four ETL-dependent benchmark drivers
- [ ] Update `docs/api/molix.md`, delete `docs/molix/user-guide/data.md` with its `zensical.toml` nav entry, and retarget the remaining `molix.data` mentions in the molix and molzoo guides at `molix.batch` / MolHub
- [ ] Update CLAUDE.md § Architecture (module dependency graph, two-tier data contract conversion point, cache-format path, Extxyz sentence) to describe `molix.batch` / `molix.store` and MolHub-owned acquisition
- [ ] Verify zero references repo-wide and remove the now-dead `molix/data`/`molix/datasets` entries from `scripts/check_test_mirror.py` and the `scripts/matpes_port/run_nve.py` docstring
- [ ] Add regression example `regressions/molix-data-dissolution-04-etl-exit.py` (public API only; hard-coded goldens, no third-party runtime)
- [ ] Run full check + test suite

## Testing strategy

Unit tests only under `tests/`; the surviving suite is the safety net — after
deleting the ETL layer it must pass unchanged, which is the proof that nothing
training-side depended on it.

- `tests/test_repository_boundaries.py` → one function per rule:
  `test_molnex_library_does_not_import_molhub` (existing, unchanged),
  `test_molix_data_package_is_gone` (`importlib.util.find_spec("molix.data")
  is None` and likewise `molix.datasets`), and
  `test_no_molix_data_references` (AST/textual scan over `src/`, `tests/`,
  `scripts/`, `regressions/`, `benchmarks/` reporting every offending
  `path:line`, so a reintroduction fails loudly with an address).
- Edge case covered by the scan: a docstring or comment mention counts as a
  violation — stale prose pointing at a deleted module is the exact rot this
  chain exists to remove.
- No domain validation: nothing physical is touched.

Regression example: `regressions/molix-data-dissolution-04-etl-exit.py` — public
API only. It asserts `molix.data` / `molix.datasets` do not resolve; asserts the
surviving public entry points import and compose
(`from molix.store import PackedCache, MmapDataset`, `from molix.batch import
DataModule, NeighborList, TargetSchema, collate_molecules`, `from molix.io
import parse_extxyz_frames`); and runs one end-to-end hard-coded scenario —
two literal molecules through `NeighborList` → `PackedCache` → `MmapDataset` →
`DataModule` — pinning literal goldens for batch shapes, `graphs.num_atoms` and
`edge_dist`. No third-party software is invoked.

## Out of scope

- Any molhub-side work — phase 05 (repo: molhub).
- Refreshing `.claude/notes/architecture.md` (6 stale `molix.data` mentions) —
  named follow-up: run `/mol:map` after this phase lands.
- Updating molhub's `.claude/notes/streaming-data-tiers.md`, which cites
  `molnex:src/molix/data/sampler.py:267` and `molnex:src/molix/data/cache.py:91`
  — named follow-up on the molhub side, tracked in phase 05.
- Restoring the deleted benchmarks in any form.
