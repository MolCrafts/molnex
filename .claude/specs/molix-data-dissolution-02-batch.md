---
title: Dissolve molix.data — collate, sampling and DataModule into molix.batch
status: approved
slug: molix-data-dissolution-02-batch
created: 2026-08-14
---

# Dissolve molix.data — collate, sampling and DataModule into molix.batch

## Summary

Second phase of the `molix.data` dissolution: the batch-construction half moves
into its fixed new home `src/molix/batch/` — `collate.py` (the post-collate
schema is a repo hard contract), `sampler.py`, `datamodule.py`, `group_split.py`
— together with the post-collate transform surface (`BatchTask`,
`PadMolecularBatch`). This phase also severs the one real coupling from
training-side code into the ETL pipeline DSL: `DataModule` currently takes
`batch_nodes: Sequence[Node]` and calls `Node.apply`, which would leave molnex
importing a molhub symbol. It becomes `batch_tasks: Sequence[BatchTask]` calling
`task.execute`, and `PipelineSpec` stops routing post-collate tasks at all.

## Design

No physics is introduced or changed; `$META.science.required` does not apply.
The collated batch schema (`atoms` / `edges` / `graphs` / `bonds`, per-namespace
`batch_size`), the `INDEX_KEYS` rebase registry, the packed fast path, the
token-budget packing rule and `PadMolecularBatch`'s ghost-atom zero-contribution
contract are all unchanged.

**New package `molix.batch`** — one responsibility: *samples → the collated
batch the model forward receives, and the loader machinery that feeds it*.
Modules moved verbatim except for import lines:

- `batch/collate.py` ← `data/collate.py`
- `batch/group_split.py` ← `data/group_split.py`
- `batch/sampler.py` ← `data/sampler.py`
- `batch/datamodule.py` ← `data/datamodule.py`
- `batch/pad.py` ← `data/tasks/pad.py`
- `batch/task.py` — **new module, moved concept**: the `BatchTask` base
  (`task_id` property, `execute`, `__call__`), lifted out of
  `data/task.py`. It no longer inherits the ETL `Task`: a post-collate
  transform is not a pipeline transform, and the ETL hierarchy is leaving the
  repo. Same name, same three members, same semantics.

`batch/__init__.py` exports, alphabetized: `BatchSamplerSource`, `BatchTask`,
`DEFAULT_TARGET_SCHEMA`, `DataModule`, `DataModuleProtocol`, `MoleculeCollator`,
`PadMolecularBatch`, `TargetSchema`, `TaskMixtureBatchSampler`,
`TokenBudgetBatchSampler`, `TokenBudgetPacker`, `collate_molecules`,
`group_split_indices`. Surface parity with today's `molix.data` — nothing new
is promoted (`collate_packed`, `INDEX_KEYS`, `rebase` stay module-level, as now).

**Dependency direction:** `molix.batch` → `molix.store` only
(`sampler`/`datamodule` import `BaseDataset`; `collate` imports `PackedView`
under `TYPE_CHECKING`). `molix.store` never imports `molix.batch`.

**`batch_nodes` → `batch_tasks` (the coupling cut).** `DataModule`,
`_CollateFn` and `_PackedCollateFn` currently hold `Sequence[Node]` and call
`entry.apply(batch)`. `Node` belongs to the pipeline DSL that leaves for
molhub, and molnex may never import molhub. They now hold
`Sequence[BatchTask]` and call `task.execute(batch)` — the same dispatch
`Node.apply` performed for a `Runnable`, minus the DAG wrapper. Callers pass
transform instances directly: `DataModule(..., batch_tasks=[PadMolecularBatch(
max_atoms=..., max_edges=...)])`.

Correspondingly, `PipelineSpec.batch_nodes` and the `BatchTask` branch of
`Pipeline._validate` are **removed**, and `BatchTask` is deleted from
`data/task.py`. This is not cosmetic: if the pipeline kept routing by
`isinstance(node.task, BatchTask)` while `PadMolecularBatch` subclassed the
`molix.batch` base, a padded pipeline would match neither `is_cacheable` nor
`batch_nodes` and the padding task would be **silently dropped** — a wrong-shape
training run with no error. Post-collate configuration belongs to the
DataModule; the pipeline stops pretending to own it.

`DataLoaderProfiler(pipeline=PipelineSpec)` used the spec for exactly one thing
(`self.pipeline.batch_nodes`), so it takes `batch_tasks: Sequence[BatchTask] =
()` on the same contract; `_ProfilerCollate` follows.

**e2e test under `tests/` (found rot, fixed here).**
`tests/test_molix/test_data/test_e2e_workers.py` drives
source → Pipeline → cache → DataModule → worker iteration, which CLAUDE.md
forbids under `tests/`. Its load-bearing assertion (the collate callable must
survive `pickle` for `spawn`/`forkserver` workers) is a single-method test and
is preserved as `TestDataModule::test_collate_fn_is_picklable` in
`tests/test_molix/test_batch/test_datamodule.py`, with the cache built directly
by `PackedCache.save(...)` instead of a Pipeline. The worker-iteration half is
likewise rebuilt on `PackedCache` as a single-method test of
`DataModule.train_dataloader`. The two `Pipeline`-based tests inside
`test_datamodule.py` (lines 403, 431) get the same treatment.

**No shims.** `molix/data/__init__.py` loses these exports; no aliases.
Rewrite map:

| old | new |
|---|---|
| `molix.data.collate` | `molix.batch.collate` |
| `molix.data.datamodule` | `molix.batch.datamodule` |
| `molix.data.sampler` | `molix.batch.sampler` |
| `molix.data.group_split` | `molix.batch.group_split` |
| `molix.data.tasks.pad` / `molix.data.tasks.PadMolecularBatch` | `molix.batch.pad` / `molix.batch.PadMolecularBatch` |
| `molix.data.task.BatchTask` | `molix.batch.task.BatchTask` |
| `DataModule(..., batch_nodes=pipe.batch_nodes)` | `DataModule(..., batch_tasks=[...])` |

### Reuse decision

No `librarian_report` was supplied; candidates resolved from a direct sweep of
all `molix.data` importers.

- `collate_molecules` / `MoleculeCollator` / `TargetSchema` / `collate_packed` —
  **reuse**, moved verbatim. The batch schema is a hard contract; nothing is rewritten.
- `TokenBudgetPacker` / `TokenBudgetBatchSampler` / `TaskMixtureBatchSampler` /
  `BatchSamplerSource` — **reuse**, moved verbatim.
- `DataModule` / `DataModuleProtocol` — **generalize**: the type stays, its
  post-collate parameter is retyped off the ETL `Node` onto `BatchTask` so one
  implementation serves both the in-repo caller and molhub-built pipelines.
- `BatchTask` — **generalize**: promoted out of the ETL `Task` tree into
  `molix.batch.task` so it serves the DataModule and the profiler without the
  pipeline DSL.
- `PadMolecularBatch` — **reuse**, moved verbatim (only its base-class import changes).
- `Node` — **new — not adopted**: deliberately *not* reproduced in molnex; the
  DAG wrapper is ETL-only and its one training-side use is replaced by direct
  `BatchTask` dispatch.
- Closest pattern: `molix.store` from phase 01 — same package shape,
  alphabetized `__all__`, one-sentence responsibility docstring.

## Files to create or modify

- `src/molix/batch/__init__.py` (new)
- `src/molix/batch/collate.py` (new — content of `src/molix/data/collate.py`)
- `src/molix/batch/group_split.py` (new — content of `src/molix/data/group_split.py`)
- `src/molix/batch/sampler.py` (new — content of `src/molix/data/sampler.py`)
- `src/molix/batch/datamodule.py` (new — content of `src/molix/data/datamodule.py`)
- `src/molix/batch/task.py` (new — `BatchTask`, lifted from `src/molix/data/task.py`)
- `src/molix/batch/pad.py` (new — content of `src/molix/data/tasks/pad.py`)
- `src/molix/data/collate.py` (delete)
- `src/molix/data/group_split.py` (delete)
- `src/molix/data/sampler.py` (delete)
- `src/molix/data/datamodule.py` (delete)
- `src/molix/data/tasks/pad.py` (delete)
- `src/molix/data/__init__.py`
- `src/molix/data/task.py`
- `src/molix/data/tasks/__init__.py`
- `src/molix/data/pipeline.py`
- `src/molix/core/trainer.py`
- `src/molix/compile.py`
- `src/molix/config.py`
- `src/molix/schema.py`
- `src/molix/core/losses/molecular.py`
- `src/molix/profiler/dataloader.py`
- `src/molix/profiler/dataset.py`
- `src/molix/md/neighbors.py`
- `src/molix/datasets/qm9.py`
- `src/molix/datasets/revmd17.py`
- `src/molix/datasets/threebpa.py`
- `src/molix/datasets/water_les.py`
- `src/molix/datasets/molrec.py`
- `src/molpot/derivation/protocol.py`
- `tests/test_molix/test_batch/__init__.py` (new)
- `tests/test_molix/test_batch/test_collate.py` (new — from `tests/test_molix/test_data/test_collate.py`)
- `tests/test_molix/test_batch/test_collate_packed.py` (new — from `tests/test_molix/test_data/test_collate_packed.py`)
- `tests/test_molix/test_batch/test_bond_collate.py` (new — from `tests/test_molix/test_data/test_bond_collate.py`)
- `tests/test_molix/test_batch/test_valence_collate.py` (new — from `tests/test_molix/test_data/test_valence_collate.py`)
- `tests/test_molix/test_batch/test_datamodule.py` (new — from `tests/test_molix/test_data/test_datamodule.py` + the picklability/worker tests of `test_e2e_workers.py`)
- `tests/test_molix/test_batch/test_sampler.py` (new — from `tests/test_molix/test_data/test_sampler.py`)
- `tests/test_molix/test_batch/test_group_split.py` (new — from `tests/test_molix/test_data/test_group_split.py`)
- `tests/test_molix/test_batch/test_dtype_propagation.py` (new — from `tests/test_molix/test_data/test_dtype_propagation.py`)
- `tests/test_molix/test_batch/test_pad.py` (new — from `tests/test_molix/test_data/test_tasks/test_pad.py`)
- `tests/test_molix/test_batch/test_task.py` (new)
- `tests/test_molix/test_data/test_e2e_workers.py` (delete)
- `tests/test_molix/test_data/test_tasks/` (delete)
- `tests/test_molix/test_data/test_pipeline.py`
- `tests/test_molix/test_core/test_losses_molecular.py`
- `tests/test_molix/test_profiler/test_dataloader_profiler.py`
- `regressions/universal-class-i-09-packer.py`
- `regressions/universal-class-i-10-sampler.py`
- `regressions/learnable-classical-ff-02-valence-topology.py`
- `regressions/molix-data-dissolution-02-batch.py` (new)
- `scripts/mm_param_learning/espaloma_mm_protocol_train.py`
- `scripts/check_test_mirror.py`
- `benchmarks/train_pinet_ef_padded.py`
- `benchmarks/train_pinet_ef_compare.py`
- `benchmarks/bench_pinet_ef_full.py`
- `docs/molix/user-guide/data-modules.md`
- `docs/molix/user-guide/data-loading.md`
- `docs/molix/user-guide/profiling.md`
- `docs/molix/tutorials/train-a-graph-model.md`
- `docs/api/molix.md`

## Tasks

- [ ] Write failing batch-package tests: move `tests/test_molix/test_data/{test_collate,test_collate_packed,test_bond_collate,test_valence_collate,test_datamodule,test_sampler,test_group_split,test_dtype_propagation}.py` and `test_tasks/test_pad.py` into `tests/test_molix/test_batch/` on `molix.batch.*` imports, and add `test_task.py` covering `BatchTask.execute` dispatch plus `DataModule(batch_tasks=...)` (RED)
- [ ] Move `collate.py`, `group_split.py`, `sampler.py`, `datamodule.py` into `src/molix/batch/` and add `src/molix/batch/__init__.py` with alphabetized `__all__`, repointing internal imports at `molix.store`
- [ ] Implement `BatchTask` in `src/molix/batch/task.py` and move `tasks/pad.py` to `src/molix/batch/pad.py` on that base
- [ ] Replace `batch_nodes: Sequence[Node]` with `batch_tasks: Sequence[BatchTask]` in `DataModule`, `_CollateFn` and `_PackedCollateFn`, dispatching via `task.execute`
- [ ] Remove `BatchTask` from `src/molix/data/task.py` and delete `PipelineSpec.batch_nodes` plus the `BatchTask` branch of `Pipeline._validate`, with their now-invalid tests in `tests/test_molix/test_data/test_pipeline.py`
- [ ] Replace `DataLoaderProfiler(pipeline=...)` with `batch_tasks=...` and repoint the profiler collate imports at `molix.batch`
- [ ] Rebuild the two `Pipeline`-based `DataModule` tests and the worker/pickle tests of `test_e2e_workers.py` on `PackedCache.save`, then delete `tests/test_molix/test_data/test_e2e_workers.py`
- [ ] Rewrite the remaining `molix.data.{collate,datamodule,sampler,group_split,tasks.pad}` references across `src/`, `tests/`, `regressions/`, `scripts/`, `benchmarks/` and `docs/`, including `PINET_SPINE` in `scripts/check_test_mirror.py`
- [ ] Add regression example `regressions/molix-data-dissolution-02-batch.py` (public API only; hard-coded goldens, no third-party runtime)
- [ ] Run full check + test suite

## Testing strategy

Unit tests only under `tests/`, mirroring `src/molix/batch/`; the moved tests
are the safety net and keep their assertions verbatim.

- `test_collate.py` → `collate_molecules`, `MoleculeCollator`, `INDEX_KEYS`,
  `rebase`: namespace layout, per-namespace `batch_size`, atom-offset rebase of
  `edge_index` / `bond_index`, target routing by `TargetSchema`.
- `test_collate_packed.py` → `collate_packed` and the DataModule fast path:
  leaf-equality against `collate_molecules`, fast-path routing, `batch_tasks`
  application on the packed path, dtype cast.
- `test_datamodule.py` → one method per test: `train_dataloader`,
  `val_dataloader`, `on_epoch_start` reseeding, `_resolve_collation`,
  `_make_collate_fn` picklability (the preserved `spawn`-worker regression).
- `test_sampler.py` / `test_group_split.py` → packer budget arithmetic, mixture
  proportions, deterministic group splits.
- `test_pad.py` → ghost atoms contribute exactly zero; padding edges are
  self-loops on the first ghost atom; overflow raises.
- `test_task.py` (new) → `BatchTask.task_id` default, `execute` raising on the
  base, `__call__` delegating, and `DataModule` applying `batch_tasks` in order.
- Edge cases: empty `batch_tasks`, a batch with zero edges, single-graph batch,
  overflow past `max_atoms`.

Regression example: `regressions/molix-data-dissolution-02-batch.py` — public
API only (`from molix.batch import DataModule, PadMolecularBatch,
TargetSchema, collate_molecules`; `from molix.store import PackedCache,
MmapDataset`). Two hard-coded molecules (3 and 4 atoms) are collated; the file
pins literal expected values for `atoms.Z`, the rebased `edges.edge_index`
second-molecule offset, `graphs.num_atoms == [3, 4]`, and — after
`PadMolecularBatch(max_atoms=8, max_edges=8)` via `batch_tasks` — the padded
shapes, `atoms.mask` and the unchanged real-atom rows. It also asserts
`PipelineSpec` has no `batch_nodes` attribute. No third-party software runs.

## Out of scope

- The NeighborList task, valence-column helpers and extxyz codec — phase 03.
- Deleting `src/molix/data/` / `src/molix/datasets/` — phase 04.
- Any change to the collated batch schema itself (a CLAUDE.md hard contract).
- Refreshing `.claude/notes/architecture.md` — `/mol:map` after phase 04.
