---
title: Dissolve molix.data — packed cache and dataset readers into molix.store
status: approved
slug: molix-data-dissolution-01-store
created: 2026-08-14
---

# Dissolve molix.data — packed cache and dataset readers into molix.store

## Summary

`molix.data` mixes two unrelated jobs: training-side storage/batching and
ETL/data acquisition. The operator's boundary is binding — 跟训练有关的留在
molix；跟数据有关的给 molhub — and molix must end up with no `data/` package at
all. This first phase moves the storage half into its fixed new home
`src/molix/store/`: `PackedCache` (run-scoped charter unchanged) and the reader
family (`BaseDataset`, `MmapDataset`, `CachedDataset`, `SubsetDataset`,
`PackedView`). It also deletes `PesBundle` outright — an orphan container with
zero production consumers — and rehomes the one contract sentence that hung off
it. No behaviour changes: the moved unit tests are the safety net and must pass
unmodified apart from their import lines and their new mirror paths.

## Design

No physics is introduced or changed; `$META.science.required` does not apply to
this spec. The `.pt` payload layout, cache keys, readiness polling, mmap
semantics and every `@cached_property` statistic are byte-for-byte unchanged.

**New package `molix.store`** — one responsibility: *bytes on disk ↔ tensors in
memory*. Two modules, moved verbatim:

- `src/molix/store/cache.py` ← `src/molix/data/cache.py` (`PackedCache` v1;
  single-file packed buckets; the "do not migrate to `TensorDict.memmap_()`"
  inode argument stays in the module docstring).
- `src/molix/store/dataset.py` ← `src/molix/data/dataset.py` (readers over a
  `PackedCache` file).

`store/__init__.py` re-exports `BaseDataset`, `CachedDataset`, `MmapDataset`,
`PackedCache`, `PackedView`, `SubsetDataset` with `__all__` alphabetized
(notes.md 2026-08-09). `PackedView` is exported because `collate_packed` takes
one as a parameter; nothing else is added to the public surface.

**Layering, one-way.** `molix.store` imports nothing from `molix.data`,
`molix.batch` or `molix.datasets`. The only reverse edge is
`collate.py`'s `TYPE_CHECKING` import of `PackedView`, which is a type-only
edge from the batch side down to the store side and therefore fine. The
upcoming `streaming-data-tiers-02` block store also lands in `molix.store`;
this spec does not design it.

**No shims.** `molix/data/__init__.py` loses the cache/dataset/PesBundle
exports; there is no alias, no re-export, no deprecation path (stage:
experimental). Every importer is rewritten in the same commit. Rewrite map:

| old | new |
|---|---|
| `molix.data.cache` | `molix.store.cache` |
| `molix.data.dataset` | `molix.store.dataset` |
| `from molix.data import PackedCache \| BaseDataset \| CachedDataset \| MmapDataset \| SubsetDataset` | `from molix.store import …` |
| `molix.data.pes_bundle` / `PesBundle` | *(deleted)* |

**PesBundle deletion.** `src/molix/data/pes_bundle.py`, its unit test and
`regressions/universal-class-i-04-bundle.py` go. Verified: no `src/` consumer
outside its own `__init__` export and one docstring. The single load-bearing
sentence it owned — the flat Cartesian Hessian layout `i = 3a + c`
(`x0, y0, z0, x1, …`) for `graphs.hessian` `(3N, 3N)` — moves into
`ProjectedInternalHessianLoss` (`src/molix/core/losses/molecular.py:271`),
which is the actual consumer; `src/molpot/derivation/hessian.py:19` is
repointed at that class instead of the deleted one.

**Shared test fixture.** `tests/test_molix/test_data/conftest.py`'s
`equal_count_samples` is consumed by tests that now split across
`test_store/` (cache, dataset) and `test_batch/` (collate_packed, phase 02).
It moves up to the nearest common ancestor, `tests/test_molix/conftest.py`,
imported by package path per CLAUDE.md.

**Rot found while surveying (named, not silently left).**
`src/molix/data/README.md:31` documents `DataModule.from_cached_pipeline`,
which does not exist in the code. The README is deleted wholesale in phase 04;
this spec does not patch a file scheduled for deletion, but records the finding
here so it is not mistaken for a lost feature.

### Reuse decision

No `librarian_report` was supplied with this task, so reuse candidates were
resolved from a direct inventory sweep of every `molix.data` importer (88 files).

- `PackedCache` — **reuse**, moved verbatim. No reimplementation, no new cache type.
- `BaseDataset` / `MmapDataset` / `CachedDataset` / `SubsetDataset` / `PackedView`
  — **reuse**, moved verbatim.
- `PesBundle` — **delete** (operator verdict 删除); not generalized, not replaced.
- `group_split_indices` — untouched here; it is batch-side and moves in phase 02.
- Closest pattern followed: `molix.io` (an existing single-responsibility leaf
  package with an alphabetized `__all__` and a package docstring naming its one
  job) — `molix.store` is written to read the same way.

## Files to create or modify

- `src/molix/store/__init__.py` (new)
- `src/molix/store/cache.py` (new — content of `src/molix/data/cache.py`)
- `src/molix/store/dataset.py` (new — content of `src/molix/data/dataset.py`)
- `src/molix/data/cache.py` (delete)
- `src/molix/data/dataset.py` (delete)
- `src/molix/data/pes_bundle.py` (delete)
- `src/molix/data/__init__.py`
- `src/molix/data/pipeline.py`
- `src/molix/data/datamodule.py`
- `src/molix/data/sampler.py`
- `src/molix/data/collate.py`
- `src/molix/profiler/dataloader.py`
- `src/molix/profiler/dataset.py`
- `src/molix/config.py`
- `src/molix/io/writer.py`
- `src/molix/datasets/qm9.py`
- `src/molix/datasets/molrec.py`
- `src/molix/core/losses/molecular.py`
- `src/molpot/derivation/hessian.py`
- `tests/test_molix/conftest.py` (new)
- `tests/test_molix/test_store/__init__.py` (new)
- `tests/test_molix/test_store/test_cache.py` (new — from `tests/test_molix/test_data/test_cache.py`)
- `tests/test_molix/test_store/test_cache_valence.py` (new — from `tests/test_molix/test_data/test_cache_valence.py`)
- `tests/test_molix/test_store/test_dataset.py` (new — from `tests/test_molix/test_data/test_dataset.py`)
- `tests/test_molix/test_data/conftest.py` (delete)
- `tests/test_molix/test_data/test_cache.py` (delete)
- `tests/test_molix/test_data/test_cache_valence.py` (delete)
- `tests/test_molix/test_data/test_dataset.py` (delete)
- `tests/test_molix/test_data/test_pes_bundle.py` (delete)
- `tests/test_molix/test_data/test_collate_packed.py`
- `tests/test_molix/test_data/test_datamodule.py`
- `tests/test_molix/test_data/test_sampler.py`
- `tests/test_molix/test_data/test_bond_collate.py`
- `tests/test_molix/test_data/test_valence_collate.py`
- `tests/test_molix/test_data/test_dtype_propagation.py`
- `tests/test_molix/test_data/test_pipeline.py`
- `tests/test_molix/test_profiler/test_dataset.py`
- `regressions/universal-class-i-04-bundle.py` (delete)
- `regressions/dataset-profiler-salvage.py`
- `regressions/universal-class-i-09-packer.py`
- `regressions/universal-class-i-10-sampler.py`
- `regressions/learnable-classical-ff-02-valence-topology.py`
- `regressions/molix-data-dissolution-01-store.py` (new)
- `scripts/mm_param_learning/espaloma_mm_protocol_train.py`
- `docs/molix/user-guide/profiling.md`
- `docs/api/molix.md`

## Tasks

- [ ] Write failing store tests by moving `tests/test_molix/test_data/{test_cache,test_cache_valence,test_dataset}.py` to `tests/test_molix/test_store/` with `molix.store.*` imports (RED: package absent)
- [ ] Move `cache.py` and `dataset.py` into `src/molix/store/` and add `src/molix/store/__init__.py` with alphabetized `__all__`
- [ ] Rewrite every `molix.data.cache` / `molix.data.dataset` importer under `src/` (`data/{__init__,pipeline,datamodule,sampler,collate}.py`, `profiler/{dataloader,dataset}.py`, `config.py`, `io/writer.py`, `datasets/{qm9,molrec}.py`) and drop the store exports from `molix/data/__init__.py`
- [ ] Rewrite the store references in `tests/`, `regressions/`, `scripts/mm_param_learning/espaloma_mm_protocol_train.py`, `docs/molix/user-guide/profiling.md` and `docs/api/molix.md`
- [ ] Delete `src/molix/data/pes_bundle.py`, `tests/test_molix/test_data/test_pes_bundle.py` and `regressions/universal-class-i-04-bundle.py`
- [ ] Move the `(3N,3N)` `x0,y0,z0` Cartesian-layout contract sentence into `ProjectedInternalHessianLoss` and repoint `src/molpot/derivation/hessian.py:19` at it
- [ ] Move `equal_count_samples` to `tests/test_molix/conftest.py` and delete the `test_data` copy
- [ ] Add regression example `regressions/molix-data-dissolution-01-store.py` (public API only; hard-coded goldens, no third-party runtime)
- [ ] Run full check + test suite

## Testing strategy

Unit tests only under `tests/`, mirroring source; the **moved tests are the
safety net** — their assertions are unchanged, only import lines and file paths
move, so any behavioural drift shows up as a failure rather than an edit.

- `tests/test_molix/test_store/test_cache.py` → `TestPackedCache`: save/load
  round-trip, atomic write + readiness, bucket-identity under
  `n_edges == n_atoms` (`equal_count_samples`), schema inference, cache-key
  stability. Happy path + the degenerate-count edge case.
- `tests/test_molix/test_store/test_dataset.py` → `TestMmapDataset` /
  `TestCachedDataset` / `TestSubsetDataset`: single-method targets —
  `__getitem__`, `__len__`, `split`, `packed_view`, `stats`, and the
  `avg_num_neighbors` / `max_atoms` / `max_edges` cached properties.
- `tests/test_molix/test_store/test_cache_valence.py`: valence-namespace
  round-trip through the packed layout.
- Edge cases retained: empty edge set, single-sample cache, `SubsetDataset` over
  a non-packed dataset raising from `packed_view()`.
- No new domain validation: this spec adds no physics and changes no numbers.

Regression example: `regressions/molix-data-dissolution-01-store.py` — public
API only (`from molix.store import CachedDataset, MmapDataset, PackedCache`).
It writes three hard-coded 2-atom samples (goldens literal in the file, no
oracle of any kind), reads them back through both readers, asserts exact tensor
equality on `Z`, `pos`, `edge_index`, `edge_diff`, `edge_dist` and
`targets.U0`, asserts `avg_num_neighbors == 1.0`, and asserts that
`importlib.util.find_spec("molix.data.cache")` is `None` and that
`from molix.data import PesBundle` raises `ImportError`.

## Out of scope

- Moving collate / sampler / datamodule / group_split — phase 02.
- Moving the NeighborList task, valence-column helpers or the extxyz codec — phase 03.
- Deleting `src/molix/data/` or `src/molix/datasets/` — phase 04.
- Designing the `streaming-data-tiers-02` block store (it will also live in `molix.store`).
- Refreshing `.claude/notes/architecture.md` — named follow-up: run `/mol:map` after phase 04.
