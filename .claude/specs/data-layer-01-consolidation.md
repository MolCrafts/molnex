---
title: One data layer — molix.data; delete the unused on-disk format
status: in-progress
grilled: false
created: 2026-09-17
chain: data-layer
---

# One data layer — `molix.data`; delete the unused on-disk format

## Summary

`molix.data`, `molix.data` and `molix.data` are three packages for one
job. They are merged into **`molix.data`**. The bespoke `molix.block/1` on-disk
format — `FrameStoreWriter` / `FrameStoreReader` / `FrameStoreView` /
`FrameStorePlan` and the block-aware collate route threaded through the live
`DataModule` — is **deleted**: it has no producer, no consumer and no instance
on disk. `TensorFrame` and the neighbour list leave the data layer for
`molix/frame.py` and `molix/neighbor.py`, because one is what the *model* reads
and the other is a geometry component, not loading machinery.

## Evidence that the format is dead (2026-09-17)

- `find <workspace> -name commit.log` → **0**. No store has ever been written.
- `FrameStoreWriter(` outside tests → one call site,
  `scripts/store/bench_block_read.py`, a benchmark.
- `FrameStoreReader(` / `FrameStoreDataset(` outside tests → only inside the
  store modules themselves and that same benchmark.
- The Class-I production path reads **LMDB**: measured call stack
  `molhub/io/lmdb.py:94 get` → `omol25.py:_frame_from_record` (cProfile, job
  2577512).
- molhub's imports still name the **pre-rename** path (`molix.data.block`, 20
  occurrences): the downstream has been broken since 2026-09-08 and nobody
  noticed, because nobody uses it.
- `DataModule._make_dataloader` probes `_block_capable(dataset)` on every
  loader build for a route that cannot fire.

CLAUDE.md claims "`molix.block/1` is molnex's **one** on-disk format … MolHub
writes these; molnex only reads them". Neither half is true. That text is
corrected as part of this slug.

## Design

```
molix/data/
  dataset.py       BaseDataset (+ .nrows), SubsetDataset      <- store/
  frame_dataset.py FrameDataset                               <- store/
  collator.py      FrameCollator                              <- dataloader/
  datamodule.py    DataModule (single collate route)          <- dataloader/
  collate.py  pad.py  sampler.py  mixture.py  group_split.py
  residual.py  task.py  charge.py  valence_columns.py
  bond_adapter.py                                             <- batch/
molix/frame.py     TensorFrame                                <- batch/frame.py
molix/neighbor.py  cutoff graph over (B, N, 3)                <- batch/neighbor.py
```

`molix.data`, `molix.data` and `molix.data` cease to exist.

### `nrows` becomes polymorphism, not a dispatcher

`molix/data/frame_nrows.py` is a 122-line five-way `isinstance` dispatcher over
source kinds, two of which are the deleted ones. The quantity it returns is
already a molpy member (`block.nrows`, used by molpy's own `selector.py:155`).
What the three live callers (`sampler.py:270`, `datamodule.py:427`,
`profiler/dataset.py:756`) actually want is a per-index vector for *a dataset*,
so it becomes `BaseDataset.nrows(table)` with `SubsetDataset` and `DataMixture`
overriding. The module and the free function go.

### `FrameCollection` is deleted

It is `FrameCollection = Sequence[Frame]`, a type alias exported in `__all__` as
if it were a type. Call sites write `Sequence[Frame]`.

## Out of scope

- Renaming `molrs::store` (`store::block::{Block, DType}`, `store::frame::Frame`,
  `store::schema`, `store::keys` and the PyO3 bindings under
  `molrs-python/src/core/store/`). That namespace is live and load-bearing —
  molpy's entire core data model — so there is nothing dead to delete there; the
  word is a separate rename across four crates and an FFI surface.
- `src/molpy/io/store/` — an empty leftover directory, deleted here because it
  costs nothing, but molpy is otherwise untouched.
- The neighbour-list redesign itself (dense `(B, N, 3)`, no 10 Å truncation).
  This slug only *moves* `neighbor.py`; the redesign is its own slug.

## Passes (mechanical rename discipline)

1. Delete the dead format + surgery on the live files that reference it.
2. Create `molix/data/`, move the live modules.
3. Rewrite imports and package names.
4. Prose: CLAUDE.md, `docs/`, `.claude/notes/`, and molhub's stale imports.

Full unit tier between passes. Baseline before pass 1: 1515 passed, 8 skipped,
0 failed (one pre-existing failure, a stale `frame.meta[...].value` assertion in
`test_mixture.py`, fixed first).
