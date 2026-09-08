# Data Loading

MolNex speaks two data shapes and nothing else: a molpy `Frame` at rest, and a
nested `TensorDict` in the model.

1. a molpy `Frame` is the unit of data at rest — on disk in a `molix.block/1`
   store, or built in memory
2. `DataLoader(collate_fn=collate_molecules)` turns frames into one nested
   `TensorDict`; `molix.batch.TensorFrame` is its typed view
3. `Trainer` passes that batch to the model and loss function

Step 1 is where the boundary sits: producing those frames — downloading a
dataset, parsing it, preprocessing it and writing the stores molnex reads — is
**dataset acquisition**, which lives in
[MolHub](https://github.com/molcrafts/molhub). molnex owns steps 2 and 3, plus
the store reader in `molix.store`.

## Reading Prepared Blocks

MolHub's `prepare(name, split=..., output_dir=...)` is the producing half of
that boundary: one call writes `output_dir` as a directory of sealed
`molix.block/1` block roots. molnex reads that directory back:

```python
from molix.store import FrameStoreDataset

ds = FrameStoreDataset.load_from_disk("blocks/")
```

`load_from_disk` does the discovery: the roots are the immediate children of the
directory carrying a `manifest.json`, **sorted by child name**, so one prepared
directory always yields the same frame order; children without a manifest (a
`README.md`, a scratch directory) are ignored, and a directory carrying
`manifest.json` itself is taken as the single root of a one-block view. Nothing
already discovered is ever skipped, so every construction refusal still applies
— `FrameStoreNotReadyError` (a root unreadable or still unsealed),
`FrameStoreSplitMismatchError` (roots from more than one split),
`FrameStoreSchemaMismatchError` (roots that are not concatenable) and a plain
`ValueError` when no root is found at either level.

One directory is one split: `prepare` exports a single split per call and blocks
are split-pure (`ds.split` is that one manifest split), so build one
`FrameStoreDataset` per split rather than pointing one view at both. `ds[i]`
returns frame `i` as a one-frame batch `TensorDict` — a store yields the
model's structure directly, with no per-sample object in between — and
`ds.batch(frames)` composes several frames in one read. The producing side is
documented in MolHub's
[`docs/reference/python.md`](https://github.com/molcrafts/molhub/blob/main/docs/reference/python.md).

## Collation

```python
from torch.utils.data import DataLoader
from molix.batch.collate import collate_molecules

loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_molecules)
```

`collate_molecules` produces a plain nested `TensorDict` whose namespaces each
carry their own batch size:

- Atom-level fields (`Z`, `pos`, `batch`) → `atoms` (batch_size=[N_total])
- Edge fields (`edge_index`, `edge_diff`, `edge_dist`) → `edges` (batch_size=[E_total])
- Graph-level metadata (`num_atoms`) + graph targets → `graphs` (batch_size=[B])
- Covalent topology (`bond_index`, `bond_types`), when present → `bonds` (batch_size=[])

Read it with tuple keys: `batch["atoms", "Z"]`, `batch["edges", "edge_index"]`,
`batch["graphs", "energy"]`.

There is one collate path. `collate_molecules` takes molpy `Frame` objects
and returns the batch; reading a store goes straight from its mmapped columns
to that batch with one gather per column. The `collate_packed` fast path that
used to sit beside it was deleted on 2026-09-07 — it re-gathered every tensor
through an identity index, which was a full copy of the batch for nothing.

## Preprocessing Tasks

Preprocessing that belongs to dataset acquisition is scheduled by MolHub.
molnex keeps exactly one step of its own, because the edge convention it writes
is a molnex contract:

- **`molix.batch.NeighborList`** is a `BatchTask`: `execute(batch)` writes
  `edge_index`, `edge_diff` and `edge_dist` for every pair within `cutoff`,
  rebased onto the batch. Callers holding raw tensors rather than a batch use
  `pairs(pos, box_vectors)` instead.

Dataset-level steps (an atomic-dress baseline fitted over the training split,
unit conversion, constant labels) are MolHub's. molnex reads back whatever fit
the producer recorded in the block manifest.

For the full batch structure, see [Batch Schema](../explanation/batch-schema.md).
