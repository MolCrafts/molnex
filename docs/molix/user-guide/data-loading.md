# Data Loading

MolNex data flows from plain dict samples to nested TensorDict batches:

1. `dataset[i]` returns a single-sample dict (`Z`, `pos`, `targets`, ...)
2. `DataLoader(collate_fn=collate_molecules)` merges samples into one nested `TensorDict`
3. `Trainer` passes that batch to the model and loss function

Step 1 is where the boundary sits: producing those sample dicts — downloading a
dataset, parsing it, preprocessing it and writing the packed cache molnex reads
— is **dataset acquisition**, which lives in
[MolHub](https://github.com/molcrafts/molhub). molnex owns steps 2 and 3, plus
the cache reader in `molix.store`.

## Reading Prepared Blocks

MolHub's `prepare(name, split=..., output_dir=...)` is the producing half of
that boundary: one call writes `output_dir` as a directory of sealed
`molix.block/1` block roots. molnex reads that directory back:

```python
from molix.store import BlockDataset

ds = BlockDataset.load_from_disk("blocks/")
```

`load_from_disk` does the discovery: the roots are the immediate children of the
directory carrying a `manifest.json`, **sorted by child name**, so one prepared
directory always yields the same frame order; children without a manifest (a
`README.md`, a scratch directory) are ignored, and a directory carrying
`manifest.json` itself is taken as the single root of a one-block view. Nothing
already discovered is ever skipped, so every construction refusal still applies
— `BlockNotReadyError` (a root unreadable or still unsealed),
`BlockSplitMismatchError` (roots from more than one split),
`BlockSchemaMismatchError` (roots that are not concatenable) and a plain
`ValueError` when no root is found at either level.

One directory is one split: `prepare` exports a single split per call and blocks
are split-pure (`ds.split` is that one manifest split), so build one
`BlockDataset` per split rather than pointing one view at both. `ds[i]` returns
the same flat sample dict `MmapDataset[i]` does, so it feeds the collation below
unchanged. The producing side is documented in MolHub's
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

Two collate paths exist and are required to agree leaf for leaf.
`collate_molecules` walks a list of per-sample dicts; `collate_packed` builds
the same batch directly out of the packed `PackedCache` tensors by gathering
rows, skipping the unpack-then-repack round trip. `collate_molecules` is the
equivalence oracle for `collate_packed`.

## Preprocessing Tasks

Preprocessing runs *before* collation, on flat sample dicts, and is scheduled
by MolHub. molnex keeps exactly one such step, because the edge convention it
writes is a molnex contract:

- **`molix.batch.NeighborList`** sees one molecule at a time and adds
  `edge_index`, `edge_diff` and `edge_dist` for every pair within `cutoff`.

Dataset-level steps (an atomic-dress baseline fitted over the training split,
unit conversion, constant labels) are MolHub's. molnex reads back whatever fit
they recorded through `MmapDataset.stats()`.

For the full batch structure, see [Batch Schema](../explanation/batch-schema.md).
