# Data Loading

MolNex data flows from plain dict samples to nested TensorDict batches:

1. `DataSource.__getitem__` returns a single-sample dict (`Z`, `pos`, `targets`, ...)
2. `DataLoader(collate_fn=collate_molecules)` merges samples into one nested `TensorDict`
3. `Trainer` passes that batch to the model and loss function

## Collation

```python
from torch.utils.data import DataLoader
from molix.data.collate import collate_molecules

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

Tasks run *before* collation, on flat sample dicts, and are composed into a
`Pipeline`. They come in two flavours:

- **`NeighborList`** is a `SampleTask`: it sees one molecule at a time and adds
  `edge_index`, `edge_diff` and `edge_dist` for every pair within `cutoff`.
- **`AtomicDress`** is a `DatasetTask`: it needs the whole training set, so it
  runs in two phases — `fit` solves a least-squares problem for a per-element
  baseline energy, then `execute` subtracts that baseline from each sample's
  scalar target.

For the full batch structure, see [Batch Schema](../explanation/batch-schema.md).
