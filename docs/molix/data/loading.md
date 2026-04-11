# Data Loading

MolNex data flows from plain dict samples to nested TensorDict batches:

1. `DataSource.__getitem__` returns a single-sample dict (`Z`, `pos`, `targets`, ...)
2. `DataLoader(collate_fn=collate_molecules)` merges samples into a `GraphBatch` (nested TensorDict)
3. `Trainer` passes the `GraphBatch` to the model and loss function

## Collation

```python
from torch.utils.data import DataLoader
from molix.data.collate import collate_molecules

loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_molecules)
```

`collate_molecules` produces a nested `GraphBatch`:

- Atom-level fields (`Z`, `pos`) → `AtomData` (batch_size=[N_total])
- Edge fields (`edge_index`, `bond_diff`, `bond_dist`) → `EdgeData` (batch_size=[E_total])
- Graph-level metadata + targets → `GraphData` (batch_size=[B])

## Preprocessing Tasks

- **NeighborList**: Compute neighbor edges in the `prepare()` stage
- **AtomicDress**: Remove atomic baselines from graph-level scalar targets
