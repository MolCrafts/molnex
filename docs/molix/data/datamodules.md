# DataModules

`DataModule` integrates the full data pipeline: source → pipeline → collation → DataLoader.

The minimal protocol requires:

- `setup(stage)` - Prepare datasets
- `train_dataloader()` - Returns iterable of `GraphBatch`
- `val_dataloader()` - Returns iterable of `GraphBatch`

Each batch element is a `GraphBatch` (nested TensorDict, see [Batch Schema](../../tensordict_schema.md)).

## Using the Built-in DataModule

```python
from molix.data import DataModule, Pipeline, NeighborList, AtomicDress
from molix.datasets import QM9Source

source = QM9Source(root="./data/qm9", total=1000)

pipe = (
    Pipeline("qm9")
    .add(NeighborList(cutoff=5.0))
    .add(AtomicDress(target_key="U0"))
    .build()
)

dm = DataModule(source=source, pipeline=pipe, batch_size=32)
dm.setup("fit")

for batch in dm.train_dataloader():
    # batch is a GraphBatch
    Z = batch["atoms", "Z"]           # (N_total,)
    pos = batch["atoms", "pos"]       # (N_total, 3)
    energy = batch["graphs", "U0"]    # (B,)
    break
```

## Minimal Custom DataModule

```python
from torch.utils.data import DataLoader
from molix.data.collate import collate_molecules

class MyDataModule:
    def __init__(self, train_set, val_set, batch_size=32):
        self.train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_molecules,
        )
        self.val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_molecules,
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
```
