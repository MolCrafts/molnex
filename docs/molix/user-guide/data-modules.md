# Data Modules

`DataModule` is the last stage of the data pipeline: it wraps two pre-built
datasets in DataLoaders that collate, shard across DDP ranks, and prefetch.
Everything upstream — downloading, per-sample transforms, caching, splitting —
happens before it.

The minimal protocol requires:

- `setup(stage)` - Prepare datasets/samplers for a stage (e.g. `"fit"`)
- `train_dataloader()` - Returns an iterable of collated training batches
- `val_dataloader()` - Returns an iterable of collated validation batches

`DataModuleProtocol` also declares `on_epoch_start(epoch)`, which the `Trainer`
calls when the module defines it (e.g. to reseed a sampler).

Each item yielded by those loaders is a nested `TensorDict` with `atoms` /
`edges` / `graphs` namespaces, see
[Batch Schema](../explanation/batch-schema.md).

## Using the Built-in DataModule

The wiring is source → `PipelineSpec.cache` → dataset → split → `DataModule`:

```python
from molix.data import (
    AtomicDress, DataModule, NeighborList, Pipeline, SubsetDataset, SubsetSource,
)
from molix.datasets import QM9Source

source = QM9Source(root="./data/qm9", total=1000)

train_idx = list(range(800))
val_idx = list(range(800, 1000))

# AtomicDress fits its per-element baseline with a least-squares solve over a
# whole dataset. Fit it on the training indices only, or the val split leaks
# into the baseline.
train_source = SubsetSource(source, train_idx)

pipe = (
    Pipeline("qm9")
    .add(NeighborList(cutoff=5.0))
    .add(AtomicDress(elements=(1, 6, 7, 8, 9), target_key="U0"))
    .build()
)

dag = pipe.cache(source, base_dir="./cache", fit_source=train_source)
full = dag.dataset(mmap=True)
train_ds = SubsetDataset(full, train_idx)
val_ds = SubsetDataset(full, val_idx)

dm = DataModule(
    train_ds,
    val_ds,
    target_schema=QM9Source.TARGET_SCHEMA,
    batch_nodes=pipe.batch_nodes,
    batch_size=32,
)
dm.setup("fit")

for batch in dm.train_dataloader():
    # batch is a nested TensorDict
    Z = batch["atoms", "Z"]           # (N_total,)
    pos = batch["atoms", "pos"]       # (N_total, 3)
    energy = batch["graphs", "U0"]    # (B,)
    break
```

`target_schema` decides which target names land under `graphs` and which under
`atoms`; `batch_nodes` carries any `BatchTask` nodes in the pipeline, which run
after collation and so cannot be cached with the rest.

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
