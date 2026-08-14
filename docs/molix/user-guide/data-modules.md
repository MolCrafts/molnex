# Data Modules

`DataModule` is the last stage of the data path: it wraps two pre-built
datasets in DataLoaders that collate, shard across DDP ranks, and prefetch.
Everything upstream — downloading, per-sample transforms, caching, splitting —
happens before it.

Dataset **acquisition** — downloading, parsing, per-sample preprocessing and
writing the packed cache — is not part of molnex; it lives in
[MolHub](https://github.com/molcrafts/molhub). molnex starts from the cache
file MolHub wrote, read through `molix.store`.

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

The molnex-side wiring is packed cache → dataset → split → `DataModule`. The
cache was produced upstream by MolHub, which owns the source, the per-sample
transforms (neighbor list, atomic dress) and the fit of any dataset-level
statistic — including the rule that such a fit sees training indices only, or
the val split leaks into the baseline:

```python
from molix.batch import DataModule, PadMolecularBatch, TargetSchema
from molix.store import MmapDataset, SubsetDataset

full = MmapDataset("./cache/qm9.pt")  # written by MolHub

train_idx = list(range(800))
val_idx = list(range(800, 1000))
train_ds = SubsetDataset(full, train_idx)
val_ds = SubsetDataset(full, val_idx)

dm = DataModule(
    train_ds,
    val_ds,
    target_schema=TargetSchema(graph_level=frozenset({"U0"}), atom_level=frozenset()),
    batch_tasks=[PadMolecularBatch(max_atoms=1024, max_edges=16384)],
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
`atoms`; `batch_tasks` carries `molix.batch.BatchTask` instances, applied in
order after collation — post-collate work is configured on the DataModule, not
upstream, because it cannot be cached with the rest.

## Minimal Custom DataModule

```python
from torch.utils.data import DataLoader
from molix.batch.collate import collate_molecules

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
