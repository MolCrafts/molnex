# molix.data

Molecular data pipeline with nested TensorDict batch types.

- `types`: TensorDict subclasses — `AtomData`, `EdgeData`, `GraphData`, `GraphBatch`
- `source`: Data source abstraction (`DataSource` protocol)
- `pipeline`: Task-based preprocessing pipeline (`SampleTask`, `DatasetTask`, `BatchTask`)
- `tasks/`: Built-in tasks (`NeighborList`, `AtomicDress`)
- `collate`: `collate_molecules` converts sample dicts → nested `GraphBatch`
- `dataset`: `CachedDataset` wraps pre-computed sample lists
- `datamodule`: DDP-aware `DataModule` integrating pipeline + collation + DataLoader
