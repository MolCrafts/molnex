# Molix Data

`molix.data` provides the molecular data pipeline:

- **Types** (`types.py`): Nested TensorDict subclasses — `AtomData`, `EdgeData`, `GraphData`, `GraphBatch`
- **Sources**: `DataSource` protocol and `InMemorySource` / `SubsetSource` implementations
- **Pipeline**: Task-based preprocessing pipeline (sample-level, dataset-level, batch-level)
- **Tasks**: Built-in preprocessing tasks (`NeighborList`, `AtomicDress`)
- **Collation**: `collate_molecules` converts sample dicts into nested `GraphBatch`
- **DataModule**: DDP-aware data module integrating pipeline + collation + DataLoader

Recommended reading order:

1. [Batch Schema Reference](../../tensordict_schema.md)
2. [Data Loading](loading.md)
3. [DataModules](datamodules.md)
