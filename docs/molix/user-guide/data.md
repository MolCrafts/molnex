# Data Pipeline

`molix.data` provides the molecular data pipeline:

- **Sources** (`source.py`): `DataSource` protocol and `InMemorySource` / `SubsetSource` implementations
- **Tasks** (`task.py`, `tasks/`): transform primitives (`SampleTask`, `DatasetTask`, `BatchTask`) and the built-ins `NeighborList`, `AtomicDress`, `UnitConvert`, `ConstantLabel`, `PadMolecularBatch`
- **Pipeline** (`pipeline.py`): declarative `Pipeline` / `PipelineSpec` container — which tasks run, in what order, under what cache identity
- **Cache** (`cache.py`): `PackedCache`, the single-file packed store a materialized pipeline writes
- **Datasets** (`dataset.py`): `MmapDataset` / `CachedDataset` / `SubsetDataset` readers over a `PackedCache`
- **Collation** (`collate.py`): `collate_molecules` turns sample dicts into a plain nested `TensorDict` with `atoms` / `edges` / `graphs` (and `bonds`) namespaces; `collate_packed` is the equivalent fast path straight off packed cache tensors
- **DataModule** (`datamodule.py`): DDP-aware data module integrating pipeline + collation + DataLoader

Recommended reading order:

1. [Batch Schema](../explanation/batch-schema.md)
2. [Data Loading](data-loading.md)
3. [Data Modules](data-modules.md)
