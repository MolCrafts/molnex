# Data Pipeline

`molix.data` provides the molecular data pipeline:

- **Sources** (`source.py`): `DataSource` protocol and `InMemorySource` / `SubsetSource` implementations
- **Tasks** (`task.py`, `tasks/`): transform primitives (`SampleTask`, `DatasetTask`) and the built-ins `NeighborList`, `AtomicDress`, `UnitConvert`, `ConstantLabel`. Post-collate transforms (`BatchTask`, `PadMolecularBatch`) live in `molix.batch`
- **Pipeline** (`pipeline.py`): declarative `Pipeline` / `PipelineSpec` container — which tasks run, in what order, under what cache identity

Batch construction lives next door in `molix.batch`:

- **Collation** (`batch/collate.py`): `collate_molecules` turns sample dicts into a plain nested `TensorDict` with `atoms` / `edges` / `graphs` (and `bonds`) namespaces; `collate_packed` is the equivalent fast path straight off packed cache tensors
- **Post-collate transforms** (`batch/task.py`, `batch/pad.py`): `BatchTask` and the shipped `PadMolecularBatch`, configured as `DataModule(batch_tasks=[...])`
- **Batch policies** (`batch/sampler.py`, `batch/group_split.py`): token-budget and task-mixture batch samplers, group-aware index splits
- **DataModule** (`batch/datamodule.py`): DDP-aware data module integrating datasets + collation + DataLoader

Storage lives next door in `molix.store`:

- **Cache** (`store/cache.py`): `PackedCache`, the single-file packed store a materialized pipeline writes
- **Datasets** (`store/dataset.py`): `MmapDataset` / `CachedDataset` / `SubsetDataset` readers over a `PackedCache`

Recommended reading order:

1. [Batch Schema](../explanation/batch-schema.md)
2. [Data Loading](data-loading.md)
3. [Data Modules](data-modules.md)
