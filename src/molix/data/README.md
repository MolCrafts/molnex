# molix.data

Molecular data pipeline with nested TensorDict batch types.

- `types`: TensorDict subclasses — `AtomData`, `EdgeData`, `GraphData`, `GraphBatch`
- `task`: Task hierarchy — `SampleTask`, `DatasetTask`, `BatchTask`
- `source`: Data source abstraction (`DataSource` protocol, `InMemorySource`, `SubsetSource`)
- `pipeline`: **Declarative** container of tasks — `Pipeline` builder, `PipelineSpec`. No execution, no IO.
- `execute`: Free functions that run a pipeline — `run`, `transform`, `collect_task_states`.
- `cache`: Short-lived scratch cache for pipeline output — `cache`, `cache_key`, `is_ready`, `save`, `load`. One `.pt` file per cache, atomic commit.
- `ddp`: Opt-in distributed helpers — `rank`, `wait_for_ready`.
- `tasks/`: Built-in tasks (`NeighborList`, `AtomicDress`).
- `collate`: `collate_molecules` converts sample dicts → nested `GraphBatch`.
- `dataset`: `MmapDataset` / `CachedDataset` read cache files; `SubsetDataset` is an index view.
- `datamodule`: DDP-aware `DataModule` integrating datasets + collation + post-collate `BatchTask`s.
