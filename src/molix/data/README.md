# molix.data

Molecular data pipeline with nested TensorDict batch types.

## Two-tier data contract

The pipeline speaks two shapes at different stages — **not** a single nested
contract end-to-end. Get this wrong and `KeyError(('edges', 'edge_index'))` is
the first thing you'll see.

| Stage | Shape | Key access |
|---|---|---|
| `DataSource[i]`, pipeline `SampleTask` / `DatasetTask` I/O, `MmapDataset[i]` | **flat `dict`** | `sample["Z"]`, `sample["edge_index"]`, `sample["targets"]["U0"]` |
| Post-`collate_molecules` (what the model forward receives) | **nested `TensorDict`** (plain, no subclass) | `batch["atoms", "Z"]`, `batch["edges", "edge_index"]`, `batch["graphs", "energy"]` |

The single conversion point is `molix.batch.collate.collate_molecules`
(invoked by `molix.batch.datamodule._CollateFn`). Stats / diagnostics that operate on individual
cached samples use flat keys; model / loss / metric code that operates on
batches uses nested tuple-keys. See `docs/tensordict_schema.md` for the full
schema on both sides.

## Modules

- `task`: Task hierarchy — `SampleTask`, `DatasetTask`. Post-collate transforms are **not** here: `BatchTask` lives in `molix.batch.task`.
- `source`: Data source abstraction (`DataSource` protocol, `InMemorySource`, `SubsetSource`)
- `pipeline`: Declarative container **and** orchestrator — `Pipeline` builder, `PipelineSpec` with `.run` / `.cache_key` / `.cache` / `.build_cache` methods. Owns DDP-aware cache materialisation.
- `tasks/`: Built-in tasks (`NeighborList`, `AtomicDress`, `UnitConvert`, `ConstantLabel`).

Neighbouring packages: storage (`PackedCache`, `MmapDataset` / `CachedDataset` /
`SubsetDataset`) lives in `molix.store`; batch construction (`collate_molecules`,
`DataModule`, the batch samplers, `BatchTask` / `PadMolecularBatch`) lives in
`molix.batch`.
