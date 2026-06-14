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

The single conversion point is `collate.collate_molecules` (invoked by
`DataModule._CollateFn`). Stats / diagnostics that operate on individual
cached samples use flat keys; model / loss / metric code that operates on
batches uses nested tuple-keys. See `docs/tensordict_schema.md` for the full
schema on both sides.

## Modules

- `task`: Task hierarchy — `SampleTask`, `DatasetTask`, `BatchTask` (no built-in `BatchTask` subclasses; it's a post-collate extension point)
- `source`: Data source abstraction (`DataSource` protocol, `InMemorySource`, `SubsetSource`)
- `pipeline`: Declarative container **and** orchestrator — `Pipeline` builder, `PipelineSpec` with `.run` / `.cache_key` / `.cache` / `.build_cache` methods. Owns DDP-aware cache materialisation.
- `cache`: `PackedCache` class — one atomic `.pt` file per cache; owns readiness, mmap load, and rank-aware polling. **Do not migrate to `TensorDict.memmap_()`** — see that module's docstring for the inode-budget argument.
- `tasks/`: Built-in tasks (`NeighborList`, `AtomicDress`, `UnitConvert`).
- `collate`: `collate_molecules` converts sample dicts → nested `TensorDict`.
- `dataset`: `MmapDataset` / `CachedDataset` read `PackedCache` files; `SubsetDataset` is an index view. Connectivity statistics (`avg_num_neighbors`, `max_atoms`, `max_edges`) are exposed as `@cached_property` on every dataset — derived directly from packed-cache pointers, no user-facing task needed. Fitted `DatasetTask` state (e.g. `AtomicDress` baselines) lives under `.stats()`.
- `datamodule`: DDP-aware `DataModule` integrating datasets + collation + post-collate `BatchTask`s. `DataModule.from_cached_pipeline` is the one-shot factory.
