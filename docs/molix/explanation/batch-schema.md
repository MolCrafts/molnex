# Batch Schema

MolNex speaks **two** shapes at different stages of the data pipeline. The
shapes are intentionally different — you must know which side you're on:

| Stage | Container | Example access |
|---|---|---|
| Pre-collate (source, pipeline task I/O, `MmapDataset[i]`) | **flat `dict`** | `sample["Z"]`, `sample["edge_index"]` |
| Post-collate (`GraphBatch` from `collate_molecules`) | **nested `TensorDict`** | `batch["atoms", "Z"]`, `batch["edges", "edge_index"]` |

The single conversion point is `collate_molecules` (invoked by
`DataModule._CollateFn`). Tuple-key access like `batch["atoms", "Z"]` is a
`TensorDict`-only feature and **does not work** on a raw sample dict — that's
why `sample["edges", "edge_index"]` raises `KeyError`.

Nested `TensorDict` subclasses (defined in `molix.data.types`) are the
batch-side containers. Each level carries its own batch size, enabling
natural per-atom, per-edge, and per-graph operations.

## Sample Schema (pre-collate, single molecule, plain flat dict)

Individual samples from `DataSource.__getitem__`, pipeline task I/O, and
`MmapDataset[i]` / `CachedDataset[i]` are plain Python dicts with **flat
top-level keys** (no `"atoms"` / `"edges"` nesting):

- `Z`: `LongTensor[N]` - Atomic numbers
- `pos`: `FloatTensor[N, 3]` - Atom positions
- `edge_index` (optional, added by `NeighborList`): `LongTensor[E, 2]` - Edge source-target pairs
- `bond_diff` (optional, added by `NeighborList`): `FloatTensor[E, 3]` - Edge vectors
- `bond_dist` (optional, added by `NeighborList`): `FloatTensor[E]` - Edge distances
- `targets` (optional): `dict[str, Tensor]` - Target labels

Access with flat keys: `sample["Z"]`, `sample["edge_index"]`,
`sample["targets"]["U0"]`. The nested tuple-key syntax below is for the
post-collate `GraphBatch` only.

## Batch Schema (nested TensorDict)

`collate_molecules` converts a list of sample dicts into a `GraphBatch`:

```
GraphBatch (batch_size=[])
├── "atoms": AtomData (batch_size=[N_total])
│   ├── Z: LongTensor[N_total]
│   ├── pos: FloatTensor[N_total, 3]
│   ├── batch: LongTensor[N_total]       # graph membership
│   └── <atom-level targets, e.g. forces>
├── "edges": EdgeData (batch_size=[E_total])
│   ├── edge_index: LongTensor[E_total, 2]
│   ├── bond_diff: FloatTensor[E_total, 3]
│   └── bond_dist: FloatTensor[E_total]
└── "graphs": GraphData (batch_size=[B])
    ├── num_atoms: LongTensor[B]
    └── <graph-level targets, e.g. energy, U0>
```

## Type Hierarchy

| Type | Extends | batch_size | Purpose |
|------|---------|------------|---------|
| `AtomData` | `TensorDict` | `[N]` | Per-atom tensors (encoder adds `node_features` in place) |
| `EdgeData` | `TensorDict` | `[E]` | Per-edge tensors (encoder adds `edge_features` in place) |
| `GraphData` | `TensorDict` | `[B]` | Per-graph tensors + targets |
| `GraphBatch` | `TensorDict` | `[]` | Top-level container |

Encoder outputs are written into the existing `AtomData` / `EdgeData`
sub-dicts by key addition — no subclass swap.

## Access Patterns

```python
batch["atoms", "Z"]           # atomic numbers (N_total,)
batch["atoms", "pos"]         # positions (N_total, 3)
batch["edges", "edge_index"]  # edge pairs (E_total, 2)
batch["graphs", "energy"]     # graph-level target (B,)
```

## Conventions

- Graph-level targets (energy, U0, etc.) are stored in `GraphData`, shape `[B]`.
- Atom-level targets (forces) are stored in `AtomData`, shape `[N_total, ...]`.
- `edge_index` is always `[E, 2]` with `[:, 0] = source`, `[:, 1] = destination`.
- Models receive the `GraphBatch` directly and access nested keys as needed.
- Loss functions receive `(predictions, batch)` and read targets from the batch.

## Related Pages

- [Data Loading](../user-guide/data-loading.md)
- [Data Modules](../user-guide/data-modules.md)
- [Train a Graph Model](../tutorials/train-a-graph-model.md)
