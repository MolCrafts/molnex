# Batch Schema Reference

MolNex uses nested `TensorDict` subclasses (defined in `molix.data.types`) as its batch format. Each level has its own batch size, enabling natural per-atom, per-edge, and per-graph operations.

## Sample Schema (single molecule, plain dict)

Individual samples from `DataSource.__getitem__` are plain Python dicts:

- `Z`: `LongTensor[N]` - Atomic numbers
- `pos`: `FloatTensor[N, 3]` - Atom positions
- `edge_index` (optional): `LongTensor[E, 2]` - Edge source-target pairs
- `bond_diff` (optional): `FloatTensor[E, 3]` - Edge vectors
- `bond_dist` (optional): `FloatTensor[E]` - Edge distances
- `targets` (optional): `dict[str, Tensor]` - Target labels

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
| `AtomData` | `TensorDict` | `[N]` | Per-atom tensors |
| `NodeRepAtoms` | `AtomData` | `[N]` | Adds `node_features` from encoder |
| `EdgeData` | `TensorDict` | `[E]` | Per-edge tensors |
| `EdgeRepEdges` | `EdgeData` | `[E]` | Adds `edge_features` from encoder |
| `GraphData` | `TensorDict` | `[B]` | Per-graph tensors + targets |
| `GraphBatch` | `TensorDict` | `[]` | Top-level container |

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
