# Batch Schema

MolNex speaks **two** shapes at different stages of the data pipeline. The
shapes are intentionally different — you must know which side you're on:

| Stage | Container | Example access |
|---|---|---|
| Pre-collate (source, pipeline task I/O, `MmapDataset[i]`) | **flat `dict`** | `sample["Z"]`, `sample["edge_index"]` |
| Post-collate (output of `collate_molecules`) | **nested `TensorDict`** | `batch["atoms", "Z"]`, `batch["edges", "edge_index"]` |

The single conversion point is `collate_molecules` (invoked by
`DataModule._CollateFn`). Tuple-key access like `batch["atoms", "Z"]` is a
`TensorDict`-only feature and **does not work** on a raw sample dict — that's
why `sample["edges", "edge_index"]` raises `KeyError`.

The batch side is a **plain `tensordict.TensorDict`** — there is no subclass and
no `@tensorclass` wrapper. What makes it a "molecular batch" is the namespace
layout (`atoms` / `edges` / `graphs`, plus `bonds` when the samples carry
covalent topology), not a Python type. Each namespace is itself a `TensorDict`
carrying its own `batch_size`, which is what lets per-atom, per-edge and
per-graph tensors of different lengths live in one container.

## Sample Schema (pre-collate, single molecule, plain flat dict)

Individual samples from `DataSource.__getitem__`, pipeline task I/O, and
`MmapDataset[i]` / `CachedDataset[i]` are plain Python dicts with **flat
top-level keys** (no `"atoms"` / `"edges"` nesting):

- `Z`: `LongTensor[N]` - Atomic numbers
- `pos`: `FloatTensor[N, 3]` - Atom positions
- `edge_index` (optional, added by `NeighborList`): `LongTensor[E, 2]` - Edge source-target pairs
- `edge_diff` (optional, added by `NeighborList`): `FloatTensor[E, 3]` - Edge vectors
- `edge_dist` (optional, added by `NeighborList`): `FloatTensor[E]` - Edge distances
- `targets` (optional): `dict[str, Tensor]` - Target labels

Access with flat keys: `sample["Z"]`, `sample["edge_index"]`,
`sample["targets"]["U0"]`. The nested tuple-key syntax below is for the
post-collate batch only.

## Batch Schema (nested TensorDict)

`collate_molecules` converts a list of sample dicts into one nested
`TensorDict`:

```
TensorDict (batch_size=[])
├── "atoms": TensorDict (batch_size=[N_total])
│   ├── Z: LongTensor[N_total]
│   ├── pos: FloatTensor[N_total, 3]
│   ├── batch: LongTensor[N_total]       # graph membership
│   └── <atom-level targets, e.g. forces>
├── "edges": TensorDict (batch_size=[E_total])
│   ├── edge_index: LongTensor[E_total, 2]
│   ├── edge_diff: FloatTensor[E_total, 3]
│   └── edge_dist: FloatTensor[E_total]
├── "graphs": TensorDict (batch_size=[B])
│   ├── num_atoms: LongTensor[B]
│   └── <graph-level targets, e.g. energy, U0>
└── "bonds": TensorDict (batch_size=[])   # only when samples carry bonds
    ├── bond_index: LongTensor[2, N_bonds]
    └── bond_types: Tensor[N_bonds]
```

## Namespaces

| Namespace | batch_size | Purpose |
|------|------------|---------|
| `atoms` | `[N_total]` | Per-atom tensors (encoder adds `node_features` in place) |
| `edges` | `[E_total]` | Per-edge tensors (encoder adds `edge_features` in place) |
| `graphs` | `[B]` | Per-graph tensors + graph-level targets |
| `bonds` | `[]` | Covalent topology, present only when samples supply it |
| *(top level)* | `[]` | Container holding the namespaces above |

Every level is a plain `TensorDict`; the batch sizes differ because the
number of atoms, edges, graphs and bonds in a batch are unrelated counts.
`bonds` is the exception with `batch_size=[]`: `bond_index` is COO-shaped
`[2, N_bonds]`, so its leading dimension is 2, not the bond count, and it
cannot share a batch axis with `bond_types` `[N_bonds]`.

Encoder outputs are written into the existing `atoms` / `edges` sub-dicts by
key addition — the batch object is mutated in place, never replaced.

## Access Patterns

```python
batch["atoms", "Z"]           # atomic numbers (N_total,)
batch["atoms", "pos"]         # positions (N_total, 3)
batch["edges", "edge_index"]  # edge pairs (E_total, 2)
batch["graphs", "energy"]     # graph-level target (B,)
```

## Conventions

- Graph-level targets (energy, U0, etc.) go under `graphs`, shape `[B]`.
- Atom-level targets (forces) go under `atoms`, shape `[N_total, ...]`.
  Which target name is routed where is declared by the `TargetSchema` passed to
  `collate_molecules`; names in `atom_level` go to `atoms`, everything else is
  flattened to `[B]` under `graphs`.
- `edge_index` is always `[E, 2]` with `[:, 0] = source`, `[:, 1] = target`, and
  `edge_diff = pos[target] - pos[source]`. Per-molecule edge indices are rebased
  onto the concatenated atom numbering during collation.
- Models receive the whole batch and access nested keys as needed.
- Loss functions receive `(predictions, batch)` and read targets from the batch.

## Related Pages

- [Data Loading](../user-guide/data-loading.md)
- [Data Modules](../user-guide/data-modules.md)
- [Train a Graph Model](../tutorials/train-a-graph-model.md)
