# Batch Schema

MolNex speaks **two** shapes at different stages of the data pipeline. The
shapes are intentionally different — you must know which side you're on:

| Stage | Container | Example access |
|---|---|---|
| At rest (a store on disk, or built in memory) | **molpy `Frame`** | `frame["atoms"]["x"]`, `frame.meta["energy"]` |
| In the model (output of `collate_molecules`) | **nested `TensorDict`** | `batch["atoms", "Z"]`, `batch["edges", "edge_index"]` |

The single conversion point is `collate_molecules` (invoked by
`DataModule._CollateFn`). Tuple-key access like `batch["atoms", "Z"]` is a
`TensorDict`-only feature and does not work on a `Frame`.

The batch is a **`tensordict.TensorDict`**; `molix.batch.TensorFrame` is its
typed view, declaring the namespaces as fields and wrapping an existing batch
without copying. Neither owns storage or behaviour — what makes a batch a
"molecular batch" is the namespace layout (`atoms` / `edges` / `graphs`, plus
`bonds` / `angles` / `propers` / `impropers` when the frames carry them). Each
namespace is itself a `TensorDict` carrying its own `batch_size`, which is what
lets per-atom, per-edge and per-graph tensors of different lengths live in one
container.

## Frame Schema (at rest, single molecule)

A molpy `Frame` holds its columns in blocks, and where a column sits is what
decides where it lands in the batch — nothing is declared:

- `atoms.number` → `atoms.Z`, atomic numbers
- `atoms.x` / `.y` / `.z` → `atoms.pos`, positions
- `atoms.fx` / `.fy` / `.fz` → `atoms.forces`
- any other `atoms` column → `atoms.<name>`
- a `meta` scalar (`energy=…`) → `graphs.<name>`
- `bonds.atomi` / `.atomj` → `bonds.bond_index`, rebased onto the batch;
  `bonds.type_id` → `bonds.bond_types`
- an `angles` / `propers` / `impropers` block → its own namespace of 1-D
  index columns (improper centre is `atomi`)

`FrameStoreDataset[i]` skips the `Frame` entirely and returns frame `i` as a
one-frame batch `TensorDict`, read straight off the store's mmapped columns.

The cutoff graph is not part of a frame: it is a function of positions and a
cutoff, so `NeighborList` builds it downstream.

## Batch Schema (nested TensorDict)

`collate_molecules` converts a list of molpy `Frame` objects into one nested
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
  Nothing declares the routing: where a column sits in the molpy `Frame`
  decides where it lands. A column of the frame's `atoms` block becomes an
  `atoms` leaf; a frame `meta` scalar becomes a `graphs` leaf. A new target is
  a new frame column and nothing else.
- `edge_index` is always `[E, 2]` with `[:, 0] = source`, `[:, 1] = target`, and
  `edge_diff = pos[target] - pos[source]`. Per-molecule edge indices are rebased
  onto the concatenated atom numbering during collation.
- Models receive the whole batch and access nested keys as needed.
- Loss functions receive `(predictions, batch)` and read targets from the batch.

## Related Pages

- [Data Loading](../user-guide/data-loading.md)
- [Data Modules](../user-guide/data-modules.md)
- [Train a Graph Model](../tutorials/train-a-graph-model.md)
