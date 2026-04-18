# Component Taxonomy

MolNex code is organized by functional layer:

1. `molix`: Training and flow control
2. `molix.data`: Data loading, preprocessing, batching, TensorDict types
3. `molrep`: Representation learning components
4. `molpot`: Potential functions and readout components
5. `molzoo`: Pre-built encoder assemblies (MACE / Allegro)

## Data Flow

```
DataSource (plain dict samples)
  → collate_molecules
    → GraphBatch (nested TensorDict)
      → Model (receives GraphBatch)
        → predictions
          → loss_fn(predictions, batch)
```

## Unified Interface

- Samples are plain dicts: `{"Z": tensor, "pos": tensor, "targets": {...}}`
- Batches are nested `GraphBatch` TensorDicts with per-level batch sizes
- Models receive `GraphBatch` and access: `batch["atoms", "Z"]`, `batch["edges", "bond_dist"]`
- Loss function signature: `loss_fn(predictions, batch)`

## Where to Place New Code

- Data-related changes go in `molix.data`
- New operators go in `molrep/interaction` or `molix/F`
- New end-to-end models go in `molzoo`
- Reusable potentials and readout go in `molpot`
