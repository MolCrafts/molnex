# Data Documentation

Molnex provides a robust set of tools for handling molecular data.

## Guides

- [**Data Loading Patterns**](loading.md)
    - Learn the standard `Frame -> Dataset -> DataLoader` pipeline.
    - understand how `molpy.Frame` and `nested_collate_fn` work together.

- [**DataModules**](datamodules.md)
    - Learn how to encapsulate your data pipeline for reproducible training with the `Trainer`.

- [**Atomic TensorDict**](datamodules.md#atomic-tensordict)
    - Understand the `AtomicTD` data structure used by models.
