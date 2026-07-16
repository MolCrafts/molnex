# Molix

Molix is the training and execution package in MolNex. Use it when you need
training loops, data modules, hooks, checkpoints, metrics, or the nested
`GraphBatch` data contract used by models and losses.

## Tutorials

- [Quick Start](tutorials/quick-start.md): train a small PyTorch model with
  `Trainer`.
- [Train a Graph Model](tutorials/train-a-graph-model.md): use a nested
  molecular `GraphBatch` end to end.

## User Guide

- [Trainer](user-guide/trainer.md): configure the training loop.
- [Hooks](user-guide/hooks.md): add logging, metrics, checkpointing, and custom
  lifecycle behavior.
- [Data Pipeline](user-guide/data.md): understand sources, preprocessing,
  caching, collation, and data modules.
- [Data Loading](user-guide/data-loading.md): convert flat samples into
  `GraphBatch` objects.
- [Data Modules](user-guide/data-modules.md): wire datasets into `Trainer`.

## Explanation

- [Execution Model](explanation/execution-model.md): how Molix separates
  trainer, steps, hooks, and state.
- [Batch Schema](explanation/batch-schema.md): the raw sample and post-collate
  `TensorDict` shapes.
- [Throughput & `torch.compile`](explanation/throughput-and-compilation.md):
  why PiNet training is launch-bound, the front/back-end compile sweep, the
  fixed-length-padding + CUDA-graphs ~5x path, and the `max-autotune`
  shared-memory failure (and what GPU "shared memory" is vs HBM).

## API

See [molix API Reference](../api/molix.md).
