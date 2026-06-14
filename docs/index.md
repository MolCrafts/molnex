# MolNex Documentation

MolNex is a Python framework for molecular machine learning. It is split into
four packages that can be used together or independently:

- `molix` provides training, data loading, state, hooks, execution utilities
  (`torch.compile` / CUDA-graph, AOT export), an in-process MD driver, and
  trajectory diagnostics.
- `molrep` provides representation learning modules for molecular structure.
- `molpot` provides potential, composition, pooling, head, and derivation layers.
- `molzoo` provides assembled reference encoder families such as MACE, Allegro,
  and PiNet.

## Start Here

If this is your first time using MolNex:

1. Install the project with [Installation](installation.md).
2. Learn the training loop with [Molix Quick Start](molix/tutorials/quick-start.md).
3. Train with molecular graph batches in
   [Train a Graph Model](molix/tutorials/train-a-graph-model.md).
4. Build model parts with [MolRep](molrep/index.md), [MolPot](molpot/index.md), and
   [MolZoo](molzoo/index.md).

## Package Guides

Each package follows the same documentation shape:

- Tutorials: short, task-focused introductions.
- User Guide: practical usage details for day-to-day work.
- Explanation: concepts and design background.
- API Reference: generated from source with `mkdocstrings`.

## Common Tasks

- Use the trainer: [Molix Trainer](molix/user-guide/trainer.md)
- Load molecular batches: [Molix Data Loading](molix/user-guide/data-loading.md)
- Understand batch structure: [Molix Batch Schema](molix/explanation/batch-schema.md)
- Build an encoder: [MolRep Build an Encoder](molrep/tutorials/build-an-encoder.md)
- Build a potential: [MolPot Build a Potential](molpot/tutorials/build-a-potential.md)
- Use MolZoo encoders: [MolZoo Tutorial](molzoo/tutorials/index.md)
- Look up Python objects: [API Reference](api/index.md)
