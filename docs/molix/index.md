# Molix

Molix is the training layer of MolNex.

Its job is to make experimentation repeatable without turning the training system into the center of the project. Molix handles orchestration, lifecycle, and extensibility, while staying intentionally smaller than the modeling layers it supports.

## Design Role

Molix exists to answer a simple question: how should training infrastructure support molecular research without taking it over?

The answer in MolNex is to keep the execution layer clear, lightweight, and extensible. Training should provide structure, not lock the rest of the stack into one worldview.

## What Molix Optimizes For

- a training loop that stays understandable
- extension points for real research workflows
- separation between orchestration and model design
- enough structure to support growth without unnecessary framework weight

## Suggested Reading

1. [Quickstart](quickstart.md)
2. [The Trainer](core/trainer.md)
3. [Hooks System](core/hooks.md)
4. [Data Overview](data/index.md)
