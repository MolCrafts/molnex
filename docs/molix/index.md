# Molix

Molix is the training infrastructure layer. Core objects:

- `Trainer`: Main training loop controller
- `TrainState`: Epoch, step, and stage tracking
- `Step`: Train/eval batch computation protocol
- `Hook`: Lifecycle callbacks (logging, metrics, checkpoint, profiling)

The data interface is unified as plain batch dictionaries (`dict[str, Tensor | dict]`).

Recommended reading order:

1. [Quickstart](quickstart.md)
2. [The Trainer](core/trainer.md)
3. [Hooks System](core/hooks.md)
4. [Data Overview](data/index.md)
