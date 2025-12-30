# The Machine Learning Trainer

The `Trainer` is the heart of Molix. It abstracts away the boilerplate of the training loop—device management, gradient accumulation, validation intervals—so you can focus on your model and data.

## Basic Usage

At its simplest, the Trainer just needs a model, a loss function, and an optimizer factory.

```python
from molix.core.trainer import Trainer

trainer = Trainer(
    model=my_model,
    loss_fn=torch.nn.MSELoss(),
    optimizer_factory=lambda p: torch.optim.Adam(p, lr=1e-3)
)

trainer.train(datamodule, max_epochs=100)
```

The trainer handles the rest: iterating over epochs, switching between train/eval modes, and calling hooks at the right times.

## The Training Loop

Under the hood, the Trainer implements a standard, robust optimization loop:

1.  **Epoch Start**: Reset meters, call `on_epoch_start` hooks.
2.  **Training Phase**:
    *   Iterate over the training dataloader.
    *   Move batch to device.
    *   **Step Execution**: Forward pass $\rightarrow$ Compute Loss $\rightarrow$ Backward pass $\rightarrow$ Optimizer Step.
    *   Call batch-level hooks.
3.  **Validation Phase**:
    *   Switch to eval mode (no grad).
    *   Iterate over validation dataloader.
    *   Compute metrics.
4.  **Epoch End**: Save checkpoints, call `on_epoch_end`.

## Understanding TrainState

The `TrainState` object is passed to every hook and step. It acts as the "source of truth" for the training progress. It tracks:

*   `epoch`: The current epoch index (0-based).
*   `global_step`: The total number of optimizer steps taken across all epochs.
*   `stage`: The current execution phase (`Stage.TRAIN` or `Stage.EVAL`).

Hooks read this state to decide when to act (e.g., "log to WandB every 50 steps").

## Customizing the Loop: Steps

Sometimes, the standard "forward-backward-step" loop isn't enough. You might be training a GAN (requires alternating updates), doing Reinforcement Learning, or using a complex gradient accumulation schedule.

Molix allows you to replace the inner loop logic by providing custom `TrainStep` and `EvalStep` objects.

```python
from molix.steps.train_step import TrainStep

class GANTrainStep(TrainStep):
    def run(self, state, batch):
        # 1. Update Discriminator
        ...
        # 2. Update Generator
        ...
        return {"loss_g": loss_g, "loss_d": loss_d}

# Pass your custom logic to the Trainer
trainer = Trainer(train_step=GANTrainStep(), ...)
```

This modularity ensures that `molix` scales from simple regression tasks to cutting-edge research without requiring you to maintain a separate codebase for "weird" training loops.
