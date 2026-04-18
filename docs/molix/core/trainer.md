# The Machine Learning Trainer

The `Trainer` is the heart of Molix. It manages the training loop — epoch/step iteration, train/eval stage transitions, hook dispatch, and state tracking — so you can focus on your model and data.

## What the Trainer Owns

- **Loop control**: epoch and step iteration
- **State management**: `TrainState` with epoch, global_step, stage
- **Hook dispatch**: lifecycle callbacks at epoch/batch boundaries
- **Stage transitions**: switching between `Stage.TRAIN` and `Stage.EVAL`
- **Step delegation**: forwarding batches to `DefaultTrainStep` / `DefaultEvalStep`

The Trainer intentionally does **not** own device transfer, AMP, gradient accumulation, checkpoint saving, or scheduler stepping. These concerns belong in custom steps or hooks.

## Basic Usage

```python
from molix.core.trainer import Trainer

trainer = Trainer(
    model=my_model,
    loss_fn=my_loss_fn,
    optimizer_factory=lambda p: torch.optim.Adam(p, lr=1e-3),
)

trainer.train(datamodule, max_epochs=100)
```

## The Training Loop

1.  **Epoch Start**: Call `on_epoch_start` hooks.
2.  **Training Phase**:
    *   Iterate over the training dataloader.
    *   Delegate to `train_step.on_train_batch(trainer, state, batch)`.
    *   Call batch-level hooks.
3.  **Validation Phase**:
    *   Switch to eval mode.
    *   Delegate to `eval_step.on_eval_batch(trainer, state, batch)`.
    *   Call batch-level hooks.
4.  **Epoch End**: Call `on_epoch_end` hooks.

## Understanding TrainState

The `TrainState` object is passed to every hook and step. It tracks:

*   `epoch`: The current epoch index (0-based).
*   `global_step`: The total number of optimizer steps taken across all epochs.
*   `stage`: The current execution phase (`Stage.TRAIN` or `Stage.EVAL`).

Hooks read this state to decide when to act (e.g., "log every 50 steps").

## Customizing the Loop: Steps

Molix allows you to replace the inner loop logic by providing custom step objects that implement the `Step` protocol.

```python
from molix.core.steps import DefaultTrainStep

class GANTrainStep:
    def on_train_batch(self, trainer, state, batch):
        # 1. Update Discriminator
        ...
        # 2. Update Generator
        ...
        return {"loss_g": loss_g, "loss_d": loss_d}

    def on_eval_batch(self, trainer, state, batch):
        ...

trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    optimizer_factory=opt_factory,
    train_step=GANTrainStep(),
)
```

This modularity ensures that `molix` scales from simple regression tasks to cutting-edge research without requiring you to maintain a separate codebase for non-standard training loops.
