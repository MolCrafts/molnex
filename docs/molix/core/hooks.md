# Mastering the Hook System

The real power of Molix lies in its Hook System. While simple scripts work for basic models, research code often requires unique monitoring, intervention, or complex workflows. Hooks allow you to inject custom logic into the training loop without rewriting the core `Trainer` class.

## Built-in Hooks

Molix comes with several essential hooks out of the box. You will almost always use these in your training runs.

### TensorBoard Logging
The `TensorBoardHook` automatically logs loss curves, epoch progress, and learning rates. It connects seamlessly with PyTorch's `SummaryWriter`.

```python
from molix.core.hooks import TensorBoardHook

# Logs to ./runs/experiment_1
hook = TensorBoardHook(log_dir="./runs/experiment_1", log_every_n_steps=10)
```

### Checkpointing
The `CheckpointHook` handles model saving. It ensures you don't lose progress if your job crashes and allows you to resume training later.

```python
from molix.core.hooks import CheckpointHook

# Saves every 5 epochs, and always keeps the latest 'last.pt'
ckpt = CheckpointHook(checkpoint_dir="./ckpt", save_every_n_epochs=5)
```

### Progress Monitoring
The `ProgressBarHook` wraps your training loop with `tqdm`, giving you a real-time estimate of time remaining and current loss values.

```python
from molix.core.hooks import ProgressBarHook
pbar = ProgressBarHook(desc="Training Methane Model")
```

## Writing Custom Hooks

When built-in tools aren't enough, you can write your own. A Hook is simply a class that overrides specific event methods. You don't need to implement every method—just the ones you care about. We recommend inheriting from `BaseHook`.

### Example: The "Early Stopper" 

Suppose we want to stop training if the loss explodes (becomes NaN).

```python
from molix.core.hooks import BaseHook

class NaNStopperHook(BaseHook):
    def on_train_batch_end(self, trainer, state, batch, outputs):
        loss = outputs.get("loss")
        if loss is not None and torch.isnan(loss):
            print(f"❌ Loss became NaN at step {state.global_step}!")
            # We can signal the trainer to stop (hypothetically)
            # or simply raise an error to halt execution
            raise RuntimeError("Training Diverged")
```

### Accessing Context
Every hook method receives two key arguments:
*   `trainer`: Gives you access to the `model`, `optimizer`, and other hooks.
*   `state`: A snapshot of the current `epoch`, `global_step`, and `stage`.

This context allows your hooks to be state-aware. You can change learning rates based on `state.epoch`, or freeze model layers by accessing `trainer.model`.

## Execution Order

In complex pipelines, the order of hooks matters. For example, you should save a checkpoint *before* uploading it to a cloud bucket, but *after* the optimizer step.

Molix runs hooks in the order they are provided. to enforce a specific order regardless of the list position, you can use **Priority Tuples**.

```python
hooks = [
    (CriticalSetup(), 1),     # Runs FIRST (Priority 1)
    NormalLogging(),          # Runs MIDDLE (Default Priority 100)
    (Cleanup(), 999)          # Runs LAST (Priority 999)
]
```

This system ensures robust pipelines where critical infrastructure (like setting up distributed process groups) always happens before user logic.
