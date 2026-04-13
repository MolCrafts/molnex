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

## State-Key Naming Convention

Hooks communicate through a shared `TrainState` dict. A *producer* hook writes
scalars during the training loop; *consumer* hooks (`Log`, `TensorBoardHook`, …)
read those scalars by key. Because keys are the public contract between hooks,
they need a naming convention.

### Flat keys with a namespace prefix

State keys are always flat strings of the form `"<namespace>/<name>"`. Nested
dicts (`state["train"]["loss"]`) are **not** used — a single `state.get(key)`
call must return the value.

| Namespace     | Meaning                                                                 |
|---------------|-------------------------------------------------------------------------|
| `train/`      | Values produced during training batches (one per step).                 |
| `val/`        | Values produced during validation passes.                               |
| `test/`       | Values produced during test passes.                                     |
| `performance/`| Hardware-agnostic throughput / timing (step/s, samples/s, …).           |
| `gpu/`        | CUDA-device telemetry (memory, utilisation, …).                         |
| `system/`     | Host-side telemetry (CPU, RAM, disk, network). Reserved.                |
| `opt/`        | Optimiser-derived scalars other than gradients (e.g. `opt/lr`).         |

If a scalar does not fit any of the above, pick a new top-level namespace
rather than overloading an existing one. Don't put loss under `performance/`.

### Name form within a namespace

- Use **lower_snake_case** for descriptive names: `performance/step_per_second`.
- Use bracket notation to embed units directly: `GPU-State/peak[GB]`,
  `GPU-State/alloc[GB]`. A reader should not need to open the hook to know
  what "1.23" means.
- Use the **class name verbatim** (no casefolding) when the name *is* an
  identity, e.g. metrics: `train/MAE`, `val/RMSE`. This matches the
  `class.__name__` that `MetricsHook` writes.
- Keep names stable. Renaming a key is a breaking change for every
  downstream consumer.

### Advertising produced keys (`ScalarHook`)

Any hook that writes to `state` and expects another hook to consume those
values **must** subclass `ScalarHook` and declare `scalar_keys`:

```python
from molix.core.hooks import ScalarHook

class GPUMemoryHook(ScalarHook):
    scalar_keys = ("GPU-State/alloc[GB]", "GPU-State/resv[GB]", "GPU-State/peak[GB]")

    def on_train_batch_end(self, trainer, state, batch, outputs):
        state["GPU-State/alloc[GB]"] = torch.cuda.memory_allocated() / 1e9
        ...
```

For hooks whose key set depends on runtime configuration (e.g. `MetricsHook`
whose names come from the metric list), override `scalar_keys` as a
`@property` that returns the computed tuple.

`scalar_keys` is the single source of truth:

- **`Log(keys=...)`** and **`TensorBoardHook(keys=..., eval_keys=...)`**
  accept either flat strings or `ScalarHook` instances. Passing the hook
  itself expands to its `scalar_keys`, so call sites don't need to hard-code
  key names.
- Linting / introspection tools can diff *declared* keys against keys
  actually written to catch typos.

### Canonical keys produced by built-in hooks

| Producer                    | Key(s)                                                               |
|-----------------------------|----------------------------------------------------------------------|
| `Trainer` (DefaultTrainStep)| `train/loss`                                                         |
| `MetricsHook`               | `{prefix_train}/{MetricCls}`, `{prefix_val}/{MetricCls}` (dynamic)   |
| `StepSpeedHook`             | `performance/step_per_second`                                        |
| `GPUMemoryHook`             | `GPU-State/alloc[GB]`, `GPU-State/resv[GB]`, `GPU-State/peak[GB]`    |
| `GradClipHook`              | `train/grad_norm`                                                    |

Default `TrainState` keys (written by the trainer, not by hooks) are
`epoch`, `global_step`, `stage`, `steps_since_last_eval`. These are reserved
and must not be overwritten by hooks.

### Rule of thumb

> If another hook might want this value, namespace it and put it in `state`.
> If it is an internal detail of the hook, keep it on `self`.

This keeps `state` a tidy, introspectable dashboard rather than a dumping
ground.
