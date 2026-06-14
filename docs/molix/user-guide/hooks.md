# Hooks

Hooks add behavior around the training loop without replacing `Trainer`.
Use them for logging, metrics, checkpoints, profiling, telemetry, learning-rate
events, and custom lifecycle logic.

## Registration

```python
from molix.core.hooks import Log, TensorBoardHook
from molix.core.trainer import Trainer

trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    optimizer_factory=opt_factory,
    hooks=[
        Log(every_n_steps=100),
        TensorBoardHook(log_dir="runs/experiment-1"),
    ],
)
```

Hooks run in registration order by default. To force an order, pass
`(hook, priority)` tuples. Lower priorities run earlier.

```python
hooks = [
    (setup_hook, 10),
    logging_hook,
    (cleanup_hook, 900),
]
```

## Lifecycle Methods

Subclass `BaseHook` and override only the methods you need:

```python
from molix.core.hooks import BaseHook


class NaNStopperHook(BaseHook):
    def on_train_batch_end(self, trainer, state, batch, outputs):
        loss = outputs.get("loss")
        if loss is not None and not loss.isfinite():
            raise RuntimeError(f"Non-finite loss at step {state.global_step}")
```

Common hook points include:

- `on_train_start`
- `on_train_end`
- `on_epoch_start`
- `on_epoch_end`
- `on_train_batch_start`
- `on_train_batch_end`
- `on_eval_batch_start`
- `on_eval_batch_end`
- `on_after_backward`

Hook exceptions propagate. A hook that detects an invalid run should raise
instead of silently logging and continuing.

## State Writes

`TrainState` keeps scalar values in namespace sub-dicts. Write with nested dict
access:

```python
state["train"]["loss"] = loss.item()
state["eval"]["MAE"] = mae
state["performance"]["step_per_second"] = rate
state["gpu"]["peak_gib"] = peak
```

Do not write slash or tuple paths:

```python
state["train/loss"] = loss.item()       # raises ValueError
state[("train", "loss")] = loss.item()  # raises ValueError
```

Reads support all three forms:

```python
state["eval"]["MAE"]
state["eval/MAE"]
state[("eval", "MAE")]
```

## ScalarHook

Hooks that produce scalar values for other hooks to consume should subclass
`ScalarHook` and declare `scalar_keys`.

```python
from molix.core.hooks import ScalarHook


class StepSpeedHook(ScalarHook):
    scalar_keys = (("performance", "step_per_second"),)
```

The `Log` hook can use these paths to decide which columns to render.
