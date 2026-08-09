# Hooks

Hooks add behavior around the training loop without replacing `Trainer`.
Use them for logging, metrics, checkpoints, profiling, telemetry, learning-rate
events, and custom lifecycle logic.

Hooks live in two modules, and the split matters when you write imports:

- `molix.core.hook` (singular) — the *contract* layer: the `Hook` protocol, the
  `BaseHook` no-op base class, and `ScalarHook`.
- `molix.hooks` (plural) — every *concrete* implementation (`Log`, `TensorBoardHook`,
  `MetricsHook`, `StepSpeedHook`, `CheckpointHook`, `JournalHook`,
  `GradClipHook`, `ProgressBarHook`, `ProfilerHook`, `GPUMemoryHook`,
  `GPUUtilsHook`, `MolRecMetricsHook`, `ActivationCheckpointingHook`,
  `EarlyStop`).

## Registration

```python
from molix.core.trainer import Trainer
from molix.hooks import Log, StepSpeedHook, TensorBoardHook

speed = StepSpeedHook()

trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    optimizer_factory=opt_factory,
    hooks=[
        speed,
        Log(every_n_steps=100, keys=[("train", "loss"), speed]),
        TensorBoardHook(every_n_steps=100, log_dir="runs/experiment-1"),
    ],
)
```

`Log` has no default column set: `keys` is required and each entry is either a
state path (`("train", "loss")`, or the equivalent slash string `"train/loss"`)
or a `ScalarHook` instance, in which case `Log` expands that hook's
`scalar_keys`. At `on_train_start`, `Log` rejects any key that is neither a
built-in state path nor advertised by a registered `ScalarHook`, so a typo
fails loudly instead of printing a column of dashes.

`TensorBoardHook` needs no key list — it scans the `train`, `performance` and
`gpu` namespaces on each logged train step and `eval` on each eval completion,
writing every numeric or 0-d tensor value it finds.

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
from molix.core.hook import BaseHook


class NaNStopperHook(BaseHook):
    def on_train_batch_end(self, trainer, state, batch, outputs):
        loss = outputs.get("loss")
        if loss is not None and not loss.isfinite():
            raise RuntimeError(f"Non-finite loss at step {state.global_step}")
```

The default steps return `{"loss": ..., "predictions": ...}` (the train step
adds `"optimizer_applied"`), which is what `outputs` holds in the batch-end
callbacks.

The full set of callbacks the `Trainer` dispatches:

- `on_train_start`
- `on_train_end`
- `on_epoch_start`
- `on_epoch_end`
- `on_train_batch_start`
- `on_train_batch_end`
- `on_after_backward`
- `on_eval_phase_start`
- `on_eval_batch_start`
- `on_eval_batch_end`
- `on_eval_step_complete`

`on_eval_phase_start` and `on_eval_step_complete` bracket **every** eval phase —
both the step-based one triggered by `eval_every_n_steps` and the epoch-end one.
Accumulating hooks should reset their eval buffers in `on_eval_phase_start` and
publish in `on_eval_step_complete`; publishing there (rather than in
`on_epoch_end`) also puts the value in `state` before the LR scheduler reads it.

Hook exceptions propagate: `Trainer._call_hooks` logs the failure with a
traceback and then re-raises. A hook that detects an invalid run should raise
instead of silently logging and continuing.

## State Writes

`TrainState` keeps scalar values in namespace sub-dicts. Write with nested dict
access:

```python
state["train"]["loss"] = loss.detach()
state["eval"]["MAE"] = mae
state["performance"]["step_per_second"] = rate
state["gpu"]["peak_gib"] = peak
```

Note the `detach()` rather than `item()` on the per-step value: `.item()` forces
a CPU↔GPU synchronization, and doing that on every training step drains the GPU
queue and serializes an otherwise launch-bound loop. Everything on the per-step
path (`DefaultTrainStep`, `GradClipHook`, `MetricsHook`'s train side) therefore stores
the 0-d device tensor and leave materialization to the consumers, which sample
on their own throttled cadence. Cold-path values — an eval metric published once
per eval phase — are converted to `float` at the write, because the LR scheduler
and `CheckpointHook` compare and serialize them as plain numbers.

Do not write slash or tuple paths:

```python
state["train/loss"] = loss.detach()       # raises ValueError
state[("train", "loss")] = loss.detach()  # raises ValueError
```

Reads support all three forms:

```python
state["eval"]["MAE"]
state["eval/MAE"]
state[("eval", "MAE")]
```

Which namespace a hook may write depends on the callback it is writing from:

- `on_train_batch_end` / `on_after_backward` → `train`, `performance`, `gpu`
- `on_eval_batch_end` / `on_eval_step_complete` → `eval`

A hook that reports the same metric on both sides must hold two independent
accumulators (`MetricsHook` deep-copies its metrics into `train_metrics` and
`val_metrics` for exactly this reason) — sharing one buffer across phases lets
an eval-side `reset()` corrupt the train-side value.

## ScalarHook

Hooks that produce scalar values for other hooks to consume should subclass
`ScalarHook` and declare `scalar_keys` — a tuple of state paths, where each path
is either a top-level string key or a `(namespace, name)` tuple.

```python
from molix.core.hook import ScalarHook


class LearningRateHook(ScalarHook):
    scalar_keys = (("train", "lr"),)

    def on_train_batch_end(self, trainer, state, batch, outputs):
        state["train"]["lr"] = trainer.optimizer.param_groups[0]["lr"]
```

`Log` reads `scalar_keys` to expand a hook passed in its `keys` list into
columns, and uses the union of all registered hooks' `scalar_keys` (plus the
built-in paths `epoch`, `global_step`, `stage`, `steps_since_last_eval`,
`best_metric`, `train/loss`, `eval/loss`) to validate the keys you asked for.

If the paths depend on runtime configuration — `MetricsHook` derives them from
the metric class names it was given — override `scalar_keys` as a `@property`
instead of setting it as a class attribute.

Shipped `ScalarHook` subclasses and the paths they publish:

| Hook | Writes |
|---|---|
| `StepSpeedHook` | `performance/step_per_second` |
| `GradClipHook` | `train/grad_norm` (pre-clip L2 norm) |
| `MetricsHook` | `train/<Metric>` and `eval/<Metric>`, one per metric class |
| `GPUMemoryHook` | `gpu/alloc_gib`, `gpu/resv_gib`, `gpu/peak_gib` (selected subset) |
| `GPUUtilsHook` | `gpu/util_pct`, `gpu/mem_util_pct` (selected subset) |
