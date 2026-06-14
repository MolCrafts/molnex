# Execution Model

Molix separates the outer training loop from the inner batch computation.

## Trainer

`Trainer` owns run orchestration:

- epoch and step iteration
- train/eval stage transitions
- model and batch device placement
- hook dispatch
- checkpoint resume plumbing
- `TrainState` counters

The trainer should not contain model-specific training logic. That logic belongs
in the model, the loss function, hooks, or custom steps.

## Steps

Steps own one batch of computation. The default train step:

1. Extracts model inputs from the batch.
2. Runs the model.
3. Calls `loss_fn(predictions, batch)`.
4. Runs backward.
5. Calls `on_after_backward` hooks.
6. Steps the optimizer.
7. Writes `state["train"]["loss"]`.

Custom steps are the extension point for non-standard optimization.

## Hooks

Hooks observe lifecycle events and can write scalar values into `TrainState`.
They are best for side effects such as logging, checkpointing, metrics,
profiling, and runtime checks.

## State

`TrainState` contains counters and namespaced scalar dictionaries. Writes are
nested to keep ownership explicit:

```python
state["train"]["loss"] = loss.item()
state["eval"]["MAE"] = mae
```

Reads are ergonomic:

```python
state["train/loss"]
state[("eval", "MAE")]
```

This lets loggers and checkpoint rules use path-like strings while producer
hooks still write to explicit namespaces.
