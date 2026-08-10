# Trainer

`Trainer` owns the outer training loop: epoch and step iteration, train/eval
stage transitions, batch device transfer, hook dispatch, checkpoint resume, and
`TrainState` updates.

## Basic Usage

```python
from molix.core.trainer import Trainer

trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    optimizer_factory=lambda params: torch.optim.Adam(params, lr=1e-3),
)

state = trainer.train(datamodule, max_epochs=100)
```

`datamodule` must provide `train_dataloader()` and may provide
`val_dataloader()`.

## Loss Function Contract

The default train and eval steps pass the batch through unmodified:

```python
predictions = model(batch)
loss = loss_fn(predictions, batch)
```

The whole batch object goes to `model(batch)` — no key filtering, no unpacking
into keyword arguments — so the model and the loss see exactly the same object.
For the collated nested `TensorDict` that means the model reads its inputs with
tuple keys (`batch["atoms", "Z"]`) and the loss reads its targets from the same
batch (`batch["graphs", "energy"]`).

`loss_fn` must return a scalar tensor; the step calls `.backward()` on it (after
dividing by `accumulate_grad_batches`) and records the un-scaled value in
`state["train"]["loss"]`.

## Loop Control

Use `max_epochs`, `max_steps`, or both:

```python
trainer.train(datamodule, max_epochs=10)
trainer.train(datamodule, max_steps=1000)
trainer.train(datamodule, max_epochs=10, max_steps=1000)
```

When both are set, training stops at the first limit reached.

## Evaluation

By default, validation runs at epoch boundaries when the data module provides a
validation loader. To add step-based validation:

```python
trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    optimizer_factory=opt_factory,
    eval_every_n_steps=500,
)
```

## Device

Pass `device` to move the model and batches inside the training loop:

```python
trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    optimizer_factory=opt_factory,
    device="cuda",
)
```

## Custom Steps

Replace the inner training or evaluation behavior by passing objects that
implement the `Step` protocol.

```python
class CustomTrainStep:
    def on_train_batch(self, trainer, state, batch):
        ...


trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    optimizer_factory=opt_factory,
    train_step=CustomTrainStep(),
)
```

Use custom steps for non-standard optimization, multi-loss updates, gradient
accumulation, or model-specific control flow.

## TrainState

`trainer.state` is a `TrainState` dict with counters and scalar namespaces:

- `epoch`
- `global_step`
- `stage`
- `steps_since_last_eval`
- `train`
- `eval`
- `performance`
- `gpu`

Hooks and steps write scalars into nested namespace dicts, for example
`state["train"]["loss"] = value`. Reads can use nested access, tuple paths, or
slash paths: `state["train"]["loss"]`, `state[("train", "loss")]`, and
`state["train/loss"]`.
