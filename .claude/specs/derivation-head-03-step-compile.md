---
title: Compile the training step — forward + loss as one region, backward from AOTAutograd
status: draft
created: 2026-09-11
chain: derivation-head
depends: derivation-head-04-molix-derivative
---

## Summary

Operator directive (2026-09-11): 训练编译整个 step. `Trainer.compile` compiles the
model only (`trainer.py:656`); the loss, backward and optimizer stay eager, and
`Trainer.smoke` re-implements the step body inline. Measured on GH200 (matpes
fused, 256 atoms, fp64, `.scratch/compile-seam/repro/probe_design.py`, job
2293234): forward + force loss as one `fullgraph` region runs 13.4–14.5 ms per
step vs 41 ms eager (3×); leaf placement is worth ≤ 8 % and on stock torch the
head's leaf costs 2 graph breaks until pytorch/pytorch#196653 (0 breaks with
it) — accepted.

## Design

1. `DefaultTrainStep.forward_loss(trainer, batch) -> loss` is the one traced
   function: `predictions = model(batch)`, `loss = loss_fn(predictions, batch)`.
   `on_train_batch` calls it, then `backward` / hooks / optimizer exactly as
   today (Python cadence stays outside). `Trainer.smoke` calls the same
   `forward_loss` — the inline copy goes.
2. `Trainer.compile(..., step=True)` (default `True` once green) compiles
   `train_step.forward_loss` and `eval_step.forward_loss` with the `Compiler`
   config; `step=False` keeps today's model-only compile for steps whose loss
   cannot trace (`MultiTaskTrainStep`'s router host sync — documented, not
   compiled).
3. AMP: the autocast context wraps the compiled call (dynamo traces it);
   `GradScaler` stays eager.
4. Losses must be pure: `energy_force_mse` / `energy_mse` already are; the
   group-loop losses (`center_by_group`, `TorsionProfileLoss`, ...) are marked
   non-compilable in their docstrings and rejected by `compile(step=True)`
   with a clear error naming the loss.
5. Static shapes for `cuda_graphs=True` come from spec 02 (padding + mask).

## Files

`src/molix/core/steps/{base,train,eval,multitask}.py`, `src/molix/core/trainer.py`,
`src/molix/core/losses/molecular.py` (docstrings), `docs/molix/explanation/throughput-and-compilation.md`;
mirrored tests (`tests/test_molix/test_core/test_compile.py`, `test_steps/*`).

## Testing

- `test_compile.py`: `compile(step=True)` swaps `forward_loss` on both steps,
  `smoke` and `train` run the same callable (identity check on the wrapper).
- slow: `Compiler.explain(train_step.forward_loss, trainer, batch)` with a tiny
  MACE + `energy_force_mse` → `graph_break_count == 0` or exactly the leaf
  break (`force.py`) on stock torch; parameter gradients equal eager to 1e-12.
- GH200 benchmark row: steps/s eager vs model-only compile vs step compile.

## Out of scope

Compiled optimizer (`torch.compile` on `optimizer.step`) — separate;
`MultiTaskTrainStep` routing.
