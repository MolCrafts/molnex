---
title: Multi-task train/eval Step with per-batch task routing
status: approved
created: 2026-08-14
grilled: true
---

# Multi-task train/eval Step with per-batch task routing

## Summary

Class-I parameter learning trains one shared model against four heterogeneous QM
tasks (energy+forces, projected internal Hessian, torsion profile, MBIS charges)
whose losses are not comparable term-by-term and whose reference levels of theory
differ. This spec adds `src/molix/core/steps/multitask.py`: a train Step and an
eval Step that read the per-batch task label riding as a graph-level target
(`batch["graphs", "task"]`, produced by `TargetSchema` routing in
`src/molix/data/collate.py:25`), dispatch the batch to one injected
`{task: loss_fn}` entry, and publish that task's scalar into the correct
`TrainState` namespace. Both steps **inherit** `DefaultTrainStep` /
`DefaultEvalStep` and override only a new `compute_loss` seam, so AMP autocast,
gradient accumulation, `optimizer_applied` scheduler-desync detection and eval
grad-context semantics are reused byte-for-byte rather than forked. The model
arrives as `trainer.model` and the losses arrive pre-constructed from the
caller's script, so `molix` gains no `molpot` dependency: cross-task routing is
the only new concern, gradient clipping stays with `GradClipHook`, per-task
metrics stay with `MetricsHook(name_prefix=…)`, and per-task scalars stay
detached device tensors so the launch-bound hot path keeps exactly one host sync
per batch (the label read itself).

## Domain basis

The step is task-agnostic code, but it is the single place where the v0 task
mixture becomes operational, so the v0 science contract is recorded here and
enforced by the module docstring (ac-010).

**v0 task set** (names are caller-chosen; codes are caller-assigned integers):

| task | prediction keys (from 08-adapter) | QM reference | residual unit | normalization |
|---|---|---|---|---|
| `energy_force` | `energy`, `forces` | B3LYP-D3(BJ)/DZVP | kcal mol⁻¹, kcal mol⁻¹ Å⁻¹ | fixed physical denominator per term |
| `hessian` | `hessian` | B3LYP-D3(BJ)/DZVP | kcal mol⁻¹ Å⁻² (bonds), kcal mol⁻¹ rad⁻² (angles) | fixed physical denominator |
| `torsion` | `energy` | B3LYP-D3(BJ)/DZVP | kcal mol⁻¹ | fixed physical denominator |
| `charge` | `q` | PBE0/def2-TZVPD, MBIS partitioning | e | separate fixed denominator |

Binding v0 constraints:

1. **[CRITICAL] LJ is frozen.** `LJParameterHead` parameters are excluded from
   optimization in v0. Gas-phase intramolecular energies, forces, Hessians and
   torsion scans do not determine σ/ε; espaloma-0.3 inherits openff-2.0.0 vdW and
   ByteFF pins GAFF-2.2 vdW for the same reason. This step never introduces a
   task that would supply a σ/ε gradient signal.
2. **[HIGH] The charge task keeps its own `reference_method` and its own
   normalization denominator.** PBE0/def2-TZVPD MBIS charges and
   B3LYP-D3(BJ)/DZVP energetics must never share one metadata field or one
   denominator. The step therefore carries **no** shared scale / normalization /
   reference-method surface: each `TaskSpec` owns a fully-constructed loss whose
   `ScaleNormalizedLoss` denominator was supplied by the caller
   (`src/molix/core/losses/molecular.py:416`).
3. **Normalization is fixed physical denominators, not learned or dataset-scanned
   weights** (ForceBalance discipline). After normalization, espaloma-0.3 uses
   unit weights across tasks, with charge weight 1e-3 in their hyperparameter
   search. Because a batch is single-task (10-sampler), any residual per-task
   weight is expressible as a denominator rescale; the step exposes no weight
   vector so two competing weighting surfaces cannot drift apart.
4. **Energy centering is per group (scan / conformer set), exactly.**
   `center_by_group` (`molecular.py:50`) is applied inside the caller's
   `molecule_centered_energy_mse` / `TorsionProfileLoss`; the step never mixes
   groups because it never re-partitions a batch.
5. **Declared v0 biases.** (a) MM energies/forces/Hessians are evaluated at the
   QM constrained-torsion geometries **without MM reoptimization**, so torsion
   parameters absorb relaxation error; (b) EEM charges remain topology-only (no
   geometry dependence). Both are recorded, not corrected, in v0.
6. **Units.** Per-task state scalars are dimensionless (post
   `ScaleNormalizedLoss`); the step performs no unit conversion and imports
   nothing from `molix.units`, so no conversion float is restated.

References:

- Takaba et al., "Machine-learned molecular mechanics force fields from
  large-scale quantum chemical data" (espaloma-0.3), arXiv:2307.07085; SI §C.2.3
  (post-normalization unit weights, charge weight 1e-3).
- Zheng et al., "Data-Driven Parametrization of Molecular Mechanics Force Fields
  for Expansive Chemical Space Coverage" (ByteFF), arXiv:2408.12817 (GAFF-2.2 vdW
  pinned).
- Qiu et al., *J. Chem. Theory Comput.* **17**, 6262 (2021), §2.1.7 —
  ForceBalance objective normalization. https://doi.org/10.1021/acs.jctc.1c00571
- Boothroyd et al., *J. Chem. Theory Comput.* **19**, 3251 (2023) — openff-2.0.0
  vdW provenance. https://doi.org/10.1021/acs.jctc.3c00039
- Verstraelen et al., *J. Chem. Theory Comput.* **12**, 3894 (2016) — MBIS.
  https://doi.org/10.1021/acs.jctc.6b00456
- Eastman et al., *Sci. Data* **10**, 11 (2023) — SPICE (PBE0/def2-TZVPD charge
  level). https://doi.org/10.1038/s41597-022-01882-6
- Wilson, *J. Chem. Phys.* **9**, 76 (1941) — B matrix behind the Hessian task's
  units. https://doi.org/10.1063/1.1750829

## Design

### A. `compute_loss` seam on the default steps (no fork)

`DefaultTrainStep.on_train_batch` (`src/molix/core/steps/train.py:47`) currently
inlines `loss = trainer.loss_fn(predictions, batch)` inside the autocast block.
Change exactly two things in `train.py` and the same two in `eval.py`:

- add `compute_loss(self, trainer, state, batch, predictions) -> torch.Tensor`
  whose default body is `assert trainer.loss_fn is not None; return
  trainer.loss_fn(predictions, batch)` (the assert **moves** here from the top of
  `on_train_batch` / `on_eval_batch`, so a step that owns its losses may run with
  `Trainer(loss_fn=None)`);
- replace the inlined call with `self.compute_loss(trainer, state, batch,
  predictions)`, still **inside** the `with ctx:` / `with grad_ctx, amp_ctx:`
  block so per-task losses see the same autocast dtype as today.

Untouched: accumulation-window arithmetic (`train.py:68-70`), `zero_grad`
placement, scaler `scale/unscale_/step/update` order, the
`optimizer_applied = scaler.get_scale() >= scale_before` inf/nan detection
(`:96-110`), the `on_after_backward` fire point, and
`state["train"]["loss"] = loss.detach()`. The `Step` Protocol
(`steps/base.py:16`) gains **no** required method — `compute_loss` is an optional
seam, duck-typed by callers — so third-party steps keep working. `base.py`'s
"See Also" gains the two new steps and one sentence pointing at the seam as the
sanctioned way to change the computation without changing the cadence
(`base.py:32-38`).

### B. `TaskSpec` + `TaskRouter` (new module)

```python
@dataclass(frozen=True)
class TaskSpec:
    name: str      # state-scalar suffix: state[ns]["loss_<name>"]
    code: int      # integer label as it rides in batch[task_key]
    loss: Callable[[Mapping[str, Any], Any], torch.Tensor]
```

Exactly three fields — no `weight`, no `scale`, no `reference_method` (see Domain
basis §2/§3). `TaskRouter(tasks: Sequence[TaskSpec], *, task_key: tuple[str, str]
= ("graphs", "task"))` validates at construction: non-empty `tasks`, unique
`name`s, unique `code`s, non-empty names; it builds a `{code: TaskSpec}` index
once. `task_key` is a *graph namespace* path, deliberately named `task` and not
`task_id` so it can never be confused with the pipeline cache-identity
`task_id`.

`TaskRouter.route(batch) -> TaskSpec` resolves the label with two `Mapping.get`
steps (`batch.get("graphs")` then `.get("task")`, so plain nested dicts used by
unit tests and `TensorDict` both work and a missing label never surfaces as a
bare `KeyError`), then:

- label missing → `ValueError` naming the full path `("graphs", "task")` and how
  to declare it via `TargetSchema`;
- `numel() == 0` → `ValueError`;
- floating dtype → `ValueError` naming the dtype (a float label would silently
  truncate);
- `codes = torch.unique(labels).tolist()` — **the one and only host sync per
  batch**; `len(codes) != 1` → `ValueError` naming every code found and the
  10-sampler single-task-batch contract it violates;
- `code` not declared → `ValueError` naming the code and the declared
  `{code: name}` map.

`route` reads nothing but the label: predictions and batch are handed to the
routed loss **verbatim**, so per-task key/namespace adaptation (e.g. supplying
`target["B"]` for `ProjectedInternalHessianLoss`, or
`{"q": batch["atoms","q"]}` for `ChargeMSELoss`) stays in the caller's closure.

### C. `MultiTaskTrainStep` / `MultiTaskEvalStep`

`MultiTaskTrainStep(DefaultTrainStep)` with `namespace = "train"`;
`MultiTaskEvalStep(DefaultEvalStep)` with `namespace = "eval"` and
`__init__(self, router, *, no_grad: bool = False)` forwarding `no_grad` (must
stay `False` for force/Hessian tasks — their losses need a live graph). Neither
overrides `on_train_batch` / `on_eval_batch`; each overrides only:

```python
def compute_loss(self, trainer, state, batch, predictions):
    spec = self.router.route(batch)
    loss = spec.loss(predictions, batch)
    state[self.namespace][f"loss_{spec.name}"] = loss.detach()
    return loss
```

Three duplicated lines beat a mixin here (the only difference is one constant).
Writes are nested (`state["train"]["loss_hessian"]`), never flat slash keys
(rejected at `state.py:86`), and are **detached device tensors** — never
`.item()` (`train.py:40-44`). The steps never call `clip_grad_norm_`;
`GradClipHook` (`src/molix/hooks/training.py:19`) keeps sole ownership of
clipping via `on_after_backward`, which the inherited `on_train_batch` still
fires at the unchanged point. `WeightedLoss`
(`src/molix/core/losses/combined.py:9`) evaluates *every* term (`:17-22`), so it
may only appear **inside** one `TaskSpec.loss`; cross-task selection is this
router's job and nothing else's.

Each step exposes `scalar_keys` — `tuple((namespace, f"loss_{t.name}") for t in
router.tasks)` — mirroring the `ScalarHook.scalar_keys` convention. This makes
per-task scalars first-class for the generic consumers that already walk the
namespaces (`JournalHook` / `TensorBoardHook`, `journal.py:173`,
`tensorboard.py:33`) and for `Log`.

### D. Two dependency fixes on the surface this step rides (iron law)

1. `Log._validate_keys` (`src/molix/hooks/progress.py:230`) unions only
   `trainer.hooks` `ScalarHook.scalar_keys` plus `_BUILTIN_STATE_PATHS`
   (`:109`, which already whitelists the *step*-written `("train","loss")`).
   Step-written per-task scalars would therefore be rejected as "not advertised",
   making the feature invisible in the stdout thermo table. Fix: after the hook
   loop, also `advertised.update(getattr(step, "scalar_keys", ()) or ())` for
   `getattr(trainer, "train_step"/"eval_step", None)`. `getattr` keeps the
   existing hook-only fake trainers in `test_log_hook.py:116` working. **No new
   hook type** is introduced.
2. `Trainer.smoke` (`src/molix/core/trainer.py:324`, loss at `:406`) re-implements
   forward/loss/backward instead of delegating, so it silently warms up a
   *different* objective than the configured Step — and raises `TypeError` when a
   step owns its losses (`loss_fn=None`). Fix, minimally: resolve
   `compute = getattr(self.train_step, "compute_loss", None)` once and use
   `compute(self, self.state, batch, predictions)` when present, else the current
   `self.loss_fn(predictions, batch)`. The non-finite-loss abort (`:407`) is
   unchanged. **Named remaining debt (not fixed here):** `smoke` still duplicates
   the AMP / zero_grad / optimizer sequence of `DefaultTrainStep`; collapsing that
   duplication belongs to a follow-up `/mol:refactor`, not to this spec.

### Reuse decision

No `librarian_report` accompanied this request; candidates were resolved from the
caller's `interaction_points` and a direct read of the touched modules.

- `reuse DefaultTrainStep` / `reuse DefaultEvalStep` — inherited; cadence code is
  not copied (identity of `on_train_batch` / `on_eval_batch` is asserted).
- `generalize DefaultTrainStep.on_train_batch` / `DefaultEvalStep.on_eval_batch`
  — promoted to serve both callers via the `compute_loss` seam (spawns the
  "Extract `compute_loss` seam…" task instead of a parallel implementation).
- `generalize Log._validate_keys` — extended to accept step-advertised scalar
  paths, serving both hook-written and step-written scalars.
- `reuse GradClipHook` — clipping stays there; the step contains no
  `clip_grad_norm_`.
- `reuse MetricsHook(name_prefix=…)` — per-task metric publication; no new hook.
- `reuse ScaleNormalizedLoss` / `WeightedLoss` / `ProjectedInternalHessianLoss` /
  `TorsionProfileLoss` / `ChargeMSELoss` / `energy_force_mse` /
  `molecule_centered_energy_mse` — constructed by the caller and injected as
  `TaskSpec.loss`; this spec adds no loss.
- `reuse TargetSchema` collate routing — the label rides as a graph-level target;
  `collate.py` is **not** modified.
- `reuse JournalHook` / `TensorBoardHook` namespace walks — per-task scalars are
  recorded with zero new code.
- `new TaskSpec` / `TaskRouter` / `MultiTaskTrainStep` / `MultiTaskEvalStep` —
  no existing symbol dispatches a batch to one of several losses by label;
  `WeightedLoss` deliberately cannot (it evaluates every term). Naming,
  construction (plain `__init__`, no factory) and eager actionable `ValueError`
  messages follow the closest pattern in-tree, `collate.py` / `hooks/scalar.py`.

## Files to create or modify

- src/molix/core/steps/multitask.py (new)
- src/molix/core/steps/train.py
- src/molix/core/steps/eval.py
- src/molix/core/steps/base.py
- src/molix/core/steps/__init__.py
- src/molix/core/trainer.py
- src/molix/hooks/progress.py
- tests/test_molix/test_core/test_steps/__init__.py (new)
- tests/test_molix/test_core/test_steps/test_multitask.py (new)
- tests/test_molix/test_core/test_steps/test_train.py (new)
- tests/test_molix/test_core/test_steps/test_eval.py (new)
- tests/test_molix/test_core/test_trainer_smoke.py (new)
- tests/test_molix/test_core/test_log_hook.py
- regressions/universal-class-i-11-step.py (new)

## Tasks

- [ ] Write failing unit tests for the `compute_loss` seam (tests/test_molix/test_core/test_steps/test_train.py → TestDefaultTrainStep, tests/test_molix/test_core/test_steps/test_eval.py → TestDefaultEvalStep, tests/test_molix/test_core/test_trainer_smoke.py → TestTrainerSmoke): default body delegates to `trainer.loss_fn`, asserts when it is None, is invoked from inside the autocast block, and `Trainer.smoke` routes through it while keeping the non-finite abort
- [ ] Extract the `compute_loss` seam in src/molix/core/steps/train.py and src/molix/core/steps/eval.py (move `assert trainer.loss_fn is not None` into it; AMP / accumulation-window / `optimizer_applied` / `on_after_backward` code unchanged) and route src/molix/core/trainer.py:406 `Trainer.smoke` through `getattr(train_step, "compute_loss", None)` with fallback
- [ ] Write failing unit tests for multitask routing (tests/test_molix/test_core/test_steps/__init__.py, tests/test_molix/test_core/test_steps/test_multitask.py → TestTaskSpec / TestTaskRouter / TestMultiTaskTrainStep / TestMultiTaskEvalStep) and for step-advertised Log keys (tests/test_molix/test_core/test_log_hook.py), using stub losses and a stub model so no molpot import is needed
- [ ] Implement `TaskSpec` and `TaskRouter` in src/molix/core/steps/multitask.py (construction-time validation; single-sync `torch.unique(...).tolist()` label read; eager `ValueError` naming the offending label / dtype / code / mixed codes)
- [ ] Implement `MultiTaskTrainStep` and `MultiTaskEvalStep` in src/molix/core/steps/multitask.py (override `compute_loss` only, nested detached per-task writes, `scalar_keys` property, `no_grad` forwarding)
- [ ] Extend `Log._validate_keys` in src/molix/hooks/progress.py to union `getattr(trainer.train_step / trainer.eval_step, "scalar_keys", ())` alongside the existing `ScalarHook` loop
- [ ] Export `MultiTaskEvalStep`, `MultiTaskTrainStep`, `TaskRouter`, `TaskSpec` from src/molix/core/steps/__init__.py with `__all__` re-alphabetized, and add the seam sentence + See Also entries in src/molix/core/steps/base.py
- [ ] Add Google-style docstrings per doc.style to src/molix/core/steps/multitask.py recording units (dimensionless post-normalization scalars, per-task residual units) and the v0 science constraints (LJ frozen, separate MBIS reference method + denominator, per-group centering, constrained-geometry and EEM biases)
- [ ] Add regression example regressions/universal-class-i-11-step.py (public API only: `Trainer` + duck-typed datamodule, two hand-built single-task batches, `accumulate_grad_batches=2`, `GradClipHook`, `Log` with per-task keys; analytic hard-coded goldens, no third-party runtime)
- [ ] Run full check + test suite

## Testing strategy

Unit tests mirror the source tree; every test targets a single function/method;
no e2e under `tests/`.

`tests/test_molix/test_core/test_steps/test_multitask.py`

- `TestTaskRouter` — happy path: four hand-built single-task batches (codes 0..3,
  labels as `(B,)` int64 under `("graphs","task")`) each route to their own
  `TaskSpec`; a plain nested `dict` batch routes identically to a `TensorDict`
  one.
- `TestTaskRouter` — edge cases, one `pytest.raises(ValueError, match=…)` each:
  missing label (message names `("graphs", "task")`), empty label tensor,
  float-dtype label (message names the dtype), undeclared code `7` (message names
  `7` and the declared code→name map), mixed labels `[0, 1]` (message names both
  codes), and construction-time empty `tasks` / duplicate `name` / duplicate
  `code`.
- `TestMultiTaskTrainStep` — no-fork guarantee:
  `MultiTaskTrainStep.on_train_batch is DefaultTrainStep.on_train_batch` and
  `MultiTaskEvalStep.on_eval_batch is DefaultEvalStep.on_eval_batch`;
  `MultiTaskTrainStep().on_eval_batch` still raises `NotImplementedError`.
- `TestMultiTaskTrainStep` — state contract: after one batch,
  `state["train"]["loss_<task>"]` exists, `isinstance(..., torch.Tensor)`,
  `requires_grad is False`, `grad_fn is None`, is not a `float`;
  `state["train"]["loss"]` also written; no key containing `/` exists anywhere in
  `state`; `scalar_keys` equals the expected `("train", "loss_<name>")` tuple.
- `TestMultiTaskTrainStep` — hot-path contract: with `torch.Tensor.item`
  monkeypatched to raise and `torch.nn.utils.clip_grad_norm_` monkeypatched to
  raise, one train step completes; with `GradClipHook` registered,
  `state["train"]["grad_norm"]` is written by the hook.
- `TestMultiTaskTrainStep` — parity on a single-task stream, `fp64` + fixed seed,
  `accumulate_grad_batches=2`: `DefaultTrainStep` with `trainer.loss_fn = f` and
  `MultiTaskTrainStep` with one `TaskSpec(loss=f)` produce the same
  `optimizer_applied` sequence `[False, True]`, the same `state["train"]["loss"]`
  values (`torch.equal`), the same `on_after_backward` fire count, and identical
  parameters after the window.
- `TestMultiTaskEvalStep` — writes `state["eval"]["loss_<task>"]` (detached) and,
  with default `no_grad=False`, the returned loss has `requires_grad is True`
  from a positions-dependent stub model, so a force/Hessian loss can backprop.
- Static guards: `src/molix/core/steps/multitask.py` source contains no
  `molpot` / `molrep` / `molzoo` import, no `clip_grad_norm_`, no `.item()`, and
  no `reference_method` / `normalization` / `weight` constructor parameter;
  `TaskSpec` dataclass fields are exactly `("name", "code", "loss")`.

`tests/test_molix/test_core/test_steps/test_train.py`, `…/test_eval.py` — the
seam only: default `compute_loss` returns `trainer.loss_fn(predictions, batch)`,
asserts when `loss_fn is None`, and is the call site used by
`on_train_batch` / `on_eval_batch` (verified with a subclass spy that also
records `torch.is_autocast_enabled()` parity with `config["use_amp"]`).

`tests/test_molix/test_core/test_trainer_smoke.py` — `Trainer.smoke` calls
`train_step.compute_loss` when present (spy step), falls back to `loss_fn` for a
step without the seam, and still raises `RuntimeError` on a non-finite loss.

`tests/test_molix/test_core/test_log_hook.py` — a fake trainer whose `train_step`
advertises `("train", "loss_hessian")` passes `Log` startup validation; the same
`Log` key with no advertising step still raises `ValueError(match="not
advertised")`.

**Regression example** — `regressions/universal-class-i-11-step.py` (repo-root
`regressions/`, never under `tests/`): public API only
(`Trainer`, `MultiTaskTrainStep`, `TaskRouter`, `TaskSpec`, `GradClipHook`,
`Log`, `ScaleNormalizedLoss`, `energy_force_mse`, `ChargeMSELoss`), `fp64`, a
fixed-weight tiny model, a duck-typed datamodule yielding two hand-built
single-task batches (`energy_force` code 0, `charge` code 1) and
`accumulate_grad_batches=2`, `loss_fn=None`. Asserts hard-coded goldens:
`state["train"]["loss_energy_force"]`, `state["train"]["loss_charge"]`,
`state["train"]["loss"]`, `state["global_step"] == 1`,
`state["train"]["grad_norm"]` present. Goldens are **closed-form analytic**
values with the derivation written in the module docstring (fixed weights →
exact MSE arithmetic), not captured from any third-party tool and not imported
at runtime; tolerance `atol=1e-12` in float64; non-zero exit on mismatch.

## Out of scope

- σ/ε (LJ) optimization and any task that would supply a vdW gradient — frozen
  in v0 by Domain basis §1.
- The molpot prediction adapter producing `{"energy","forces","hessian","q"}`
  (08-adapter) and any conditional / lazy per-task forward: the step never
  reshapes or conditions the forward pass.
- Batch construction, task-label assignment and mixture schedule (10-sampler);
  `src/molix/data/collate.py` is not modified.
- `TorsionProfileLoss` extensions (12): they land inside one `TaskSpec.loss`, so
  the routing contract defined here does not change when 12 lands.
- Cross-task weighting surfaces: adaptive/learned task weights (GradNorm,
  uncertainty weighting), per-task learning rates, multiple optimizers. Per-task
  weight folds into the caller's fixed denominator.
- Mixed-task batches (rejected with a `ValueError`, not supported), and any
  gradient-surgery/PCGrad-style conflict resolution.
- New hook types, `Log` column auto-discovery beyond the `scalar_keys` union, and
  checkpoint-schema changes.
- Collapsing the remaining `Trainer.smoke` duplication of the AMP / optimizer
  sequence (named debt in Design §D2) — route to `/mol:refactor`.
- Relocating the pre-existing, mirror-misplaced `tests/test_molix/test_core/
  test_log_hook.py` (Log's tests belong at `tests/test_molix/test_hooks/
  test_progress.py`) — named, deliberately not moved here.
- The multi-task tiny-set overfit and calibration gates
  (`.claude/notes/production-training-checklist.md` §3) — executed by the
  caller's training script, not by this step.
