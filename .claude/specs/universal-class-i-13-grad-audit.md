---
title: Per-task gradient-balance audit primitive + flat-pool trainer retirement
status: approved
created: 2026-08-14
chain: universal-class-i
grilled: true
---

# Per-task gradient-balance audit primitive + flat-pool trainer retirement

## Summary

Deliver the evidence primitive that justifies fixed unit task weights for the
Universal Class-I v0 objective, and retire the last hand-rolled training loop
that would otherwise carry that objective. Given a model, an injected
`{task: loss_fn}` mapping, and a bounded stream of real task-labeled packed
batches, `TaskGradientProbe` records per-task loss and per-task,
per-parameter-group gradient norm `‖∂L_task/∂θ_G‖₂` without changing a single
parameter; `GradientBalanceReport` turns two such snapshots (at init and after
N optimizer steps, N configurable, default 100) into a JSON-serializable ratio
table that the calibration script — not the library — hands to molexp as
E4/E5 evidence for `universal-class-i-v0-calibration-gh200`. In the same
change, `scripts/mm_param_learning/espaloma_mm_protocol_train.py` loses its
flat conformer pool, hand-rolled epoch loop, inline gradient clipping and
inline early-stop bookkeeping in favour of a script-level composition of the
chain's sampler + multi-task step + `Trainer` + hooks, and the orphaned
flat-pool smoke trainer `train_phalkethoh_mm_energy.py` is deleted. This spec
runs last in the chain and depends on `universal-class-i-10-sampler` and
`universal-class-i-11-step` for the real batching path — the library primitive
itself depends on neither.

## Domain basis

Notation: `θ_G` is the parameter subset of group `G` (default: one group per
direct child module — encoder, bond head, angle head, proper head, atom head,
charge model), `L_t` is the task loss for task `t`.

**Per-task, per-group gradient norm**

```
g_{t,G} = ‖∂L_t/∂θ_G‖₂ = sqrt( Σ_{θ_i ∈ θ_G} (∂L_t/∂θ_i)² )
```

measured at `θ(0)` (initialization) and `θ(N)` (default `N = 100` optimizer
steps), averaged over the per-batch values of the bounded probe stream —
per-batch, because the per-batch norm is what the optimizer actually sees per
step.

**Ratio table and imbalance**

```
r_{t,G} = g_{t,G} / max_{t'} g_{t',G}          ∈ (0, 1]
I_G     = max_t g_{t,G} / min_t g_{t,G}        defined only when min_t g_{t,G} > 0
```

A group with `min_t g_{t,G} = 0` is **task-specific** (a head only one task
touches); `I_G` is reported as `null` there and excluded from the verdict — a
head receiving gradient from one task only is correct, not an imbalance. The
verdict is taken over **shared** groups only:

```
fixed unit weights (w_t = 1 ∀t) are justified  ⟺  I_G ≤ 10 (one decade)
for every shared group G, at both probe points.
```

**Units (binding precondition).** `g_{t,G}` carries units `[L_t]·[θ_G]⁻¹`.
Comparing two tasks inside one group is only dimensionally meaningful when all
`[L_t]` agree — energy MSE is `kcal²·mol⁻²`, force MSE is `(kcal·mol⁻¹·Å⁻¹)²`,
charge MSE is `e²`. ForceBalance-style normalization makes every task loss
dimensionless (`[L_t] = 1`); in this repo that is
`molix.core.losses.ScaleNormalizedLoss` (`L/s²`, delivered by
`universal-class-i-03-losses`). The probe therefore records a caller-supplied
unit label per task (default `"1"`), and `GradientBalanceReport.balanced()`
refuses a verdict when the labels are not all identical. Ratios `r_{t,G}` and
`I_G` are dimensionless.

**References**

- Qiu, Y. *et al.* "Development and Benchmarking of Open Force Field v1.0.0."
  *J. Chem. Theory Comput.* **17**, 6262–6280 (2021).
  DOI: 10.1021/acs.jctc.1c00571 — ForceBalance objective normalization
  (§2.1.7 per the operator citation): each residual is divided by a reference
  denominator so that unit weights are defensible.
- Takaba, K. *et al.* "Machine-learned molecular mechanics force fields from
  large-scale quantum chemical data." arXiv:2307.07085 — espaloma-0.3 sets
  **all task weights to 1** (SI §C.2.3); this audit is the local reproduction
  of the evidence behind that choice.
- Escalation path only — **not v0**: Chen, Z. *et al.* "GradNorm."
  arXiv:1711.02257; Kendall, A., Gal, Y., Cipolla, R. "Multi-Task Learning
  Using Uncertainty to Weigh Losses." arXiv:1705.07115. If a shared group
  exceeds one decade, these are the recorded next steps; nothing in this spec
  implements adaptive weighting.

Cross-reference: `.claude/notes/production-training-checklist.md` §2 ("every
intended trainable group receives a finite, nonzero gradient") and §3 ("the
configured multi-task mixture overfits … without one loss or gradient
contribution overwhelming the others") are the checklist items this primitive
produces evidence for.

## Design

### New library surface (three primitives, one hook)

**`src/molix/core/audit.py` (new)**

- `TaskGradientSnapshot` — frozen dataclass; one probe point. Fields:
  `label: str` (`"step_0"`, `"step_100"`), `at_step: int`,
  `loss: dict[str, float]` (task → mean loss over that task's probe batches),
  `grad_norm: dict[str, dict[str, float]]` (task → group → `g_{t,G}`),
  `n_batches: dict[str, int]`, `units: dict[str, str]`. `to_dict()` returns
  plain JSON types.
- `TaskGradientProbe` — the measurement type.
  `TaskGradientProbe(model, losses, *, groups=None, units=None)` where
  `losses: Mapping[str, Callable[[predictions, batch], Tensor]]` uses the
  molix `loss_fn(predictions, batch)` convention, and `groups` defaults to
  `{name: params for name, module in model.named_children() if module has
  trainable params}`, falling back to `{"model": model.parameters()}`. Groups
  are resolved once in `__init__` to `dict[str, tuple[nn.Parameter, ...]]`.
  One method: `run(batches, *, label, at_step=0) -> TaskGradientSnapshot`,
  consuming a caller-bounded iterable of `(task_name, batch)` pairs — the
  label shape the chain's task-mixture sampler yields (task → molecule →
  configuration). Per pair: `zero_grad(set_to_none=True)` → `model(batch)` →
  `losses[task](pred, batch)` → `backward()` → read per-group norms. One loss
  per batch, so no `retain_graph`.
- Refusals (no fake evidence): unknown task label in the stream → `ValueError`;
  a task in `losses` that received **no** batch → `ValueError` (a silent zero
  row would be fabricated evidence); non-finite loss or non-finite group norm
  → `ValueError` naming task and group.
- `GradientBalanceReport` — the ratio table. `add(snapshot)` (duplicate label
  → `ValueError`), `snapshots` property, `ratios()` → `label → group → task →
  r_{t,G}`, `imbalance()` → `label → group → I_G | None`,
  `balanced(*, max_ratio=10.0) -> bool`, and `to_dict()` →
  `{"schema": "molix.grad_balance/1", "snapshots": [...], "ratios": ...,
  "imbalance": ...}`. `balanced()` raises `ValueError` when fewer than two
  tasks were probed, when no shared group exists, or when task unit labels
  differ — all three mean "no comparison is possible", and returning `True`
  there would launder a non-result into a GO.
- Deliberately **no** `print_report()`: rendering is the caller's job, and
  borrowing `molix.profiler._utils._fmt_table` would invert the leaf direction
  (`molix.core → molix.profiler`).

**Read-only contract (stated choice).** The probe operates **in place on the
caller's model** and is state-preserving; it does not deep-copy. On entry it
captures a CPU clone of `model.state_dict()` (parameter bytes only, no
optimizer state — HBM-neutral on the GH200 path), a clone of every existing
`.grad` buffer (including which ones are `None`), `model.training`, and RNG
state via `molix.core.checkpoint.rng.capture_rng_states()`. A `finally` block
restores all four. It never constructs an optimizer, never calls
`optimizer.step()`, never touches `global_step`, and runs outside autocast /
`GradScaler` so recorded norms are unscaled and comparable. Rejected
alternative: `copy.deepcopy(model)` — doubles resident parameter memory and
severs parameter identity with the caller's optimizer param groups.

**`src/molix/hooks/audit.py` (new)** — `GradientBalanceHook(ScalarHook)`:
`GradientBalanceHook(probe, batches, *, at_steps=(0, 100))`. Fires the probe
from `on_train_start` (label `step_0`) and from
`on_train_batch_end` when `state["global_step"]` reaches each remaining entry
of `at_steps` (label `step_<n>`), appending to a `report` property. It writes
`state["train"]["grad_imbalance"]` (max shared-group `I_G`, or the single-task
placeholder omitted) through the nested form only — never a slash key — and
advertises `scalar_keys = (("train", "grad_imbalance"),)`. The same fixed
`batches` sequence is used at both points so the two snapshots are comparable.
Because the probe restores `.grad` byte-exactly, probing from
`on_train_batch_end` is safe under gradient accumulation. Re-exported from
`src/molix/hooks/__init__.py`.

**Why this is not a duplicate Step.** The multi-task step from
`universal-class-i-11-step` backwards the *summed* objective once; per-task
gradient isolation requires one backward per task loss, which that step
deliberately does not expose. The probe therefore owns only
forward + per-task backward + norm read + restore; cadence, accumulation,
scheduling and the optimizer stay with `Trainer`/`Step`. The library primitive
imports nothing from siblings 10/11 — only `nn.Module` and the loss callables
— so its unit tests stand alone.

**Placement.** `molix.core` holds the training contracts (`steps/`, `state`,
`hook`, `losses`, `metrics`); a training-time numerical measurement with no
I/O belongs there. Rejected: `src/molix/profiler/` — its charter is explicitly
performance ("identifying performance bottlenecks … No Trainer, no hooks"),
while this is numerical evidence with a hook-driven cadence. Its
`XProfiler` + immutable `XResult` shape is nevertheless the pattern followed
here (measure → frozen result → caller renders).

### Reuse decision

No `librarian_report` was supplied with this request; the scan below was
performed directly against the tree and every candidate is resolved.

- `reuse molix.core.checkpoint.rng.capture_rng_states / restore_rng_states` —
  probe RNG preservation; no second RNG helper.
- `reuse molix.core.hook.ScalarHook` — hook base and `scalar_keys` contract.
- `reuse molix.core.losses.ScaleNormalizedLoss / WeightedLoss` — the injected
  losses and the unit-weight object this audit justifies; the audit never
  combines or re-weights losses itself.
- `reuse molix.hooks.GradClipHook` — replaces the script's inline
  `clip_grad_norm_(…, 1.0)`; also the reason the probe does **not** clip.
- `reuse molix.hooks.CheckpointHook(save_best=True, best_metric_name=…,
  best_metric_mode="min")` — replaces the script's hand-rolled `best_state`
  clone/restore.
- `reuse molix.hooks.MetricsHook`, `molix.hooks.EarlyStop(if_nan=True)`,
  `molix.core.trainer.Trainer`, `molix.data.datamodule.DataModuleProtocol` —
  the retirement composition; no new training loop is written anywhere.
- `reuse` the task-mixture batch sampler from `universal-class-i-10-sampler`
  and the multi-task train step from `universal-class-i-11-step` — the script
  composes them; this spec adds no sampler and no step.
- `pattern molix.profiler.ModuleProfiler / ModuleResult` — naming and
  measure→frozen-result shape mirrored; placement deliberately not shared
  (see above).
- `new TaskGradientProbe / TaskGradientSnapshot / GradientBalanceReport /
  GradientBalanceHook` — nothing in-tree decomposes gradients by task or by
  parameter group: `GradClipHook` computes one global norm of the summed
  objective and mutates gradients by design; `molix.profiler` measures time
  and memory; `molix.core.metrics` measures predictions, not gradients.

### Debt named and retired (CLAUDE.md iron law)

1. `scripts/mm_param_learning/espaloma_mm_protocol_train.py` `train()` (:458)
   — flat conformer pool (:483-486) forbidden by checklist §1 ("no undeclared
   source or flat concatenated frame pool enters the loader"), hand-rolled
   epoch loop and early stop (:493, :554), inline clipping (:508), inline
   best-state clone (:531). **Retired here** by a script-level composition:
   the chain's task-mixture sampler behind a `DataModuleProtocol` object,
   `Trainer(model, loss_fn=…, optimizer_factory=lambda p:
   torch.optim.Adam(p, lr=args.lr), train_step=<multi-task step>,
   hooks=[GradClipHook(1.0), MetricsHook, CheckpointHook(save_best=True),
   EarlyStop(if_nan=True), GradientBalanceHook(...)])`. No library façade is
   added — composition stays in the script, as CLAUDE.md requires.
   **The pinned Espaloma protocol is not re-fit by this change**: objective
   (`MSE(E_pred, E_ref)`, absolute kcal/mol), Adam lr, batch size, split seed
   2666 / 8:1:1, `eval_every` cadence, metric definitions and every CLI flag
   (including `--early-stop-patience`, consumed by
   `run_class_i_espaloma_scale_gh200.sbatch`) stay behaviourally identical;
   only the loop structure changes. `eval_split` is kept as the script's final
   test-split reporter. The `GradientBalanceHook` is attached opt-in behind a
   new `--grad-audit-steps N` flag (`0` = off); on this single-task protocol
   run it yields the checklist §2 per-group finite/nonzero-gradient table
   rather than a multi-task verdict, and `balanced()` is not called.
2. `scripts/mm_param_learning/train_phalkethoh_mm_energy.py` — the same
   anti-pattern (`_run_epoch` :64, `for epoch in range` :145) in a bond-only
   "Validation B2 smoke" trainer with **zero in-repo references**, fully
   superseded by the protocol trainer. **Deleted here**, not left behind.
3. `molix` has no patience-based early stop (only
   `EarlyStop(if_nan=True)`), so `--early-stop-patience` is re-homed as a
   ~20-line script-local `_ValPatienceStop(BaseHook)` that raises to unwind
   like `EarlyStop` does. This is named, not hidden: promoting a general
   patience hook into `molix.hooks` is a separate spec (see Out of scope), and
   the script carries a comment pointing at it.

### Evidence hand-off (no façade)

`report.to_dict()` is plain JSON. The **script** — never the primitive —
writes `<run>/artifacts/grad_balance_audit.json` and calls
`ctx.set_result("grad_balance_audit", report.to_dict())` inside the existing
molexp `Run.start()` block, alongside the existing `setup` / `final` results
and `MetricsWriter` scalars. `molix.core.audit` imports nothing from molexp,
`molix.io`, or the filesystem.

## Files to create or modify

- `src/molix/core/audit.py` (new) — `TaskGradientSnapshot`,
  `TaskGradientProbe`, `GradientBalanceReport`.
- `src/molix/hooks/audit.py` (new) — `GradientBalanceHook`.
- `src/molix/hooks/__init__.py` — re-export `GradientBalanceHook`
  (alphabetized `__all__`).
- `tests/test_molix/test_core/test_audit.py` (new).
- `tests/test_molix/test_hooks/test_audit.py` (new).
- `scripts/mm_param_learning/espaloma_mm_protocol_train.py` — replace the
  `train()` path with the Trainer composition; add `--grad-audit-steps`;
  keep CLI surface and molexp recording.
- `scripts/mm_param_learning/train_phalkethoh_mm_energy.py` (delete).
- `regressions/universal-class-i-13-grad-audit.py` (new).

## Tasks

- [ ] Write failing unit tests for `TaskGradientProbe` (`tests/test_molix/test_core/test_audit.py` → `TestTaskGradientProbe`)
- [ ] Implement `TaskGradientProbe` + `TaskGradientSnapshot` in `src/molix/core/audit.py` (new) with Google-style docstrings carrying gradient-norm units and the DOI/arXiv references
- [ ] Write failing unit tests for `GradientBalanceReport` (`tests/test_molix/test_core/test_audit.py` → `TestGradientBalanceReport`)
- [ ] Implement `GradientBalanceReport` (`ratios`, `imbalance`, `balanced`, `to_dict`) in `src/molix/core/audit.py`
- [ ] Write failing unit tests for `GradientBalanceHook` (`tests/test_molix/test_hooks/test_audit.py` → `TestGradientBalanceHook`)
- [ ] Implement `GradientBalanceHook` in `src/molix/hooks/audit.py` (new) and re-export it from `src/molix/hooks/__init__.py`
- [ ] Replace the hand-rolled `train()` in `scripts/mm_param_learning/espaloma_mm_protocol_train.py` with the sampler + multi-task step + `Trainer` + hooks composition, preserving every CLI flag and the molexp recording path, and add `--grad-audit-steps`
- [ ] Delete the superseded flat-pool smoke trainer `scripts/mm_param_learning/train_phalkethoh_mm_energy.py`
- [ ] Add regression example `regressions/universal-class-i-13-grad-audit.py` (public API only; hard-coded goldens, no third-party runtime)
- [ ] Run full check + test suite

## Testing strategy

Unit tests only under `tests/`, mirroring source paths
(`src/molix/core/audit.py` → `tests/test_molix/test_core/test_audit.py`;
`src/molix/hooks/audit.py` → `tests/test_molix/test_hooks/test_audit.py`),
types mirrored (`TaskGradientProbe` → `TestTaskGradientProbe`). Each test
targets a single method. No e2e under `tests/`.

**Analytic two-task fixture (shared trunk, hard-coded goldens).** A toy
`nn.Module` with three `nn.Linear(1, 1, bias=False)` children —
`trunk` (`w_t = 2.0`), `head_a` (`w_a = 1.0`), `head_b` (`w_b = 1.0`) —
`forward(batch) → {"a": head_a(trunk(x)), "b": head_b(trunk(x))}` with
`x = [[1.0]]`; losses `L_a = a²` and `L_b = 100·b²` (both labelled unit `"1"`).
Closed form at `h = 2.0`, `a = b = 2.0`:

| group | `g_{a,G}` | `g_{b,G}` | `I_G` |
|---|---|---|---|
| `trunk` | `∂L_a/∂w_t = 2·a·w_a·x = 4.0` | `∂L_b/∂w_t = 200·b·w_b·x = 400.0` | **100.0** (shared) |
| `head_a` | `2·a·h = 8.0` | `0.0` | `null` (task-specific) |
| `head_b` | `0.0` | `200·b·h = 800.0` | `null` (task-specific) |

with `L_a = 4.0`, `L_b = 400.0`. All asserted to `atol=1e-9` in float64.

`TestTaskGradientProbe` — happy path (the table above, one batch per task);
state restoration (`state_dict` tensors `torch.equal` before/after, a
pre-seeded `.grad` restored exactly, a `None` grad still `None`,
`model.training` restored); refusal cases (task in `losses` never sampled →
`ValueError`; unknown label in the stream → `ValueError`; non-finite loss →
`ValueError` naming the task).

`TestGradientBalanceReport` — `ratios()` normalizes to the per-group max
(`trunk`: `a = 0.01`, `b = 1.0`); `imbalance()` gives `trunk = 100.0` and
`null` for both heads; `balanced(max_ratio=10.0) is False` and
`balanced(max_ratio=1000.0) is True`; `balanced()` raises for a single-task
report, for a report with no shared group, and for mixed unit labels;
`add()` rejects a duplicate label; `json.loads(json.dumps(report.to_dict()))
== report.to_dict()` with `schema` and `units` present.

`TestGradientBalanceHook` — `on_train_start` records the `step_0` snapshot;
`on_train_batch_end` records `step_100` at the default `at_steps=(0, 100)`
and at nothing else; the hook writes `state["train"]["grad_imbalance"]`
nested (a slash write would raise) and declares it in `scalar_keys`. Driven
by a real `TrainState` plus a minimal trainer stub exposing `.model`.

**Regression example** — `regressions/universal-class-i-13-grad-audit.py`,
public API only (`from molix.core.audit import TaskGradientProbe,
GradientBalanceReport`), rebuilding the same analytic fixture, asserting the
hard-coded goldens (`4.0 / 400.0 / 8.0 / 800.0`, `I_trunk = 100.0`,
`balanced(10.0) is False`, `balanced(1000.0) is True`), and JSON round-tripping
the report. Goldens are analytic (closed-form derivatives written in the
module docstring, 2026-08-14) — no third-party oracle is used, imported, or
subprocessed at any time.

## Out of scope

- Running the GH200 calibration, or any training submission — the
  `.claude/notes/production-training-checklist.md` gate is untouched by this
  spec and no launcher behaviour changes.
- Registering the `universal-class-i-v0-calibration-gh200` molexp experiment,
  its slug constant, or its sbatch launcher (separate calibration spec); this
  spec only makes the evidence dict available for it.
- Any GO/NO-GO decision or checklist sign-off.
- MolHub dataset composition (closed in MolNex; lives in MolHub as
  `universal-class-i-01-dataset-composition`).
- Adaptive task weighting (GradNorm arXiv:1711.02257, uncertainty weighting
  arXiv:1705.07115) — recorded escalation path, not v0. Alternatives
  considered and rejected for v0 precisely because this audit is the cheaper
  falsifiable test of the unit-weight assumption.
- A library-level patience `EarlyStop` hook — a real gap, named in Design §3
  above, carried in the interim by a script-local hook and routed to its own
  spec; explicitly **not** silently dropped.
- Per-batch gradient *conflict* metrics (pairwise cosine similarity, PCGrad
  surgery) — a different question from magnitude balance; considered and
  deferred rather than folded in.
- Changing the pinned Espaloma protocol numbers (objective, lr, batch size,
  split, metrics) while retiring its loop.
