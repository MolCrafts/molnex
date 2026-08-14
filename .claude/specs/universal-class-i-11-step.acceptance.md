---
slug: universal-class-i-11-step
criteria:
  - id: ac-001
    summary: Multi-task steps inherit cadence semantics instead of forking them
    type: code
    pass_when: |
      Tests in tests/test_molix/test_core/test_steps/test_multitask.py assert
      MultiTaskTrainStep.on_train_batch is DefaultTrainStep.on_train_batch and
      MultiTaskEvalStep.on_eval_batch is DefaultEvalStep.on_eval_batch, and the
      git diff for src/molix/core/steps/train.py and eval.py changes only the
      loss call site, the relocation of `assert trainer.loss_fn is not None`
      into the new compute_loss, and docstrings — no change to the accumulation
      window arithmetic, zero_grad placement, scaler order, optimizer_applied
      computation, or on_after_backward fire point.
    status: pending
  - id: ac-002
    summary: Default compute_loss seam delegates to trainer.loss_fn inside autocast
    type: code
    pass_when: |
      tests/test_molix/test_core/test_steps/test_train.py and test_eval.py pass:
      DefaultTrainStep.compute_loss / DefaultEvalStep.compute_loss return
      trainer.loss_fn(predictions, batch), raise AssertionError when
      trainer.loss_fn is None, and a spy subclass records that on_train_batch /
      on_eval_batch call compute_loss with torch.is_autocast_enabled() matching
      config["use_amp"].
    status: pending
  - id: ac-003
    summary: Router dispatches each single-task batch to its own TaskSpec loss
    type: code
    pass_when: |
      A test in tests/test_molix/test_core/test_steps/test_multitask.py builds
      four batches labelled 0..3 at ("graphs","task") with distinguishable stub
      losses and asserts each batch invokes exactly its own loss (call recorder),
      for both a TensorDict batch and an equivalent plain nested dict.
    status: pending
  - id: ac-004
    summary: Missing, empty, float, unknown, or mixed task labels raise naming ValueError
    type: code
    pass_when: |
      Six pytest.raises(ValueError, match=...) cases pass in
      tests/test_molix/test_core/test_steps/test_multitask.py: missing label
      (message contains "graphs" and "task"), zero-element label, floating-dtype
      label (message contains the dtype), undeclared code 7 (message contains
      "7" and the declared code-to-name map), mixed labels [0, 1] (message
      contains both codes), plus construction-time empty tasks / duplicate name /
      duplicate code.
    status: pending
  - id: ac-005
    summary: Per-task scalars are nested detached device tensors, never flat keys
    type: code
    pass_when: |
      Tests assert state["train"]["loss_<task>"] and state["eval"]["loss_<task>"]
      exist after one train / one eval batch, each isinstance(torch.Tensor) with
      requires_grad False and grad_fn None and not a Python float, that
      state["train"]["loss"] is still written, and that no key containing "/"
      exists at any level of the TrainState.
    status: pending
  - id: ac-006
    summary: Step performs no host sync and no gradient clipping
    type: code
    pass_when: |
      A test monkeypatches torch.Tensor.item and torch.nn.utils.clip_grad_norm_
      to raise, runs one MultiTaskTrainStep.on_train_batch, and it completes;
      a second test with GradClipHook registered shows state["train"]["grad_norm"]
      written by the hook (pre-clip 0-d tensor), not by the step.
    status: pending
  - id: ac-007
    summary: Single-task parity with DefaultTrainStep under accumulation
    type: code
    pass_when: |
      A float64 seeded test with accumulate_grad_batches=2 shows
      MultiTaskTrainStep (one TaskSpec wrapping f) and DefaultTrainStep
      (trainer.loss_fn = f) produce the same optimizer_applied sequence
      [False, True], torch.equal state["train"]["loss"] values per micro-batch,
      the same on_after_backward fire count, and torch.equal model parameters
      after the window closes.
    status: pending
  - id: ac-008
    summary: molix stays molpot-free and exposes no shared weight or reference field
    type: code
    pass_when: |
      A static test reads src/molix/core/steps/multitask.py and asserts no
      "molpot" / "molrep" / "molzoo" import, no "clip_grad_norm_", no ".item()",
      that dataclasses.fields(TaskSpec) names == ("name", "code", "loss"), and
      that no public constructor in the module accepts a weight, scale,
      normalization, or reference_method parameter.
    status: pending
  - id: ac-009
    summary: Eval step keeps a live graph so force and Hessian losses backprop
    type: code
    pass_when: |
      A test with a positions-dependent stub model and default no_grad=False
      shows MultiTaskEvalStep.on_eval_batch returns a loss with requires_grad
      True and writes the detached scalar into state["eval"]; passing
      no_grad=True is forwarded to DefaultEvalStep.
    status: pending
  - id: ac-010
    summary: Log accepts step-advertised per-task scalar keys
    type: code
    pass_when: |
      tests/test_molix/test_core/test_log_hook.py passes: a fake trainer whose
      train_step advertises ("train","loss_hessian") lets Log(1, keys=[that])
      start without raising, while the same key with no advertising step still
      raises ValueError matching "not advertised"; the pre-existing hook-only
      fake-trainer tests remain green.
    status: pending
  - id: ac-011
    summary: Trainer.smoke warms up the configured step's objective
    type: code
    pass_when: |
      tests/test_molix/test_core/test_trainer_smoke.py passes: with a spy step
      exposing compute_loss, Trainer.smoke invokes it once per warmup batch and
      never touches trainer.loss_fn; with a step lacking the seam it falls back
      to loss_fn; a non-finite loss still raises RuntimeError naming the warmup
      step index.
    status: pending
  - id: ac-012
    summary: New public symbols carry Google docstrings with units and v0 constraints
    type: docs
    pass_when: |
      TaskSpec, TaskRouter, MultiTaskTrainStep, MultiTaskEvalStep and the two
      compute_loss seams each carry Google-style docstrings with Args/Returns/
      Raises; the multitask.py module docstring records dimensionless
      post-normalization scalars plus per-task residual units, the frozen-LJ v0
      rule, the separate MBIS reference method and denominator, per-group
      centering, and the constrained-geometry / EEM-topology-only biases with
      the Domain-basis citations; ruff check passes.
    status: pending
  - id: ac-013
    summary: Regression example reproduces hard-coded analytic goldens
    type: runtime
    pass_when: |
      `PYTHONPATH=src python regressions/universal-class-i-11-step.py` exits 0
      and reproduces its in-file hard-coded analytic goldens for
      state["train"]["loss_energy_force"], state["train"]["loss_charge"],
      state["train"]["loss"], state["global_step"] == 1 and a present
      state["train"]["grad_norm"], within atol=1e-12 in float64, with the
      closed-form derivation in the module docstring and no third-party import
      or subprocess at runtime.
    status: pending
  - id: ac-014
    summary: Full check and test suite pass
    type: runtime
    pass_when: |
      `ruff check src/ tests/ scripts/ regressions/ && ruff format --check src/
      tests/ scripts/ regressions/ && ty check src/ --exit-zero-on-warning` and
      `python -m pytest tests/ -q -n 12 --dist worksteal` both exit 0 on the
      branch containing this spec.
    status: pending
---

# Acceptance criteria

ac-001, ac-002 and ac-007 together are the no-fork contract: the multi-task
steps must inherit AMP autocast, gradient accumulation, `optimizer_applied`
scheduler-desync detection and the `on_after_backward` fire point rather than
re-implement them, and must be observationally identical to `DefaultTrainStep`
on a single-task stream. ac-003 and ac-004 bind routing correctness and
error-message quality (a mis-labelled batch must fail loudly, naming the label —
never silently train the wrong objective). ac-005 and ac-006 bind the hot-path
and namespace contracts: nested detached device tensors only, one host sync per
batch, clipping left to `GradClipHook`. ac-008 is the layering and
single-weighting-surface guard (`molix` never imports `molpot`; no shared
`reference_method` / normalization field that could merge the PBE0/def2-TZVPD
charge task with the B3LYP-D3(BJ)/DZVP energetics). ac-009 preserves the live
eval graph needed by force and Hessian tasks. ac-010 and ac-011 close the two
dependency holes this feature rides on (`Log` key advertisement, `Trainer.smoke`
objective drift). ac-012 records the v0 science constraints where an implementer
will actually read them; ac-013 is the public-API regression with analytic
goldens; ac-014 is the full gate.

No `type: scientific` criterion appears here: this spec ships routing
infrastructure, and the v0 physics claims it records (frozen LJ, separated
charge reference method, fixed physical denominators, declared
constrained-geometry bias) are enforced as documentation and code-shape
constraints (ac-008, ac-012) plus the numeric regression (ac-013). Validation
against QM reference data belongs to the training-run calibration evidence in
`.claude/notes/production-training-checklist.md`, not to this step.
