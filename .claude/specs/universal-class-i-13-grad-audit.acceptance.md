---
slug: universal-class-i-13-grad-audit
criteria:
  - id: ac-001
    summary: Probe records per-task loss and per-group gradient norms (analytic golden)
    type: code
    evaluator_hint: "pytest tests/test_molix/test_core/test_audit.py -k TaskGradientProbe"
    pass_when: |
      TestTaskGradientProbe builds the float64 shared-trunk fixture (trunk
      w=2.0, head_a w=1.0, head_b w=1.0, x=[[1.0]], L_a=a**2,
      L_b=100*b**2) and TaskGradientProbe.run over one batch per task returns
      a TaskGradientSnapshot with loss {"a": 4.0, "b": 400.0} and grad_norm
      {"a": {"trunk": 4.0, "head_a": 8.0, "head_b": 0.0},
       "b": {"trunk": 400.0, "head_a": 0.0, "head_b": 800.0}},
      all within atol=1e-9, and n_batches == {"a": 1, "b": 1}.
    status: pending
  - id: ac-002
    summary: Probe is read-only w.r.t. model, gradients, RNG and mode
    type: code
    pass_when: |
      A test seeds a non-None .grad on one parameter, leaves another at None,
      sets model.train(), snapshots state_dict and RNG, runs
      TaskGradientProbe.run, and asserts afterwards: every state_dict tensor
      torch.equal to its pre-run clone, the seeded .grad torch.equal to its
      pre-run value, the None .grad still None, model.training unchanged, and
      torch.random.get_rng_state() equal to the pre-run state. No optimizer is
      constructed anywhere in src/molix/core/audit.py (asserted by source scan
      for "optimizer" / ".step()").
    status: pending
  - id: ac-003
    summary: Probe refuses to fabricate evidence
    type: code
    pass_when: |
      TaskGradientProbe.run raises ValueError for (a) a task present in
      `losses` that received no batch in the stream, (b) an unknown task label
      appearing in the stream, and (c) a non-finite loss or non-finite group
      norm, with the offending task (and group, for case c) named in the
      message. No case returns a snapshot.
    status: pending
  - id: ac-004
    summary: Report ratio table normalizes per group; task-specific groups are null
    type: code
    pass_when: |
      For the ac-001 snapshot, GradientBalanceReport.ratios() gives
      trunk -> {"a": 0.01, "b": 1.0} (atol 1e-9) and each head's max task at
      1.0, while imbalance() gives {"trunk": 100.0, "head_a": None,
      "head_b": None}.
    status: pending
  - id: ac-005
    summary: balanced() verdict thresholds and refusal semantics
    type: code
    pass_when: |
      On the ac-001 report, balanced(max_ratio=10.0) is False and
      balanced(max_ratio=1000.0) is True. balanced() raises ValueError for a
      report with fewer than two probed tasks, for a report whose groups are
      all task-specific (no shared group), and for a report whose task unit
      labels are not all identical. add() raises ValueError on a duplicate
      snapshot label.
    status: pending
  - id: ac-006
    summary: Report is JSON round-trippable evidence with schema and units
    type: code
    pass_when: |
      json.loads(json.dumps(report.to_dict())) == report.to_dict() for a
      two-snapshot report, and to_dict() contains "schema" ==
      "molix.grad_balance/1", a "snapshots" list carrying label, at_step,
      loss, grad_norm, n_batches and units, plus "ratios" and "imbalance"
      keys. src/molix/core/audit.py imports nothing from molexp,
      molix.io, pathlib or json-writing helpers (source scan).
    status: pending
  - id: ac-007
    summary: Hook snapshots at step 0 and the configured step, writes nested scalar
    type: code
    evaluator_hint: "pytest tests/test_molix/test_hooks/test_audit.py"
    pass_when: |
      With default at_steps=(0, 100), GradientBalanceHook.on_train_start
      appends a snapshot labelled "step_0" and on_train_batch_end appends
      exactly one further snapshot labelled "step_100" when
      state["global_step"] reaches 100 (and none at other steps). The hook
      writes state["train"]["grad_imbalance"] via the nested form, declares
      ("train", "grad_imbalance") in scalar_keys, and
      `from molix.hooks import GradientBalanceHook` succeeds.
    status: pending
  - id: ac-008
    summary: Flat-pool trainer path retired from espaloma_mm_protocol_train.py
    type: code
    pass_when: |
      scripts/mm_param_learning/espaloma_mm_protocol_train.py contains no
      `for epoch in range(`, no `train_pool`, no
      `torch.nn.utils.clip_grad_norm_`, no `opt.step()` / `optimizer.step()`
      and no hand-rolled best_state clone; it instantiates
      molix.core.trainer.Trainer with the chain's task-mixture sampler behind
      a DataModuleProtocol object, the multi-task train step, and hooks
      including GradClipHook and CheckpointHook(save_best=True). No new
      library façade (no molix/molpot symbol added by this task).
    status: pending
  - id: ac-009
    summary: CLI surface and molexp recording preserved across the retirement
    type: code
    pass_when: |
      `python scripts/mm_param_learning/espaloma_mm_protocol_train.py --help`
      still accepts every flag used by
      scripts/mm_param_learning/run_class_i_espaloma_scale_gh200.sbatch
      (--data-root --energy-scope --n-mols --max-confs --selection
      --use-official-splits --epochs --batch-size --hidden --n-blocks --lr
      --eval-every --early-stop-patience --device --workspace --project
      --experiment --run-id --dataset-coordinate) plus the new
      --grad-audit-steps (default 0), and the molexp path still calls
      Run.start(), ctx.set_result("final"/"setup"/"checkpoint"),
      mark_succeeded(), MetricsWriter scalars and harvest_run; with
      --grad-audit-steps > 0 it additionally writes
      <run>/artifacts/grad_balance_audit.json and
      ctx.set_result("grad_balance_audit", <report dict>).
    status: pending
  - id: ac-010
    summary: Superseded flat-pool smoke trainer deleted with no dangling references
    type: code
    pass_when: |
      scripts/mm_param_learning/train_phalkethoh_mm_energy.py no longer
      exists and a repo-wide search for "train_phalkethoh_mm_energy" returns
      no hits in src/, tests/, scripts/, regressions/, docs/ or .claude/.
    status: pending
  - id: ac-011
    summary: Audit module documented per Google style with units and references
    type: docs
    pass_when: |
      Every public symbol in src/molix/core/audit.py and
      src/molix/hooks/audit.py carries a Google-style docstring with Args /
      Returns / Raises; the module docstring of audit.py states the
      g_{t,G} = ||dL_t/dtheta_G||_2 definition, the ratio and imbalance
      formulas, the dimensional precondition (identical task loss units,
      ScaleNormalizedLoss), and cites DOI 10.1021/acs.jctc.1c00571 and
      arXiv:2307.07085, marking arXiv:1711.02257 / arXiv:1705.07115 as the
      non-v0 escalation path.
    status: pending
  - id: ac-012
    summary: Regression example reproduces hard-coded gradient-balance goldens
    type: runtime
    evaluator_hint: "PYTHONPATH=src python regressions/universal-class-i-13-grad-audit.py"
    pass_when: |
      `PYTHONPATH=src python regressions/universal-class-i-13-grad-audit.py`
      exits 0, reproducing from the public API alone the hard-coded goldens
      loss {"a": 4.0, "b": 400.0}, grad norms trunk 4.0 / 400.0, head_a 8.0,
      head_b 800.0, imbalance trunk 100.0 (all atol 1e-9),
      balanced(max_ratio=10.0) False, balanced(max_ratio=1000.0) True, and a
      JSON round-trip of report.to_dict(). The script imports only torch and
      molix; it never imports or subprocesses espaloma, openff, openmm or any
      other third-party oracle.
    status: pending
  - id: ac-013
    summary: Full check and unit-test suite pass
    type: runtime
    evaluator_hint: "build.check then build.test from CLAUDE.md mol_project"
    pass_when: |
      `ruff check src/ tests/ scripts/ regressions/ && ruff format --check
      src/ tests/ scripts/ regressions/ && ty check src/
      --exit-zero-on-warning` succeeds and
      `python -m pytest tests/ -q -n 12 --dist worksteal` reports no failures
      and no new skips.
    status: pending
---

# Acceptance criteria

- **ac-001 — ac-003** bind the measurement primitive: the analytic golden
  table, the read-only guarantee (state, gradients, RNG, mode), and the three
  refusal paths that keep the audit from manufacturing evidence.
- **ac-004 — ac-006** bind the ratio table: per-group normalization,
  `null` for task-specific groups, the one-decade verdict with its three
  refusal cases, and the JSON evidence shape (with the "no molexp writes from
  the primitive" boundary asserted by source scan).
- **ac-007** binds the cadence hook (init + step N, default N = 100) and the
  nested `TrainState` write contract.
- **ac-008 — ac-010** bind the debt retirement: the flat conformer pool and
  hand-rolled loop are gone, the pinned protocol CLI and molexp recording
  survive untouched, and the orphan smoke trainer is deleted rather than left.
- **ac-011** binds documentation, units and citations.
- **ac-012 — ac-013** bind delivery: the offline-golden regression example and
  the full check + unit-test suite.
