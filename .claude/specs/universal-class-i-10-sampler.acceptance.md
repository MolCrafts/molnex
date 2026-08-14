---
slug: universal-class-i-10-sampler
criteria:
  - id: ac-001
    summary: Every yielded batch is single-task — flat frame pool unreachable
    type: code
    pass_when: |
      A test in tests/test_molix/test_data/test_sampler.py iterates a
      TaskMixtureBatchSampler over a multi-task fixture and asserts, for every
      batch, that {task_labels[i] for i in batch} has exactly one element and
      len(batch) >= 1; no batch mixes two task labels.
    status: pending
  - id: ac-002
    summary: Sampling units are never split across batches (scan integrity)
    type: code
    pass_when: |
      For a fixture containing multi-frame torsion-scan units, a test asserts
      each unit's full frame-index set appears within exactly one batch of the
      plan, including a unit whose aggregate cost exceeds the budget (kept as a
      whole-unit singleton batch, not split and not dropped), and that exactly
      one UserWarning naming that oversize unit (task label + group id) is
      emitted by the sampler — TokenBudgetPacker itself stays silent.
    status: pending
  - id: ac-003
    summary: Realized batch mixture matches the configured mixture within the apportionment bound
    type: code
    pass_when: |
      For the Stage B {equilibrium 0.35, torsion 0.45, relax 0.20} and Stage C
      {equilibrium 0.25, torsion 0.30, relax 0.35, electrostatic 0.10} mixtures
      over a plan of >= 200 batches, a test asserts
      abs(batches_per_task()[t] - p_t * len(sampler)) <= 1 for every task where
      no anti-starvation override fired, and <= 1 + K for donor tasks when K
      rare tasks were rescued.
    status: pending
  - id: ac-004
    summary: Rare tasks with nonzero probability never get zero batches
    type: code
    pass_when: |
      With Stage C and a deliberately scarce electrostatic task (p=0.10, few
      units) at a small epoch length, a test asserts
      batches_per_task()["electrostatic"] >= 1 and that the donating task's
      deviation stays within the documented 1 + K bound.
    status: pending
  - id: ac-005
    summary: Token budgets honoured via the 09 packer; no second packing loop exists
    type: code
    pass_when: |
      (1) A test asserts every batch's summed atom and edge counts are within
      the sampler's max_atoms / max_edges budgets except documented
      oversize-unit singletons; and (2) a source-level assertion over
      src/molix/data/sampler.py shows the next-fit (single open bin) token
      packing loop exists exactly once — inside the universal-class-i-09-packer
      primitive TokenBudgetPacker — and that TaskMixtureBatchSampler contains no
      second packing loop, delegating instead to one transient TokenBudgetPacker
      per task built over that task's aggregate unit-cost vectors.
    status: pending
  - id: ac-006
    summary: Plan construction and iteration never unpack a sample
    type: code
    pass_when: |
      Using the existing _CountingDataset pattern, a test asserts
      dataset.getitem_calls == 0 after constructing TaskMixtureBatchSampler and
      consuming one full epoch; unit costs come from atom_counts / edge_counts
      pointer arithmetic only.
    status: pending
  - id: ac-007
    summary: Hard-coded golden plan reproduces for a fixed seed
    type: code
    pass_when: |
      A test asserts list(sampler) equals a hard-coded list[list[int]] and that
      batches_per_task() / frames_per_task() equal hard-coded dicts, for the
      documented fixture at seed=20260814.
    status: pending
  - id: ac-008
    summary: Determinism, pickle round-trip, and seed+epoch resume derivability
    type: code
    pass_when: |
      Tests assert (a) two instances with identical arguments yield identical
      plans, (b) seed vs seed+1 yield different plans, (c) iterating one
      instance twice yields the same plan, (d)
      pickle.loads(pickle.dumps(sampler)) yields an identical plan, and (e) a
      fresh instance built with seed+1 reproduces the epoch-1 plan of a
      sampler advanced from seed, mirroring datamodule.py:238 semantics.
    status: pending
  - id: ac-009
    summary: Eager ValueErrors name the offending argument, its value, and the fix
    type: code
    pass_when: |
      pytest.raises(ValueError) tests pass for each of: neither max_atoms nor
      max_edges given (message naming both arguments) or a given budget <= 0;
      mixture not summing to 1.0 within 1e-9; a probability outside [0, 1] or an
      empty mixture; a task label present in the split but absent from mixture;
      a task with p > 0 and zero units; len(task_labels) or len(group_ids) !=
      len(dataset); empty or out-of-range indices — and each message contains
      the argument name and the offending value.
    status: pending
  - id: ac-010
    summary: Task exhaustion emits exactly one UserWarning and renormalizes
    type: code
    pass_when: |
      With a forced batches_per_epoch that exhausts the scarcest task,
      pytest.warns(UserWarning) records exactly one matching warning per
      sampler instance (naming the exhausted task), len(sampler) equals the
      hand-computed realized length, and the post-exhaustion composition
      matches the hand-computed renormalized shares.
    status: pending
  - id: ac-011
    summary: Audit accessors report realized per-task batch and frame counts as copies
    type: code
    pass_when: |
      A test asserts sum(batches_per_task().values()) == len(sampler),
      sum(frames_per_task().values()) == total frames in the plan, per-task
      values match a direct recount of the plan, and mutating a returned dict
      leaves subsequent calls and the plan unchanged; no combined accessor and
      no tolerance/verdict method exists on the class.
    status: pending
  - id: ac-012
    summary: Docstrings carry the no-precedent statement, both citations, and the round-off rules
    type: docs
    pass_when: |
      TaskMixtureBatchSampler and its public methods carry Google-style
      docstrings (Args/Returns/Raises) that state plainly that no published
      task-group-configuration sampler precedent exists, cite
      arXiv:2408.12817 for stage structure only and DOI:10.1039/D2SC02739A for
      molecule-level grouping, list the Stage A/B/C mixtures, and document the
      largest-remainder round-off rule, the anti-starvation rule, the
      batch-share-vs-frame-share distinction, and the seed+epoch convention;
      `ruff check src/ tests/ regressions/` passes.
    status: pending
  - id: ac-013
    summary: Regression example reproduces the hard-coded plan via public API only
    type: runtime
    pass_when: |
      `PYTHONPATH=src python regressions/universal-class-i-10-sampler.py`
      exits 0 and prints OK, reproducing its embedded hard-coded golden plan
      (exact integer index lists) and hard-coded batches_per_task() /
      frames_per_task() dicts for the Stage B mixture at seed=20260814, while
      also asserting single-task batches and whole-unit scan integrity; the
      script imports only molnex public API plus torch/tensordict, records
      command/sha/torch version/date/device in its header, and neither imports
      nor subprocesses any third-party oracle.
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

ac-001 / ac-002 are the structural core: the batch is single-task and the
sampling unit is atomic (a torsion scan is never split, and the oversize
warning is the sampler's since 09's packer is silent by design). ac-003 /
ac-004 bind mixture fidelity and the anti-starvation round-off rule as hard
deterministic bounds rather than statistical ones, which is possible only
because apportionment is largest-remainder rather than i.i.d. multinomial.
ac-005 / ac-006 protect the two reuse boundaries: the next-fit (single open
bin) packing loop stays owned by `universal-class-i-09-packer` — this sampler
builds one transient packer per task over aggregate unit costs — and cost
derivation stays pointer-only. ac-007 / ac-008 lock determinism, pickling and
resume derivability (`seed + epoch`, `datamodule.py:238`). ac-009 / ac-010 bind
message quality for the fatal and degraded paths. ac-011 keeps the audit
surface primitive — counts only; the tolerance comparison (calibration evidence
E2) is the caller's job. ac-012 covers the documentation obligations, including
the explicit statement that no published precedent exists. ac-013 is the
in-repo regression example; ac-014 is the full gate.

No `scientific`-type criterion appears: this sampler computes no physical
quantity, and the scientific claim that per-task gradient contributions track
the configured mixture is validated by the sibling
`universal-class-i-13-grad-audit` spec, not here. No `ui_runtime` criterion and
no UI verification section — nothing here touches a frontend.
