---
slug: streaming-data-tiers-04-scheduler-integration-01-tables
criteria:
  - id: ac-001
    summary: BlockManifest exposes typed units/rows over the sealed document
    type: code
    pass_when: |
      Tests in tests/test_molix/test_store/test_block.py::TestBlockManifest assert,
      on the sealed fixture block: manifest.units is an int equal to the number of
      appended units; manifest.rows is a Mapping[str, int] equal entry-for-entry to
      json.loads(manifest.json)["rows"]; and its "molecule", "molecule_atom" and
      "frame_atom" entries equal hand-computed literals (M, sum N_m, sum n_m*N_m)
      for the two-unit fixture. Both accessors read only the parsed document — no
      new state, no __slots__ entry added to BlockManifest.
    status: pending
  - id: ac-002
    summary: units/rows never hash bytes; verify() stays the only verifier
    type: code
    pass_when: |
      A test monkeypatches molix.store.block._sha256_file to raise RuntimeError,
      then reads BlockManifest.read(sealed_block).units and .rows successfully;
      BlockReader.verify() on the same block still passes with the patch removed.
      git diff for this spec shows no change to BlockReader.verify or to
      BlockManifest._derive_block_digest.
    status: pending
  - id: ac-003
    summary: units/rows are authoritative only on a sealed manifest
    type: code
    pass_when: |
      A test appends >=2 units of the same task to a BlockWriter without sealing and
      asserts BlockManifest.read(root).units < writer.unit_count (the manifest is
      rewritten only on schema freeze, task registration and seal), and that both
      accessor docstrings name `sealed` as the authority.
    status: pending
  - id: ac-004
    summary: FrameLabelTable round-trips both label representations
    type: code
    pass_when: |
      tests/test_molix/test_batch/test_sampler.py::TestFrameLabelTable passes:
      from_labels(TASK_LABELS, GROUP_IDS) yields (F,) integral code tensors in
      first-appearance order with len(table) == F; task_name(code) and
      group_label(code) return the original strings; group_label falls back to
      str(code) when no group-name dictionary was supplied; codes_at(indices)
      returns the two Python int lists for exactly those indices, in the given
      order.
    status: pending
  - id: ac-005
    summary: FrameLabelTable rejects bad tables with ValueError naming key and remedy
    type: code
    pass_when: |
      One pytest.raises(ValueError) test per case, each asserting the message names
      the offending argument and a remedy: unequal task/group table lengths (message
      contains both argument names and both lengths); a 2-D, floating-dtype, or
      non-CPU code tensor; a negative code; a task code >= len(task_names);
      task_names omitted with integer codes; task_names supplied together with
      string labels.
    status: pending
  - id: ac-006
    summary: Both table forms produce byte-identical plans and audit dicts
    type: code
    pass_when: |
      A test builds TaskMixtureBatchSampler over the Stage B fixture twice at
      seed=TASK_SEED — once with the string sequences, once with the equivalent
      (F,) code tensors plus task_names — and asserts list(sampler_a) ==
      list(sampler_b), batches_per_task() equal, and frames_per_task() equal. The
      oversize warning text names "torsion/tor_a" in the string form and in the code
      form when group names are supplied.
    status: pending
  - id: ac-007
    summary: One sampler, no representation fork downstream of construction
    type: code
    pass_when: |
      A source-inspection test asserts that inspect.getsource of _validate_tables,
      _bucket_units, _pack_tasks, _interleave, batches_per_task and frames_per_task
      contains no isinstance(...) check against str, Sequence or torch.Tensor (the
      dispatch lives only in FrameLabelTable), that the module still contains
      "current.append(" exactly once and only inside TokenBudgetPacker, and that
      TaskMixtureBatchSampler source still shows no "batches.append(current)" or
      "totals =" marker.
    status: pending
  - id: ac-008
    summary: Sampler retains no (F,)-length table; audit dicts are mutation-proof
    type: code
    pass_when: |
      A test constructs the sampler from code tensors, mutates those tensors in
      place afterwards, and asserts list(sampler), batches_per_task() and
      frames_per_task() are unchanged; a second assertion scans vars(sampler) (and
      any __slots__) and finds no retained object whose len() equals len(dataset);
      the existing pickle round-trip test (test_pickle_round_trip_gives_an_identical_plan)
      stays green.
    status: pending
  - id: ac-009
    summary: Existing str-sequence goldens and the espaloma caller keep working verbatim
    type: code
    pass_when: |
      git diff for this spec leaves test_hard_coded_golden_plan,
      test_hard_coded_golden_audit_dicts, test_task_labels_length_mismatch_raises_naming_both_lengths,
      test_group_ids_length_mismatch_raises_naming_both_lengths and
      scripts/mm_param_learning/espaloma_mm_protocol_train.py unmodified, all four
      tests pass, and a test constructs TaskMixtureBatchSampler exactly as
      EnergyTaskSamplerSource.batch_sampler does (tuple[str, ...] task_labels and
      group_ids, indices=range(len(dataset)), single-task mixture) and obtains a
      non-empty plan.
    status: pending
  - id: ac-010
    summary: Crossover bench measures both representations and records a verdict
    type: performance
    evaluator_hint: "script: scripts/batch/bench_plan_tables.py"
    pass_when: |
      `python scripts/batch/bench_plan_tables.py` exits 0 and prints, for each
      measured F in {1e6, 1e7, 1e8} and both representations (list-form, code-form),
      table-build seconds, plan-build seconds and peak RSS (from the cell's own
      subprocess VmHWM), plus a verdict line naming the smallest F at which
      list-form peak RSS >= 2x code-form peak RSS or stating "no crossover below
      <F_max>"; the run aborts loudly if the two representations produce different
      plans, and skips (with a printed projection) any cell exceeding
      --rss-budget-gb. The printed numbers, host, torch version and date appear in
      .claude/notes/notes.md under a dated topic note, and the chosen representation
      is written into the spec's Domain basis successor note.
    status: pending
  - id: ac-011
    summary: Regression reproduces the hard-coded block-plan goldens through both forms
    type: runtime
    pass_when: |
      `PYTHONPATH=src python regressions/streaming-data-tiers-04-scheduler-integration-01-tables.py`
      exits 0: it writes and seals a block with BlockWriter, opens a BlockReader,
      constructs TaskMixtureBatchSampler once from decoded string labels and once
      from reader.task_codes / reader.group_codes / reader.task_names at the same
      seed, and asserts both plans equal the hard-coded golden list[list[int]] and
      both audit dicts equal the hard-coded goldens, plus manifest.units and
      manifest.rows equal hard-coded literals. No third-party package is imported or
      subprocessed at run time; the docstring records capture command, commit, torch
      version and date.
    status: pending
  - id: ac-012
    summary: New public symbols carry Google docstrings with shapes and dtypes
    type: docs
    pass_when: |
      FrameLabelTable (class, __init__, from_labels, task_names, task_name,
      group_label, codes_at) and BlockManifest.units / BlockManifest.rows each carry
      Google-style docstrings with Args/Returns/Raises where applicable and explicit
      ``(F,)`` shape plus dtype annotations; the TaskMixtureBatchSampler class
      docstring's "the instance holds only ..." inventory and its task_labels /
      group_ids / task_names Args entries describe the widened contract; ruff check
      and ruff format --check pass on src/, tests/, scripts/, regressions/.
    status: pending
  - id: ac-013
    summary: Full check and test suite pass
    type: runtime
    pass_when: |
      `python -m ruff check src/ tests/ scripts/ regressions/`,
      `python -m ruff format --check src/ tests/ scripts/ regressions/`,
      `OMP_NUM_THREADS=1 python -m pytest tests/ -q -n 4` and
      `python scripts/check_test_mirror.py` all exit 0 on the branch containing
      this spec (login-node gate discipline: -n 4, OMP_NUM_THREADS=1).
    status: pending
---

# Acceptance criteria

ac-001..ac-003 fence the block-manifest surface: the two accessors are a typed read of digest-covered document fields (ac-001), they never touch column bytes so `BlockReader.verify()` remains the single verifier (ac-002), and their staleness before `seal()` is a tested contract rather than a footnote a successor discovers in production (ac-003).

ac-004..ac-008 are the generalization contract. ac-004/ac-005 pin the new table type's behaviour and its error quality; ac-006 is the binding equivalence bar — two representations, one plan; ac-007 is the structural "one sampler, never a fork" guard, phrased over source inspection because that is the failure mode (a second code path added later under time pressure); ac-008 records the actual R1 win, namely that the sampler stops carrying a per-frame table at all.

ac-009 is the binding back-compat bar named by the split verdict: the pre-existing string-sequence goldens and the in-tree `EnergyTaskSamplerSource` caller must keep working with zero edits.

ac-010 is the only `performance` criterion; it produces the number `02-view` and `03-plan` are allowed to design against. ac-011 is the delivery-time runtime witness over a real sealed block, with goldens embedded as literals. ac-012 and ac-013 are the documentation and full-gate criteria.
