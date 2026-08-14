---
slug: streaming-data-tiers-04-scheduler-integration-03-plan
criteria:
  - id: ac-001
    summary: Global plan drives consumption unchanged when K >= n_blocks
    type: code
    pass_when: |
      A test in tests/test_molix/test_batch/test_block_plan.py builds a
      BlockFrameView whose max_open >= len(block_ids) and whose planned
      batches are all block-pure, and asserts
      [list(b) for b in BlockPlanSource(...).batch_sampler(view, seed=S)] equals
      [list(b) for b in TaskMixtureBatchSampler(view, indices=range(len(view)),
      <same tables/mixture/max_atoms>, seed=S)] batch-for-batch, with
      plan.boundary_splits == 0.
    status: pending
  - id: ac-002
    summary: Every emitted batch is block-pure; straddlers divided, counted, never dropped
    type: code
    pass_when: |
      A test with a planned batch spanning two blocks asserts each emitted batch
      maps to exactly one block ordinal via view.locate, that concatenating the
      emitted batches is a permutation (multiset equality) of the concatenated
      planned batches — nothing dropped or duplicated — that no sampling unit
      (group code) is split across two emitted batches, and that
      plan.boundary_splits equals the number of divided planned batches.
    status: pending
  - id: ac-003
    summary: K-window ordering bounds residency and degenerates correctly
    type: code
    pass_when: |
      Tests assert: with max_open == 1 the emitted block sequence is
      strictly block-major (each block's batches contiguous); with
      1 < K < n_blocks the visited blocks form consecutive windows of at most K
      distinct blocks and within a window plan order is preserved; with K >= n_blocks
      the order is the plan order (identity).
    status: pending
  - id: ac-004
    summary: Visit-order export is a stdlib-json document; no molhub import
    type: code
    pass_when: |
      A test calls BlockPlan.write_visit_order(tmp_path/"v.json"), re-reads it with
      json.load, and asserts format == "molix.visit-order/1", split/seed/window
      match the plan, order equals plan.visit_order() as a list of block identity
      strings, and coverage rows carry block/units/frames/batches; the existing
      tests/test_repository_boundaries.py guard stays green (no molhub import
      anywhere under src/).
    status: pending
  - id: ac-005
    summary: Coverage report accounts every planned block, unit, frame and batch
    type: code
    pass_when: |
      A test asserts the BlockCoverage rows cover exactly the blocks touched by the
      plan, that sum(row.frames) equals the total planned frame count, sum(row.batches)
      equals len(plan), and each row.units equals the number of distinct group codes
      among that block's planned frames; BlockPlan.__post_init__ raises ValueError on
      a deliberately inconsistent coverage tuple.
    status: pending
  - id: ac-006
    summary: Block collate stamps ("graphs","task") int64 and asserts single-task
    type: code
    pass_when: |
      A test in tests/test_molix/test_batch/test_datamodule.py::TestBlockCollation
      collates a batch of frames from one block and asserts
      batch["graphs","task"] is a (B,) int64 tensor equal to the block's
      reader.task_codes for those frames; a second test builds a frame list spanning
      two task codes of one block and asserts pytest.raises(ValueError) whose message
      names the block and the single-task requirement.
    status: pending
  - id: ac-007
    summary: Mixed-block batch refused; no skip, no substitution
    type: code
    pass_when: |
      A test hands _BlockCollateFn a batch of frames from two different blocks and
      asserts pytest.raises(ValueError) naming both block identities and pointing at
      BlockPlan; a second test asserts that when view.reader(b) raises for an
      unavailable block, the error propagates with the block identity named and no
      batch from any other block is produced in its place.
    status: pending
  - id: ac-008
    summary: DataModule routes a block view to the block collate path only
    type: code
    pass_when: |
      Tests assert _resolve_collation returns (_IndexDataset, _BlockCollateFn) for a
      BlockFrameView (block route probed before the packed route), still returns
      (_IndexDataset, _PackedCollateFn) for a PackedCache-backed dataset, still
      returns (dataset, _CollateFn) for a non-packed dataset, and that
      batch_sampler_source under a monkeypatched _is_distributed() still raises the
      existing DDP ValueError.
    status: pending
  - id: ac-009
    summary: Post-collate contract and picklability match the packed path
    type: code
    pass_when: |
      A test registers a marker BatchTask and a non-default ftype and asserts the
      block-path batch shows the task's effect, all floating leaves equal the captured
      ftype, and batch["graphs","task"] stays int64; a pickle.dumps/loads round-trip of
      _BlockCollateFn produces correct batches and its __getstate__ contains no payload
      tensors.
    status: pending
  - id: ac-010
    summary: Eager refusals — edge budget, wrong dataset, task in atom_level
    type: code
    pass_when: |
      pytest.raises(ValueError) for BlockPlanSource(max_edges=...) with a message
      naming BlockReader.edge_counts and the downstream graph builder;
      pytest.raises(TypeError) for batch_sampler() on an object that is not a
      BlockFrameView; pytest.raises(ValueError) for _BlockCollateFn constructed with a
      TargetSchema whose atom_level contains "task".
    status: pending
  - id: ac-011
    summary: Regression reproduces plan, stamping, export and coverage goldens
    type: runtime
    pass_when: |
      `PYTHONPATH=src python regressions/streaming-data-tiers-04-scheduler-integration-03-plan.py`
      exits 0, writes three BlockWriter blocks, plans over them, iterates one
      DataModule epoch offline, and its assertions against the literals embedded in
      the script all hold: emitted batch index lists, visit_order() tuple, coverage
      rows, per-batch ("graphs","task") codes and int64 dtype, per-batch
      graphs.num_atoms and graphs.targets.energy values (exact equality — the fixture
      is closed-form), the JSON document re-parsed with stdlib json, and plan equality
      against a direct TaskMixtureBatchSampler over the same metadata. No network, no
      subprocess, no third-party oracle at runtime.
    status: pending
  - id: ac-012
    summary: New public symbols carry Google docstrings with shapes and count units
    type: docs
    pass_when: |
      BlockFrameView, BlockCoverage, BlockPlan, BlockPlanSource, _block_capable and
      _BlockCollateFn each carry Google-style docstrings with Args/Returns/Raises where
      applicable, tensor shape annotations (e.g. ``(F,)``, ``(B,)``) and explicit count
      units (frames / units / batches / blocks); molix.batch.__init__ exports the four
      public symbols with an alphabetized __all__; `ruff check` passes.
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

ac-001 is the chain's inherited exit criterion from the strategy of record
(`molhub:.claude/specs/streaming-data-tiers-01-strategy.md` § 12): the eager plan of
`src/molix/batch/sampler.py:267` drives consumption unchanged at small scale.
ac-002 and ac-003 are I8 (plan authority): the block-windowed order is a permutation
of the planned batches, block-pure, unit-preserving, and residency-bounded.
ac-004 is the MolHub handoff — a file, not an import (`tests/test_repository_boundaries.py`).
ac-005 is I10 (coverage reported, never implicit).
ac-006 and ac-007 close the binding `("graphs","task")` gap against `TaskRouter`
(`src/molix/core/steps/multitask.py:113`) without revising the block format, and pin
the no-skip / no-substitute rule.
ac-008 to ac-010 constrain the `DataModule` wiring, the post-collate contract shared
with `_PackedCollateFn` (`src/molix/batch/datamodule.py:556`), and error quality.
ac-011 is the runtime regression; ac-012 documentation; ac-013 the full gate.

No `scientific` criterion is emitted: this link introduces no physical quantity or
equation — the sampling-discipline claims (unit integrity, split purity, task purity,
mixture apportionment) are structural and are covered by ac-001, ac-002 and ac-005.
No `ui_runtime` criterion exists (no frontend). The mixture-locality cost of
block-windowed consumption is recorded as risk R2 in the spec body and is inspectable
through `BlockPlan.window` / `BlockPlan.coverage`; it is deliberately not gated here.
