---
slug: universal-class-i-09-packer
criteria:
  - id: ac-001
    summary: TokenBudgetPacker owns packing as two primitives, decoupled from dataset/RNG/warnings
    type: code
    pass_when: |
      src/molix/data/sampler.py defines TokenBudgetPacker with exactly the
      primitives `pack(order) -> list[list[int]]` and `oversize_mask() ->
      torch.Tensor` (bool, shape (n_samples,)) plus `__init__(budgets:
      Sequence[tuple[int, torch.Tensor]])`; the class body references neither
      `warnings`, nor `BaseDataset`/`self._dataset`, nor `make_generator`; a test
      constructs it from bare torch tensors with no dataset and
      tests/test_molix/test_data/test_sampler.py::TestTokenBudgetPacker passes.
    status: pending
  - id: ac-002
    summary: Packing semantics pinned by hand-computed next-fit expectations
    type: code
    pass_when: |
      TestTokenBudgetPacker asserts hard-coded expectations: counts [7,6,5] with
      budget 12 packs to [[0],[1,2]] (next-fit, single open bin); counts [3,20,4]
      with budget 10 packs to [[0],[1],[2]] and emits no warning under
      warnings.simplefilter("error"); a dual-budget case where the edge budget
      closes the batch first matches its hand-computed plan; pack([]) == [];
      oversize_mask() equals the hard-coded bool list for the same counts.
    status: pending
  - id: ac-003
    summary: Packer rejects an empty budget list eagerly with an actionable message
    type: code
    pass_when: |
      pytest.raises(ValueError) passes for TokenBudgetPacker(budgets=[]) and the
      message states that at least one (budget, counts) pair is required.
    status: pending
  - id: ac-004
    summary: Post-refactor batch plans equal the pre-refactor goldens for three configs
    type: code
    pass_when: |
      tests/test_molix/test_data/test_sampler.py::TestSamplerGoldenPlan holds
      module-level literals for the permutation and the batch plan of a
      16-sample varied-count cache under (max_atoms=16), (max_atoms=16,
      max_edges=20) and (max_atoms=8, under pytest.warns(UserWarning)), captured
      from the pre-refactor tree with the capture command, date and torch version
      recorded in a comment; every config asserts both
      list(TokenBudgetBatchSampler(...)) == golden_plan and
      TokenBudgetPacker(pairs).pack(golden_perm) == golden_plan, and all pass.
    status: pending
  - id: ac-005
    summary: TokenBudgetBatchSampler observable surface is unchanged
    type: code
    pass_when: |
      A test asserts sorted(vars(sampler)) == ["_batches", "_dataset",
      "_max_atoms", "_max_edges", "_seed"] (no packer stored) and asserts the
      oversize UserWarning message text verbatim, fired exactly once per
      instance; the pre-existing classes TestSamplerBudgetCompliance,
      TestSamplerCoverage, TestSamplerOversize, TestSamplerDeterminism,
      TestSamplerValidation, TestSamplerNoUnpack and TestSamplerPickle pass with
      no edits and no weakened assertions (git diff shows no deletions or
      relaxations inside those classes).
    status: pending
  - id: ac-006
    summary: Injected batch sampler reaches the loader and receives seed + epoch
    type: code
    pass_when: |
      tests/test_molix/test_data/test_datamodule.py::TestBatchSamplerInjection
      passes: with a stub BatchSamplerSource returning a fixed list of index
      groups, train_dataloader().batch_sampler is that object and
      loader.batch_size is None; iterating the loader yields exactly the injected
      groups as verified by the graphs.U0 identity values; after
      on_epoch_start(k) the stub recorded seed == dm.seed + k.
    status: pending
  - id: ac-007
    summary: Seam guards — DDP rejection and eager ValueErrors on misuse
    type: code
    pass_when: |
      With molix.data.datamodule._is_distributed monkeypatched True, an injected
      source raises ValueError matching (?i)ddp|distributed naming
      batch_sampler_source, and the budget path still raises with both
      max_atoms_per_batch and max_edges_per_batch named; DataModule(...,
      batch_sampler_source=src, max_atoms_per_batch=16) and
      DataModule(..., batch_sampler_source=object()) each raise ValueError from
      __init__ with messages naming the conflict / the required
      `batch_sampler(dataset, *, seed)` method.
    status: pending
  - id: ac-008
    summary: Default DataModule behavior unchanged when the new keyword is omitted
    type: code
    pass_when: |
      TestDataModuleBudgetOptIn, TestDataModuleBackwardCompat,
      TestDataModuleEpochRederivation and TestDataModuleDDP pass unmodified: no
      keyword plus a budget still yields a TokenBudgetBatchSampler batch_sampler,
      and the non-budget loader still shows batch_size==4, RandomSampler,
      BatchSampler, a non-None generator and drop_last False; git diff shows no
      change to the fixed-batch_size branch of train_dataloader other than the
      `_epoch_batch_sampler()` dispatch.
    status: pending
  - id: ac-009
    summary: New public symbols documented Google-style; first-fit wording corrected
    type: docs
    pass_when: |
      TokenBudgetPacker (class + pack + oversize_mask), BatchSamplerSource,
      _epoch_batch_sampler, _reject_ddp_batch_sampler and the
      batch_sampler_source kwarg each carry Google-style docstrings with
      Args/Returns/Raises where applicable, `(n_samples,)` shape annotations and
      an explicit note that counts are dimensionless integers; no docstring in
      src/molix/data/sampler.py still calls the algorithm "first-fit" (grep for
      "first-fit" in that file returns nothing); ruff check passes on both
      modified modules.
    status: pending
  - id: ac-010
    summary: Regression example reproduces hard-coded packing and injection goldens
    type: runtime
    pass_when: |
      `PYTHONPATH=src python regressions/universal-class-i-09-packer.py` exits 0
      and prints its OK line, asserting (a) TokenBudgetPacker over a hard-coded
      count table and explicit order reproduces the hard-coded expected batches
      exactly (next-fit closure plus one oversize singleton) and (b) a DataModule
      with an injected stub source over a temporary PackedCache saw seed == 44
      after on_epoch_start(2) and emitted exactly the injected index groups; the
      script imports only molnex/torch (no third-party oracle, no subprocess).
    status: pending
  - id: ac-011
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

ac-001 through ac-003 bind the extracted unit: two primitives, no hidden coupling to dataset / RNG / warning machinery, and hand-computed packing semantics — ac-002 is the one that actually distinguishes next-fit from first-fit, so a "cleaner" rewrite that reopens closed bins fails here rather than silently changing training batch composition.

ac-004 and ac-005 are the behavior-preservation contract, and they are the reason this spec exists in refactor-only form: goldens captured *before* the change, plus an unchanged instance field set (pickle payload shape) and an unchanged warning text fired exactly once. Any diff in the emitted plan for a fixed seed is a failure, not an improvement.

ac-006 through ac-008 cover the seam: it must pass through, it must carry `seed + epoch`, it must fail closed under DDP and on misuse, and it must be invisible to every existing caller.

ac-009 is the documentation gate and also carries the one honest wording fix (the loop is next-fit, not first-fit). ac-010 is the public-API regression with hard-coded goldens and no live third party. ac-011 is the full gate.

This spec contains no physics, so there is no Domain basis with DOI/arXiv refs and no `scientific` criterion; there is likewise no `performance` criterion — the refactor is output-identical by construction and makes no throughput claim.
