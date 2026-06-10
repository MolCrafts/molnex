---
slug: dynamic-batching-packed-collate-01-sampler
criteria:
  - id: ac-001
    summary: Count accessors derive per-sample sizes from pointers, never __getitem__
    type: code
    pass_when: |
      _CacheBacked.atom_counts / edge_counts return long tensors of shape
      (n_samples,) equal to ptr[1:]-ptr[:-1]; a test monkeypatching
      __getitem__ with a counter shows zero calls during sampler
      construction and full iteration.
    status: pending
  - id: ac-002
    summary: SubsetDataset counts remap local indices to packed indices
    type: code
    pass_when: |
      For a split() subset (and a nested subset), subset.atom_counts[i]
      equals atom_ptr[j+1]-atom_ptr[j] where j is the subset's packed index
      for local i; values are NOT the full-dataset count vector.
    status: pending
  - id: ac-003
    summary: Missing pointers or unsupported dataset raise actionable ValueError
    type: code
    pass_when: |
      atom_counts on a cache without atom_ptr, and TokenBudgetBatchSampler
      over a non-packed BaseDataset subclass, both raise ValueError whose
      message names MmapDataset, CachedDataset and SubsetDataset as the
      supported types.
    status: pending
  - id: ac-004
    summary: Sampler validates budgets eagerly in __init__
    type: code
    pass_when: |
      Constructing with neither max_atoms nor max_edges, or with any budget
      <= 0, raises ValueError at __init__ time (before iteration) naming the
      offending parameter and the fix.
    status: pending
  - id: ac-005
    summary: No batch exceeds active budgets except documented singletons
    type: code
    pass_when: |
      For atom-only, edge-only, and dual-budget configurations, every
      yielded batch satisfies counts[batch].sum() <= budget for each active
      budget, with the sole exception of single-sample batches whose own
      count exceeds the budget.
    status: pending
  - id: ac-006
    summary: Oversize sample forms singleton batch, never dropped, warns once
    type: code
    pass_when: |
      A sample with count > budget appears alone in exactly one batch;
      pytest.warns records exactly one warning per sampler instance even
      with multiple oversize samples.
    status: pending
  - id: ac-007
    summary: Every index appears exactly once per epoch; __len__ matches
    type: code
    pass_when: |
      Concatenated batch indices for one epoch sort to range(len(dataset))
      with no duplicates, and len(sampler) equals the number of yielded
      batches for that permutation.
    status: pending
  - id: ac-008
    summary: Same seed deterministic; different epoch seeds reshuffle
    type: code
    pass_when: |
      Two samplers with identical (dataset, budgets, seed) yield identical
      batch sequences; seeds seed+0 vs seed+1 yield different batch
      compositions on a dataset large enough to make collision negligible.
    status: pending
  - id: ac-009
    summary: Resume at epoch k re-derives epoch k's batch composition
    type: code
    pass_when: |
      A fresh DataModule given on_epoch_start(k) then train_dataloader()
      produces batch index lists identical to a DataModule that advanced
      continuously through epochs 0..k.
    status: pending
  - id: ac-010
    summary: Opt-in wiring uses batch_sampler and omits exclusive kwargs
    type: code
    pass_when: |
      With max_atoms_per_batch and/or max_edges_per_batch set, the returned
      train DataLoader has a TokenBudgetBatchSampler as batch_sampler and
      DataLoader construction passed none of batch_size/shuffle/sampler/
      drop_last/generator (no ValueError from DataLoader's mutual-exclusion
      check; loader iterates successfully).
    status: pending
  - id: ac-011
    summary: Default path DataLoader config unchanged (backward compat)
    type: code
    pass_when: |
      With both new kwargs left None, train_dataloader() yields a DataLoader
      whose batch_size, sampler, batch_sampler, drop_last and generator
      configuration is identical to the pre-change behavior asserted in
      existing test_datamodule.py tests (which must still pass unmodified).
    status: pending
  - id: ac-012
    summary: DDP plus budget raises eager ValueError at dataloader build
    type: code
    pass_when: |
      With dist.is_available/is_initialized monkeypatched True and a budget
      set, train_dataloader() raises ValueError mentioning DDP/
      DistributedSampler incompatibility and how to proceed.
    status: pending
  - id: ac-013
    summary: Sampler picklable; symbol re-exported from molix.data
    type: code
    pass_when: |
      pickle.loads(pickle.dumps(sampler)) yields a sampler producing the
      same batches, and `from molix.data import TokenBudgetBatchSampler`
      succeeds with the name present in molix.data.__all__.
    status: pending
  - id: ac-014
    summary: Full check and test suite pass
    type: runtime
    pass_when: |
      `ruff check src/ && ruff format --check src/` and
      `python -m pytest tests/ -v` both exit 0 on the final tree.
    status: pending
---

# Acceptance criteria

ac-001/ac-002/ac-003 锁定指针统计访问器的契约（向量化、子集重映射、可操作报错）；ac-004 至 ac-008 锁定采样器本体（eager 校验、预算遵守、singleton 不丢样、全覆盖、确定性与重洗）；ac-009 至 ac-012 锁定 DataModule 接线（resume 可重推导、互斥 kwargs、默认路径向后兼容、DDP 抛错）；ac-013 锁定可腌制性与 re-export；ac-014 是最终的全量门禁。
