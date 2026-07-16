---
slug: dynamic-batching-packed-collate-02-collate
criteria:
  - id: ac-001
    summary: collate_packed equals collate_molecules on multi-sample batches with edges
    type: code
    pass_when: |
      Test in tests/test_molix/test_data/test_collate_packed.py builds a
      PackedCache-backed dataset, picks >=2 indices with edges, and asserts the
      fast-path TensorDict matches collate_molecules over the same per-sample
      dicts: identical key sets, torch.equal per leaf, identical dtypes, and
      identical batch_size on atoms/edges/graphs and top level.
    status: verified
    last_checked: 2026-06-10
  - id: ac-002
    summary: Edge-less handling matches oracle, incl. empty-edge fallback
    type: code
    pass_when: |
      Tests pass for (1) a batch mixing zero-edge and edged samples and (2) a
      schema with no edge keys, where the fast-path edges TensorDict equals the
      collate.py:149-156 fallback (edge_index zeros(0,2) long, bond_diff
      zeros(0,3), bond_dist zeros(0), batch_size=[0]) and equals the oracle output.
    status: verified
    last_checked: 2026-06-10
  - id: ac-003
    summary: TargetSchema routing identical to oracle for atom/graph/scalar targets
    type: code
    pass_when: |
      Tests with a TargetSchema containing both atom_level (e.g. forces) and
      graph_level (e.g. energy) targets, plus a python-scalar target, show
      atom-level targets concatenated under atoms and graph-level targets
      reshape(-1) under graphs, torch.equal to collate_molecules output.
    status: verified
    last_checked: 2026-06-10
  - id: ac-004
    summary: SubsetDataset views and singleton batches collate equivalently
    type: code
    pass_when: |
      Tests collate via SubsetDataset.packed_view() (split-produced shuffled
      indices) and via a single-index batch; both equal the oracle leaf-for-leaf
      with correct local-to-packed index remapping.
    status: verified
    last_checked: 2026-06-10
  - id: ac-005
    summary: cache.py stays IO-only — no tensordict or TargetSchema import
    type: code
    pass_when: |
      A test (or equivalent static assertion) reads src/molix/data/cache.py
      source and asserts neither "tensordict" nor "TargetSchema" appears in its
      import statements; git diff for this spec leaves cache.py unmodified.
    status: verified
    last_checked: 2026-06-10
  - id: ac-006
    summary: DataModule routes packed datasets to fast path, falls back otherwise
    type: code
    pass_when: |
      With a packed-capable dataset and num_workers=0, DataModule dataloaders
      use _PackedCollateFn (asserted via collate_fn type or unpack_sample call
      count == 0) and emitted batches equal the slow-path batches; with a
      non-packed BaseDataset, _CollateFn + collate_molecules is used and output
      is unchanged.
    status: verified
    last_checked: 2026-06-10
  - id: ac-007
    summary: Spawn-worker integration (num_workers=2) yields correct batches
    type: code
    pass_when: |
      An integration test mirroring test_e2e_workers.py iterates one epoch
      through DataModule with num_workers=2 (spawn) on a packed dataset; every
      batch is a TensorDict with correct atoms/edges/graphs shapes and values
      equal to the num_workers=0 fast-path output for the same index order.
    status: verified
    last_checked: 2026-06-10
  - id: ac-008
    summary: batch_nodes and batch_to(ftype) post-steps apply on fast path
    type: code
    pass_when: |
      A test registers a marker batch node and a non-default ftype; fast-path
      batches show the node's effect and all floating-point leaves cast to the
      captured ftype, matching _CollateFn semantics.
    status: verified
    last_checked: 2026-06-10
  - id: ac-009
    summary: Collate callable pickles without capturing payload tensors
    type: code
    pass_when: |
      pickle.dumps/loads round-trip of _PackedCollateFn succeeds and the
      restored callable produces correct batches; pickled state excludes the
      lazily built PackedView (asserted via __getstate__ contents), payload
      reached only through the dataset reference.
    status: verified
    last_checked: 2026-06-10
  - id: ac-010
    summary: Eager actionable ValueError on empty indices or missing Z/pos
    type: code
    pass_when: |
      pytest.raises(ValueError) tests pass for collate_packed with an empty
      index list and with a payload schema lacking Z or pos; messages name the
      offending condition.
    status: verified
    last_checked: 2026-06-10
  - id: ac-011
    summary: New public symbols carry Google docstrings with tensor shapes
    type: docs
    pass_when: |
      collate_packed, PackedView, packed_view, _IndexDataset, _PackedCollateFn
      each have Google-style docstrings annotating tensor shapes (e.g.
      ``(E, 2)``, ``(N,)``) and Args/Returns/Raises sections where applicable;
      ruff check passes.
    status: verified
    last_checked: 2026-06-21
    note: |
      Audited green by the documenter agent (all five symbols carry
      Google-style docstrings with tensor shapes; ruff clean). Manually
      reviewed + approved 2026-06-21: collate_packed / PackedView /
      packed_view carry Args/Returns/Raises with (N,)/(E,)/(B,) shape
      annotations; _IndexDataset / _PackedCollateFn carry class docstrings
      (tensor shapes N/A — they pass bare int indices). `ruff check
      src/molix/data/collate.py` clean; ac-012 full ruff+pytest verified.
  - id: ac-012
    summary: Full check and test suite pass
    type: runtime
    pass_when: |
      `ruff check src/ && ruff format --check src/` and
      `python -m pytest tests/ -v` both exit 0 on the branch containing this spec.
    status: verified
    last_checked: 2026-06-10
---

# Acceptance criteria

ac-001 至 ac-004 共同构成等价性 oracle 合同：快速路径的任何输出差异（键、值、dtype、batch_size）即为失败。ac-005 是分层守护（cache = IO/格式，零 collate 依赖）。ac-006 至 ac-009 覆盖 DataModule 接线、spawn worker 与 pickle 机制。ac-010 与 ac-011 分别约束错误信息质量与文档规范。ac-012 是全量门禁。

注：本规格不含物理内容，故无 Domain basis 与 scientific 类验收项；性能声明以 ac-006 中 `unpack_sample` 零调用的代码级冒烟体现（`$META` 无 bench 配置）。
