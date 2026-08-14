---
slug: streaming-data-tiers-04-scheduler-integration-02-view
criteria:
  - id: ac-001
    summary: Global index maps frames to (block, local) in the given block order
    type: code
    pass_when: |
      Tests in tests/test_molix/test_store/test_block_view.py build the three-block
      fixture (A: 5 frames/2 units, B: 5/1, C: 5/2, all split "train") and assert
      len(view) == 15 and the vectorized view.locate([0,4,5,9,10,14]) returns
      block ordinals (0,0,1,1,2,2) and local rows (0,4,0,4,0,4) as two (6,) long
      tensors; out-of-range and negative frames raise IndexError/ValueError naming
      the offending value.
    status: pending
  - id: ac-002
    summary: Missing or unsealed block stalls with BlockNotReadyError, never skipped
    type: code
    pass_when: |
      pytest.raises(BlockNotReadyError) passes for (a) a root with no manifest.json,
      (b) a root whose manifest has sealed == False, (c) a root deleted after
      construction and then reached COLD through reader()/frame_payload — the
      block must never have been opened (POSIX mmap survives unlink, so the
      guard is pinned on a never-mapped block); each message
      names the block row, its path and its state, and in every case no BlockDataset
      instance exists that reports a length computed from only the surviving blocks.
    status: pending
  - id: ac-003
    summary: Mixing manifest splits raises BlockSplitMismatchError naming the blocks
    type: code
    pass_when: |
      Constructing BlockDataset over one split="train" and one split="valid" block
      raises BlockSplitMismatchError whose message lists both roots with their split
      values; constructing the two per-split views separately succeeds and each
      reports its own split property.
    status: pending
  - id: ac-004
    summary: Concatenated tables equal the hand-computed literals and are torch.long
    type: code
    pass_when: |
      For the fixture, view.atom_counts equals (3,3,4,4,4,4,4,4,4,4,3,4,4,4,4),
      view.group_codes equals (0,0,1,1,1,2,2,2,2,2,3,4,4,4,4) and view.task_codes
      equals (0,0,1,1,1,2,2,2,2,2,1,2,2,2,2) via torch.equal, all with dtype
      torch.long and shape (15,); the set of group_codes equals set(range(5)),
      i.e. group ids are globally unique and partition [0, total_units).
    status: pending
  - id: ac-005
    summary: Task dictionaries merge collision-free; shape mismatch raises typed error
    type: code
    pass_when: |
      view.task_names == ("equilibrium", "relax", "torsion") (first-appearance merge)
      and block C's local codes (0,1) map to global (1,2) — a non-identity remap; a
      block whose task column carries a code >= len(manifest.task_names) raises
      BlockSchemaMismatchError naming the block and the offending code, and blocks
      whose routed column sets differ raise BlockSchemaMismatchError naming the
      symmetric difference.
    status: pending
  - id: ac-006
    summary: edge_counts refuses; never zeros; propagates through sampler construction
    type: code
    pass_when: |
      pytest.raises(ValueError, match="NeighborList") passes for view.edge_counts;
      TokenBudgetBatchSampler(view, max_edges=...) raises that same ValueError
      (not the AttributeError-derived "use MmapDataset" message) while
      TokenBudgetBatchSampler(view, max_atoms=...) succeeds; grep of
      src/molix/store/block_view.py shows no zeros_like/zeros construction of an
      edge-count vector.
    status: pending
  - id: ac-007
    summary: Cross-block frame_payload preserves caller order and matches the oracle
    type: code
    pass_when: |
      collate_packed(PackedView(view.frame_payload([12,1,7])), range(3)) equals
      collate_molecules([view[12], view[1], view[7]]) leaf-for-leaf (identical key
      sets, batch_size on atoms/edges/graphs and top level, dtypes, torch.equal),
      with atoms.batch_size == (11,), graphs.batch_size == (3,), edges.batch_size
      == (0,); the reversed frame list yields the reversed batch; a single-block
      frame list returns the owning BlockReader's payload unchanged.
    status: pending
  - id: ac-008
    summary: Reader pool stays bounded at K and results are independent of K
    type: code
    pass_when: |
      With max_open=1 over three blocks, len(view.open_blocks) <= 1 after every
      access during a full sweep of all 15 frames; the payload leaves for the same
      cross-block frame list are torch.equal between a max_open=1 view and a
      max_open=8 view; max_open < 1 raises ValueError.
    status: pending
  - id: ac-009
    summary: __getitem__ flat sample agrees with the packed route
    type: code
    pass_when: |
      For every frame of the fixture, collate_molecules([view[i]]) equals
      collate_packed(PackedView(view.frame_payload([i])), range(1)) leaf-for-leaf,
      confirming __getitem__ is the one-frame payload unpacked (no second data
      route) including nested targets, bonds, angles and propers.
    status: pending
  - id: ac-010
    summary: Unroutable internal family is refused, not silently dropped
    type: code
    pass_when: |
      A block whose units declare the `internal` valence family causes
      BlockDataset construction to raise BlockSchemaMismatchError naming the family
      and the absent collate/unpack route; no test observes a payload or sample dict
      in which an internals bucket is present in the block but missing downstream.
    status: pending
  - id: ac-011
    summary: Exports and Google docstrings with shapes and units are in place
    type: docs
    pass_when: |
      BlockDataset, BlockNotReadyError, BlockSchemaMismatchError and
      BlockSplitMismatchError are importable from molix.store, listed in its
      __all__ and mentioned in the package docstring; every new public symbol has a
      Google-style docstring with Args/Returns/Raises where applicable and tensor
      shape annotations ((F,), (B+1,), (2, T)) plus the stored units it re-exposes
      (pos in angstrom, energy in kcal/mol); `ruff check src/molix/store/` passes.
    status: pending
  - id: ac-012
    summary: Regression reproduces the hard-coded multi-block goldens
    type: runtime
    pass_when: |
      `PYTHONPATH=src python regressions/streaming-data-tiers-04-scheduler-integration-02-view.py`
      exits 0, using only public API, and reproduces the literals embedded in the
      script: frame_ptr (0,5,10,15), unit_ptr (0,2,3,5), len 15, the six locate
      pairs, atom_counts / group_codes / task_codes / task_names of the fixture,
      exact collate parity (atoms 11, graphs 3, edges 0) for frames [12,1,7] and
      its reverse, a deterministic single-task group-integral sampler epoch covering
      relax = 4 and torsion = 9 frames, the three typed refusals, and max_open=1
      vs 8 payload identity — no third-party oracle imported or subprocessed at
      runtime.
    status: pending
  - id: ac-014
    summary: Module layering guard — block_view imports no tensordict or molix.batch
    type: code
    pass_when: |
      An ast-level test in tests/test_molix/test_store/test_block_view.py asserts
      src/molix/store/block_view.py imports neither tensordict nor any molix.batch
      symbol at any depth (stdlib + torch + molix.store.* only).
    status: pending
  - id: ac-015
    summary: Unit-level sampler-integration epoch holds the discipline invariants
    type: code
    pass_when: |
      A test constructs TaskMixtureBatchSampler over the fixture view with
      max_atoms=8, max_edges=None, mixture {"equilibrium": 0.0, "relax": 0.5,
      "torsion": 0.5}, seed=42, passing task_labels=view.task_codes,
      group_ids=view.group_codes, task_names=view.task_names, and asserts: every
      yielded batch is single-task; every group's frames land whole in one batch;
      two identical constructions yield identical plans; per-task frame totals
      equal relax=4 and torsion=9.
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

ac-001, ac-004, ac-005, ac-007 and ac-009 are the correctness core: the global index algebra, the three concatenated sampler tables (with a deliberately non-identity task-dictionary remap so an identity bug cannot pass), caller-order preservation across blocks, and the single-data-route guarantee for `__getitem__` — all pinned against the `collate_molecules` equivalence oracle already used by the packed collate and the block-format regression.

ac-002 and ac-003 are invariants I8 and I7 verbatim: a planned block that is missing, unsealed or deleted mid-epoch stalls loudly by named type and is never skipped or substituted; blocks of different splits are never silently mixed. ac-005 (second clause) and ac-010 cover the "not concatenable" refusals, including the `internal` family, which has no downstream route and would otherwise be dropped silently — refusing is the honest behaviour and is asserted, not merely documented.

ac-006 is the honesty bar for edge counts: the view delegates to the block store's own refusal and must never fabricate zeros. The contrast case (`scripts/mm_param_learning/espaloma_mm_protocol_train.py:492`) is legitimate only because that dataset genuinely has no edges; a block deliberately excludes edge columns for a graph that will be rebuilt downstream, so a zero would be a false budget. Samplers over this view are consequently built with `max_edges=None`.

ac-008 pins the mmap lifecycle: at most K resident readers, with results provably independent of K because every tensor handed out is a gathered copy, so eviction can never invalidate a live batch.

ac-011 is the documentation/export bar, ac-012 the public-API regression with hard-coded, hand-derived goldens (no external oracle at capture or at run time), and ac-013 the full gate. No `ui_runtime` criterion exists — this spec touches no frontend. No `scientific` criterion exists either: the spec introduces index algebra and I/O lifecycle only, converts no units and asserts no physical quantity; the physical content it carries through (positions in Å, energies in kcal mol⁻¹) is verified by the byte-exact parity assertions in ac-007, ac-009 and ac-012.
