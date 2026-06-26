---
slug: graph-connectivity-alignment-01-offset
criteria:
  - id: ac-001
    summary: collate_molecules rebases edge_index identically via registry
    type: code
    pass_when: |
      A 2-molecule edged batch through collate_molecules yields the same
      globally-offset edge_index as the pre-refactor hardcoded path; the
      pinned regression in tests/test_molix/test_data/test_collate.py asserts
      the second molecule's edges == [[2, 3], [2, 4]] and passes.
    status: pending
  - id: ac-002
    summary: packed==molecules equivalence oracle still holds on edged batch
    type: code
    pass_when: |
      The edged-batch assertion in
      tests/test_molix/test_data/test_collate_packed.py shows
      collate_packed(view, indices) equals
      collate_molecules([dataset[i] for i in indices]) leaf-for-leaf
      (keys, torch.equal values, dtypes, batch_size) after edge_index is
      routed through rebase, and passes.
    status: pending
  - id: ac-003
    summary: rebase generalizes to a synthetic cat_dim=1 mock bond_index
    type: code
    pass_when: |
      A unit test feeds a synthetic [2, N] mock bond_index and a per-count
      offset vector through rebase(tensor, offset, "bond_index") and asserts
      every column is offset by its segment value broadcast across both rows
      (offset.unsqueeze(0)); the test passes without any phase-03 production
      code present.
    status: pending
  - id: ac-004
    summary: registry pre-declares bond/angle/dihedral with correct axes
    type: code
    pass_when: |
      INDEX_KEYS in src/molix/data/collate.py maps edge_index->(0, 1),
      bond_index->(1, 0), angle_index->(1, 0), dihedral_index->(1, 0); a test
      asserts these exact tuples so sub-specs 02/03 can import a stable
      contract.
    status: pending
---

# Acceptance criteria

- **ac-001 (code)** — Regression pin: the existing `collate_molecules` offset behavior is preserved exactly when driven by `INDEX_KEYS` + `rebase`. Verified by `tests/test_molix/test_data/test_collate.py`.
- **ac-002 (code)** — The fast-path/slow-path equivalence oracle is unbroken by the routing change, on a batch that actually carries edges. Verified by `tests/test_molix/test_data/test_collate_packed.py`.
- **ac-003 (code)** — The mechanism is proven to handle the differing `cat_dim=1` / `index_axis=0` case before any real producer exists, so phase 03 only needs to emit `bond_index`. Verified by a direct `rebase` unit test on a synthetic `[2, N]` tensor.
- **ac-004 (code)** — The pre-declared registry entries are present with the precise `(cat_dim, index_axis)` tuples that sub-specs 02 (rename) and 03 (produce) will rely on.

Note for callers: `INDEX_KEYS` and `rebase` are intentionally kept in `molix.data.collate` (single home, ponytail rule). Sub-specs 02/03 should `from molix.data.collate import INDEX_KEYS, rebase`; only lift to a new `src/molix/data/_index_keys.py` when `cache.py` schema inference becomes a second consumer.
