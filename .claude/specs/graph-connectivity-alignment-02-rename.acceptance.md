---
slug: graph-connectivity-alignment-02-rename
criteria:
  - id: ac-001
    summary: No bond_diff/bond_dist key-string reads remain in src/ or tests/
    type: code
    pass_when: |
      ripgrep for the literals "bond_diff", "bond_dist", 'bond_diff',
      'bond_dist' across src/ and tests/ returns zero matches; the
      edges namespace is accessed only via edge_diff / edge_dist.
    status: pending
  - id: ac-002
    summary: Sign invariant and reverse-edge negation pinned post-rename
    type: scientific
    pass_when: |
      tests/test_molix/test_data/test_neighbor_sign.py passes, asserting
      edge_diff[e] == pos[edge_index[e,1]] - pos[edge_index[e,0]] within
      1e-6 and that symmetry=True yields reverse edges whose edge_diff is
      the exact negation of the forward edge.
    status: pending
  - id: ac-003
    summary: Rotation and permutation equivariance suites green after rename
    type: scientific
    pass_when: |
      tests/test_molzoo/test_symmetry.py and the symmetry_helpers-backed
      equivariance checks for MACE/Allegro/PiNet pass with the new keys,
      energies invariant and forces equivariant within existing tolerances.
    status: pending
  - id: ac-004
    summary: Full check + test suite passes with no KeyError after atomic sweep
    type: runtime
    pass_when: |
      the repo check + full pytest suite run clean; no test raises
      KeyError(("edges", "bond_diff")) or KeyError(("edges", "bond_dist")).
    status: pending
  - id: ac-005
    summary: format_version=2 cache loads via alias yielding edge_* keys
    type: code
    pass_when: |
      tests/test_molix/test_data/test_cache_alias.py passes: a PackedCache
      written with format_version=2 (bond_diff/bond_dist buckets) loads and
      exposes edge_diff/edge_dist with a DeprecationWarning, while a freshly
      written cache reports FORMAT_VERSION=3 and carries edge_* natively.
    status: pending
---

# Acceptance criteria

- **ac-001 (grep gate)**: Static proof the rename is complete. The four quoted literals must be absent from `src/` and `tests/`. Docs and `.claude/specs` are out of this gate.
- **ac-002 (sign invariant)**: Guards the load-bearing convention `edge_diff = target − source` and the parity-required reverse-edge negation, so the rename cannot silently flip signs.
- **ac-003 (equivariance)**: Confirms downstream physics (rotation invariance of energy, equivariance of forces, permutation symmetry) is preserved byte-for-byte through the key rename.
- **ac-004 (suite green)**: The atomicity check — a partial sweep surfaces as a `KeyError` somewhere in the suite; full green proves the commit is whole.
- **ac-005 (cache alias)**: Verifies the chosen migration path (read-time alias for one version, no forced HPC rebuild) and the `FORMAT_VERSION` 2→3 bump both work.
