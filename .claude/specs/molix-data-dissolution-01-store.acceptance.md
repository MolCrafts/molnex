---
slug: molix-data-dissolution-01-store
criteria:
  - id: ac-001
    summary: molix.store exposes the cache and reader family
    type: code
    pass_when: |
      `python -c "import molix.store as s; print(s.__all__)"` prints exactly
      ['BaseDataset', 'CachedDataset', 'MmapDataset', 'PackedCache',
      'PackedView', 'SubsetDataset'] (alphabetized), and src/molix/store/
      contains __init__.py, cache.py, dataset.py.
    status: pending
  - id: ac-002
    summary: no reference to the old cache/dataset module paths survives
    type: code
    pass_when: |
      `grep -rn "molix\.data\.cache\|molix\.data\.dataset" src/ tests/ scripts/
      regressions/ benchmarks/ docs/ CLAUDE.md` returns no matches, and
      src/molix/data/cache.py and src/molix/data/dataset.py no longer exist.
    status: pending
  - id: ac-003
    summary: PesBundle is gone from source, tests and regressions
    type: code
    pass_when: |
      `grep -rni "pesbundle\|pes_bundle" src/ tests/ regressions/ scripts/`
      returns no matches, and regressions/universal-class-i-04-bundle.py and
      tests/test_molix/test_data/test_pes_bundle.py are deleted.
    status: pending
  - id: ac-004
    summary: Hessian Cartesian-layout contract has a live owner
    type: docs
    pass_when: |
      The flat layout i = 3a + c ("x0, y0, z0, x1, ...") for graphs.hessian
      (3N,3N) is stated in the ProjectedInternalHessianLoss docstring in
      src/molix/core/losses/molecular.py, and src/molpot/derivation/hessian.py
      cites that class instead of molix.data.pes_bundle.PesBundle.
    status: pending
  - id: ac-005
    summary: moved store tests pass unmodified apart from import lines
    type: code
    pass_when: |
      `python -m pytest tests/test_molix/test_store/ -q` passes, and
      `git diff -M --stat` shows the three moved test files as renames whose
      only content changes are import statements.
    status: pending
  - id: ac-006
    summary: shared packed-bucket fixture lives at one address
    type: code
    pass_when: |
      equal_count_samples is defined only in tests/test_molix/conftest.py
      (`grep -rn "def equal_count_samples" tests/` yields exactly one hit) and
      every consumer imports it as `from tests.test_molix.conftest import
      equal_count_samples`.
    status: pending
  - id: ac-007
    summary: store regression reproduces hard-coded packed goldens
    type: runtime
    pass_when: |
      `PYTHONPATH=src python regressions/molix-data-dissolution-01-store.py`
      exits 0, reproducing its literal goldens (Z=[1,6], pos=[[0,0,0],[1,0,0]],
      edge_dist=[1.0,1.0], targets.U0 = sample index, avg_num_neighbors == 1.0)
      through molix.store.PackedCache -> MmapDataset and CachedDataset with
      exact equality, and asserting molix.data.cache is not importable. No
      third-party software is invoked at runtime.
    status: pending
  - id: ac-008
    summary: full suite and lint gate green
    type: code
    pass_when: |
      `python -m pytest tests/ -q -n 12 --dist worksteal` passes and
      `ruff check src/ tests/ scripts/ regressions/ && ruff format --check src/
      tests/ scripts/ regressions/` is clean.
    status: pending
  - id: ac-009
    summary: docs point at the new store module paths
    type: docs
    pass_when: |
      docs/molix/user-guide/profiling.md imports PackedCache/CachedDataset from
      molix.store, and docs/api/molix.md renders `::: molix.store` (plus
      molix.store.cache / molix.store.dataset) with no `::: molix.data.cache`
      or `::: molix.data.dataset` entry left.
    status: pending
---

# Acceptance criteria

ac-002, ac-003 and ac-005 together are the "behaviour preserved, addresses
changed" bar: the tests may move but may not be weakened, and no alias may be
left behind (stage: experimental — no back-compat shims).
