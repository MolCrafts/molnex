---
slug: molix-data-dissolution-04-etl-exit
criteria:
  - id: ac-001
    summary: molix has no data/ or datasets/ package
    type: code
    pass_when: |
      src/molix/data and src/molix/datasets do not exist on disk, and
      `python -c "import importlib.util as u; assert u.find_spec('molix.data')
      is None and u.find_spec('molix.datasets') is None"` exits 0.
    status: pending
  - id: ac-002
    summary: zero references to the dissolved packages anywhere
    type: code
    pass_when: |
      `grep -rn "molix\.data\b\|molix\.datasets\b\|molix/data/\|molix/datasets/"
      src/ tests/ scripts/ regressions/ benchmarks/ docs/ CLAUDE.md zensical.toml`
      returns no matches.
    status: pending
  - id: ac-003
    summary: boundary tests enforce the dissolution and the molhub direction
    type: code
    pass_when: |
      `python -m pytest tests/test_repository_boundaries.py -q` passes with
      three tests: no molhub import under src/, molix.data/molix.datasets
      unresolvable, and no textual reference to them across src/ tests/
      scripts/ regressions/ benchmarks/ (failures report path:line).
    status: pending
  - id: ac-004
    summary: surviving suite passes unchanged after the deletion
    type: code
    pass_when: |
      `python -m pytest tests/ -q -n 12 --dist worksteal` passes with no test
      skipped, xfailed or weakened as part of this phase; the only test files
      removed are those under tests/test_molix/test_data/ and
      tests/test_molix/test_datasets/.
    status: pending
  - id: ac-005
    summary: no ETL-dependent benchmark is left broken
    type: code
    pass_when: |
      benchmarks/ contains no file importing molix.data or molix.datasets, and
      every remaining benchmark script imports successfully under
      `python -c "import ast,pathlib; [ast.parse(p.read_text()) for p in
      pathlib.Path('benchmarks').glob('*.py')]"` with its imports resolvable.
    status: pending
  - id: ac-006
    summary: API docs and nav describe the packages that exist
    type: docs
    pass_when: |
      docs/api/molix.md contains `::: molix.batch` and `::: molix.store` and no
      `::: molix.data*` or `::: molix.datasets` block;
      docs/molix/user-guide/data.md is deleted and its "Data Pipeline" entry is
      gone from zensical.toml; the surviving data-loading / data-modules pages
      state that dataset acquisition lives in MolHub.
    status: pending
  - id: ac-007
    summary: CLAUDE.md architecture matches the new package tree
    type: docs
    pass_when: |
      CLAUDE.md's module dependency graph names molix.batch and molix.store (not
      molix.data / molix.datasets), the two-tier data contract names
      molix.batch.collate.collate_molecules as the single conversion point, the
      cache-format section cites src/molix/store/cache.py, and the Extxyz rule
      cites molix.io.extxyz.
    status: pending
  - id: ac-008
    summary: exit regression proves the surviving surface composes
    type: runtime
    pass_when: |
      `PYTHONPATH=src python regressions/molix-data-dissolution-04-etl-exit.py`
      exits 0: molix.data and molix.datasets are unimportable, and two literal
      molecules flow NeighborList -> PackedCache -> MmapDataset -> DataModule
      reproducing the file's hard-coded goldens for batch shapes,
      graphs.num_atoms and edge_dist. No third-party software is invoked.
    status: pending
  - id: ac-009
    summary: lint and mirror gates clean
    type: code
    pass_when: |
      `ruff check src/ tests/ scripts/ regressions/ && ruff format --check src/
      tests/ scripts/ regressions/ && ty check src/ --exit-zero-on-warning` is
      clean, and `python scripts/check_test_mirror.py --strict-pinet` exits 0
      with no molix/data or molix/datasets entries left in its tables.
    status: pending
---

# Acceptance criteria

ac-002 and ac-003 are the binding pair: the grep must be zero *and* a test must
enforce it, otherwise the dissolved names creep back through a docstring the
next time someone copies an old example.
