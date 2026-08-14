---
slug: molix-data-dissolution-03-transforms
criteria:
  - id: ac-001
    summary: NeighborList lives in molix.batch and keeps its cache identity
    type: code
    pass_when: |
      `from molix.batch import NeighborList` succeeds, the class defines
      `task_id` and `execute` and inherits no molix.data base, and
      NeighborList(cutoff=5.0).task_id == "nlist:cut=5.0:max=512:pbc=False:sym=True"
      (byte-identical to the pre-move string).
    status: pending
  - id: ac-002
    summary: edge-sign convention preserved bit-for-bit
    type: code
    pass_when: |
      `python -m pytest tests/test_molix/test_batch/test_neighbor.py
      tests/test_molix/test_md/test_neighbors.py -q` passes with assertions
      unchanged from the pre-move files, including
      molix.md.neighbors.NeighborListTask is molix.batch.neighbor.NeighborList.
    status: pending
  - id: ac-003
    summary: extxyz codec is a molix.io surface
    type: code
    pass_when: |
      `from molix.io import parse_extxyz_frames, write_extxyz_frames,
      ExtxyzFrame` succeeds, molix/io/__init__.py __all__ is alphabetized, and
      `grep -rn "molix\.datasets\._extxyz" src/ tests/ scripts/ regressions/
      benchmarks/ docs/` returns no matches.
    status: pending
  - id: ac-004
    summary: connectivity column adapters live in molix.batch
    type: code
    pass_when: |
      `grep -rn "molix\.datasets\._valence_columns\|molix\.datasets\._bond_adapter"
      src/ tests/ scripts/ regressions/ docs/` returns no matches, and
      molix.batch exports stack_angle_index, stack_proper_index,
      stack_improper_index and bond_index_from_columns.
    status: pending
  - id: ac-005
    summary: profiler no longer depends on the pipeline DSL
    type: code
    pass_when: |
      `grep -rn "molix\.data" src/molix/profiler/` returns no matches, and
      tests/test_molix/test_profiler/test_task.py passes, covering an
      execute-only object, a plain callable and the fit branch.
    status: pending
  - id: ac-006
    summary: stale molpot docstring references repaired
    type: docs
    pass_when: |
      `grep -rn "AvgNumNeighborsStat" src/` returns no matches, and
      src/molpot/heads/{edge,multipole,charge_bond}.py name only symbols that
      exist at their stated addresses after this phase.
    status: pending
  - id: ac-007
    summary: numerical regression suites still parse their fixtures
    type: code
    pass_when: |
      `python -m pytest -m regression tests/regression/test_values_ewald.py
      tests/regression/test_values_dipole.py -q` passes with the extxyz import
      repointed at molix.io.extxyz and reference values unchanged.
    status: pending
  - id: ac-008
    summary: transforms regression reproduces hard-coded goldens
    type: runtime
    pass_when: |
      `PYTHONPATH=src python regressions/molix-data-dissolution-03-transforms.py`
      exits 0, reproducing its literals: for the 3-atom chain the full
      bidirectional edge_index, edge_diff = pos[target] - pos[source] and
      edge_dist; the [3, N] angle COO stack; and an extxyz write/parse
      round-trip recovering the written lattice and positions exactly. No
      third-party software is invoked.
    status: pending
  - id: ac-009
    summary: mirror gate and prune regression follow the new addresses
    type: code
    pass_when: |
      scripts/check_test_mirror.py ALLOW_MISSING no longer lists
      molix/datasets/_extxyz.py or molix/datasets/_bond_adapter.py,
      `python scripts/check_test_mirror.py --strict-pinet` exits 0, and
      `PYTHONPATH=src python regressions/md-neighborlist-skin-02-prune.py`
      exits 0 against molix.batch.NeighborList.
    status: pending
  - id: ac-010
    summary: full suite and lint gate green
    type: code
    pass_when: |
      `python -m pytest tests/ -q -n 12 --dist worksteal` passes and
      `ruff check src/ tests/ scripts/ regressions/ && ruff format --check src/
      tests/ scripts/ regressions/` is clean.
    status: pending
---

# Acceptance criteria

ac-001/ac-002 exist because this is the one move that touches a physics
contract: the neighbour list's sign, symmetry and cutoff behaviour must be
identical after the rehome, and the `task_id` string must not drift or every
existing on-disk cache key silently changes meaning.
