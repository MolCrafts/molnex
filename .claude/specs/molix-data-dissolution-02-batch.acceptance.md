---
slug: molix-data-dissolution-02-batch
criteria:
  - id: ac-001
    summary: molix.batch exposes the collation and loader surface
    type: code
    pass_when: |
      `python -c "import molix.batch as b; print(b.__all__)"` prints exactly
      ['BatchSamplerSource', 'BatchTask', 'DEFAULT_TARGET_SCHEMA', 'DataModule',
      'DataModuleProtocol', 'MoleculeCollator', 'PadMolecularBatch',
      'TargetSchema', 'TaskMixtureBatchSampler', 'TokenBudgetBatchSampler',
      'TokenBudgetPacker', 'collate_molecules', 'group_split_indices'].
    status: pending
  - id: ac-002
    summary: no reference to the moved batch module paths survives
    type: code
    pass_when: |
      `grep -rn "molix\.data\.collate\|molix\.data\.datamodule\|molix\.data\.sampler\|molix\.data\.group_split\|molix\.data\.tasks\.pad" src/ tests/ scripts/ regressions/ benchmarks/ docs/ CLAUDE.md`
      returns no matches.
    status: pending
  - id: ac-003
    summary: DataModule takes batch_tasks, not pipeline Nodes
    type: code
    pass_when: |
      `grep -rn "batch_nodes" src/ tests/ scripts/ regressions/ benchmarks/ docs/`
      returns no matches; inspect.signature(molix.batch.DataModule.__init__)
      has a `batch_tasks` keyword and no `batch_nodes`; molix/batch/*.py
      contains no import of molix.data.
    status: pending
  - id: ac-004
    summary: pipeline no longer routes post-collate tasks
    type: code
    pass_when: |
      PipelineSpec has no `batch_nodes` attribute, molix/data/task.py defines no
      BatchTask, and Pipeline._validate rejects a PadMolecularBatch instance
      only by the plain-callable branch — verified by the tests in
      tests/test_molix/test_data/test_pipeline.py, which no longer assert
      BatchTask routing.
    status: pending
  - id: ac-005
    summary: PadMolecularBatch zero-contribution contract still holds
    type: code
    pass_when: |
      `python -m pytest tests/test_molix/test_batch/test_pad.py -q` passes with
      its assertions unchanged from the pre-move file: ghost atoms add exactly
      0 to per-graph energy, padding edges are self-loops on the first ghost
      atom, and overflow raises rather than truncating.
    status: pending
  - id: ac-006
    summary: no e2e test remains under tests/ for the data path
    type: code
    pass_when: |
      tests/test_molix/test_data/test_e2e_workers.py is deleted and no test
      under tests/test_molix/test_batch/ constructs a Pipeline; the
      spawn-worker pickle assertion survives as a single-method test in
      tests/test_molix/test_batch/test_datamodule.py.
    status: pending
  - id: ac-007
    summary: DataLoaderProfiler configures post-collate tasks directly
    type: code
    pass_when: |
      DataLoaderProfiler.__init__ accepts `batch_tasks` and no `pipeline`
      parameter, src/molix/profiler/ imports nothing from molix.data.pipeline,
      and tests/test_molix/test_profiler/test_dataloader_profiler.py covers a
      profiler run with one BatchTask applied.
    status: pending
  - id: ac-008
    summary: batch regression reproduces hard-coded collate and padding goldens
    type: runtime
    pass_when: |
      `PYTHONPATH=src python regressions/molix-data-dissolution-02-batch.py`
      exits 0, reproducing its literal goldens: graphs.num_atoms == [3, 4],
      the second molecule's edge_index rebased by +3, padded shapes
      (8 atoms / 8 edges) with atoms.mask = [1,1,1,1,1,1,1,0]-style layout and
      bitwise-unchanged real-atom rows. No third-party software is invoked.
    status: pending
  - id: ac-009
    summary: test mirror gate knows the new pad path
    type: code
    pass_when: |
      scripts/check_test_mirror.py PINET_SPINE lists "molix/batch/pad.py" and
      not "molix/data/tasks/pad.py", and `python scripts/check_test_mirror.py
      --strict-pinet` exits 0.
    status: pending
  - id: ac-010
    summary: docs teach batch_tasks and the new module paths
    type: docs
    pass_when: |
      docs/molix/user-guide/data-modules.md, data-loading.md, profiling.md and
      tutorials/train-a-graph-model.md import from molix.batch and construct
      DataModule with batch_tasks=[...]; docs/api/molix.md renders
      `::: molix.batch` (plus collate/datamodule submodules) with no
      `::: molix.data.collate` or `::: molix.data.datamodule` left.
    status: pending
  - id: ac-011
    summary: full suite and lint gate green
    type: code
    pass_when: |
      `python -m pytest tests/ -q -n 12 --dist worksteal` passes and
      `ruff check src/ tests/ scripts/ regressions/ && ruff format --check src/
      tests/ scripts/ regressions/` is clean.
    status: pending
---

# Acceptance criteria

ac-003 and ac-004 are the load-bearing pair: the `Node` coupling must be gone
*and* the pipeline must stop claiming post-collate routing, otherwise a padded
pipeline would silently drop its padding task once the DSL leaves the repo.
