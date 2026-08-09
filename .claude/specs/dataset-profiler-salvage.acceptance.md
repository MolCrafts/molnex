---
slug: dataset-profiler-salvage
criteria:
  - id: ac-001
    summary: one count helper per data tier, no duplicate definitions left
    type: code
    pass_when: |
      `rg -n "_extract_batch_counts|_extract_counts" src/` returns no match.
      src/molix/profiler/_utils.py defines exactly one nested-tier helper
      `batch_counts(batch)` (reads batch["atoms","Z"] / batch["graphs","num_atoms"])
      and one flat-tier helper `sample_counts(sample)` (reads sample["Z"] /
      sample["edge_index"]), each docstring naming its tier per the CLAUDE.md
      two-tier data contract. dataloader.py and module.py import them from
      molix.profiler._utils, and tests/test_molix/test_profiler/
      test_dataloader_profiler.py + test_module_profiler.py pass unmodified.
    status: pending
  - id: ac-002
    summary: DatasetProfiler passes the CLAUDE.md shape check
    type: code
    pass_when: |
      molix.profiler.dataset exposes DatasetProfiler(n_samples=200, stride=1,
      n_warmup=3) taking all config in __init__ and run(data) -> DatasetResult;
      src/molix/profiler/dataset.py contains no make_*/create_*/build_* symbol
      and no module-level profile_dataset()-style facade. DatasetProfiler(
      n_samples=0), DatasetProfiler(stride=0) and DatasetProfiler().run([]) each
      raise ValueError whose message contains a remediation sentence naming a
      valid input (e.g. "CachedDataset").
    status: pending
  - id: ac-003
    summary: size stats come from packed pointers, exact and unpack-free
    type: code
    pass_when: |
      For a CachedDataset of 100 literal samples profiled with
      DatasetProfiler(n_samples=5).run(ds): result.counts_exact is True,
      result.max_atoms == ds.max_atoms, result.max_edges == ds.max_edges,
      result.avg_num_neighbors == ds.avg_num_neighbors, and
      result.atom_stats.mean == float(ds.atom_counts.double().mean()) (all 100
      samples, not the 5 sampled), while result.n_sampled == 5 and access_ms is
      built from 5 measurements. A cache lacking edge_ptr yields
      edge_stats is None plus a [WARN] line and does not raise.
    status: pending
  - id: ac-004
    summary: field layout reads packed schema, falls back audibly, names disambiguated
    type: code
    pass_when: |
      For a cache-backed dataset, {(f.key, f.axis, f.dtype, f.extra_shape) for f
      in result.fields} equals the entries of ds.packed_view().payload["schema"]
      and result.fields_exact is True; for a plain list[dict] input the profiler
      returns a non-empty result.fields with fields_exact is False and raises
      nothing. `rg -n "FieldSchema" src/ tests/` returns no match, and the
      FieldSpec and TargetStat docstrings each disambiguate against
      molix.data.collate.TargetSchema and the packed payload["schema"].
    status: pending
  - id: ac-005
    summary: __init__ additive + alphabetized, trainer.py untouched, flattener divergence documented
    type: code
    pass_when: |
      git diff on src/molix/profiler/__init__.py adds only the dataset imports,
      the two names DatasetProfiler/DatasetResult and one docstring bullet — no
      previously exported name is removed — and `__all__ == sorted(__all__)`
      holds per notes.md 2026-08-09. git diff on src/molix/profiler/trainer.py
      and src/molix/data/cache.py is empty. dataset.py::_flatten_leaves has a
      docstring naming molix.data.cache._flatten and stating the divergence
      (no reserved-key validation, never raises), and
      test_flatten_leaves_does_not_raise_on_reserved_key passes.
    status: pending
  - id: ac-006
    summary: MockSource/MockBatch seeds are actually reproducible
    type: code
    pass_when: |
      src/molix/profiler/mock.py::_resolve takes an explicit random.Random and
      `rg -n "random\.(randint|sample|Random\(\))" src/molix/profiler/mock.py`
      shows no module-level RNG draw inside _resolve. Two MockSource(n_samples=8,
      n_atoms=(5,20), seed=0) built in the same process give identical
      [len(s["Z"]) for s in ...]; two MockBatch(n_atoms=(8,32), n_edges=(16,64),
      seed=0)() give identical shapes (tests/test_molix/test_profiler/
      test_mock.py).
    status: pending
  - id: ac-007
    summary: regression example reproduces hard-coded dataset-profile goldens
    type: runtime
    pass_when: |
      `PYTHONPATH=src python regressions/dataset-profiler-salvage.py` prints OK
      and exits 0. The script builds its PackedCache from 5 sample dicts written
      as literals in the file, then asserts n_total, atom_stats.mean, max_atoms,
      max_edges, avg_num_neighbors, the sorted field-key list with axes, and the
      targets.U0 mean/min/max against hard-coded literals (math.isclose,
      rel_tol=1e-12); it asserts no timing value, imports nothing beyond molix +
      torch + stdlib, and makes no network call or subprocess. Its header records
      goldens=analytic (no oracle), commit sha, torch version, date,
      device=cpu/float32 per regressions/README.md.
    status: pending
  - id: ac-008
    summary: profiling guide lands under user-guide and matches the real TrainerProfiler API
    type: docs
    pass_when: |
      docs/molix/user-guide/profiling.md exists with one section each for
      TaskProfiler, ModuleProfiler, DataLoaderProfiler, TrainerProfiler and
      DatasetProfiler. The TrainerProfiler section documents only the in-tree
      surface — TrainerProfiler(model, loss_fn, device, hooks), run(n_steps,
      n_warmup, batch, top), raw-loop baseline / Trainer overhead /
      baseline-subtracted hotspots — and `rg -n "cuda_sync|profile_eval|phase
      table" docs/molix/user-guide/profiling.md` returns no match. zensical.toml
      contains { "Profiling" = "molix/user-guide/profiling.md" } inside Molix →
      User Guide, and docs/molix/profiling.md does not exist.
    status: pending
  - id: ac-009
    summary: stray branch retired with its commit kept reachable
    type: code
    pass_when: |
      After the port has landed, `git branch --list worktree-molix-profiler-suite`
      prints nothing, `git worktree list` shows no worktree for it, and
      `git rev-parse archive/worktree-molix-profiler-suite` resolves to 92760f0
      (tag created before the delete). `git branch --list` shows no other stray
      worktree-* branch on the unified baseline.
    status: pending
  - id: ac-010
    summary: full check + test suite green with both new mirrors present
    type: runtime
    pass_when: |
      `ruff check src/ tests/ scripts/ regressions/ && ruff format --check src/
      tests/ scripts/ regressions/` is clean, `ty check src/
      --exit-zero-on-warning` reports no error, `PYTHONPATH=src python -m pytest
      tests/ -v` is green including test_dataset.py and test_mock.py, and
      `python scripts/check_test_mirror.py` reports a missing count no higher
      than before the change with neither molix/profiler/dataset.py nor
      molix/profiler/mock.py in the missing list.
    status: pending
---

# Acceptance criteria

- **ac-001 / ac-002 / ac-004 / ac-005** 守的是「移植不留债、不复刻反模式」:计数函数收敛成两层各一个,`DatasetProfiler` 从空 `__init__` 改造成配置/数据分离的 OOP 形态,`FieldSchema` 这个歧义名消失,`__init__.py` 与 `trainer.py` 不被分支版本污染,唯一的第二个 flattener 有白纸黑字的分歧说明与单测背书。
- **ac-003 / ac-004** 是本次移植相对分支的真实升级:规模统计与字段布局走 packed 指针 / packed schema 的精确零解包路径,逐样本循环退回它本来该管的事(访问延迟、内存足迹、标签统计)。两条都要求「有 cache 时精确、无 cache 时降级且自报家门」。
- **ac-006** 是 iron-law 修复项:seed 承诺可复现却从全局 RNG 取数,会让任何 mock 基线 golden 失效,本 spec 内修并补上此前缺失的 mock.py mirror。
- **ac-007** 的 golden 全部是从文件内字面量样本手算出来的解析值,不存在第三方 oracle,因此可以永久硬编码;计时量一律不入断言。
- **ac-009** 是用户的收尾诉求:统一基线上不再有游离分支,但 92760f0 通过 archive tag 保持可达,不被 gc 吞掉。
