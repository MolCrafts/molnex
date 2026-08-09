---
title: Salvage DatasetProfiler from worktree-molix-profiler-suite into the unified baseline
status: approved
created: 2026-08-09
grilled: true
---

# Salvage DatasetProfiler from worktree-molix-profiler-suite

## Summary

把游离分支 `worktree-molix-profiler-suite`(92760f0,2026-06-26)里唯一未落地的产物 —— `DatasetProfiler` —— 按文件拣选移植进当前树,使 `molix.profiler` 四件套补齐为「任务 / 模块 / DataLoader / Trainer / 数据集」五件套:给定一个数据集(`CachedDataset` / `MmapDataset` / `SubsetDataset` / 普通 `Sequence[dict]` / `MockSource`),一次调用即可读出规模分布(原子/边计数、`avg_num_neighbors`、padding 倾斜)、单样本访问延迟与内存足迹、字段布局(每个 key 的 axis/dtype/尾部形状)、以及目标标签的数值统计与非有限值告警。移植过程中顺带收敛掉本次触碰面上已存在的两处技术债(两份逐字重复的 batch 计数函数、`MockSource`/`MockBatch` 的全局 RNG 泄漏),并把分支里那份记录了不存在 API 的 `docs/molix/profiling.md` 重写为符合 diátaxis 布局的 `docs/molix/user-guide/profiling.md`。落地后删除该分支,统一基线上不再有游离分支。

分支中的 `trainer.py` 及其旧 API 测试**不移植**:树内 `TrainerProfiler`(raw-loop-baseline 设计)已取代它,回退即是倒退;整分支 merge 已试并 abort(trainer.py add/add 冲突),本 spec 走按文件拣选路线。

## Design

### 新增实体

`src/molix/profiler/dataset.py`(新)承载四个类型,全部走 `dataloader.py` 已确立的 pattern:**配置进 `__init__`、数据进 `run()`、结果对象自带 `print_report()`**。

| 符号 | 形态 | 职责 |
|---|---|---|
| `DatasetProfiler` | class,`__init__(n_samples=200, stride=1, n_warmup=3)` + `run(data) -> DatasetResult` | 唯一入口;无 `make_*` / `profile_dataset()` 自由函数 |
| `DatasetResult` | `@dataclass` + `print_report()` | 分组结果 + 分节报表 |
| `FieldSpec` | `@dataclass(frozen=True)`:`key / axis / dtype / extra_shape` | 单个字段的打包布局描述 |
| `TargetStat` | `@dataclass(frozen=True)`:`key / stat: ValueStat / min / max / n_nonfinite` | 单个标签列的数值统计 |

`DatasetProfiler.run` 是**两条路径的合流**,这是本次移植相对分支实现的核心升级:

1. **精确快路径(规模统计)** —— 若对象暴露 `atom_counts` / `edge_counts` / `avg_num_neighbors` / `max_atoms` / `max_edges`(即 `_CacheBacked` 及 `SubsetDataset` 的转发),直接读这些属性:它们由 `atom_ptr` / `edge_ptr` cumsum 向量化推出,覆盖**全体样本**且**零解包**。`edge_counts` 在无 `edge_ptr` 的 cache 上会抛 `ValueError`(这是合法情形:pipeline 未跑 `NeighborList`),捕获后降级为 `edge_stats=None` 并追加一行 `[WARN]`,不外抛。此时 `counts_exact=True`。
2. **抽样慢路径(访问与足迹)** —— 只用于本质上必须逐样本才能测的量:冷启动首访延迟 `cold_access_ms`、稳态访问延迟 `access_ms`(`Timer` 包住 `data[i]`,丢弃 `n_warmup` 次)、单样本字节足迹 `sample_bytes`(叶子张量 `numel()*element_size()` 求和)与外推 `est_total_mb`、标签统计 `targets`。样本按 `stride` 跨步取前 `n_samples` 个。非 cache 对象(`list[dict]` / `MockSource`)走这条路径同时补出规模统计,并置 `counts_exact=False`。
3. **字段布局** —— 有 `packed_view()` 时直接读 `data.packed_view().payload["schema"]`:那是打包时跨全体样本推断出的 `(axis, dtype, extra_shape)`,精确且免费,`fields_exact=True`;`packed_view()` 缺失或抛 `AttributeError`(`SubsetDataset` 包非 cache 数据集时)则退回抽样推断,`fields_exact=False`。
4. **已拟合任务状态** —— `data.stats()` 存在时原样收进 `task_states`,报表里列出任务名与其 state key(如 `AtomicDress`),让「这份 cache 是用什么 pipeline 烤出来的」在同一屏可见。

退化输入抛 `ValueError`,消息必须带补救句:`__init__` 的 `n_samples <= 0` / `stride <= 0`;`run()` 的 `len(data) == 0`("…point it at a prepared cache (PipelineSpec.run(...) → CachedDataset) or a non-empty Sequence[dict].");既无 `__len__` 又无 `__getitem__` 的对象。诊断类信息一律走 `[WARN]` 行,**永不抛**:非有限标签值、`p95/p50 > 3` 的原子数倾斜(padding 浪费)、`fields_exact=False`、缺 `edge_ptr`。

报表沿用 `─`*72 分节框、`_fmt_table` 表格、`f"Data: {desc}"` 头(`desc` 由 `getattr(data, "describe", lambda: type(data).__name__)()` 取),与 `dataloader.py` / `module.py` 逐行同构。模块头 `from __future__ import annotations`,Google docstring 带张量形状标注。单位不做任何换算:位置 Å、能量 eV 由数据集自身定义,profiler 原样打印并在 docstring 中声明「不做单位转换、不猜单位」。

### 命名消歧(三个 "schema" 只隔一次 import)

`molix.data.collate.TargetSchema`(collate 时的标签路由)、`PackedCache` payload 的 `"schema"`(每 key 的 axis/dtype/尾形)、分支里的 `FieldSchema`(报表用字段描述)三者同名不同义。处理:分支的 `FieldSchema` **改名 `FieldSpec`**;`FieldSpec` 与 `TargetStat` 的 docstring 各写一句显式消歧(`FieldSpec` 是 packed `payload["schema"]` 的**报表视图**;`TargetStat` 与 `TargetSchema` 无关,前者是标签数值统计,后者是路由规则)。本树不再出现 `FieldSchema` 这个符号。

### Reuse decision(逐条落定 librarian 报告)

- **reuse `src/molix/profiler/_utils.py` 的 `Timer` / `TimingStat` / `ValueStat` / `_fmt_table`** —— 分支文件本就 import 这四个,原样沿用,不新增任何统计原语。
- **reuse cache 精确统计 `_CacheBacked.stats()` / `avg_num_neighbors` / `atom_counts` / `edge_counts` / `max_atoms` / `max_edges`(`SubsetDataset` 转发)** —— 即上文快路径 1;规模统计**不得**用逐样本循环重算。
- **reuse `dataset.packed_view().payload["schema"]`** —— 即上文路径 3;抽样推断只作为无 packed view 时的兜底,且必须自报 `fields_exact=False`。
- **generalize `dataloader.py:151 _extract_batch_counts` 与 `module.py:208 _extract_counts`(逐字重复)** —— 合并进 `_utils.py`,按 CLAUDE.md 的两层数据契约**每层各一个**:`batch_counts(batch)` 读 collate 后嵌套 TensorDict 层(`batch["atoms","Z"]` / `batch["graphs","num_atoms"]`),`sample_counts(sample)` 读扁平原始样本层(`sample["Z"]` / `sample["edge_index"]`,取代分支的 `_atom_count` / `_edge_count`)。两处旧定义删除,`dataloader.py` / `module.py` 改为 import 复用;末状态全树只剩这两个计数函数。
- **generalize `cache.py:491 _flatten` vs 分支 `_walk`** —— 取「保留 + 写明分歧」一支,并改名为 `dataset.py::_flatten_leaves`。理由两条:(a) 语义确实不同 —— `cache._flatten` 是打包期**校验器**,遇到与保留 key 冲突会 `raise ValueError`,而 profiler 的铁律是「诊断不抛」,必须在任意畸形样本上活着走完;(b) CLAUDE.md「第二次真实使用前不抽取」—— `_flatten_leaves` 只有一个调用点,提升进 `_utils.py` 属于过早抽象。分歧不是静默的:其 docstring 点名 `molix.data.cache._flatten` 并写清「不校验保留 key、不抛异常、仅供只读遍历」,且由一条单测钉住(保留 key 样本不抛)。`cache._flatten` 本身一个字不动。
- **reuse-do-not-regress `src/molix/profiler/__init__.py` 与 `trainer.py`** —— `trainer.py` 零改动;`__init__.py` 仅加 `DatasetProfiler` / `DatasetResult` 两行 import、`__all__` 两个名字、docstring 一条 bullet。**绝不** cherry-pick 分支的 `__init__.py`。插入 `__all__` 时按 notes.md 2026-08-09「`__all__` 保持字母序」整体重排(只增不删,现有导出一个不少,分组注释合并为一条说明)。
- **docs** —— 新建 `docs/molix/user-guide/profiling.md`(diátaxis user-guide 层),**不**沿用分支的扁平 `docs/molix/profiling.md`。分支文档的 TrainerProfiler 章节描述的是不存在的 API(phase 表 / `cuda_sync` / `profile_eval`),必须照树内 `trainer.py` 真实接口重写:`TrainerProfiler(model, loss_fn, device, hooks)` + `run(n_steps, n_warmup, batch, top)`,输出三行(raw loop baseline / Trainer loop / Trainer overhead)加 baseline-subtracted hotspot 表。DatasetProfiler / ModuleProfiler 章节可整段搬运后校对。`zensical.toml` 的 Molix → User Guide 增一行 nav。
- **pattern** —— 以 `dataloader.py` 为范本(多组 Result + 分节 `print_report`、`__init__` 收配置 / `run()` 收数据、退化输入 `ValueError` 带补救句、`[WARN]` 不抛、`from __future__ import annotations`、Google docstring、`─`*72 框)。分支那个空 `__init__` 的 `DatasetProfiler` 过不了 CLAUDE.md 的 shape check,按 pattern 改造。

### Iron-law 发现(本次触碰面)

`src/molix/profiler/mock.py:45 _resolve` 从**模块级** `random` 取随机数,绕开了 `MockBatch._rng` / `MockSource` 的 `random.Random(seed)`:`MockSource(seed=0)` 的 docstring 承诺「reproducible」,实际同进程内两次构造得到不同的原子数序列,`MockBatch(seed=0)` 同理形状不稳。这直接毁掉任何以 mock 数据为基准的 golden。**本 spec 内修**(局部、三行级):`_resolve(value, rng: random.Random)` 显式传参,`MockBatch.__call__` 传 `self._rng`,`MockSource.__init__` 传其局部 `rng`;补 `tests/test_molix/test_profiler/test_mock.py`(该模块此前无 mirror,`check_test_mirror` 的 missing 数因此 −1)。本 spec 的单测与 regression 另行只用**字面量样本**构造,不把 golden 建在 RNG 上。

另有两处**不在本次触碰面上**的 docs 腐坏,仅点名并路由 `/mol:fix`,不在此 spec 内改:`docs/molix/user-guide/data-loading.md:18-22` 仍在描述 2026-05-18 已删除的 `GraphBatch` / `AtomData` / `EdgeData` / `GraphData` 子类;`docs/molix/user-guide/md.md` 存在但不在 `zensical.toml` nav 中(可能是有意未发布,不擅自补)。

### 分支注销

移植验收通过后执行 `git tag archive/worktree-molix-profiler-suite 92760f0` 再 `git branch -D worktree-molix-profiler-suite`。先打 tag 是必要的:本次是按文件手工移植而非 merge,直接 `-D` 会让 92760f0 变为不可达并最终被 gc,与「内容保留在历史里」的意图相悖。

## Files to create or modify

- `src/molix/profiler/dataset.py` (new) —— `FieldSpec` / `TargetStat` / `DatasetResult` / `DatasetProfiler` / `_flatten_leaves` / `_sample_bytes`
- `src/molix/profiler/_utils.py` —— 新增 `batch_counts` / `sample_counts`
- `src/molix/profiler/dataloader.py` —— 删 `_extract_batch_counts`,改用 `_utils.batch_counts`
- `src/molix/profiler/module.py` —— 删 `_extract_counts`,改用 `_utils.batch_counts`
- `src/molix/profiler/mock.py` —— `_resolve` 显式接收 `random.Random`
- `src/molix/profiler/__init__.py` —— 追加两个导出 + `__all__` 字母序重排 + docstring 一条 bullet
- `tests/test_molix/test_profiler/test_dataset.py` (new)
- `tests/test_molix/test_profiler/test_mock.py` (new)
- `docs/molix/user-guide/profiling.md` (new)
- `zensical.toml` —— Molix → User Guide 增 `{ "Profiling" = "molix/user-guide/profiling.md" }`
- `regressions/dataset-profiler-salvage.py` (new)

## Tasks

- [ ] Write failing unit tests for DatasetProfiler and the shared count helpers (tests/test_molix/test_profiler/test_dataset.py → TestDatasetProfiler / TestDatasetResult)
- [ ] Generalize `_extract_batch_counts` + `_extract_counts` into `batch_counts` and add `sample_counts` in src/molix/profiler/_utils.py, rebinding src/molix/profiler/dataloader.py and src/molix/profiler/module.py
- [ ] Implement `FieldSpec`, `TargetStat`, `DatasetResult.print_report` and `DatasetProfiler.run` (cache-backed exact fast path + strided access loop + `_flatten_leaves`) in src/molix/profiler/dataset.py
- [ ] Write failing unit test for MockSource / MockBatch seed determinism (tests/test_molix/test_profiler/test_mock.py → TestMockSource / TestMockBatch)
- [ ] Fix the module-level RNG leak in `_resolve` in src/molix/profiler/mock.py by threading the instance `random.Random`
- [ ] Export `DatasetProfiler` / `DatasetResult` from src/molix/profiler/__init__.py and re-alphabetize `__all__` per notes.md 2026-08-09
- [ ] Write docs/molix/user-guide/profiling.md (TrainerProfiler section rewritten against the in-tree raw-loop-baseline API) and add the zensical.toml nav entry
- [ ] Add regression example regressions/dataset-profiler-salvage.py (public API only; hard-coded goldens, no third-party runtime)
- [ ] Delete branch worktree-molix-profiler-suite after tagging `archive/worktree-molix-profiler-suite` at 92760f0
- [ ] Run full check + test suite

## Testing strategy

单测全部落在 `tests/` 镜像路径下,每条只打一个函数/方法;类型镜像(`DatasetProfiler` → `TestDatasetProfiler`,`DatasetResult` → `TestDatasetResult`,`MockSource` → `TestMockSource`)。所有夹具用**字面量样本 dict** 构造(不依赖 RNG),因此期望值可硬编码。

`tests/test_molix/test_profiler/test_dataset.py`

- `TestDatasetProfiler::test_run_uses_exact_cache_counts` —— 100 条字面量样本 → `PackedCache.save` → `CachedDataset`;`DatasetProfiler(n_samples=5).run(ds)` 后 `counts_exact is True`,`max_atoms` / `avg_num_neighbors` 等于 `ds.max_atoms` / `ds.avg_num_neighbors`,`atom_stats.mean` 等于全部 100 条的均值(而非抽的 5 条)。
- `TestDatasetProfiler::test_run_samples_only_n_samples_for_access` —— `n_sampled == 5`,`access_ms` 基于 5 次测量,`cold_access_ms > 0`。
- `TestDatasetProfiler::test_run_fields_come_from_packed_schema` —— `fields` 的 key/axis/dtype/extra_shape 与 `ds.packed_view().payload["schema"]` 逐项一致,`fields_exact is True`。
- `TestDatasetProfiler::test_run_on_plain_sequence_falls_back` —— 输入 `list[dict]`:`counts_exact is False`、`fields_exact is False`、仍返回完整 `DatasetResult`,无异常。
- `TestDatasetProfiler::test_run_without_edge_ptr_warns_not_raises` —— 无 `edge_index` 的 cache:`edge_stats is None`,`warnings` 非空,不抛。
- `TestDatasetProfiler::test_run_rejects_empty_dataset` —— `ValueError`,消息含补救句关键词 `CachedDataset`。
- `TestDatasetProfiler::test_init_rejects_degenerate_config` —— `n_samples=0` / `stride=0` 各抛 `ValueError`。
- `TestDatasetResult::test_print_report_sections` —— `capsys` 断言 `─`*72 框与 Size / Access / Footprint / Fields / Targets 五节标题齐备。
- `TestDatasetResult::test_print_report_flags_nonfinite_targets` —— 含 `nan` 的标签列产生 `[WARN]` 行且 `print_report()` 不抛。
- `test_flatten_leaves_does_not_raise_on_reserved_key` —— 模块级;样本含 `"schema"` 这类 packed 保留 key 时 `_flatten_leaves` 正常返回,钉住与 `molix.data.cache._flatten`(会抛)的有意分歧。

`tests/test_molix/test_profiler/test_mock.py`

- `TestMockSource::test_seed_makes_atom_counts_reproducible` —— 同进程两次 `MockSource(n_samples=8, n_atoms=(5,20), seed=0)` 的 `[len(s["Z"])]` 完全相同。
- `TestMockBatch::test_seed_makes_shapes_reproducible` —— 同进程两次 `MockBatch(n_atoms=(8,32), n_edges=(16,64), seed=0)()` 的形状序列相同。

计数辅助函数的重绑定由既有 `test_dataloader_profiler.py`(`throughput_graphs_per_sec > 0` 必须靠 `batch_counts` 才成立)与 `test_module_profiler.py` 无改动通过来守,不新增重复用例。`_utils.py` 在 `scripts/check_test_mirror.py` 的 `ALLOW_MISSING` 中,无需 mirror 文件。

**回归示例**:`regressions/dataset-profiler-salvage.py` —— 唯一的公开 API 场景脚本。5 条写死在文件内的样本(`Z` / `pos` / `edge_index` / `edge_dist` / `targets.U0`)→ `PackedCache(tmp).save(...)` → `CachedDataset` → `DatasetProfiler(n_samples=5).run(ds)`,逐项对**解析求出的硬编码字面量**断言:`n_total`、`atom_stats.mean`、`max_atoms`、`max_edges`、`avg_num_neighbors`(整数比,`math.isclose(rel_tol=1e-12)`)、排序后的字段 key 列表与其 axis、`targets` 中 `targets.U0` 的 mean/min/max。**不断言任何计时量**(本质非确定)。goldens 为解析值,无任何第三方 oracle、无网络、无 subprocess;文件头注释按 `regressions/README.md` 记录来源(analytic,无 oracle)、commit sha、torch 版本、日期、device=cpu / float32。运行:`PYTHONPATH=src python regressions/dataset-profiler-salvage.py`,成功打印 OK 退出 0,漂移退出 1。

## Out of scope

- **不移植** 分支的 `src/molix/profiler/trainer.py` 与 `tests/.../test_trainer.py`:树内 `TrainerProfiler`(raw-loop-baseline + baseline-subtracted hotspots)已取代该设计,回退即倒退;整分支 merge 的 add/add 冲突正源于此。
- **不移植** 分支的 `test_suite_smoke.py`:那是跨 profiler 的 e2e,CLAUDE.md 禁止 `tests/` 下出现 e2e。若日后确需跨 profiler 冒烟覆盖,另开 spec 放进 `regressions/` 并配硬编码 golden —— 本 spec 判定其**不在范围内**(五个 profiler 的联合冒烟价值低于其维护成本,各自单测已覆盖各自契约)。
- **不移植** 分支的 `conftest.py`(`TinyGraphModel` 无消费者,`test_dataset` 用不到)。
- **不移植** 分支的 `docs/molix/profiling.md` 路径(扁平布局违反 diátaxis),也不新建 `docs/molix/profiling.md`。
- **不动** `molix.data.cache._flatten`、`PackedCache` 单文件布局、`_CacheBacked` 的任何既有属性语义 —— 本 spec 只读不写这些面。
- **不做**:CLI 子命令、接进 Trainer hook 生命周期、GPU 侧 dataset profiling、`DatasetProfiler` 与 `DataLoaderProfiler` 的自动串联建议器。
- **无 Domain basis 章节**:本 spec 不引入任何物理方程或数值物理量;`avg_num_neighbors` 系直接复用既有精确属性,profiler 不做单位换算、不猜单位。
- 已点名但路由到 `/mol:fix` 的相邻腐坏(不在本次触碰面):`docs/molix/user-guide/data-loading.md:18-22` 引用已删除的 `GraphBatch` / `AtomData` / `EdgeData` / `GraphData`;`docs/molix/user-guide/md.md` 未进 `zensical.toml` nav。
