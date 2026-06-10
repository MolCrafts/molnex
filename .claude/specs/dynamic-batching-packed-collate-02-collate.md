---
title: Packed 感知的 collate 快速路径（packed-aware collate fast path）
status: approved
created: 2026-06-10
---

# Packed 感知的 collate 快速路径

## Summary

当前数据通路对每个 batch 先经 `PackedCache.unpack_sample` 逐样本重建 flat dict，再由 `collate_molecules` 重新 `torch.cat` 拼回大张量 —— 对本就以 packed 连续布局存盘的缓存而言是一次"拆了再拼"的纯开销。本规格新增 `collate_packed` 快速路径：给定样本索引列表，直接按 `atom_ptr` / `edge_ptr` 切片 PackedCache 的 packed 张量、向量化重定基 `edge_index`、按 `TargetSchema` 路由 targets，一步构出嵌套 `atoms/edges/graphs` TensorDict。其输出与对相同索引逐样本 dict 调用 `collate_molecules` 的结果**逐叶完全相等**（等价性 oracle，`src/molix/data/collate.py:62-174`）。数据集侧新增只读 `packed_view()` 访问器，使 collate 代码不再触碰私有 `_payload`；`DataModule` 在构造期检测 packed 能力并自动路由，非 packed 数据集与 DDP 场景完全回退到现有逐样本路径，用户可见行为（batch 内容、dtype、`batch_nodes` 后处理、spawn worker 可 pickle 性）不变。

## Design

**等价性 oracle 是唯一正确性标准。** `collate_packed(view, indices, target_schema)` 的输出必须与 `collate_molecules([dataset[i] for i in indices], target_schema)` 在键集合、每叶 `torch.equal`、dtype、各层 `batch_size` 上全部一致。`collate_molecules` 只消费 `Z`、`pos`、`edge_index`、`bond_diff`、`bond_dist`、`targets.*` 并忽略其余键 —— 快速路径同样只处理这组键。

**(a) `collate_packed`（`src/molix/data/collate.py`）。** 设 `idx` 为重映射后的 packed 索引张量，`B = len(indices)`：

- 原子级：`counts = atom_ptr[idx+1] - atom_ptr[idx]`；`Z`/`pos` 取 `torch.cat` 的逐样本切片（视图 cat，无中间 dict），`Z` 与 `edge_index` 镜像 oracle 施加 `.long()`。`batch = repeat_interleave(arange(B), counts)`，偏移 `offsets = cumsum(counts) - counts`。
- 边级：`e_counts = edge_ptr[idx+1] - edge_ptr[idx]`；拼接的 `(E, 2)` `edge_index` 两列同加 `repeat_interleave(offsets, e_counts).unsqueeze(1)`（向量化重定基）；`bond_diff` / `bond_dist` 同法切片拼接。schema 中无 edge 键时，构造与 `collate.py:149-156` **逐字节一致**的空边回退 TensorDict（`edge_index zeros(0,2) long`、`bond_diff zeros(0,3)`、`bond_dist zeros(0)`、`batch_size=[0]`）；含 0 边样本的混合 batch 走正常拼接分支（与 oracle 行为一致）。
- targets：剥去点分前缀 `targets.`，按 `TargetSchema` **名称**路由（与 oracle 完全一致）：`atom_level` → cat 进 `atoms`；其余 → 从 `graphs` 桶 `index_select(0, idx)` 后复现 oracle 的 `reshape(-1)` / cat 语义，scalar 桶（python 标量 targets）经 `torch.tensor` 复现 oracle 的转换路径。`graphs` 层固定含 `num_atoms`（long）。
- 急切报错：空 `indices` 抛与 oracle 同文案的 `ValueError`；schema 缺 `Z` 或 `pos` 时抛出指明 cache 来源与缺键的可操作 `ValueError`。

**(b) `PackedView` 只读访问器（`src/molix/data/dataset.py`）。** 轻量 dataclass，暴露快速路径所需的最小面：payload 句柄、`atom_ptr` / `edge_ptr`、点分键 schema（即 `cache.py:332` `_infer_schema_across` 产出、存于 `payload["schema"]`）、`map_indices(local) -> packed` 重映射。`_CacheBacked.packed_view()` 返回恒等映射视图；`SubsetDataset.packed_view()` 经 `self._indices` 重映射后委托底层数据集（底层无 packed 能力则不提供该方法，由 `__getattr__` 语义自然处理）。**`cache.py` 零改动**——不 import `tensordict` / `TargetSchema`，分层维持 cache = IO/格式、dataset = 视图、collate = 批构造。

**(c) DataLoader 接线（`src/molix/data/datamodule.py`）—— 选定方案：索引恒等包装数据集。** 两个候选中选 `_IndexDataset`（`__getitem__(i) -> i`、`__len__` 同真实数据集）+ `_PackedCollateFn` 接收 `list[int]`，**不选** `__getitems__` 钩子：后者要求数据集在取数时就完成 collate，迫使 dataset 知晓 `TargetSchema`（DataModule 的配置关切）并 import tensordict，破坏 (b) 刚建立的分层；而恒等包装对 sampler / shuffle / `batch_sampler` 完全透明（它们只依赖 `__len__`），与 chain link -01 引入的可选动态 batch sampler 接线正交组合——`-01` 的 sampler 工厂产出的 batch_sampler 原样传给 DataLoader，无论 dataset 实参是真实数据集（回退路径）还是 `_IndexDataset`（快速路径）。`DataModule` 在构造 dataloader 时按数据集逐个检测（`callable(getattr(ds, "packed_view", None))`）：命中且非 DDP（`_is_distributed()` 为假，DDP 不在本规格范围）→ `DataLoader(_IndexDataset(len(ds)), ..., collate_fn=_PackedCollateFn(ds, schema, batch_nodes))`；否则维持现状 `_CollateFn` + `collate_molecules`，逐样本 map-style 路径不动。`_PackedCollateFn` 复用 `_CollateFn` 的后处理契约：构造期捕获 `config["ftype"]`，每个 batch 依次过 `batch_nodes` 再 `batch_to(dtype=ftype)`。可 pickle 性：callable 的 pickled state 只含**数据集引用**（payload mmap 经数据集 pickle 在 worker 内重新打开，复用 `test_e2e_workers.py` 已验证的 `MmapDataset` 机制），`PackedView` 在首次 `__call__` 时惰性构建并由 `__getstate__` 剔除，绝不直接捕获 payload 张量。

## Files to create or modify

- src/molix/data/collate.py
- src/molix/data/dataset.py
- src/molix/data/datamodule.py
- tests/test_molix/test_data/test_collate_packed.py (new)

## Tasks

- [ ] Write failing equivalence tests for collate_packed vs collate_molecules (tests/test_molix/test_data/test_collate_packed.py): multi-sample with edges, edge-less samples mixed in, all-edge-less fallback, graph+atom targets per TargetSchema, scalar targets, SubsetDataset views, singleton batches — asserting torch.equal per leaf, same key sets, same dtypes, same batch_size attrs
- [ ] Implement PackedView and packed_view() accessors on _CacheBacked and SubsetDataset in src/molix/data/dataset.py (read-only; local→packed remap; cache.py untouched, no tensordict/TargetSchema import there)
- [ ] Implement collate_packed in src/molix/data/collate.py (vectorized edge_index rebase via repeat_interleave, batch column, TargetSchema routing, empty-edge fallback identical to collate.py:149-156, eager actionable ValueError)
- [ ] Write failing integration tests for DataModule fast-path routing in tests/test_molix/test_data/test_collate_packed.py: num_workers=0 and spawn num_workers=2 end-to-end, fallback for non-packed dataset, batch_nodes + ftype cast applied, collate callable pickles without captured payload tensors
- [ ] Implement _IndexDataset and _PackedCollateFn wiring in src/molix/data/datamodule.py (construction-time detection, DDP and non-packed fallback to _CollateFn, batch_nodes + batch_to post-steps, lazy view excluded from __getstate__, -01 batch_sampler passthrough preserved)
- [ ] Add Google-style docstrings with tensor shapes for collate_packed, PackedView, packed_view, _IndexDataset, _PackedCollateFn in src/molix/data/collate.py, src/molix/data/dataset.py, src/molix/data/datamodule.py
- [ ] Run full check + test suite

## Testing strategy

- **等价性（happy path）**：构造经 `PackedCache` round-trip 的小数据集，对同一索引列表比较 `collate_packed` 与 `collate_molecules`：键集合相同、每叶 `torch.equal`、dtype 一致、`atoms/edges/graphs` 与顶层 `batch_size` 一致。覆盖多样本含边 batch、graph+atom 双路 targets、scalar targets。
- **边界**：混入 0 边样本的 batch；全体无 edge 键时的空边回退（与 `collate.py:149-156` 字段、dtype、`batch_size=[0]` 逐项相等）；单样本 batch；`SubsetDataset` 视图（含 split 产出的乱序索引）；空索引列表与缺 `Z`/`pos` schema 的急切 `ValueError`。
- **集成**：`DataModule` + packed 数据集在 `num_workers=0` 与 spawn `num_workers=2` 下各迭代一个 epoch，batch 与慢路径输出相等且浮点叶 dtype 等于捕获的 `ftype`、`batch_nodes` 生效；非 packed 数据集（如 `InMemorySource` 直连的伪数据集）回退 `_CollateFn`。
- **快速路径冒烟（性能声明的代码级体现）**：monkeypatch 计数 `PackedCache.unpack_sample`，断言快速路径 collate 期间调用次数为 0。
- **分层守护**：静态断言 `src/molix/data/cache.py` 源文本不含 `tensordict` / `TargetSchema` import。
- **可 pickle 性**：`pickle.dumps/loads(_PackedCollateFn(...))` round-trip 后仍产出正确 batch，且 pickled state 不含惰性视图字段。

## Out of scope

- 任何盘上缓存格式变更（`cache.py` 模块 docstring 明令禁止；`FORMAT_VERSION` 不动）。
- DDP 下启用快速路径（检测到分布式时回退现有路径）。
- Bucketing / 长度分桶采样策略。
- `collate_molecules` 本身的任何行为修改（它是等价性 oracle，必须保持不动）。
