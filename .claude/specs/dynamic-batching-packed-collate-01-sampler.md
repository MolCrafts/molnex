---
title: Token-budget dynamic batch sampler over PackedCache pointers
status: approved
created: 2026-06-10
---

# Token-budget dynamic batch sampler over PackedCache pointers

## Summary

为训练 DataLoader 引入基于 token 预算（每 batch 原子总数和/或边总数上限）的动态 batch 采样器。分子数据集中样本尺寸差异极大（QM9 从 3 原子到 29 原子），固定 `batch_size` 导致显存利用率随 batch 内样本组成剧烈波动；按预算贪心装填可使每个 batch 的计算量近似恒定。采样器的逐样本原子/边计数完全来自 `PackedCache` 的 `atom_ptr` / `edge_ptr` 指针向量（`ptr[idx+1] - ptr[idx]`），采样期间不调用任何 `__getitem__`、不解包样本。`DataModule` 通过新增的 opt-in 关键字参数接入：不启用时默认路径的 DataLoader 配置逐字节不变（完全向后兼容）；启用时与 DDP 互斥并在构造 dataloader 时立即抛出可操作的 `ValueError`。每个 epoch 通过 `make_generator(seed + epoch)` 重新洗牌，与现有 `DataModule.on_epoch_start` 记录 epoch 的语义一致，保证从 epoch k 恢复训练时可重新推导出 epoch k 的 batch 组成。

## Design

**新模块 `src/molix/data/sampler.py`**（单词模块名，与 `dataset` / `cache` / `collate` 一致），定义：

```python
class TokenBudgetBatchSampler:  # 迭代产出 list[int]，即 torch.utils.data.Sampler[list[int]] 形状
    def __init__(
        self,
        dataset: BaseDataset,
        *,
        max_atoms: int | None = None,
        max_edges: int | None = None,
        seed: int = 42,
    ) -> None: ...
    def __iter__(self) -> Iterator[list[int]]: ...
    def __len__(self) -> int: ...
```

- **预算校验（eager，在 `__init__`）**：`max_atoms` 与 `max_edges` 至少给定一个，二者可同时生效（一个 batch 必须同时满足两个上限）；任一给定值 `<= 0` 时抛多子句 `ValueError`，指明参数名、收到的值、修复方式。
- **计数来源**：构造时读取 `dataset.atom_counts` /（需要时）`dataset.edge_counts`（见下）。目标数据集没有该访问器（非 `PackedCache` 后端）→ eager `ValueError`，明确列出受支持类型：`MmapDataset`、`CachedDataset`、以及包装二者的 `SubsetDataset`。
- **批组成算法**：用 `make_generator(seed)`（即 `torch.Generator().manual_seed`）做 `randperm(n)`，按洗牌顺序贪心装填：当前样本加入后任一预算超限则封闭当前 batch、开新 batch。单个样本自身即超预算时，独占一个 singleton batch——**绝不丢弃**，并通过 `warnings.warn` 每个采样器实例只告警一次。
- **`__len__`**：返回当前 epoch 排列下的 batch 数。batch 列表在构造时一次性贪心算出并缓存（`O(n)`，纯指针算术），`__len__` / `__iter__` 直接读缓存。文档明确：**batch 数随 epoch 排列变化**，不同 epoch 的 `len()` 可能不同。
- **可腌制性（picklability）**：实例只持有数据集引用、整数预算、seed 与预算算出的 `list[list[int]]`——全部可 pickle，兼容 `spawn` worker。文档注明：`batch_sampler` 由 DataLoader 在主进程消费，worker 只收到索引列表，因此采样器本身不会被发送到 worker，但保持可腌制以防用户跨进程传递 DataModule。

**`src/molix/data/dataset.py` — 泛化指针统计访问器**：

- `_CacheBacked` 新增 `@cached_property atom_counts -> torch.Tensor` 与 `edge_counts -> torch.Tensor`，形状 `(n_samples,)`、dtype `long`，由 `ptr[1:] - ptr[:-1]` 一次性向量化算出（泛化 dataset.py:194-208 中 `max_atoms` / `max_edges` 的逐差数学）。对应指针缺失（cache 无 per-atom / per-edge 键）时抛 `ValueError`，说明该 cache 缺哪个指针、受支持的产生方式。现有 `max_atoms` / `max_edges` 的 0 回退契约**保持不变**（内部实现可改为复用 counts 的非异常路径，但对外行为不动）。
- `SubsetDataset` 新增同名 `@cached_property`：`self._dataset.atom_counts[torch.as_tensor(self._indices)]` —— 沿用 dataset.py:298-349 中 `avg_num_neighbors` / `max_atoms` 的 local→packed 重映射语义，且天然支持嵌套 subset。必须显式定义而不能依赖 `__getattr__` 转发（转发会返回**全集**计数，索引错位）。

**`src/molix/data/datamodule.py` — opt-in 接线**：

- `DataModule.__init__` 新增 `max_atoms_per_batch: int | None = None`、`max_edges_per_batch: int | None = None`。二者均为 `None`（默认）时走现有固定 `batch_size` 路径，传给 `DataLoader` 的参数完全不变。
- `train_dataloader()`：预算启用时，在现有 DDP 分支点（datamodule.py:171-204）首先检查 `_is_distributed()`，为真则抛 `ValueError`（说明动态 batch 与 `DistributedSampler` 不兼容、当前修复是去掉预算参数或单进程运行、DDP 支持属后续 spec）。非 DDP 时构造 `TokenBudgetBatchSampler(self.train_dataset, max_atoms=..., max_edges=..., seed=self.seed + self._epoch)` 并以 `batch_sampler=` 传入 DataLoader；此时**不得**传 `batch_size` / `shuffle` / `sampler` / `drop_last` / `generator`（PyTorch 规定与 `batch_sampler` 互斥），其余 kwargs（`num_workers`、`pin_memory`、`persistent_workers`、`prefetch_factor`、`collate_fn`、`multiprocessing_context`、`worker_init_fn`）原样保留。
- **epoch 重洗**：沿用现有 `generator = make_generator(self.seed + self._epoch)` 模式——`on_epoch_start(epoch)`（datamodule.py:256-272）记录 `self._epoch`，每次 `train_dataloader()` 调用都新建采样器，故 epoch k 恢复训练时（fresh DataModule + `on_epoch_start(k)`）重新推导出与连续训练完全相同的 batch 组成，无需 checkpoint 生成器状态。
- `val_dataloader()` 保持固定 `batch_size` 不动：评估无反向传播、显存余量大，且固定 batch 使指标累积逻辑简单；预算化 eval 留在 Out of scope。

**`src/molix/data/__init__.py`**：从 `molix.data.sampler` re-export `TokenBudgetBatchSampler` 并加入 `__all__`。

所有新公开符号按 Google docstring 风格，张量形状用 ``(n_samples,)`` 反引号标注。

## Files to create or modify

- `src/molix/data/sampler.py` (new) — `TokenBudgetBatchSampler`
- `src/molix/data/dataset.py` — `_CacheBacked.atom_counts` / `edge_counts`、`SubsetDataset.atom_counts` / `edge_counts`
- `src/molix/data/datamodule.py` — opt-in kwargs、`train_dataloader` 的 `batch_sampler` 接线与 DDP 抛错
- `src/molix/data/__init__.py` — re-export `TokenBudgetBatchSampler`
- `tests/test_molix/test_data/test_sampler.py` (new) — 全部新测试

## Tasks

- [ ] Write failing tests for pointer-derived count accessors, incl. SubsetDataset local→packed remap and nested subsets (tests/test_molix/test_data/test_sampler.py, reuse test_datamodule.py `_make_samples` + `PackedCache(...).save` conventions)
- [ ] Implement `atom_counts` / `edge_counts` as `@cached_property` on `_CacheBacked` and `SubsetDataset` in src/molix/data/dataset.py, keeping `max_atoms` / `max_edges` zero-fallback behavior unchanged
- [ ] Write failing tests for `TokenBudgetBatchSampler` in tests/test_molix/test_data/test_sampler.py: budget compliance (atom-only, edge-only, both), exact-once coverage per epoch, oversize singleton + warn-once, seed determinism vs. cross-seed reshuffle, eager ValueError for missing/non-positive budgets and non-packed datasets, picklability
- [ ] Implement `TokenBudgetBatchSampler` in src/molix/data/sampler.py (new) and re-export it from src/molix/data/__init__.py
- [ ] Write failing tests for DataModule wiring in tests/test_molix/test_data/test_sampler.py: opt-in path passes `batch_sampler` and omits batch_size/shuffle/sampler/drop_last/generator, default path DataLoader config identical to pre-change, `_is_distributed()` monkeypatched True raises ValueError, fresh-sampler-at-epoch-k equals continuous-training epoch-k batch composition
- [ ] Implement `max_atoms_per_batch` / `max_edges_per_batch` kwargs, `train_dataloader` batch_sampler wiring at the DDP decision point, and the DDP ValueError in src/molix/data/datamodule.py
- [ ] Add Google-style docstrings with tensor shapes for all new public symbols, documenting per-epoch-varying `__len__`, main-process batch_sampler consumption, and why val keeps fixed batch_size
- [ ] Run full check + test suite

## Testing strategy

- **Happy path — 预算遵守**：对已知计数分布构造 cache，断言每个产出 batch 的 `atom_counts[batch].sum() <= max_atoms`（及 `edge_counts` 同理；双预算同时启用时两者皆查），唯一允许的例外是文档化的超额 singleton。
- **Happy path — 全覆盖**：一个 epoch 内所有 batch 索引拼接后排序 == `range(len(dataset))`，每个索引恰好出现一次，无丢样。
- **边界 — 超额 singleton**：单样本计数 > 预算 → 自成 batch、不丢弃、`pytest.warns` 捕获恰好一次告警。
- **边界 — 确定性与重洗**：同 `seed` 两次构造产出逐 batch 相同的索引序列；`seed + k` 与 `seed + k'`（k ≠ k'）产出不同排列；fresh DataModule 在 `on_epoch_start(k)` 后的 `train_dataloader()` 与连续训练到 epoch k 的 batch 组成完全一致（resume 可重 derive）。
- **边界 — eager 校验**：无预算、零/负预算、非 packed 数据集（裸 `BaseDataset` 子类）、`_is_distributed()` 为真（monkeypatch `dist.is_available` / `is_initialized`）四种情形分别抛 `ValueError`，消息包含修复指引。
- **边界 — 向后兼容**：不传新 kwargs 时，检查 `train_dataloader()` 返回的 DataLoader 的 `batch_size` / `sampler` / `batch_sampler` / `drop_last` / `generator` 等属性与改动前配置一致（feature off 即字节级等价路径）。
- **边界 — SubsetDataset 重映射**：`split()` 出的子集上构造采样器，断言预算按子集自身索引的 packed 计数计算（与手算 `atom_ptr[packed_idx+1] - atom_ptr[packed_idx]` 对照），嵌套 subset 同样正确。
- **不解包约束**：monkeypatch `__getitem__` 计数器，断言采样器构造与迭代期间调用次数为 0。

## Out of scope

- **val/test dataloader 的预算化**：评估侧无反向传播、显存余量充足，固定 `batch_size` 保持指标累积与样本遍历语义简单；如后续确有需要，可在新 spec 中以同一采样器对 `val_dataloader` 做可选接线。
- **DDP 下的动态 batch**（需 DistributedSampler 等价的跨 rank 切分协议）——本 spec 只保证 eager 抛错。
- **装填最优性**：仅贪心（first-fit in shuffled order），不做 bin-packing 优化或按尺寸排序。
- **生成器状态 checkpoint**：沿用 `seed + epoch` 重推导语义，不持久化 RNG 状态。
