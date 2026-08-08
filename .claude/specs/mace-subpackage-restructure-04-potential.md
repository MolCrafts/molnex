---
title: MACE 子包重构 04 — 统一 MACEPotential（单态流水线 + 公共扁平能量核）
status: approved
created: 2026-08-08
revised: 2026-08-08
grilled: true
chain: mace-subpackage-restructure
---

# MACE 子包重构 04 — 统一 MACEPotential

## Summary

把今天分散在 `src/molzoo/mace_matpes.py` 与 `src/molzoo/mace_omol.py` 两份平铺模块里的
「能量 + 力」前向合并成 **一个** `MACEPotential`，落在 `src/molzoo/mace/potential.py`，由 02 的
`MACEMatpesSpec` / `MACEOMolSpec` 构造出两种 foundation 变体。所有分支（是否求力、是否带
charge/spin 条件化）在 `__init__` 里解析成一条绑定好的 `self._pipeline`，`forward` 只剩一行分发——
`energy_forces(..., compute_forces=bool)` 这类**每次调用再决定**的旗标从公共面消失，「只要能量」
从此等于「构造一个只算能量的 potential」（这正是 `src/molix/md/forcefield.py:158-166`
`calc_energy` 已经写死的契约）。同时把原先私有的 `_compute_energy` 提升为**公共、可编译的扁平
能量核** `energy_core(positions, Z, edge_index, batch, num_graphs, shifts=None)`（1 个 dynamo 图、
0 个 break），给 `scripts/matpes_port/run_nve.py` 和 `benchmarks/bench_mace_matpes.py` 现在偷用私
有方法的地方一个正式接缝（这两处调用点由 07 改指），并实现 batch 级 `_write_energy(batch)` 钩子，
让 `molpot.derivation.protocol.call_energy`（`protocol.py:58-68`）零成本把 MACE 接进
`EnergyReadout` / `ForceReadout`。用户可见结果：MACE 的能量/力入口只有一个类、两个接缝，数值与
今天的平铺模型在 `state_dict` 直接迁移下逐位一致。

## Domain basis

MACE 等变消息传递（Batatia et al., *MACE: Higher Order Equivariant Message Passing Neural
Networks for Fast and Accurate Force Fields*, NeurIPS 2022, arXiv:2206.07697），foundation 变体分别对
应 MACE-MP-0（arXiv:2401.00096）与 MatPES（arXiv:2503.04070）；两个 checkpoint 的 paper↔code 契约
已固化在 `src/molzoo/specs/mace_matpes.md` / `src/molzoo/specs/mace_omol.md`，本 spec 不改动其中任何
一条数学契约，只搬运宿主。

本 spec 依赖的物理约束（全部由测试兜底）：

- 能量分解与力的定义：`E = Σ_i E_i`，`F_i = -∂E/∂r_i`，解析导数由 autograd 给出（cuEq 融合核不支持
  `torch.func.grad`，见 `.claude/notes/notes.md` §"Force derivation — dual explicit backends"）。
- 周期边：`r_ij = pos[target] - pos[source] + s_ij`，`s_ij = unit_shifts @ cell`；边约定全程 `(E, 2)`，
  `[:,0]=source`、`[:,1]=target`，**端到端不转置**（`scripts/matpes_port/run_nve.py:136-138` 直接把
  邻居表的 `(capacity, 2)` 缓冲喂进能量核，转置会多一次拷贝并破坏 CUDA graph 捕获）。
- 因为 `E` 只经 `r_ij` 依赖坐标：平移/旋转不变，且 `Σ_i F_i = 0`（牛顿第三定律）。
- 单位：`graphs.energy` 为 eV `(B,)`，`atoms.forces` 为 eV/Å `(N, 3)`；MD 单位桥（amu·Å²/fs²）由
  `molix.md.forcefield` 的 `energy_scale` 负责，potential 内部不做单位换算。
- 数值不变量（重构专属）：搬家不得改变数字。fp64/CPU 下相对今天的平铺模型 `max|ΔE| ≤ 1e-12 eV`、
  `max|ΔF| ≤ 1e-12 eV/Å`（-100 eV 量级上约 1e-14 相对误差，贴着 fp64 eps；同算子序时实测应为逐位相等）。

## Design

### 实体

新增一个类 `MACEPotential(MACEEncoder)`，位于 `src/molzoo/mace/potential.py`；它**继承** 02 的
`MACEEncoder`（spec-audit 裁定 2026-08-08：链式不变量 ac-007 要求 state_dict 键**扁平**、
`load_state_dict(strict=True)` 直接搬运，"持有"会引入 `encoder.` 前缀；继承与上游
`ScaleShiftMACE(MACE)` 同形，也与 02 的设计一致），并绑定 03 的力核。构造只接受一个 spec 对象：

    MACEPotential(spec: MACEMatpesSpec | MACEOMolSpec, *, compute_forces: bool = True,
                  use_fallback: bool = False)

`compute_forces` 默认 `True`（与两个 foundation checkpoint 今天 `energy_forces(compute_forces=True)`
的默认一致，也是 MD / 评测的主用途）；`PiNetPotential` 默认 `False` 是因为它主要走训练路径——两者
不同是**有意**的，各自记在 docstring 里。`use_fallback` 默认 `False`：foundation 变体的力永远走
autograd，functorch 兼容性理由不成立，纯 torch 路径是实测 **35.7x** 的倒退
（`.claude/notes/notes.md` `mol:note:topic:cueq-use-fallback`）；CPU / 测试调用点显式传 `True`。

### 两个共存接缝（必须都在 docstring 里写清）

1. **`energy_core(...) -> (B,)` — 编译目标、扁平张量、公共。**

       def energy_core(self, positions, Z, edge_index, batch, num_graphs,
                       shifts=None, total_charge=None, total_spin=None) -> Tensor

   前 6 个参数的位置与名字**必须**与今天的 `_compute_energy` 完全一致，07 改指调用点时才是纯改名。
   尾部两个 per-graph 条件张量只有 OMol 变体消费；MatPES 变体收到非 `None` 直接 `ValueError`（不静默
   忽略）——该判断对 dynamo 是常量折叠，不产生 break。核内部重算所有由坐标派生的几何量（边矢量、
   长度、球谐、径向），因此可被 `torch.autograd.grad` 微分；不接触 TensorDict，不做 `.item()` 主机同步
   （`num_graphs` 是入参，正是为此）。目标：`torch._dynamo.explain` 下 1 图 0 break。

2. **`_write_energy(batch) -> batch` — protocol 钩子、TensorDict 级、只写能量。**
   `molpot.derivation.protocol.call_energy`（`protocol.py:58-68`）优先找 `model._write_energy`，实现它
   等于免费接入 `EnergyReadout` / `ForceReadout`。它从 batch 取张量、调 `energy_core`、经
   `protocol.write_energy` 写回，**不 detach 位置、也不 detach 能量**——它可能被外层 readout 会话在自己
   的 leaf 上调用，detach 会切断 `F = -∂E/∂x`。这一点与今天平铺
   `energy_forces(compute_forces=False)` 里的 `positions.detach()` 是**有意的分歧**：MD 侧的 detach 由
   `PotentialForceField`（`forcefield.py:154, 166`）负责，potential 不该抢。

   `_write_energy` 同时是 `compute_forces=False` 时 `self._pipeline` 绑定的那条路径。

### 单态流水线

`__init__` 末尾（照抄 `src/molzoo/pinet/potential.py:84-94` 的形状）：

    self._conditioning = self._omol_conditioning if spec_is_omol else self._no_conditioning
    self._pipeline = self._pipeline_ef if compute_forces else self._write_energy

`forward(batch)` 只有一行 `return self._pipeline(batch)`。公共面上**任何地方**不再出现
`compute_forces` 参数。

### `forward` 必须原样保留的 forcefield 契约

读 `atoms.{Z,pos,batch}`、`edges.edge_index` `(E,2)`、可选 `edges.shifts` `(E,3)`；写
`graphs.energy` `(B,)` + `atoms.forces` `(N,3)`。三个易碎细节：

- **`needs_leaf` 语义**：`needs_leaf = not (pos.requires_grad and pos.is_leaf)`；已经是 grad leaf 时
  **不得** detach 能量（训练路径靠这条把 loss 接到参数上），否则 detach。传给 03 力核的
  `detach_energy` 就是这个布尔。
- **`graphs` 命名空间的 batch_size**：缺失时以 `TensorDict({}, batch_size=[num_graphs])` 建好，**再**调
  `protocol.write_energy`。顺序不能反：`protocol.ensure_graphs`（`protocol.py:71-75`）建的是
  `batch_size=[]`，先调它会把 schema 静默降级（`graphs` 命名空间约定 `batch_size=[B]`，见 CLAUDE.md
  post-collate schema）。这是**已知债务**，`ensure_graphs` 本身由 07 修；本 spec 只在本地把顺序写对，
  并在 docstring 注明原因，绝不「顺手」按 07 的写法提前改公共 helper。
- **`num_graphs` 的取法**：优先 `td["graphs"].batch_size[0]`（静态形状），无 `graphs` 命名空间时才退回
  `int(batch.max().item()) + 1`（一次主机同步）。`forward` 不是编译目标，`energy_core` 才是。

元素表一次性校验保持今天的语义（`mace_matpes.py:206-222`）：`MACEPotential` 只持 `_elements_validated`
闸门，`validate_elements` 的**实现留在 z_table 的所有者**（02 的 encoder），不在 potential 里复制。

### Reuse decision

- `reuse` `molpot.derivation.protocol.write_energy` / `write_forces`（`protocol.py:78-95`）—— 唯一写回
  路径，不自己 `batch["graphs","energy"] = ...`。
- `reuse` `molpot.derivation.protocol.call_energy`（`protocol.py:58-68`）—— 被动满足其 `_write_energy`
  钩子契约，不新增注册机制。
- `reuse`（间接）`molpot.derivation.force.autograd_forces_from_energy`（`force.py:99-124`）—— 经 03 的
  力核调用；`potential.py` 内**不得**出现 `torch.autograd.grad` / `torch.func.grad`（第三条 force path
  是明令禁止的，见 CLAUDE.md "Force derivation (dual backends)"）。
- `reuse` 03 的共享力核（`molpot.derivation.kernels` 的公共导出，即 `grad_force_pass`）—— 以
  `detach_energy` 由 `needs_leaf` 驱动的方式绑定；若 03 落地的导出名不同以 03 为准，若 03 的核**没有**
  `detach_energy` 旋钮，那是 03 的 rework，不是这里手搓一条力路径的理由。
- `reuse` 02 的 `MACEMatpesSpec` / `MACEOMolSpec` / encoder / geometry（`src/molzoo/mace/{spec,encoder,
  geometry}.py`）—— potential 只做编排，不重算几何、不重建块。
- `reuse` `molzoo.mace.MACEEncoder.validate_elements`（02 已随 z_table 迁入 encoder）—— 遗漏则视为 02
  的 rework，在 encoder 补 4 行，不在 potential 复制实现。
- `reuse` `molix.schema` 的 `ENERGY_KEY` / `FORCES_KEY` / `POS_KEY`（`schema.py:20-26`）与
  `molix.F.scatter.scatter_sum_compile_safe`（经 encoder）。
- `pattern` `molzoo.pinet.potential.PiNetPotential`（`pinet/potential.py:84-94`）—— 沿用其单态构造形
  状、命名（`_pipeline`、`_write_energy`）与错误风格；**不**为两者抽公共基类：只有两个调用点，且
  CLAUDE.md "Inline until the second real use" / 禁一次性抽象。
- `new — MACEPotential` 本身：今天两份平铺 `forward` 是同形复制，合并是本 spec 的产出，无现存符号可复用。
- 已知缺口，**不在本 spec 修**（照 Iron law 点名并路由）：`molpot.composition.EnergyForceModel` 至今
  零生产子类（`.claude/notes/architecture.md:160`），本 spec 既不新增其子类也不删它，07 处理；
  `protocol.ensure_graphs` 的 `batch_size=[]` 缺陷由 07 修（上文已本地兜住）。

## Files to create or modify

- `src/molzoo/mace/potential.py` (new) —— `MACEPotential`（`energy_core` / `_write_energy` /
  `_pipeline_ef` / `forward`）。
- `src/molzoo/mace/__init__.py` —— 由 02 创建；本 spec 只追加 `MACEPotential` 导出与一行布局说明。
- `src/molzoo/mace/encoder.py` —— 由 02 创建；仅当其未随 z_table 暴露 `validate_elements` 时在此补该
  方法（不在 potential 复制实现）。
- `tests/test_molzoo/test_mace/test_potential.py` (new) —— 镜像单测，类名 `TestMACEPotential`。
- `regressions/mace-subpackage-restructure-04-potential.py` (new)。

（`src/molzoo/mace_matpes.py`、`src/molzoo/mace_omol.py`、`src/molzoo/__init__.py`、
`scripts/matpes_port/run_nve.py`、`benchmarks/bench_mace_matpes.py` 在本 spec **只读**：平铺模块此时仍
存活并充当 parity 参照，导入兼容归 06、调用点改指与删除归 07。）

## Tasks

- [ ] Write failing unit tests for MACEPotential contract in `tests/test_molzoo/test_mace/test_potential.py` (→ `TestMACEPotential`): forward E/F shapes, `graphs` created with `batch_size=[num_graphs]`, energy-only pipeline writes no `atoms.forces`, `needs_leaf` grad semantics, `call_energy` hits `_write_energy`, unknown-element `ValueError`, and the no-`compute_forces` public-surface guard
- [ ] Write failing parity + physics tests: `MACEPotential` vs flat `MACEMatpes` / `MACEOMol` under direct `load_state_dict` (fp64, seed 0, `use_fallback=True`), identical `state_dict` key set + shapes, `Σ F = 0`, translation invariance, finite-difference forces
- [ ] Write failing compile smoke test for `energy_core` (`torch._dynamo.explain` → 1 graph, 0 breaks, CPU-sized model, `@pytest.mark.slow`)
- [ ] Implement `MACEPotential.energy_core` and `_write_energy` in `src/molzoo/mace/potential.py` (new) — flat compile seam + protocol hook, `protocol.write_energy` write-back, no detach inside the energy path
- [ ] Implement the construction-time monomorphic pipeline in `MACEPotential.__init__` (spec-driven variant + conditioning binding, `compute_forces` resolved to `self._pipeline`, `use_fallback=False` default) binding the 03 force kernels — no `autograd.grad` / `func.grad` in this file
- [ ] Implement `MACEPotential.forward(td)` preserving the `molix.md.forcefield` contract: `(E,2)` edges untransposed, optional `edges.shifts`, `needs_leaf` detach policy, `graphs` namespace built with `batch_size=[num_graphs]` before `write_energy`, once-per-instance element gate delegating to the encoder's `validate_elements`
- [ ] Export `MACEPotential` from `src/molzoo/mace/__init__.py` and add Google-style docstrings documenting both seams (`energy_core` vs `_write_energy`), units (eV, eV/Å), the `use_fallback` rationale, and the paper refs
- [ ] Add regression example `regressions/mace-subpackage-restructure-04-potential.py` (public API only; goldens hard-coded from the in-repo flat `MACEMatpes` at implementation time with a generator comment; no third-party runtime)
- [ ] Run full check + test suite (`ruff check src/ && ruff format --check src/`, `python -m pytest tests/ -v`)

## Testing strategy

单测只放 `tests/test_molzoo/test_mace/test_potential.py`（镜像 `src/molzoo/mace/potential.py`，类
`TestMACEPotential`），每个用例只打一个方法。全部用 CPU、fp64（`molix.config`）、`torch.manual_seed(0)`、
`use_fallback=True` 的**小模型**（2 元素、`num_features=8`、`l_max=1`、2 层），秒级完成。

**前置守卫（第一条测试就断言）**：`tests/test_molzoo/test_mace/` 是包，且
`tests/test_molzoo/test_mace.py` 已不存在——两者同名会让包遮蔽模块、旧用例被静默漏收。仍然存在即
02 的 rework，硬停，不在本 spec 绕过。

- Happy path：`forward` 写出 `graphs.energy (B,)` + `atoms.forces (N,3)`；`energy_core` 直调返回 `(B,)`
  且与 `forward` 的能量一致；带 `edges.shifts` 的周期 batch 走通。
- Edge cases：无 `graphs` 命名空间的 batch → 建出的 `graphs.batch_size == (B,)`（schema 回归守卫）；
  `compute_forces=False` 的实例不写 `atoms.forces`（并由 `PotentialForceField` 抛 `RuntimeError`）；
  `pos` 已是 grad leaf → 能量不被 detach 且 loss 能回传到参数，非 leaf → 能量 detach；MatPES 变体收到
  `total_charge` 非 `None` → `ValueError`；表外原子序数 → `ValueError`；`inspect.signature` 上
  `forward` / `energy_core` / `_write_energy` 均无 `compute_forces`。
- Domain validation（硬编码期望值）：相对平铺 `MACEMatpes` / `MACEOMol` 在 `state_dict` 直接迁移下
  `max|ΔE| ≤ 1e-12 eV`、`max|ΔF| ≤ 1e-12 eV/Å`（同算子序时实测应逐位相等，测试同时打印 `torch.equal`
  结果供诊断）；`state_dict()` 的 key 集合与形状与平铺版完全相同；`|Σ_i F_i| ≤ 1e-10 eV/Å`；整体平移
  1.0 Å 后 `|ΔE| ≤ 1e-12 eV`；中心差分（`h = 1e-4 Å`，fp64）与解析力 `max|ΔF| ≤ 1e-6 eV/Å`。
- 编译冒烟（`@pytest.mark.slow`）：`torch._dynamo.explain(potential.energy_core)(...)` 的
  `graph_count == 1` 且 `graph_break_count == 0`。

**Regression example**：`regressions/mace-subpackage-restructure-04-potential.py`——纯公共 API
（`MACEMatpesSpec` → `MACEPotential` → `forward` / `energy_core`）跑一个 4 原子 fp64 体系，断言能量与逐
原子力对**内嵌的 fp64 字面量**吻合到 1e-12，并断言 `Σ F = 0`。字面量在实现期由仓库内的平铺
`MACEMatpes` 生成（非第三方 oracle），脚本头注明生成命令、commit 与日期；07 删掉平铺模块后，这个文件
就是重构前数值的**唯一冻结记忆**，所以它必须在 07 之前落地。

## Out of scope

- **不碰 `src/molix/engine/adapter.py`。** `TensorDictAdapter.read_outputs`（`adapter.py:132-134`）
  `assert isinstance(out, dict)` 与 molnex 的 TensorDict potential 返回形状不符，但当前只有一个 toy
  fixture 命中它，没有任何生产 potential 走这条路；本链**不**把 MACE 接进 `EngineForward`。
  `energy_core` 就是将来某个 engine spec 要绑的扁平入口。此处明文记录，不动代码。
- 调用点改指：`scripts/matpes_port/run_nve.py:118,124` 与 `benchmarks/bench_mace_matpes.py:129` 仍指向
  平铺模型的 `_compute_energy`——本 spec 只**创建**公共接缝，改指与删除平铺模块归 07。
- 导入兼容 / `molzoo.__init__` 的 `MACEPotential` 惰性导出、`src/molzoo/specs/*.md` 的更新：归 06。
- `protocol.ensure_graphs` 的 `batch_size=[]` 修复：归 07（本 spec 本地按正确顺序兜住）。
- `molpot.composition.EnergyForceModel` 的处置（零生产子类，`architecture.md:160`）：归 07。
- 权重加载器 `load_matpes_state_dict` / `load_omol_state_dict` 的合并：归 05-checkpoint。
- GPU 性能验证与 `benchmarks/bench_mace_matpes.py` 的 fused/fallback 守卫刷新：归 07（本 spec 的 CPU 单
  测不测吞吐）。
- 多 head、stress/virial、`torch.export` / AOTI：均不在本链。
