---
title: MACE 子包重构 01 — molrep 侧 MACE 专用块归位（降级范围）
status: approved
created: 2026-08-08
revised: 2026-08-08
grilled: true
chain: mace-subpackage-restructure
---

# MACE 子包重构 01 — molrep 侧 MACE 专用块归位

## Summary

把目前散落在 `molrep/interaction/`、`molrep/readout/` 顶层以及 `molzoo/mace.py` 内部的
**MACE 专用**构件收拢到三个带 `mace` 名字的归属地：新建包 `src/molrep/interaction/mace/`、
新模块 `src/molrep/readout/mace.py` 与 `src/molrep/embedding/mace.py`。本 spec 是**纯搬运 +
拆分**：不改任何一行 cuEquivariance 构造参数、不改数值、不改公开符号名；旧模块路径全部保留为
薄再导出 shim，因此 `molzoo.mace_omol` / `molzoo.mace_matpes` / `molzoo.mace` 以及现有测试与
docs 锚点在本段落地后立即可用。搬完后 `molrep/interaction/` 顶层只剩真正共享的块
(`contraction` / `radial` / `gate` / `linear` / `aggregation` / `product`)，为链上 02–07 段
（molzoo/mace 子包、kernels、MACEPotential、checkpoint remap、删 flat 模块、清理消费者）
腾出干净的下层。本段同时新建仓库根 `regressions/` 目录并落地链上第一个硬编码 golden 冒烟脚本。

**过程中已发现、必须点名的既有问题**：(a) `molrep/interaction/residual.py`、
`product_basis.py`、`readout/scalar.py` 三个模块**当前完全没有单测镜像** —— 本 spec 在搬运
的同时补齐（在范围内，见 Tasks 1/2）；(b) `tests/test_molrep/test_readout/test_heads.py:8`
实际测的是 `molpot.derivation.{EnergyAggregation,ForceDerivation}`，与
`tests/test_molpot/test_derivation/{test_energy,test_force}.py` 重复且违反包镜像规则 ——
属跨包测试搬家，**不在本段修**，显式路由给 07-cleanup（见 Out of scope）；
(c) `scripts/check_test_mirror.py:95` 的 fuzzy 回退会让 `molrep/interaction/mace/residual.py`
被 `tests/.../test_pinet/test_residual.py` 误配成"已镜像" —— 本段通过建**精确路径**镜像
规避，不改回退逻辑。

## Domain basis

本段不引入新物理，但搬运的每个块都是已验证的 MACE 实现，必须逐位保持：

- MACE 等变消息传递：Batatia et al., *MACE: Higher Order Equivariant Message Passing Neural
  Networks for Fast and Accurate Force Fields*, NeurIPS 2022, https://arxiv.org/abs/2206.07697
  （`ResidualInteraction`、`EquivariantProductBasis`、`ConvTP`、三个 readout）。
- MACE-MP / 密度归一变体：Batatia et al., *A foundation model for atomistic materials
  chemistry*, https://arxiv.org/abs/2401.00096（`DensityInteraction` /
  `DensityResidualInteraction`，`message/(ρ+1)`，`ρ_i = Σ_j tanh(MLP(e_ij)²)·u(r_ij)`）。
- cuEquivariance 索引线性 / skip_tp：NVIDIA cuEq MACE tutorial
  https://docs.nvidia.com/cuda/cuequivariance/tutorials/pytorch/MACE.html（`ElementUpdate`，
  `SKIP_TP_METHOD = "naive"`）。

不变量（搬运后必须继续成立，单位 eV / Å）：

1. **数值不变**：所有块对同一输入的输出在 CPU float64 下与搬运前一致到 rtol ≤ 1e-12；
   `MACEOMol` 对官方 OMOL 的移植一致性仍为 |dE| < 1e-5 eV、max|dF| < 1e-4 eV/Å
   （`mace-omol-port-01` ac-006 的门槛，本段不得抬高误差）。
2. **边约定**：`edge_index` 形状 `(E, 2)`，`[:, 0]` = source、`[:, 1]` = target，
   `edge_diff = pos[target] - pos[source]`。搬运不得改动任何 `edge_index[:, k]` 取列。
3. **等变性 / 权重可加载性**：`state_dict()` 的键名（`linear_up`、`conv_tp`、
   `conv_tp_weights`、`density_fn`、`skip_tp`、`symmetric_contractions`、`linear_1/_mid/_2`…）
   保持不变，否则官方权重直接 `load_state_dict` 迁入的路径断裂（05-checkpoint 依赖此项）。
4. **`use_fallback` 语义**：块构造器默认 `use_fallback=True`（functorch 安全），
   autograd-only 整模型默认 `False`（notes.md 2026-08-08 条目）。搬运不得改默认值。

## Design

### 前置条件（已裁定，2026-08-08 spec-audit）

`.claude/specs/cuet-force-doublebackward.md` 状态为 **in-progress**，其未完成任务 3–5 改写
`src/molrep/interaction/{residual.py, product_basis.py, linear.py, element.py}` 里的 cuet
实例化点 —— 与完整搬运清单**逐字重叠**。**操作者已选择降级方案（选项 A）**：

- 本段只搬**无冲突**部分：`density.py`、`ConvTP`/`ConvTPSpec` 拆分、三个 readout +
  `ProductHead`、`EmbeddingBlock`/`InteractionBlock` 上提；
- `residual.py`、`product_basis.py`、`element.py` **原位不动**（零改动，连 shim 都不加），
  待 `cuet-force-doublebackward` 落地后由补充段 `mace-subpackage-restructure-01b-blocks`
  完成剩余搬运（届时该段以本 spec 为模板，机械同构）；
- 02–07 不受影响：它们只消费符号名（`from molrep.interaction import ResidualInteraction`
  等旧路径在 -01b 之前继续有效），不关心块文件的物理位置。

### 目标布局

```
src/molrep/interaction/mace/
    __init__.py        # 再导出本包公开符号（降级范围：6 个）
    conv.py            # ConvTP, ConvTPSpec            ← 从 interaction/product.py 拆出
    block.py           # InteractionSpec, InteractionBlock  ← 从 molzoo/mace.py 上提
    density.py         # SKIP_TP_METHOD, DensityInteraction, DensityResidualInteraction (整体搬)
src/molrep/readout/mace.py     # _ScalarO3Linear, NonLinearBiasReadout, LinearReadout,
                               # NonLinearReadout (从 readout/scalar.py 整体搬)
                               # + ProductHead, ProductHeadSpec (从 readout/product.py 整体搬)
src/molrep/embedding/mace.py   # EmbeddingSpec, EmbeddingBlock  ← 从 molzoo/mace.py 上提

# -01b（cuet-force-doublebackward 落地后）：
#   interaction/mace/residual.py / product_basis.py / element.py
```

**为什么 embedding / interaction / readout 三处各开一个 `mace` 家，而不是把上提的两个 Block
一起塞进 `interaction/mace/`**：`molrep` 的层划分（embedding → interaction → readout）是
CLAUDE.md 里的架构契约。`EmbeddingBlock` 组合的是 `JointEmbedding` / `BesselRBF` /
`SphericalHarmonics` / `CosineCutoff`，全部是 embedding 层构件；把它放进 `interaction/` 会
让层边界第一次出现例外。三个平行的 `mace` 归属地读起来也自解释。

**符号名一律不改**（`EmbeddingBlock`、`InteractionBlock`、`LinearReadout` … 保持原名）。
改名会同时破坏向后兼容和"纯搬运"的可验证性，属于 06-wire/07-cleanup 的议题。
`molrep.embedding.__init__` **不**再导出 `EmbeddingBlock`/`EmbeddingSpec` —— 这两个名字在包级
太泛，消费者按模块路径 `molrep.embedding.mace` 导入。

### 向后兼容策略（显式选择）

**选择：旧模块文件降级为纯再导出 shim；`molrep/interaction/__init__.py` 与
`molrep/readout/__init__.py` 的 `__all__` 保持逐字不变、只把 import 源改指新路径。**

即 `src/molrep/interaction/density.py` 变成：

```python
"""Deprecated location — moved to :mod:`molrep.interaction.mace.density`.

Kept as a re-export shim for chain step mace-subpackage-restructure-01;
removed in 06-wire.
"""

from molrep.interaction.mace.density import (  # noqa: F401
    SKIP_TP_METHOD,
    DensityInteraction,
    DensityResidualInteraction,
)
```

理由：两种粒度的旧写法都在用 —— `from molrep.interaction.density import DensityInteraction`
（`molzoo/mace_matpes.py:50`、`tests/test_molrep/test_interaction/test_density.py:6`）与
`from molrep.interaction import DensityInteraction`（`__init__` 的 `__all__`）。只改
`__init__` 不足以覆盖前者。shim 文件比在 `__init__` 里堆别名更容易在 06-wire 里 `git rm`
一次性删净，也不会让 `molrep.interaction` 的导入图变复杂。`molzoo/mace.py` 同理保留
`EmbeddingBlock` / `InteractionBlock` / `EmbeddingSpec` / `InteractionSpec` 的再导出，
使 `tests/test_molzoo/test_mace.py:24` 与 `tests/test_molzoo/test_symmetry.py` 不动即通过。

导入环检查（已推演，无环）：`molrep.interaction.__init__ → .product → .mace.conv` 会先执行
`molrep.interaction.mace.__init__`，其 `.residual` 走 `molrep.interaction.{gate,radial}`
子模块导入 —— 父包此时已在 `sys.modules` 中（部分初始化），子模块导入合法。
若实测出现环，退路是把 `product.py` 的 ConvTP 再导出改为模块级 `__getattr__` 懒加载。

### 不搬的东西（共享块，硬边界）

`contraction.py`(SymmetricContraction)、`radial.py`(RadialMLP/RadialWeightMLP)、
`gate.py`(GatedNonlinearity)、`linear.py`(EquivariantLinear) 有 MACE 之外的消费者，
**保持原位、零改动**。`product.py` 保留 `EquivariantPolynomialTP`、`irreps_from_l_max`、
`sh_irreps_from_l_max`（`molzoo/allegro.py:55` 依赖，且 `readout/mace.py` 仍从这里取
`irreps_from_l_max`），仅拆走 `ConvTP`/`ConvTPSpec` 并补一行再导出。
`aggregation.py`(MessageAggregation) 唯一"消费者"是
`src/molix/profiler/module.py:344` 的一段 mock 分支 —— **本段不动它**，记在 Out of scope。

### Reuse decision（按 Move map 台账逐行裁定）

| 台账行 | 裁定 | 说明 |
|---|---|---|
| `density.py`(DensityInteraction/DensityResidualInteraction) | **reuse**（整体搬运，零改） | 唯一消费者 `molzoo/mace_matpes.py:50` |
| `residual.py`(ResidualInteraction) | **deferred → -01b** | 与 cuet-force-doublebackward 冲突，原位不动 |
| `product_basis.py`(EquivariantProductBasis) | **deferred → -01b** | 与 cuet-force-doublebackward 冲突，原位不动 |
| `element.py`(ElementUpdate/Spec) | **deferred → -01b** | 与 cuet-force-doublebackward 冲突，原位不动 |
| `product.py` 中的 `ConvTP`/`ConvTPSpec` | **reuse**（拆分搬运，零改） | 消费者 `molzoo/mace.py:52`、`tests/test_molzoo/test_mace.py:15`；同文件的 `EquivariantPolynomialTP` 留守 |
| `readout/scalar.py` 三个 readout | **reuse**（整体搬运，零改） | 消费者 omol / matpes |
| `readout/product.py`(ProductHead/Spec) | **reuse**（整体搬运，零改） | 唯一消费者 `molzoo/mace.py:58`（`molrep/__init__.py:35` 的再导出经 shim 保留） |
| `molzoo/mace.py` 的 EmbeddingSpec/Block、InteractionSpec/Block | **generalize** | 从 encoder 配方文件上提为 molrep 层可复用块；**不新建并行实现**，`molzoo/mace.py` 改为导入 + 再导出 |
| `contraction/radial/gate/linear` | **reuse 原位** | 共享，不动、不复制 |
| `aggregation.py` | **不裁定** | 出范围，见 Out of scope |
| 新符号 | **new — 无** | 本段不引入任何新公开符号 |

### 测试镜像门禁

`scripts/check_test_mirror.py` 的 `PINET_SPINE` 追加 `"molrep/interaction/mace/"`，
使新包在 `--strict-pinet` 下受硬门禁保护；同时把六个纯再导出 shim
（`molrep/interaction/{density,residual,product_basis,element}.py`、
`molrep/readout/{scalar,product}.py`）加进 `ALLOW_MISSING`（该集合的语义正是
"pure re-export"），避免 shim 期间产生假缺口。

### regressions/ 的建立

仓库根尚无 `regressions/`。本段按 CLAUDE.md（"Public-API scenarios → `regressions/` with
hard-coded goldens (no live third-party oracles)"）建立该目录，并写入链上第一个脚本：
公开构造器搭出 8 个被搬的块 → 固定图 → CPU float64 → 断言**在搬运前的父提交上采集、
以字面量写死**的输出统计量与 `state_dict()` 键列表。golden 注释须记录采集命令、
commit sha、torch 版本与日期。脚本只 import molnex 自身，不 import / 不 subprocess 任何
第三方 oracle。

## Files to create or modify

- `src/molrep/interaction/mace/__init__.py` (new)
- `src/molrep/interaction/mace/conv.py` (new)
- `src/molrep/interaction/mace/block.py` (new)
- `src/molrep/interaction/mace/density.py` (new)
- `src/molrep/embedding/mace.py` (new)
- `src/molrep/readout/mace.py` (new)
- `src/molrep/interaction/density.py`
- `src/molrep/interaction/product.py`
- `src/molrep/interaction/__init__.py`
- `src/molrep/readout/scalar.py`
- `src/molrep/readout/product.py`
- `src/molrep/readout/__init__.py`
- `src/molzoo/mace.py`
- `src/molzoo/mace_omol.py`
- `src/molzoo/mace_matpes.py`
- `scripts/check_test_mirror.py`
- `tests/test_molrep/test_interaction/test_mace/__init__.py` (new)
- `tests/test_molrep/test_interaction/test_mace/test_conv.py` (new)
- `tests/test_molrep/test_interaction/test_mace/test_block.py` (new)
- `tests/test_molrep/test_interaction/test_mace/test_density.py` (new，由 `tests/test_molrep/test_interaction/test_density.py` 搬入)
- `tests/test_molrep/test_interaction/test_product.py`
- `tests/test_molrep/test_embedding/test_mace.py` (new)
- `tests/test_molrep/test_readout/test_mace.py` (new，由 `tests/test_molrep/test_readout/test_product.py` 搬入并扩充)
- `tests/test_molrep/test_reexport_compat.py` (new)
- `regressions/README.md` (new)
- `regressions/mace-subpackage-restructure-01-blocks.py` (new)

## Tasks

- [ ] Write failing unit tests for the `molrep.interaction.mace` namespace (`git mv` tests/test_molrep/test_interaction/test_density.py → tests/test_molrep/test_interaction/test_mace/ with imports re-pointed, move TestConvTPSpec/TestConvTP out of test_product.py into test_mace/test_conv.py, add test_mace/test_block.py → TestInteractionBlock; RED: module absent)
- [ ] Write failing unit tests for `molrep.readout.mace` and `molrep.embedding.mace` (`git mv` tests/test_molrep/test_readout/test_product.py → tests/test_molrep/test_readout/test_mace.py, add TestLinearReadout / TestNonLinearReadout / TestNonLinearBiasReadout; add tests/test_molrep/test_embedding/test_mace.py → TestEmbeddingBlock; RED)
- [ ] Write failing back-compat test tests/test_molrep/test_reexport_compat.py asserting every legacy path object `is` its new-path object and `import molrep, molrep.interaction, molrep.readout, molzoo.mace` raise no ImportError (RED)
- [ ] Create src/molrep/interaction/mace/ and move density.py verbatim, split ConvTP+ConvTPSpec into mace/conv.py, promote InteractionSpec+InteractionBlock from src/molzoo/mace.py into mace/block.py (import lines and module docstrings are the only permitted edits; residual.py / product_basis.py / element.py stay untouched — deferred to -01b)
- [ ] Create src/molrep/readout/mace.py (move the three scalar readouts + _ScalarO3Linear + ProductHead/ProductHeadSpec verbatim) and src/molrep/embedding/mace.py (promote EmbeddingSpec+EmbeddingBlock verbatim)
- [ ] Add re-export shims in the three legacy modules touched (interaction/density.py, readout/scalar.py, readout/product.py) and re-point src/molrep/interaction/__init__.py, src/molrep/readout/__init__.py (identical `__all__`) plus src/molzoo/{mace,mace_matpes}.py import lines (mace_omol.py imports only untouched modules)
- [ ] Extend scripts/check_test_mirror.py (PINET_SPINE += "molrep/interaction/mace/", ALLOW_MISSING += the three shim modules) and make `python scripts/check_test_mirror.py --strict-pinet` exit 0
- [ ] Add regression example regressions/mace-subpackage-restructure-01-blocks.py plus regressions/README.md (public API only; goldens captured on the pre-move parent commit and embedded as literals with sha/date/torch-version comment)
- [ ] Verify the move is numerics-neutral: `git diff -M -U0` over the moved modules touches no cuet construction site, and tests/test_molzoo/{test_mace,test_mace_encoder,test_mace_omol,test_mace_matpes,test_symmetry}.py report the same pass/fail counts as the baseline recorded before the move
- [ ] Run full check + test suite

## Testing strategy

单测一律放 `tests/`、路径镜像 `src/`、每个测试只打一个函数/方法，不写 e2e：

- `tests/test_molrep/test_interaction/test_mace/test_density.py`（搬入，`TestDensityInteraction`
  / `TestDensityResidualInteraction`）：沿用现有 float64 小图 fixture（4 节点 / 5 边、
  `edge_index` 为 `(E,2)`），happy path 断言 `(N, ir_dim, mul)` 形状与第一层
  `skip is None`；不改断言值。
- `test_mace/test_conv.py`（从 test_product.py 搬入）：`TestConvTPSpec` 配置校验、
  `TestConvTP` 前向形状与 `weight_numel`；边缘用例保留原有 gather/scatter 一致性检查。
- `test_mace/test_element.py`（搬入）：`TestElementUpdateSpec` / `TestElementUpdate` 原样。
- `test_mace/test_residual.py`（**新增，补历史缺口**）：`TestResidualInteraction` —
  happy path 输出形状 `(N, ir_dim, mul)` + `sc` 形状；边缘用例 `.double()` 后前向不抛
  dtype 错；`state_dict()` 键集合包含 `linear_up` / `conv_tp_weights` / `skip_tp` /
  `linear_res`（权重迁移契约）。
- `test_mace/test_product_basis.py`（**新增**）：`TestEquivariantProductBasis` —
  `use_sc=True/False` 两条路径的输出形状；`num_elements=1` 元素无关权重形状。
- `test_mace/test_block.py`（新增）：`TestInteractionBlock` — 前向返回 `(node_feats, sc)`
  且 `sc is` 输入张量；`avg_num_neighbors` 归一化按比例缩放输出（hard-coded 因子 2.0 → 输出减半）。
- `tests/test_molrep/test_readout/test_mace.py`：`TestProductHead`（搬入，断言不变）+
  新增 `TestLinearReadout`（输出 `(N,1)`）、`TestNonLinearReadout`、
  `TestNonLinearBiasReadout`（各自 `state_dict` 键名 `linear_1`/`linear_mid`/`linear_2`
  —— 官方权重直载契约）。
- `tests/test_molrep/test_embedding/test_mace.py`：`TestEmbeddingBlock`，从
  `tests/test_molzoo/test_mace.py` 的同名类**复制**（原文件在 06-wire 删除，本段不动它，
  短暂重复是刻意的过渡代价）。
- `tests/test_molrep/test_reexport_compat.py`：逐符号断言
  `molrep.interaction.density.DensityInteraction is molrep.interaction.mace.density.DensityInteraction`
  等 12 条 identity，以及 `molrep.interaction.__all__` / `molrep.readout.__all__` 与
  写死的预期列表相等。
- 领域验证：本段**不引入新物理**，域验证由 regression 示例与既有
  `tests/test_molzoo/test_mace_omol.py`（对官方 OMOL 的 1e-9 级一致性）承担；不新增科学断言。
- **Regression 示例**：`regressions/mace-subpackage-restructure-01-blocks.py`（仓库根
  `regressions/`，非 `tests/`）。只走公开 API：`molix.config` 设 fp64 → `torch.manual_seed(0)`
  → 构造 `ConvTP` / `InteractionBlock` / `DensityInteraction` / `DensityResidualInteraction` /
  `ResidualInteraction` / `EquivariantProductBasis` / `ElementUpdate` / `LinearReadout` /
  `NonLinearReadout` / `NonLinearBiasReadout` / `ProductHead` / `EmbeddingBlock` → 固定
  `(E,2)` 边表前向 → 断言每块输出的 `sum()` 与 `abs().max()` 命中写死的 golden
  （rtol 1e-12），并断言 `sorted(state_dict().keys())` 等于写死的键列表。golden 取自
  **搬运前的父提交**（自基线，非第三方 oracle），注释记录采集命令、commit sha、
  torch 版本、日期、CPU/float64。
- 全量：`ruff check src/ && ruff format --check src/` 与 `python -m pytest tests/ -v`，
  与搬运前记录的基线逐条对比（`tests/test_molzoo/test_symmetry.py` 已知的
  `setup_context` 既有失败按**计数比对**处理，禁止用 skip 标记掩盖）。

## Out of scope

- **`molrep/interaction/aggregation.py`（MessageAggregation）不动**：唯一"消费者"是
  `src/molix/profiler/module.py:344` 的 mock 分支，归属未定，留给 07-cleanup 裁定。
- **`tests/test_molrep/test_readout/test_heads.py` 的错位**（实测 `molpot.derivation`，
  与 `tests/test_molpot/test_derivation/{test_energy,test_force}.py` 重复）：跨包测试搬家，
  路由 07-cleanup 或独立 `/mol:fix`。本段只点名、不顺手改，以免污染"纯搬运"的 diff 可验证性。
- 删除旧 flat 模块 / 迁移 `tests/test_molzoo/test_mace.py` 等消费者测试 → 06-wire。
- `molzoo/mace.py` 拆成 `molzoo/mace/{spec,geometry,encoder}` → 02-core。
- `molpot/derivation/kernels.py` → 03-kernels；`MACEPotential` → 04-potential；
  `CheckpointRemap` / `from_checkpoint` → 05-checkpoint；`EnergyForceModel` 移除、
  `ensure_graphs` 修复、`src/molzoo/specs/*.md` 与 `docs/api/molrep.md` 锚点刷新 → 07-cleanup。
  （注意：`docs/api/molrep.md:19,31` 的 mkdocstrings 锚点指向变成 shim 的
  `molrep.interaction.product` / `molrep.readout.product`，页面会在 shim 期间少渲染部分成员；
  这是已知、有界、已路由的过渡态，不是遗漏。）
- 任何符号改名、签名调整、`use_fallback` 默认值调整、性能优化或新 benchmark。
- `cuet-force-doublebackward` 本身的收尾（见 Design 前置条件）。
