---
title: MACE checkpoint remap — CheckpointRemap + MACEPotential.from_checkpoint
status: approved
created: 2026-08-08
revised: 2026-08-08
grilled: true
chain: mace-subpackage-restructure
---

# MACE checkpoint remap — CheckpointRemap + MACEPotential.from_checkpoint

## Summary

把目前重复实现在 `src/molzoo/mace_matpes.py:403-464`（`load_matpes_state_dict`）与
`src/molzoo/mace_omol.py:374-443`（`load_omol_state_dict`）中的官方 checkpoint 装载逻辑，
收敛成 `src/molzoo/mace/checkpoint.py` 里的**一个** `CheckpointRemap` 类型。两个函数在意图上
逐字节相同的三段（跳过 `.graph.c` / `output_mask` 图常量与 irrep mask；最长前缀 key 重映射，
`None` 表示丢弃；对 `(1,)` vs 0-d 冻结标量做 numel 守恒 reshape）合并为一份实现，两族之间**唯一**
的真实差异用一个策略旋钮 `on_unexpected: Literal["raise", "return"]` 表达：MatPES checkpoint
遇到无处安放的 key 直接 raise，OMol checkpoint 返回该列表（OMol 家族带有本移植刻意不建模的辅助头）。
两个 loader 的严格性教条原封不动保留——未被 checkpoint 填充的可学习参数、或形状不符，**永远** raise。
在此之上给 04 交付的 `MACEPotential` 增加 `from_checkpoint(...)` classmethod（本仓库第一个
`from_checkpoint`，属 CLAUDE.md 允许的"语义独立的替代构造器"，不是 `make_*` 别名）：读取官方 config
json → 构造 spec → 构造模型 → 装载权重，一次调用完成，取代
`scripts/matpes_port/run_nve.py:145-169` 那段手写 `build_model`（07 步再把脚本指过来）。

## Domain basis

本步不引入新物理，但装载的是**已训练的势能面参数**，一个被静默丢弃的张量会产出"能跑、看着正常、
但物理上是错的"模型——这正是严格教条存在的理由。

- 模型架构：Batatia, Kovács, Simm, Ortner, Csányi, *MACE: Higher Order Equivariant Message
  Passing Neural Networks for Fast and Accurate Force Fields*, NeurIPS 2022,
  https://arxiv.org/abs/2206.07697
- 基础模型族：MACE-MP-0, https://arxiv.org/abs/2401.00096；MatPES 数据集
  (Kaplan et al.), https://arxiv.org/abs/2503.04070
- 参考 checkpoint：`MACE-matpes-r2scan-omat-ft.model`（ACEsuit/mace-foundations,
  tag `mace_matpes_0`），经 `mace.cli.convert_e3nn_cueq` 离线转换后得到
  `matpes_r2scan_cueq_state.pt` + `matpes_r2scan_config.json`；OMol 家族同构。
  转换工具是**离线 oracle**，运行期不得 import（CLAUDE.md 第三方面许可清单：无 mace-torch / e3nn）。
- 单位：能量 eV，力 eV/Å，`r_max` Å，`atomic_energies`（E0）eV/atom。remap 不做任何单位换算——
  官方 checkpoint 与 molnex 的单位系一致，若将来不一致必须是显式的转换步骤而非隐式 reshape。
- 两处历史事故，本 spec 的严格性直接继承自它们，实现时必须在 docstring 中引用：
  1. `mace_matpes.py:405-410` 的教条陈述（"a silently dropped tensor is the failure mode that
     produces a model which runs, looks sane, and is quietly wrong"）。
  2. `mace_omol.py:437-439` 的 `bessel.freqs` 事故——用 key 后缀启发式（`.weight`/`.bias`）判定
     "可学习"，让一个既不以 `.weight` 也不以 `.bias` 结尾的 `nn.Parameter` 被静默豁免检查，
     官方权重与解析初值相差 ~2e-7，直接落在当时上报的 parity 残差里。**"可学习"的定义只能是
     `nn.Parameter`（`model.named_parameters()`），不得用名字启发式。**

## Design

### `CheckpointRemap`（`src/molzoo/mace/checkpoint.py`，唯一新类型）

```
CheckpointRemap(table: Mapping[str, str | None], *, on_unexpected: Literal["raise", "return"] = "raise")
  .rename(official_state: Mapping[str, Tensor]) -> dict[str, Tensor]
  .load(model: nn.Module, official_state: Mapping[str, Tensor]) -> tuple[list[str], list[str]]
```

- **构造**：`table` 是"官方 cueq key（或 key 前缀）→ molnex 名"的映射，`None` = 丢弃（由构造参数重建
  或非持久常量）。最长前缀优先（构造时预先算好按长度降序的前缀序列，作为实例私有属性，
  取代现在两个模块各自的 `_KEY_REMAP_ORDER` 模块级常量）。未列出的 key 原样透传。
- **`.rename`**：纯函数，无副作用，不碰模型——独立可测。三段合并后的唯一实现：
  跳过 `".graph.c" in key or key.endswith("output_mask")`；最长前缀替换；`None` 丢弃。
- **`.load`**：一个具名动作 = rename → 计算 `unexpected = sorted(set(remap) - set(model.state_dict()))`
  → 按策略分叉 → numel 守恒 reshape → `load_state_dict(strict=False)` → 严格性检查 → 返回
  `(missing_buffers, unexpected)`。返回**元组**而非新的 report 类型，因为
  `load_omol_state_dict` 现有契约就是 `tuple[list[str], list[str]]`，wrapper 因此是一行。
  两个列表都排序（相对 OMol 现在直接回吐 torch 的 `unexpected` 是一处**刻意的**确定性归一化，
  写进 docstring；remap 表的语义内容不变，这是链式不变量）。
- **策略旋钮**（唯一差异点）：
  - `"raise"`（默认，最严者为默认；MatPES）：`unexpected` 非空时在**触碰模型之前**
    `raise RuntimeError(f"checkpoint keys with no home in {type(model).__name__}: {unexpected}")`
    —— 保留 `"no home"` 字样，现有测试 `pytest.raises(RuntimeError, match="no home")` 不变即通过，
    且保证不留下半装载的模型。
  - `"return"`（OMol）：不 raise，`unexpected` 随返回值交给调用者检查。
- **不可协商的严格性**（与旋钮无关，两族一致）：
  - 形状不符且 `numel` 不等 → `RuntimeError("shape mismatch — model built with the wrong config? …")`。
  - `numel` 相等 → reshape（MACE 把部分冻结标量存成 `(1,)`，molnex 持 0-d buffer：内容相同，秩不同）。
  - 装载后 `unfilled = sorted(m for m in missing if m in {n for n, _ in model.named_parameters()})`
    非空 → `RuntimeError("parameters not covered by the checkpoint: …")`（保留 `"not covered"` 字样）。
- **两张表按族原样搬迁**为模块级常量 `MATPES_KEY_REMAP` / `OMOL_KEY_REMAP`（内容逐键不变，
  这是链式不变量；02/04 建立的 state_dict key parity 正是让旧表可以逐字复用的前提），
  外加两个预置实例常量 `MATPES_REMAP = CheckpointRemap(MATPES_KEY_REMAP)` 与
  `OMOL_REMAP = CheckpointRemap(OMOL_KEY_REMAP, on_unexpected="return")`。这些是**数据常量**，
  不是工厂函数。

### `MACEPotential.from_checkpoint`（`src/molzoo/mace/potential.py`）

```
@classmethod
def from_checkpoint(cls, config_path: Path, weights_path: Path, *,
                    remap: CheckpointRemap = MATPES_REMAP,
                    map_location: str | torch.device = "cpu",
                    **ctor_kwargs) -> MACEPotential
```

`**ctor_kwargs`（spec-audit 裁定 2026-08-08）原样透传给构造器——`use_fallback=False` /
`compute_forces=True` 等**运行环境开关**不属于 checkpoint 的科学内容，不进 config 翻译表；
07 的 `run_nve.py` 调用形态为
`MACEPotential.from_checkpoint(weights_dir / "matpes_r2scan_config.json",
weights_dir / "matpes_r2scan_cueq_state.pt", use_fallback=use_fallback)`。

三步、无隐藏动作：读 json → 翻译成 04 的 spec 类型 → `cls` 构造 →
`torch.load(weights_path, map_location=map_location, weights_only=True)` → `remap.load(model, state)`
→ 返回模型（**不**替调用者 `.eval()`，那是调用者的一步）。命名与形态跟随最近的既有模式
`PiNet.from_spec`（`src/molzoo/pinet/encoder.py:177-180`）与 `EngineAdapter.from_name`
（`src/molix/engine/adapter.py:53-60`）：classmethod、语义独立、`from_*` 前缀。

官方 config key → spec 字段的翻译**内联在 `from_checkpoint` 里**（CLAUDE.md "inline until the
second real use"：目前只有这一个调用点）：`max_ell`→`l_max`、`radial_MLP`→`radial_mlp`、
`atomic_inter_scale`→`scale`、`atomic_inter_shift`→`shift`，`r_max` / `num_bessel` /
`num_polynomial_cutoff` / `num_interactions` / `correlation` / `atomic_numbers` /
`atomic_energies` 同名。

**`num_features` / `max_hidden_l` / `mlp_dim` 从 config 的 `hidden_irreps` / `MLP_irreps` 解析**
（例如 `"128x0e+128x1o"` → `num_features=128, max_hidden_l=1`；`"16x0e"` → `mlp_dim=16`），
**不得**照抄 `run_nve.py:154-159` 里写死的 `128 / 1 / 16` —— 那三个硬编码正是本链要消除的
"换个 checkpoint 就静默算错"类缺陷。缺少所需 key 时列出缺失名并 raise，绝不回退到默认值。
**已知风险 / Iron-law 停机点**：若真实官方 json 不含 `hidden_irreps` / `MLP_irreps`，实现者
**停下来上报**（连同实际 json 的 key 列表），不得回填硬编码；届时的正解是让调用者显式构造
spec 并把 parity 任务标为 blocked。

### 放置理由

MACE checkpoint 格式专属 → `molzoo/mace/checkpoint.py`。**不**放 `molix/core/checkpoint/`
（`Checkpoint` / `TorchSaveBackend`，`src/molix/core/checkpoint/state.py:22`、`backend.py:18`）——
那里拥有的是训练态 round-trip（optimizer / scheduler / TrainState），与"外部预训练权重的 key 方言
翻译"是两件事。本模块只依赖 `torch` + 本包，不新增任何 molzoo→molix 依赖。

旧名 `load_matpes_state_dict` / `load_omol_state_dict` 在本步降级为**薄 wrapper**（各一行，
委派给对应预置实例），保持 `src/molzoo/__init__.py` 的 lazy 导出与外部调用点不破；06 步做别名/删除，
07 步清理（stage: experimental，允许这种两步走）。

### Reuse decision

- `load_matpes_state_dict`（`src/molzoo/mace_matpes.py:403`）→ **generalize**：提升为
  `CheckpointRemap`（`on_unexpected="raise"`），旧名保留为 wrapper 直到 06。
- `load_omol_state_dict`（`src/molzoo/mace_omol.py:374`）→ **generalize**：同一实现，
  `on_unexpected="return"`，wrapper 保留原 `(missing_buffers, unexpected)` 返回契约。
- `_KEY_REMAP` / `_KEY_REMAP_ORDER`（`mace_matpes.py:387-400`、`mace_omol.py:355-371`）→ **reuse**：
  表内容逐键原样搬迁为 `MATPES_KEY_REMAP` / `OMOL_KEY_REMAP`；排序前缀列表由构造函数内部算出，
  两个模块级 `_KEY_REMAP_ORDER` 随之消失（不再有第二份）。
- `PiNet.from_spec`（`src/molzoo/pinet/encoder.py:177`）→ **pattern**：
  `from_checkpoint` 沿用 classmethod / 语义独立 / `from_*` 前缀形态，不复制构造逻辑。
- `scripts/matpes_port/run_nve.py:145-169` `build_model` → **generalize**（本步只交付库侧能力，
  脚本改指在 07；本步不动该文件）。
- `molix.core.checkpoint.Checkpoint` / `CheckpointBackend`（`src/molix/core/checkpoint/`）→
  **new — 训练态 round-trip 与外部权重 key 翻译是不同关注点，复用会把 molzoo 的模型格式方言
  塞进 molix 的训练基础设施。**
- `tests/test_molzoo/test_mace_matpes.py::TestLoadMatpesStateDict._roundtrip_state`（:224-240）→
  **reuse（作为测试模式）**：新测试沿用"把模型自己的 state_dict 改名成官方名再装回来"的
  round-trip 手法，逐参数 `torch.equal`。

## Files to create or modify

- `src/molzoo/mace/checkpoint.py` (new) — `CheckpointRemap` + `MATPES_KEY_REMAP` /
  `OMOL_KEY_REMAP` / `MATPES_REMAP` / `OMOL_REMAP`
- `src/molzoo/mace/potential.py` — 由 04 建立；本步新增 `MACEPotential.from_checkpoint`
- `src/molzoo/mace/__init__.py` — 由 02 建立；本步导出 `CheckpointRemap` 与两个预置实例
- `src/molzoo/mace_matpes.py` — `load_matpes_state_dict` 降为薄 wrapper，删除本地 `_KEY_REMAP`
  / `_KEY_REMAP_ORDER`
- `src/molzoo/mace_omol.py` — `load_omol_state_dict` 降为薄 wrapper，删除本地 `_KEY_REMAP`
  / `_KEY_REMAP_ORDER`
- `tests/test_molzoo/test_mace/test_checkpoint.py` (new)
- `regressions/mace-subpackage-restructure-05-checkpoint.py` (new)

**前置条件（任务 1 首先核对，不满足即 Iron-law 停机上报）**：02/06 的测试迁移必须已保证
`tests/test_molzoo/test_mace/` 是包且 `tests/test_molzoo/test_mace.py` 不再遮蔽同名。同名 `.py`
与包目录并存会造成 `tests.test_molzoo.test_mace` 模块名冲突，pytest 收集将不确定。

## Tasks

- [ ] Write failing unit tests for `CheckpointRemap` (tests/test_molzoo/test_mace/test_checkpoint.py → `TestCheckpointRemap`): round-trip both families, `on_unexpected` raise-vs-return, unfilled-learnable raise, shape-mismatch raise, `(1,)`-vs-0d reshape accept; first assert `tests/test_molzoo/test_mace.py` no longer shadows the package
- [ ] Implement `CheckpointRemap` (`.rename` / `.load`) plus `MATPES_KEY_REMAP`, `OMOL_KEY_REMAP`, `MATPES_REMAP`, `OMOL_REMAP` in src/molzoo/mace/checkpoint.py, with Google-style docstrings citing arXiv:2206.07697, eV / eV·Å units, and both incidents (mace_matpes.py:405-410, mace_omol.py:437-439)
- [ ] Write failing unit tests for `MACEPotential.from_checkpoint` (tests/test_molzoo/test_mace/test_checkpoint.py → `TestFromCheckpoint`): tmp-dir config json + official-named weights round-trip, `hidden_irreps`/`MLP_irreps` → `num_features`/`max_hidden_l`/`mlp_dim` derivation, missing-config-key raise
- [ ] Implement `MACEPotential.from_checkpoint` classmethod in src/molzoo/mace/potential.py (json → inline spec translation → construct → `remap.load`; no hardcoded 128/1/16)
- [ ] Export `CheckpointRemap`, `MATPES_REMAP`, `OMOL_REMAP` from src/molzoo/mace/__init__.py
- [ ] Reduce `load_matpes_state_dict` (src/molzoo/mace_matpes.py) and `load_omol_state_dict` (src/molzoo/mace_omol.py) to thin wrappers over the preset instances, deleting both local `_KEY_REMAP` / `_KEY_REMAP_ORDER` copies, and confirm tests/test_molzoo/test_mace_matpes.py::TestLoadMatpesStateDict still passes unmodified
- [ ] Verify official-checkpoint bit-parity in tests/test_molzoo/test_mace/test_checkpoint.py (`matpes_r2scan_config.json` + `matpes_r2scan_cueq_state.pt` via `from_checkpoint` vs the flat load path, every state_dict tensor `torch.equal`), `skipif` when `MOLNEX_MACE_WEIGHTS_DIR` is unset
- [ ] Add regression example regressions/mace-subpackage-restructure-05-checkpoint.py (public API only; deterministic non-RNG weights, hard-coded E/F goldens, no third-party runtime)
- [ ] Run full check + test suite (`ruff check src/ && ruff format --check src/`, `python -m pytest tests/ -v`)

## Testing strategy

单元测试全部落在 `tests/test_molzoo/test_mace/test_checkpoint.py`（镜像
`src/molzoo/mace/checkpoint.py`），类名镜像被测类型（`CheckpointRemap` → `TestCheckpointRemap`），
每个测试只针对一个方法/一条行为，`tests/` 下不放 e2e。fixture 沿用
`tests/test_molzoo/test_mace_matpes.py` 的做法：fp64 autouse fixture（cuEq 在构造期冻结工作精度）、
小模型（`l_max=1, num_features=16, num_interactions=2`）、`use_fallback=True`（CPU 无 fused wheel）。

`TestCheckpointRemap`（happy path）
- `test_rename_is_pure`：`.rename` 不修改输入、不触碰模型，`.graph.c` 与 `output_mask` 键被丢弃。
- `test_roundtrip_restores_every_parameter_matpes`：把模型自己的 `state_dict` 改名成官方 cueq 名
  （复用 `_roundtrip_state` 手法）再装回一个新构造的模型，逐 `named_parameters()` `torch.equal`。
- `test_roundtrip_restores_every_parameter_omol`：OMol 表同样断言（此前 OMol loader **没有**任何
  单元测试，本步补上）。

`TestCheckpointRemap`（edge / 教条）
- `test_raise_policy_rejects_unexpected_key`：注入 `interactions.0.mystery_layer.weight`，
  `pytest.raises(RuntimeError, match="no home")`，且模型参数在 raise 后**未被改动**（无半装载）。
- `test_return_policy_reports_unexpected_key`：同一注入下 `on_unexpected="return"` 不抛异常，
  返回值第二项 `== ["interactions.0.mystery_layer.weight"]`。
- `test_rejects_missing_parameter`：删掉 `interactions.0.linear_up.weight` →
  `match="not covered"`；两种策略下都必须抛（教条与旋钮正交）。
- `test_rejects_missing_parameter_without_weight_suffix`：删掉 `bessel.freqs`（既非 `.weight`
  也非 `.bias` 的 `nn.Parameter`）也必须抛——`mace_omol.py:437-439` 事故的回归锁。
- `test_rejects_shape_mismatch`：`match="shape mismatch"`。
- `test_accepts_rank_difference_on_frozen_scalars`：`scale_shift.scale` 以 `(1,)` 提供，
  装载后 0-d buffer 值 `== 0.75`。
- `test_reported_lists_are_sorted`。

`TestFromCheckpoint`
- `test_builds_from_config_and_weights`：`tmp_path` 写一份最小 config json + 官方名 state_dict，
  `from_checkpoint` 得到的模型逐参数 `torch.equal` 于源模型。
- `test_derives_dims_from_irreps`：config 里 `hidden_irreps="32x0e+32x1o"`、`MLP_irreps="8x0e"`
  → 模型 `num_features=32, max_hidden_l=1, mlp_dim=8`（硬编码值一律不出现）。
- `test_rejects_config_missing_irreps`：缺 `hidden_irreps` 时 raise 且异常信息列出缺失 key。

领域验证（`science.required`）
- `test_official_checkpoint_bit_parity`：`MOLNEX_MACE_WEIGHTS_DIR` 指向真实权重目录时，
  `MACEPotential.from_checkpoint(dir/"matpes_r2scan_config.json", dir/"matpes_r2scan_cueq_state.pt")`
  与"手工构造 + `MATPES_REMAP.load`"两条路径产出的 `state_dict` 张量**逐个 `torch.equal`**
  （bit-parity，不是容差比较）；权重缺席时 `pytest.mark.skipif`。真实权重按
  `.claude/notes/` 记录位于用户 `mace_models` 目录之下，属离线资产，不入库。

回归示例（**不可 skip**）
- `regressions/mace-subpackage-restructure-05-checkpoint.py`：仅用公开 API 的最小脚本
  （`python regressions/mace-subpackage-restructure-05-checkpoint.py`，不被 `pytest tests/` 收集）。
  在 `tempfile` 目录里就地构造一个**小型合成 checkpoint**：按 04 的 spec 建小模型，用**形状派生的
  确定性规则**填参数（`torch.linspace` 之类，**不用 RNG**——避免 goldens 被 torch RNG 顺序变更打翻），
  把 `state_dict` 改名成官方 key、连同一份小 config json 存盘，再 `MACEPotential.from_checkpoint`
  读回，对固定的 5 原子团簇（fp64、CPU、`use_fallback=True`）算能量与力，与脚本内**硬编码**的
  golden 比对：`|ΔE| ≤ 1e-9 eV`，`max|ΔF| ≤ 1e-8 eV/Å`（沿用
  `tests/test_molzoo/test_mace_matpes.py:96-97` 的 fp64 容差）。goldens 由脚本自身首跑产生并粘回，
  注释注明来源（本仓库自洽 oracle、torch 版本、命令、日期）；**运行期不 import 任何第三方 oracle**
  （无 mace-torch / e3nn / ASE）。

## Out of scope

- 把 `scripts/matpes_port/run_nve.py:145-169` 的 `build_model` 改指 `from_checkpoint`（07 步）。
- 删除或 alias `load_matpes_state_dict` / `load_omol_state_dict`（06/07 步）；本步只降为 wrapper。
- 刷新 `src/molzoo/specs/mace_matpes.md` / `mace_omol.md` §5 的代码锚点：本步旧函数名仍然存在且
  语义不变，锚点依旧可解析；锚点随符号消失的那一步（07）一并刷新。
- 官方 `.model`（TorchScript pickle）直读，或在仓内做 e3nn→cueq 转换——转换仍是离线步骤。
- OMol 家族那些被刻意不建模的辅助头（`on_unexpected="return"` 存在的原因）；本步不新增建模。
- checkpoint 的**保存**方向（molnex → 官方名导出）、多头 / stress / virial 权重。
- 与 `molix.core.checkpoint` 的任何统一或互操作。
