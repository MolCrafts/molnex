---
title: MACE 子包核心层 — spec / geometry / encoder
status: approved
created: 2026-08-08
revised: 2026-08-08
grilled: true
chain: mace-subpackage-restructure
---

# MACE 子包核心层 — spec / geometry / encoder

## Summary

在 `src/molzoo/mace/` 下建立 MACE 家族的核心层：一个 torch-free 的 pydantic 配置族（`MACESpec` 基类 + `MACEMatpesSpec` / `MACEOMolSpec`），一个 MACE 专属的边几何属主（`edge_vectors` / `edge_lengths`，加性 PBC `shifts`），以及一个由配置驱动的骨干编码器 `MACEEncoder`，用它替换 `mace_matpes.py` 与 `mace_omol.py` 里两份近乎复制的层栈构造与两份内联几何。本步只搭建新包并让它与现有 flat 模块**并存**：不接能量/力 forward（04）、不接 checkpoint 载入（05）、不改 `molzoo/__init__.py` 的导出编排（06）、不删 flat 模块（07）。交付后，用同一份权重把 `MACEEncoder` 的 `state_dict` 键集/形状与今天的 flat 变体对齐，激活值可逐位复现，且现有测试全绿。

## Domain basis

MACE 消息传递与对称收缩：Batatia et al., "MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields", NeurIPS 2022, https://arxiv.org/abs/2206.07697 。基础模型变体：MACE-MP-0 https://arxiv.org/abs/2401.00096 （Agnesi 距离变换、ZBL 排斥包络）与 MatPES https://arxiv.org/abs/2503.04070 。上述引用与 `src/molzoo/mace_matpes.py:23-30`、`src/molzoo/mace_omol.py:13-15` 现有 docstring 一致，本步不新增未经在库引用的文献。

几何定义（单位：位置 Å，能量 eV，`atomic_energies` 为 eV/atom）：

- 边位移 `r_ij = pos[target] - pos[source] + S_ij`，其中 `edge_index[:,0]=source`、`edge_index[:,1]=target`（仓库全局约定）；
- PBC 位移 `S_ij = n_ij · h`（`unit_shifts @ cell`），`n_ij ∈ Z^3`，`h` 为盒矩阵；`S_ij` 对 `pos` 为常量，故 `∂r_ij/∂pos` 与开放体系一致；
- 边长 `d_ij = ‖r_ij‖`；MatPES 走 `bessel(AgnesiTransform(d, Z_s, Z_t)) * PolynomialCutoff(d)`，OMol 走 `bessel(d)` 且把 `PolynomialCutoff(d)` 单独送进 `ResidualInteraction` —— 两者的截断折叠位置不同，是本步必须保持逐位一致的语义差。

两个几何属主的数学理由（见 Design）：PiNet 的 `edge_bond_diff`（`src/molzoo/pinet/geometry.py:28-49`）计算的是"取最小镜像值、梯度走 straight-through" 的量 `detach(imaged) + (raw - detach(raw))`；MACE 的量是"原始位移 + 加性 `S_ij`"。二者梯度同为 `∂raw/∂pos`，但**值不同**（最小镜像 vs. 显式整数平移），是两个数学对象。

## Design

### 命名空间冲突（必须先处理；含测试侧同名冲突，spec-audit 裁定 2026-08-08）

**测试侧同名冲突（本步一并解决）**：`tests/test_molzoo/test_mace.py`（模块）与本步新建的
`tests/test_molzoo/test_mace/`（包）同名——Python 导入系统包优先，模块内测试会**静默不被收集**。
因此本步在创建测试包的同一任务中完成 `test_mace.py` 的迁移删除：TestEmbeddingBlock /
TestInteractionBlock（及其 Equivariance 类）随 01 的 molrep 归属进
`tests/test_molrep/`（01 已建镜像）或折进本步的 `test_mace/test_encoder.py`（仍属
research 编码器的用例）；TestProductHead* 并入 `tests/test_molrep/test_readout/test_mace.py`
（01 已建，去重）；`test_mace.py` 删除。06 的测试迁移范围相应缩小（只剩
test_mace_encoder / test_mace_matpes / test_mace_omol 三个文件）。收集数守恒由
`pytest --collect-only` 前后对比保证。

### 源码侧命名空间冲突

`src/molzoo/mace.py` 与新建的包目录 `src/molzoo/mace/` 同名；CPython 的路径查找中**包优先于同名模块**，因此一旦 `src/molzoo/mace/__init__.py` 出现，`src/molzoo/mace.py` 立即不可达，`src/molzoo/__init__.py:9` 的 `from molzoo.mace import MACE, MACESpec` 会 ImportError，进而拖垮所有 `from molzoo import ...` 的测试。链式不变量"现有测试保持通过"要求本步同时解决它：

- `git mv src/molzoo/mace.py src/molzoo/mace/research.py`（内容整体搬迁，不改架构；顺带把 `mace.py:545` 的 `self.config.num_interactions` 改为 `__init__` 里固化的 `self.num_interactions` 普通属性 —— 这是同一条 dynamo graph-break 隐患，位于本步触碰的表面，按 iron law 一并修掉）；
- `src/molzoo/mace/__init__.py` 作为过渡兼容层，重新导出 `MACE` / `EmbeddingBlock` / `InteractionBlock`，并按 `molzoo/__init__.py:19-24` 的 PEP 562 惰性模式懒加载 `research` 与 `encoder`，使 `from molzoo.mace import MACEMatpesSpec` 不触发 cuEq 栈导入。既有调用点 `benchmarks/bench_trainer_throughput.py:26`、`tests/test_molzoo/test_mace.py:24`、`src/molzoo/__init__.py:9` 因此**无需改动**。

### `MACESpec` 语义迁移（显式声明）

今天的 `MACESpec`（`src/molzoo/mace.py:577`）是**研究版编码器**的配置（含 `node_attr_specs`）。本步之后，`molzoo.mace.MACESpec` 指向**基础模型变体的共享配置基类**；研究版配置在 `research.py` 内更名为 `MACEResearchSpec` 并同名导出。用户提出的兼容承诺只覆盖 `MACE` / `MACEMatpes` / `MACEOMol` 三个名字，它们全部保持可导入；`MACESpec` 的语义变更是**有意的**，本步同步 `src/molzoo/README.md:62-67` 的示例与说明。

`node_attr_specs` **不进入** spec 模型：`molrep.embedding.node`（`src/molrep/embedding/node.py:7-8`）导入 `cuequivariance` / `cuequivariance_torch`，让它进入 spec.py 会摧毁 spec.py 的零重依赖性质。它保持为编码器构造器参数（`research.MACE.__init__` 现状即如此）；本步不在 `MACEEncoder` 上预留未使用形参（"第二次真实使用前不抽取"）。

### `spec.py`（纯配置，零 torch）

模仿 `src/molzoo/pinet/spec.py:10-13`：`BaseModel` + `ConfigDict(arbitrary_types_allowed=True)` + `Field` 约束，模块内只导入 `typing` 与 `pydantic`。

`MACESpec`（共享基类）字段：`atomic_numbers: list[int]`、`atomic_energies: list[float]`（**list 而非 tensor**，张量转换在 `MACEEncoder.__init__` 内完成）、`r_max`、`num_bessel`、`num_polynomial_cutoff`、`l_max`、`num_features`、`num_interactions`、`correlation`、`mlp_dim`、`scale`、`shift`、`use_fallback`，外加五个变体开关（基类必填，子类给定默认）：`interaction: Literal["density","residual"]`、`readout: Literal["per_layer","final"]`、`distance_transform: Literal["none","agnesi"]`、`pair_repulsion: Literal["none","zbl"]`、`conditioning: Literal["none","charge_spin"]`。

`MACEMatpesSpec` 追加 `max_hidden_l`、`radial_mlp: list[int]`，默认对齐 `mace_matpes.py:89-107`（17 个构造 kwargs）；`MACEOMolSpec` 追加 `edge_channels`、`charge_classes`、`charge_offset`、`spin_classes`、`spin_offset`，默认对齐 `mace_omol.py:65-85`（18 个构造 kwargs）。

错误教义（沿用 `mace_matpes.py:206-222` 的"早失败、说清楚"）：`model_validator(mode="after")` 里立即抛 `ValueError` —— `len(atomic_energies) != len(atomic_numbers)`；`atomic_numbers` 非严格升序或有重复（`torch.searchsorted` 依赖有序表，今天**没有任何校验**，乱序会被静默吸附到邻近元素并给出错误能量）；MatPES `num_interactions < 2`；OMol `l_max < 1`（`mace_omol.py:151-153` 的 `edge_mid` 用 `range(l_max)`）。

### `geometry.py`（MACE 的唯一几何属主）

两个模块级纯函数（"无自然宿主的纯数学"例外，形态对齐 `pinet/geometry.py`）：`edge_vectors(pos, edge_index, shifts=None) -> (E,3)` 与 `edge_lengths(vectors, *, keepdim=False) -> (E,) | (E,1)`（`keepdim` 服务 OMol 现有的 `(E,1)` 写法）。二者替换 `mace_matpes.py:258-262` 与 `mace_omol.py:259-263` 的内联重复。组合由调用方完成，不提供 `compute_geometry()` 之类的一步式外观。

### `encoder.py`（一个配置驱动的骨干）

`MACEEncoder(nn.Module)`，**构造器直接吃校验后的配置**：`MACEEncoder(spec: MACESpec)`。它注册两条基础模型变体的完整模块图，属性名与今天的 flat 变体逐字一致 —— `z_table`（buffer，升序）、`node_embedding`、`spherical_harmonics`、`bessel`、`cutoff_fn`、`distance_transform`(MatPES)、`pair_repulsion`(MatPES)、`atomic_energies`、`joint_embedding` / `embedding_readout`(OMol)、`interactions`、`products`、`readouts`(MatPES) / `readout`(OMol)、`scale_shift` —— 这是链式不变量"`load_state_dict` 直接搬运"的实现方式：04 的变体类**继承** `MACEEncoder` 再加能量/力编排（与上游 `ScaleShiftMACE(MACE)` 同形），键路径因此保持扁平，不会多出 `encoder.` 前缀。

公开原语（每个一件事，组合由调用方/04 负责）：`validate_elements(Z)`、`node_attrs(Z, dtype)`、`initial_node_features(node_attrs)`、`angular_features(vectors)`、`radial_features(lengths, Z, edge_index) -> (edge_feats, cutoff)`、`conditioning(batch, *, total_spin, total_charge)`（未配置 charge/spin 时抛 `ValueError`）、`layer_features(...) -> list[Tensor]`（逐层节点特征；MatPES 各层宽度不同，故返回 list 而非堆叠张量）。

热路径纪律：`__init__` 把所有分支开关固化为普通 `bool` / `int` 属性（`self.num_interactions`、`self.fold_cutoff_into_radial`、`self.pass_cutoff_to_interaction`、`self.has_pair_repulsion` …），模仿 `mace_matpes.py:118`。配置对象仅以 `self._spec` 留作溯源（供 05 使用），**任何 forward/原语方法都不得读取它**。

`_sh_irreps`（`mace_matpes.py:55-57`）与 OMol 的内联同款字符串拼接（`mace_omol.py:96-100`）合并为 encoder.py 的一个私有 irreps 构造器。

### Reuse decision

- `src/molzoo/pinet/spec.py:10-31` 的纯配置模块形态 —— **pattern**：spec.py 逐条照抄（BaseModel / ConfigDict / Field 约束 / 零重依赖）。
- `src/molzoo/pinet/encoder.py:177-180` 的 `from_spec` —— **pattern（采纳精神，显式偏离形式）**：保留"校验后的 spec 驱动构造"，但**不**新增 `from_spec` 类方法。理由：`cls(**spec.model_dump())` 要求构造器接受两变体 kwargs 的并集（20+ 形参的 god ctor），而 `from_spec = cls(spec)` 只是 `__init__` 的别名工厂，被 CLAUDE.md "Forbid: factory functions / make_foo aliases" 明令禁止；`README.md:66` 已有 `MACE(MACESpec(...))` 的库内先例。
- `mace_matpes.py:118` 的普通属性热路径 —— **pattern**：`MACEEncoder` 与 `research.MACE` 都改用它。
- `MACEMatpes.validate_elements`（`mace_matpes.py:206-222`）—— **reuse**：实现整体迁至 `MACEEncoder`（含 docstring 与错误文案），flat 副本由 07 随模块删除。
- `tests/conftest.py:36` `make_graph_batch` —— **generalize**：加可选 `shifts=`，写入 `edges.shifts` 并折进 `edge_diff`；`rotate_graph` 需同步旋转 `shifts`（`S = n·h`，`h` 随旋转），`translate_graph` / `permute_graph` 原样携带。**不**在 `tests/test_molzoo/**/conftest.py` 里另建 batch builder。
- `molrep` / `molpot` 现有块（`BesselRBF`、`PolynomialCutoff`、`SphericalHarmonics`、`AgnesiTransform`、`ZBLRepulsion`、`DensityInteraction`、`DensityResidualInteraction`、`ResidualInteraction`、`EquivariantProductBasis`、`LinearReadout` / `NonLinearReadout` / `NonLinearBiasReadout`、`AtomicReferenceEnergy`、`GlobalRescale`、`JointFeatureEmbedding`）—— **reuse**：一律直接调用（导入路径以 01-blocks 落地后的 molrep 布局为准），本步不新写任何等价块。
- `src/molzoo/pinet/geometry.py` —— **new（两个几何属主，附理由）**：STE-最小镜像 vs. 加性 `S_ij` 是不同数学对象，强行合并会把 PiNet 的 STE 语义偷渡进 MACE 的 PBC 路径（librarian 已预授权）。

## Files to create or modify

- `src/molzoo/mace/__init__.py` (new)
- `src/molzoo/mace/spec.py` (new)
- `src/molzoo/mace/geometry.py` (new)
- `src/molzoo/mace/encoder.py` (new)
- `src/molzoo/mace/research.py` (new — 由 `src/molzoo/mace.py` 整体移动)
- `src/molzoo/mace.py` (移动源，本步后不再存在)
- `src/molzoo/README.md`
- `tests/conftest.py`
- `tests/test_molzoo/test_mace/__init__.py` (new)
- `tests/test_molzoo/test_mace/conftest.py` (new)
- `tests/test_molzoo/test_mace.py`（删除——同名遮蔽，用例迁入镜像，见 Design）
- `tests/test_molzoo/test_mace/test_spec.py` (new)
- `tests/test_molzoo/test_mace/test_geometry.py` (new)
- `tests/test_molzoo/test_mace/test_encoder.py` (new)
- `regressions/mace-subpackage-restructure-02-core.py` (new)

## Tasks

- [ ] Move `src/molzoo/mace.py` to `src/molzoo/mace/research.py` and add the `src/molzoo/mace/__init__.py` compat shim (re-export `MACE` / `EmbeddingBlock` / `InteractionBlock` / `MACESpec`; PEP 562 lazy loading per `molzoo/__init__.py:19-24`), replacing the `self.config.num_interactions` read at `mace.py:545` with a plain `self.num_interactions` attribute
- [ ] Generalize `make_graph_batch` in `tests/conftest.py` with an optional `shifts=` parameter (stored at `edges.shifts`, folded into `edge_diff`; rotated by `rotate_graph`, carried unchanged by `translate_graph` / `permute_graph`)
- [ ] Migrate and delete `tests/test_molzoo/test_mace.py` while creating the `tests/test_molzoo/test_mace/` package (fold research-encoder cases into test_mace/test_encoder.py, molrep-block cases into the 01 mirrors, dedupe TestProductHead*; assert collected-test count is preserved via `pytest --collect-only` before/after)
- [ ] Write failing unit tests for the spec models (`tests/test_molzoo/test_mace/test_spec.py` → `TestMACESpec` / `TestMACEMatpesSpec` / `TestMACEOMolSpec`)
- [ ] Implement `MACESpec` / `MACEMatpesSpec` / `MACEOMolSpec` in `src/molzoo/mace/spec.py` with Google-style docstrings (units: Å / eV), flip the `src/molzoo/mace/__init__.py` export to the new `MACESpec` (research config renamed `MACEResearchSpec`), and sync `src/molzoo/README.md:62-67`
- [ ] Write failing unit tests for MACE edge geometry (`tests/test_molzoo/test_mace/test_geometry.py` → `TestEdgeVectors` / `TestEdgeLengths`)
- [ ] Implement `edge_vectors` / `edge_lengths` in `src/molzoo/mace/geometry.py` with shape/unit-annotated docstrings and the two-geometry-owner rationale in the module docstring
- [ ] Write failing unit tests for the backbone (`tests/test_molzoo/test_mace/conftest.py`, `tests/test_molzoo/test_mace/test_encoder.py` → `TestMACEEncoder`), covering `state_dict` key/shape parity against `MACEMatpes` and `MACEOMol`
- [ ] Implement `MACEEncoder` in `src/molzoo/mace/encoder.py` (spec-driven stack, verbatim submodule names, plain-attribute hot path, `validate_elements` migrated from `mace_matpes.py:206-222`)
- [ ] Add regression example `regressions/mace-subpackage-restructure-02-core.py` (public API only; deterministic `linspace` weights, literal coordinates, hard-coded goldens generated once from today's flat variants, no import of `molzoo.mace_matpes` / `molzoo.mace_omol` at run time)
- [ ] Run full check + test suite

## Testing strategy

单元测试全部落在 `tests/` 镜像路径下，每个用例只打一个函数/方法；跨模块的组合场景一律进 `regressions/`。

`tests/test_molzoo/test_mace/conftest.py` —— **仅** MACE 专属 fixture：`atomic_numbers=[1,6,8]`、`atomic_energies=[-13.6,-1029.0,-2041.0]`、tiny `MACEMatpesSpec`（`r_max=5.0, num_bessel=4, l_max=1, num_features=16, max_hidden_l=1, num_interactions=2, correlation=2, mlp_dim=8, radial_mlp=[8], use_fallback=True`，对齐 `tests/test_molzoo/test_mace_matpes.py:38-56`）、tiny `MACEOMolSpec`、以及照搬 `test_mace_matpes.py:22-35` 的 fp64 autouse fixture。batch 构造一律用 `tests.conftest.make_graph_batch`。

`test_spec.py::TestMACESpec` / `TestMACEMatpesSpec` / `TestMACEOMolSpec`
- happy path：每个字段默认值等于对应 flat 构造器的默认值（表驱动，逐字段断言，硬编码期望）。
- 边界：`atomic_energies` 长度不匹配、`atomic_numbers` 乱序、`atomic_numbers` 重复、MatPES `num_interactions=1`、OMol `l_max=0` → 均抛 `ValueError`；`num_bessel=0` / `r_max=0` → `ValidationError`。
- 零重依赖：以 `ast.parse` 读 `src/molzoo/mace/spec.py`，断言不出现 `torch` / `cuequivariance*` / `tensordict` / `molrep` / `molix` / `molzoo.*` 的任何 import；`model_dump()` 的值全部是 JSON 原生类型（无 `Tensor`）。
- 惰性：子进程内 `import molzoo.mace; molzoo.mace.MACEMatpesSpec`，断言 `"molzoo.mace.encoder" not in sys.modules`。

`test_geometry.py::TestEdgeVectors` / `TestEdgeLengths`
- happy path：3 原子、字面坐标，`edge_vectors` 逐元素等于手算期望（硬编码）。
- PBC：边长 3.0 Å 立方盒、`unit_shift=(1,0,0)`，`edge_vectors(..., shifts=S)` 等于 `edge_vectors(...) + S`，且长度等于硬编码的最小镜像距离。
- 梯度：`torch.autograd.grad` 对 `pos` 非空，且等于解析的 `±r̂`；`shifts` 存在时梯度不变（`S` 对 `pos` 为常量）。
- 形状：`edge_lengths(..., keepdim=True).shape == (E,1)`，默认 `(E,)`。

`test_encoder.py::TestMACEEncoder`
- **键集平价（链式闸门）**：tiny MatPES / tiny OMol 两组配置下，`set(MACEEncoder(spec).state_dict())` 与同参 flat 变体的 `state_dict()` 键集完全相等，且逐键 `shape` / `dtype` 相等。
- `node_attrs`：硬编码 one-hot 期望矩阵；`validate_elements`：表外元素抛 `ValueError` 且消息里列出越界的 Z。
- `layer_features`：返回长度 `== num_interactions` 的 list，逐层 dtype 为 `config.ftype`，第 0 维 `== N`。
- 领域校验（fp64）：整体平移下最后一层（纯标量 irreps）特征不变（`atol=1e-10`）；带 `shifts` 的绕轴旋转下同一特征不变（`atol=1e-10`）—— 同时覆盖 `make_graph_batch` 对 `shifts` 的旋转处理。
- 热路径纪律：`del encoder._spec` 后所有原语仍可运行；并以 `ast.parse` 断言 `encoder.py` 的原语方法体内不出现 `self._spec`。

**回归样例**：`regressions/mace-subpackage-restructure-02-core.py`（仓库根 `regressions/`，非 `tests/`）。只用公开 API：`MACEMatpesSpec` / `MACEOMolSpec` → `MACEEncoder` → `geometry.edge_vectors/edge_lengths` → 原语 → 由脚本自己把 `E0 + scale_shift(ZBL + Σ readout)` 组合成总能量（组合是调用方的职责）。权重用确定性填充（每个参数 `linspace(-0.1, 0.1, numel).reshape(shape)`，不用 RNG），坐标为字面量的 5 原子团簇，fp64 + CPU + `use_fallback=True`。金标准（两变体的总能量、逐层特征平方和）在实现期由今天的 flat 变体生成一次后**以字面量内嵌**，注释写明生成脚本、torch 版本、命令与日期；运行期**不导入** `molzoo.mace_matpes` / `molzoo.mace_omol`（它们将在 07 被删除）。不匹配则以非零码退出。

## Out of scope

- 能量/力 forward 与 `MACEMatpes` / `MACEOMol` 变体类的新实现 —— 04-potential。
- checkpoint 键重映射与 `load_*_state_dict`（`mace_matpes.py:387-465`）—— 05-checkpoint。
- 融合 kernel / `use_fallback` 策略、`torch.compile(fullgraph=True)` 验证与性能基准 —— 03-kernels。
- `src/molzoo/__init__.py` 导出编排、flat 模块删除与调用点切换 —— 06-wire / 07-cleanup（本步靠兼容层保证零调用点改动）。
- 把研究版 `MACE` 折叠进 `MACEEncoder`：其层栈（`EmbeddingBlock` / `InteractionBlock` / `ProductHead` / `ElementUpdate` / `layer_norms`）与两个基础变体的键集完全不相交，塞进同一构造器会得到一个三分支的 god ctor；本步只统一真正重复的两个基础变体，研究版归并留给 06/07 评估（已在 Design 中显式声明这一偏离）。
- `src/molzoo/specs/mace.md:8-9`、`docs/api/molzoo.md:5`、`.claude/notes/architecture.md:65` 中对 `molzoo.mace` 模块与 `MACESpec` 的描述同步：spec 文档由 `/molzoo-spec` 技能独占，架构笔记由 `/mol:note` 独占，均随 07-cleanup 一并处理 —— 本步只同步 `src/molzoo/README.md`（已知漂移，已具名，未静默忽略）。
