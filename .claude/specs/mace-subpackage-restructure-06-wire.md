---
title: MACE 子包切换 — 导入面收口、扁平模块下线、测试迁移
status: approved
created: 2026-08-08
revised: 2026-08-08
grilled: true
chain: mace-subpackage-restructure
---

# MACE 子包切换（06-wire）

## Summary

链条 01–05 已在 `src/molzoo/mace/` 旁路建成完整的 MACE 子包（spec / geometry /
encoder / potential / checkpoint），扁平模块 `molzoo/mace_matpes.py`、
`molzoo/mace_omol.py`（以及 02 移入 `mace/research.py` 前的 `molzoo/mace.py`）仍
并存并且仍是全仓库的实际导入目标。本 spec 完成**切换**：把 `molzoo.mace` 变成一个
像 `molzoo.pinet` 那样自我说明的扁平再导出包，把 `molzoo/__init__.py` 收敛到
**一条**统一的 PEP 562 惰性导出策略，把 `MACEMatpes` / `MACEOMol` /
`load_matpes_state_dict` / `load_omol_state_dict` 做成落在新子包上的薄别名，删除
残余扁平模块，并把四个扁平测试文件折进 `tests/test_molzoo/test_mace/` 镜像目录。
切换完成后，用户可见的行为只有一条变化：`import molzoo` 不再拉起 cuEquivariance
栈；其余所有导入路径、构造签名、能量/力数值与检查点加载行为逐字保持不变。

## Domain basis

本 spec **不改动任何物理**：不新增方程、不改基组、不改单位、不改容差。它承接并
必须保持 01–05 已锁定的数值契约：

- MACE 等变消息传递：Batatia et al., NeurIPS 2022, https://arxiv.org/abs/2206.07697
- MACE-MP-0 基础模型：Batatia et al., https://arxiv.org/abs/2401.00096
- MatPES 数据集/权重：Kaplan et al., https://arxiv.org/abs/2503.04070

不变量（切换前后逐位一致，单位 eV / eV·Å）：

- `MACEMatpes.forward(td)` 与 `energy_forces(...)` 的一致性容差保持
  `1e-9 eV` / `1e-8 eV·Å`（现 `tests/test_molzoo/test_mace_matpes.py` 的判据）。
- `MACEOMol` 对官方 OMOL 的 `7e-7 eV` / `4.3e-6 eV·Å` 结论（mace-omol-port-01
  ac-006）不得因重组失效；本 spec 通过"参数张量清单不变 + 检查点仍严格加载"来
  守住它，而不是重新做外部对照（外部 oracle 离线，仓内不得引入 mace-torch）。
- 边约定 `edge_index[:,0]=source`、`edge_diff = pos[target] - pos[source]`，
  collate 后 `atoms/edges/graphs` 命名空间不变。

## Design

### 1. `src/molzoo/mace/__init__.py` — 扁平再导出面

完全对齐 `src/molzoo/pinet/__init__.py:1-31` 的形态：包 docstring 用一张"模块 →
单一职责"清单说明拆分（`spec` 配置与变体预设 / `geometry` PBC 安全的边几何 /
`encoder` molrep 块拼出的特征编码器 / `potential` 能量与力 / `checkpoint` 官方
权重重映射 / `variants` 具名基础模型 / `research` 研究版编码器），随后一句
"Public import surface is stable"示例，最后**显式** `__all__`。除
`from .x import Y` 外不含逻辑——所有实现留在子模块，这一层只负责导入面。

### 2. `src/molzoo/__init__.py` — 统一惰性策略

现状是**不对称**的：`Allegro` / `MACE` / `PiNet` 在模块级 eager 导入（因此
`import molzoo` 就付掉整个 cuEquivariance 导入代价），而 `MACEMatpes` /
`MACEOMol` 走 PEP 562 惰性（`molzoo/__init__.py:19-24`）。惰性的**理由**（避免
cuEq 导入代价）对 eager 的三个同样成立，所以策略统一为：

> **所有模型符号一律惰性。** `molzoo/__init__.py` 在模块级不导入任何模型模块；
> `_LAZY` 表覆盖 `Allegro/AllegroSpec/MACE/MACESpec/PiNet/PiNetSpec/MACEMatpes/
> MACEOMol/load_matpes_state_dict/load_omol_state_dict`；`TYPE_CHECKING` 块给静
> 态检查与 IDE 提供真实签名；`__getattr__` 未知名照旧抛 `AttributeError`；补
> `__dir__` 返回 `__all__` 以保住补全与 `from molzoo import *`。

这条策略写进模块 docstring（一句话 + 一句理由），使它成为可被 review 的显式约
定而不是三处巧合。子模块直接导入（`import molzoo.mace`、
`from molzoo.mace import MACE`）不受影响，仍是常规导入。

### 3. `src/molzoo/mace/variants.py`（新）— 具名基础模型

`MACEMatpes` / `MACEOMol` 是**既有公共 API**（`scripts/matpes_port/run_nve.py`、
`benchmarks/bench_mace_matpes.py` 直接构造），必须继续以关键字构造。它们在新布局
里退化为"预设 + 装配"，因此给它们一个**具名单一归宿** `variants.py`，而不是散在
`__init__.py` 里（后者无法被测试镜像门覆盖）。

- 两个类是薄适配器：把旧构造签名逐项映射到 04/05 已定义的 spec 预设与
  potential 类型；**不得**在此重新推导任何超参数、irreps 字符串或 E0 表——一旦
  发现需要重推，说明预设留在了错的模块，应回到 02/04/05 修，而不是在这里复制。
- 必须保持存活的消费面（已 grep 全仓确认）：
  - `MACEMatpes(**kwargs)`：`atomic_numbers, atomic_energies, r_max, num_bessel,
    num_polynomial_cutoff, l_max, num_features, max_hidden_l, num_interactions,
    correlation, mlp_dim, radial_mlp, scale, shift, use_fallback`
  - `MACEOMol(**kwargs)`：`atomic_numbers, atomic_energies, r_max, num_bessel,
    num_polynomial_cutoff, l_max, num_features, num_interactions, correlation,
    mlp_dim, edge_channels, charge_classes, charge_offset, spin_classes,
    spin_offset, scale, shift`
  - 方法：`.forward(td)`、`.energy_forces(...)`、`.validate_elements(Z)`、
    buffer `z_table`
  - **`._compute_energy(pos, Z, edge_index, batch, num_graphs, shifts)`（位置参
    数）** —— `scripts/matpes_port/run_nve.py:118` 把这个**私有**方法作为可编译能
    量核心直接绑走。这是本次切换发现的既有债：私有符号被跨包消费。本 spec 不改
    consumer（07 负责把它改绑到公开核心），但因此必须**原样保留该位置签名**，否
    则 NVE 脚本会在切换当天静默失效。别名类的 docstring 要标注这一点并指向 07。
- `load_matpes_state_dict(model, state)` / `load_omol_state_dict(model, cueq_state)`
  保留为自由函数别名，转发到 05 的 `CheckpointRemap`。按 CLAUDE.md"禁止工厂函数"
  这本不该是新符号的形态，它们**仅因既有公共 API 兼容**而保留；`variants.py`
  docstring 需注明"新代码请用 `CheckpointRemap`"。

### 4. 扁平模块下线与消费者核对

删除 `src/molzoo/{mace_matpes,mace_omol}.py`（`molzoo/mace.py` 已由 02 移入
`mace/research.py`）。全仓 grep 结果（`from|import molzoo...`）分三类：

| 消费点 | 形式 | 切换后 |
|---|---|---|
| `scripts/matpes_port/run_nve.py:45`、`benchmarks/bench_mace_matpes.py:38` | `from molzoo import MACEMatpes, load_matpes_state_dict` | 不动，惰性别名承接 |
| `benchmarks/bench_trainer_throughput.py:26` | `from molzoo.mace import MACE` | 不动，包再导出承接 |
| `tests/test_molpot/test_composition/test_composition.py:13`、`tests/test_molzoo/test_symmetry.py:23` | `from molzoo import MACE, Allegro` | 不动 |
| `tests/test_molzoo/test_mace.py:24`、`test_mace_matpes.py:16`、`test_mace_omol.py:19`、`test_mace_encoder.py:10` | 深导入扁平模块 | 由本 spec 的测试迁移一并改写 |

即：**唯一**会断的深导入面 `molzoo.mace_matpes` / `molzoo.mace_omol` 只被本 spec
自己迁移的测试使用，无需为外部保留 shim。

### 5. 测试迁移（含既有债清理）

目标镜像 `tests/test_molzoo/test_mace/`（`__init__.py` 与 test_spec / test_geometry /
test_encoder / test_potential / test_checkpoint 已由 02/04/05 建立）：

- `test_mace_matpes.py::TestMACEMatpes` / `TestDeadEdgePadding`、
  `test_mace_omol.py` 的模块级用例 → `test_mace/test_variants.py` 或
  `test_mace/test_potential.py`（按被测符号的镜像归属收拢）。
- `test_mace_matpes.py::TestLoadMatpesStateDict` 与 OMOL 加载器用例 →
  `test_mace/test_checkpoint.py`。
- `test_mace_encoder.py`（**无源镜像**，是本次要清的债）整体并入
  `test_mace/test_encoder.py`。
- `tests/test_molzoo/test_mace.py` 已由 02 迁移删除（同名遮蔽修复），不在本步范围。
- 三个私有批构造器（`test_mace_matpes.py:76 _make_batch`、
  `test_mace_omol.py:22 _full_edges` / `:34 _make_batch`）折到
  `tests/conftest.py::make_graph_batch`（02 已加 `shifts=`）+
  `tests/test_molzoo/conftest.py` 的 MACE fixture（`fp64` / 小模型 / `single_graph`）。
  迁移后 `tests/test_molzoo/` 内不得再有私有批构造器。
- 删除三个扁平测试文件（test_mace_encoder / test_mace_matpes / test_mace_omol；test_mace.py 已由 02 处理）。

### 6. 门与文档

`scripts/check_test_mirror.py` 的 `PINET_SPINE`（`:42-51`，命名是历史遗留，实为
"必须镜像"的白名单）加入 `"molzoo/mace/"`（01 已加 `"molrep/interaction/mace/"`）。
`src/molzoo/README.md` 的 MACE 段落改成与 PiNet 同构的模块表 + 一句 MACESpec 语
义说明（02 定义），并修掉现有 `MACE(MACESpec(...))` 与实际签名不符的示例；
`docs/api/molzoo.md:5` 的 `::: molzoo.mace` 在包化后只会渲染再导出层，需展开为逐
模块条目，否则 API 页面静默变空。

### Reuse decision

- `reuse` `molzoo/pinet/__init__.py` 的包 docstring + 显式 `__all__` 形态 —— 新
  `molzoo/mace/__init__.py` 逐点同构，不另发明再导出风格。
- `reuse` `molzoo/__init__.py:40-46` 现有 PEP 562 `__getattr__` 机制 —— 只扩表、
  加 `__dir__`、写明策略，不换实现方式。
- `reuse` `tests/conftest.py::make_graph_batch`（含 02 的 `shifts=` 泛化）——
  三个私有批构造器全部收敛到它，不新增测试辅助模块（`tests/` 禁止
  `helpers.py`，共享代码只走最近的 `conftest.py`）。
- `reuse` `scripts/check_test_mirror.py` 的 `PINET_SPINE` 白名单机制 —— 加一行前
  缀，不新建门脚本。
- `reuse` 05 的 `CheckpointRemap`（承接 `_KEY_REMAP` / 严格校验语义）——
  `load_*_state_dict` 仅转发，不复制重映射表。
- `new` `src/molzoo/mace/variants.py` —— 没有任何既有模块承载"具名基础模型别名"
  这一职责：`spec.py` 是配置、`potential.py` 是通用能量/力、`checkpoint.py` 是权
  重映射，把三者拼成 `MACEMatpes` / `MACEOMol` 需要一个自己的归宿；放
  `__init__.py` 会让再导出层带逻辑且落在测试镜像门之外。

## Files to create or modify

- `src/molzoo/mace/__init__.py`（02–05 已建，本 spec 重写为最终再导出面）
- `src/molzoo/mace/variants.py` (new)
- `src/molzoo/__init__.py`
- `src/molzoo/mace_matpes.py`（删除）
- `src/molzoo/mace_omol.py`（删除）
- `tests/test_molzoo/test_imports.py` (new)
- `tests/test_molzoo/test_mace/test_variants.py` (new)
- `tests/test_molzoo/test_mace/test_potential.py`（02/04 已建，本 spec 并入变体用例）
- `tests/test_molzoo/test_mace/test_checkpoint.py`（05 已建，本 spec 并入加载器用例）
- `tests/test_molzoo/test_mace/test_encoder.py`（02 已建，本 spec 并入编码器用例）
- `tests/test_molzoo/conftest.py`（02 已建，本 spec 补 MACE 变体 fixture）
- `tests/test_molrep/test_readout/`（并入 ProductHead 用例，去重）
- `tests/test_molzoo/test_mace_encoder.py`（删除）
- `tests/test_molzoo/test_mace_matpes.py`（删除）
- `tests/test_molzoo/test_mace_omol.py`（删除）
- `scripts/check_test_mirror.py`
- `src/molzoo/README.md`
- `docs/api/molzoo.md`
- `regressions/mace-subpackage-restructure-06-wire.py` (new)

## Tasks

- [ ] Capture pre-cutover goldens and write failing cutover-contract tests（在扁平模块**仍存在**时，用 `python -c` 记录小配置 `MACEMatpes` / `MACEOMol` 的参数张量数与元素总数；新增 `tests/test_molzoo/test_imports.py::TestPublicImportSurface`（顶层 6 个名 + `from molzoo.mace import MACE`；惰性策略用 `subprocess` 在干净解释器里断言 `cuequivariance_torch` 的出现时机）与 `tests/test_molzoo/test_mace/test_variants.py::TestMACEMatpes` / `TestMACEOMol`（kwargs 构造、`_compute_energy` 位置签名、`z_table`））
- [ ] Implement `MACEMatpes` / `MACEOMol` / `load_matpes_state_dict` / `load_omol_state_dict` in `src/molzoo/mace/variants.py`（薄别名，转发到 02/04/05 的 spec 预设、potential 与 `CheckpointRemap`；docstring 标注 `_compute_energy` 私有消费面属 07 的债）
- [ ] Rewrite `src/molzoo/mace/__init__.py` as the flat re-export surface（按 `molzoo/pinet/__init__.py` 形态：模块→单一职责 docstring + 稳定导入示例 + 显式 `__all__`；无逻辑）
- [ ] Rewrite `src/molzoo/__init__.py` to one unified PEP 562 lazy policy（模块级零重导入；`_LAZY` 覆盖全部模型符号；`TYPE_CHECKING` 块；补 `__dir__`；docstring 写明策略与理由）
- [ ] Delete `src/molzoo/{mace_matpes,mace_omol}.py` and verify no dangling importer（`rg -n "molzoo\.mace_(matpes|omol)" src tests scripts benchmarks docs/api` 除本 spec 迁移中的测试外零命中）
- [ ] Migrate `tests/test_molzoo/test_mace*.py` into the `tests/test_molzoo/test_mace/` mirror（变体用例→`test_variants.py`/`test_potential.py`、加载器→`test_checkpoint.py`、`test_mace_encoder.py`→`test_encoder.py`；`TestProductHead*` 去重并入 `tests/test_molrep/test_readout/` 镜像；三个私有批构造器折到 `tests/conftest.py::make_graph_batch` + `tests/test_molzoo/conftest.py` fixture；删除四个扁平测试文件）
- [ ] Extend `PINET_SPINE` in `scripts/check_test_mirror.py` with `"molzoo/mace/"` and make the gate exit 0
- [ ] Update `src/molzoo/README.md` MACE section and `docs/api/molzoo.md`（PiNet 同构模块表 + MACESpec 语义说明 + 修正过期用法示例；API 页展开逐模块 `:::` 条目）
- [ ] Add regression example `regressions/mace-subpackage-restructure-06-wire.py`（仅公开 API：导入面 + 惰性策略 + 参数清单基线；goldens 硬编码，注释记录采集命令 / commit / 日期；不调用任何第三方）
- [ ] Run full check + test suite

## Testing strategy

单元测试全部落在 `tests/` 镜像布局下，每个用例只针对一个函数/方法；端到端场景走
`regressions/`。

- `tests/test_molzoo/test_imports.py::TestPublicImportSurface`
  - happy：`from molzoo import MACE, MACESpec, MACEMatpes, MACEOMol,
    load_matpes_state_dict, load_omol_state_dict` 全部成功；
    `from molzoo.mace import MACE` 成功（`bench_trainer_throughput.py` 路径）。
  - 惰性策略（针对 `molzoo/__init__.__getattr__`）：在 `subprocess` 干净解释器里
    断言 `import molzoo` 后 `cuequivariance_torch not in sys.modules`，访问
    `molzoo.MACE` 后出现。必须用子进程——同进程内 pytest 早已导入 cuEq，同进程断
    言会假绿。
  - edge：`molzoo.NotAThing` 抛 `AttributeError`；`set(molzoo.__all__) == set(dir(molzoo))`。
- `tests/test_molzoo/test_mace/test_variants.py::TestMACEMatpes` / `::TestMACEOMol`
  - happy：以 `scripts/matpes_port/run_nve.py:148-164` 的 kwargs 形（缩小到
    `l_max=1, num_features=16, use_fallback=True`）构造成功。
  - 契约：`energy_forces` / `forward` / `validate_elements` / `z_table` 存在；
    `_compute_energy(pos, Z, edge_index, batch, num_graphs, shifts)` 可**位置**调用
    （守住 07 之前的 NVE 脚本）。
  - 等价：kwargs 形与 spec 形构造的两个实例在同一 fixed batch 上能量差 `== 0`
    （同一路径，非近似）。
  - edge：表外原子号触发 `validate_elements` 的 `ValueError`。
- `tests/test_molzoo/test_mace/test_potential.py` / `test_checkpoint.py` / `test_encoder.py`
  —— 承接迁移用例，**判据与容差逐字保留**：`forward(td)` vs `energy_forces`
  `1e-9 eV` / `1e-8 eV·Å`；死边（padding）能量不变；加载器对"无处可去的键 / 形状
  不符 / 参数未被覆盖"三种失败各有一条 `pytest.raises(RuntimeError)`。
- `tests/test_molrep/test_readout/` —— 并入后按用例去重，禁止出现
  重名 `TestProductHead`；`molrep` 测试不得 import `molzoo`。
- 迁移完整性由 `python scripts/check_test_mirror.py --strict-pinet`（含新前缀
  `molzoo/mace/`）与 `pytest --collect-only tests/`（0 error）双向守。
- **回归示例** `regressions/mace-subpackage-restructure-06-wire.py`：最小公开 API
  脚本，(1) 断言全部导入面；(2) 在干净进程语义下断言惰性策略；(3) 构造小配置
  `MACEMatpes` 与 `MACEOMol`，断言 `len(list(m.parameters()))` 与
  `sum(p.numel() for p in m.parameters())` 等于脚本内**硬编码**的切换前基线（注释
  写明采集命令、commit、日期，来源为仓内扁平模块，非第三方）。参数清单不变是
  "检查点仍能严格加载 ⇒ 官方权重数值结论仍成立"的可离线校验代理；能量数值不作
  golden（构造顺序变化会改随机初始化，天然脆）。

## Out of scope

- `scripts/matpes_port/run_nve.py` 的 `_compute_energy` 私有绑定改为公开能量核心
  ——归 07；本 spec 只保签名不改 consumer。
- `src/molzoo/specs/{mace,mace_matpes,mace_omol}.md` 的模块锚点与 §2/§3.1 行号刷新
  ——归 07（molzoo-spec 更新模式）。
- `docs/` 中除 `docs/api/molzoo.md` 之外的叙述性 MACE 引用、`docs/molzoo/index.md`
  与镜像页；`.claude/notes/architecture.md:65-68` 的模块清单由 `/mol:note` 同步。
- 任何数学、超参数、默认值、`use_fallback` 策略变更；变体预设的定义位置（属
  02/04/05）。
- `molzoo/allegro.py` 的包化（同类债，但不在本链条内；本 spec 只把它纳入统一惰性
  策略，不动其文件布局）。
- 对外部 oracle（mace-torch / e3nn）的任何重跑：离线、仓内禁止导入。
