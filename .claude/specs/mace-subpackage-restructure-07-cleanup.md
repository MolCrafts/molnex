---
title: MACE subpackage restructure — consumer re-point, EnergyForceModel removal, graphs schema fix
status: approved
created: 2026-08-08
revised: 2026-08-08
grilled: true
chain: mace-subpackage-restructure
---

# MACE 子包重构收尾：消费者重指向、EnergyForceModel 下线、graphs schema 修正

## Summary

链条 01–06 已把 MACE 家族搬进 `molzoo/mace/` 与 `molrep/interaction/mace/` 子包，并在 04 落地了公开的能量核 `energy_core`、在 05 落地了 `MACEPotential.from_checkpoint`。本收尾 spec 关掉三条尾巴：把仍在调用私有核 `model._compute_energy` 的两个树内消费者（`scripts/matpes_port/run_nve.py`、`benchmarks/bench_mace_matpes.py`）重指向到公开入口并让 `run_nve.build_model` 塌缩到 `from_checkpoint`；删除零生产子类、与本链"构造期单态"方向相冲突的 `molpot.composition.EnergyForceModel`；修正 `molpot.derivation.protocol.ensure_graphs` 创建出的 `graphs` 子 TensorDict 的 `batch_size`，使之符合 CLAUDE.md 的 `"graphs": batch_size=[B]` schema。同时刷新三份 molzoo encoder spec 与 `.claude/notes/` 中已过期的锚点。全过程对物理零改动：`run_nve.py` 的 CLI 行为逐字兼容，`bench_mace_matpes.py` 的 PASS/FAIL 语义不变，MACE 的能量与力在容差内逐位复现。

## Domain basis

本 spec 不引入新物理，但它的验收标准整体是一条**数值不变性**约束，必须按物理量书写：

- 单位：位置 Å；能量 eV；力 eV·Å⁻¹；时间 fs（`run_nve.py --dt`）。`molix.md` 内部以 `EV_PER_AMU_A2_FS2` 做能量单位桥接，本 spec 不触碰该常数。
- 被保护的关系式：`F_i = -∂E/∂r_i`，由 `molpot.derivation.force.autograd_forces_from_energy` 在编译区之外求值（cuEquivariance 的 fused 算子注册的是不带 `setup_context` 的 legacy `autograd.Function`，`torch.func.grad` 会拒绝——见 `.claude/notes/notes.md` §"Force derivation — dual explicit backends"）。
- 边约定：`edge_index (E,2)`，`[:,0]=source`、`[:,1]=target`，`edge_diff = pos[target]-pos[source]`（周期体系再加 `shifts`）。重指向后闭包传参顺序必须逐字保持。
- 模型参照：Batatia et al., *MACE*, NeurIPS 2022, https://arxiv.org/abs/2206.07697 ；*A foundation model for atomistic materials chemistry* (MACE-MP-0), https://arxiv.org/abs/2401.00096 ；数据集 Kaplan et al., *MatPES*, https://arxiv.org/abs/2503.04070 。参照实现 `ACEsuit/mace` v0.3.16（`ScaleShiftMACE`），只作为链条既有 §7.1 parity 的口径，本 spec 不重跑。
- 重构不变性容差（回归样例，fp64/CPU）：|ΔE| ≤ 1e-9 eV，max|ΔF| ≤ 1e-9 eV·Å⁻¹。该量级远小于链条既有的官方 parity 残差，因此任何真实行为改变都会被抓住。

## Design

### 1. 消费者重指向（`run_nve.py`）

`_matpes_energy_forces`（`scripts/matpes_port/run_nve.py:95-142`）现在两处取私有核：`core = model._compute_energy`（:118）与闭包默认参 `_f=model._compute_energy`（:124）。两处换成 04 落地的公开 `energy_core`，**其余逐字不变**：

- autocast 区必须继续留在 compiled callable **内部**（:120-126 的注释是实测结论：包在外面会作废 CUDA-graph 捕获，bf16 臂从 12 ms 退化到 52 ms/step）。
- `Compiler(cuda_graphs=True)(core)` 的包裹时机、`energy_fn(leaf, Z, neighbors.edge_index, batch, 1, neighbors.shifts)` 的**位置实参顺序**、`autograd.grad` 留在编译区外、`energy.sum().detach()` / `forces.detach()` 的返回形状全部保持。
- 类型标注 `model: MACEMatpes` 随 01–06 落地的公开类名更新。

`build_model`（:145-169）整体删除，调用点（:280）改为
`MACEPotential.from_checkpoint(args.weights_dir / "matpes_r2scan_config.json",
args.weights_dir / "matpes_r2scan_cueq_state.pt", use_fallback=use_fallback)`
（05 的签名：两个显式路径 + `**ctor_kwargs` 透传构造开关；`--weights-dir` 的目录约定保持不变）。两个后置条件必须显式验证：`float(model.cutoff_fn.r_cut)`（:281 附近）仍可解析；`import json`（:18）在塌缩后变成未使用导入，必须一并删除，否则 `ruff check` 报 F401。

### 2. 消费者重指向（`bench_mace_matpes.py`）

只改 `benchmarks/bench_mace_matpes.py:129` 的 `Compiler(cuda_graphs=True)(model._compute_energy)` → 公开 `energy_core`。`_build_model`（:37-54）用随机权重直接构造（perf-only，无 checkpoint），不走 `from_checkpoint`；`:115` 的 `model.energy_forces(...)` 是公开入口，本 spec 不动。`--min-speedup` 守卫逻辑、`RESULT: PASS/FAIL` 打印与 `0/1` 退出码契约保持。

### 3. `EnergyForceModel` 下线

`src/molpot/composition/energy_force.py`（112 行）+ `tests/test_molpot/test_composition/test_energy_force.py`（143 行，全部针对文件内的 `_ToyEF` 玩具子类）删除，`src/molpot/composition/__init__.py:15`（import）与 `:34`（`__all__`）两行移除。`src/molpot/__init__.py:8-18` 从未再导出它，删除不影响 `molpot` 顶层公开面。

删除依据（全部 grep 复核）：

1. **零生产子类**。`rg EnergyForceModel` 在 `src/` 下只命中定义文件与 `composition/__init__.py`；唯一的子类是测试内的 `_ToyEF`。
2. **docstring 说谎**。`energy_force.py:6` 称 "`molzoo.pinet.PiNetPotential` is the primary consumer"，实际 `PiNetPotential` 直接继承 `nn.Module` 并使用 `molpot.derivation.protocol` 的 helpers。`.claude/notes/notes.md:103-104` 也仍写着 "PiNet via `EnergyForceModel`" —— 同一条陈旧断言的第二个落点，一并修正（Iron law：在触及面上发现的陈旧不变量不得留存）。
3. **与本链方向冲突**。`forward(batch, *, compute_forces: bool | None = None)`（:56-64）是**每次调用**的多态分支；本链走的是构造期定死一条 bound 路径。
4. **返回类型错**。`energy_forward` / `forward` 返回 `dict[str, Tensor]`，而仓库契约是 in-place 写回 nested `TensorDict`。
5. **编译缝错位**。`compile_energy`（:107-111）编译的是 bound method，绕过了 `molix.compile.Compiler` 这条统一的 compile 缝。

`ForceDerivation` 本身**不动**：它是双后端力求导的唯一入口，删除的只是这层薄包装。

### 4. `ensure_graphs` schema 修正

现状 `src/molpot/derivation/protocol.py:71-75` 建出的是 `TensorDict(batch_size=[])`，违反 CLAUDE.md 的 `"graphs": TensorDict(batch_size=[B])`；正典 collate `src/molix/data/collate.py:219` 写的是 `TensorDict(graphs_dict, batch_size=[num_graphs])`。这是一处对任何新适配者都成立的潜在 schema 回归——`src/molzoo/pinet/potential.py:110` 已经在读 `batch["graphs"].batch_size[0]`，若 `graphs` 由今天的 `ensure_graphs` 创建，这一行会直接 `IndexError`（今天没炸只是因为生产路径上 `graphs` 总是由 collate 先建好）。

**取舍（本 spec 选 (a)+(b) 混合）**：

- 签名改为 `ensure_graphs(batch: TensorDict, num_graphs: int | None = None) -> TensorDict`。`num_graphs is None` 时保持今天的 `batch_size=[]`，**向后兼容**，但 Google 风格 docstring 明写该形状不符合 schema、仅为树外适配者的过渡兼容。
- 树内唯一调用方 `write_energy`（:85）迁移为传入 `num_graphs=energy.shape[0] if energy.dim() == 1 else None`，于是**仓库内每一次写入都产出合规的 `[B]`**。

为什么不直接改成必填：`ensure_graphs` 在 `protocol.__all__` 中属公开面（但 `molpot/derivation/__init__.py` 并未再导出，爆炸半径小）；全仓 grep 显示树内调用方只有 `write_energy` 一处，"全部迁移"的成本恰好是一行，而"改必填"会在毫无树内收益的前提下打破树外适配者。从 `energy.shape[0]` 推 B 不产生 host sync（`energy` 契约就是 `(B,)`，读 shape 只生成一次静态 guard），与 `energy_core(num_graphs: int)` 既有的静态 B 约定一致；0 维 energy 退回 `None` 以免 `shape[0]` 抛错。

### 5. 文档锚点刷新

三份 encoder spec 的**十节结构逐字不动**（改结构 = §10.2 breaking change），只重指向锚点并各加一行 §9 pinning：

- `src/molzoo/specs/mace.md`：头表 `| Module |` / `| Entry point |`；§8 边界表；Appendix A 的 `src/molzoo/mace.py`。
- `src/molzoo/specs/mace_matpes.md`：头表；§3.1 "Code anchor" 列；§5 crosswalk 整表（`molrep.interaction.density.*`、`molrep.interaction.product_basis.*`、`molrep.readout.scalar.*`、`molrep.embedding.*`）；§6 A5/A6 引用的 `SKIP_TP_METHOD`；§7.2 测试路径；§8。
- `src/molzoo/specs/mace_omol.md`：头表；§6 A4/A5/A6 的验证路径；§7.2；§9。
- 顺带修一处**已发现的漂移**：`docs/molzoo/specs/mace_omol.md` 与 `src/molzoo/specs/mace_omol.md` 是两份副本且已不同步。既然本 spec 正在编辑 src 副本，就把 docs 副本重新同步为其镜像，否则漂移只会更大。

`.claude/notes/architecture.md` 需改的行：模块清单（`src/molzoo/mace.py`、`src/molzoo/mace_omol.py`）、specs 清单、molzoo 导出名、known-gap 中 `EnergyForceModel` 条目标记为已关闭（`molpot.composition.pooling` 双家与 O(N²) 邻居核两条保留）。`.claude/notes/notes.md` 删除 "PiNet via `EnergyForceModel`" 的错误陈述，改为 "PiNet via `molpot.derivation.protocol` helpers"。若这些行超出手改的可靠范围，落地时可改跑 `/mol:map` 重生成，但本 spec 的验收只看结果一致。

### Reuse decision

- `reuse` `energy_core`（04 落地）—— 本 spec 的重指向目标，不另造入口。
- `reuse` `MACEPotential.from_checkpoint`（05 落地）—— `run_nve.build_model` 塌缩到它，不保留任何 `build_*` 工厂函数（CLAUDE.md §Forbid）。
- `reuse` `molix.compile.Compiler` —— 编译缝唯一入口，`EnergyForceModel.compile_energy` 正因绕过它而被删。
- `reuse` `molpot.derivation.force.autograd_forces_from_energy` —— 力求导不新增第三条路径。
- `reuse` `molpot.derivation.ForceDerivation` —— 保留不动；删掉的是包装层，不是求导层。
- `generalize` `molpot.derivation.protocol.ensure_graphs` —— 加 `num_graphs` 参数令同一符号同时服务两个调用场景，**不新增** `ensure_graphs_with_size` 之类的平行 helper。
- `reuse` `molpot.derivation.protocol.write_energy` —— 只改其内部传参，公开签名不变。
- `new — regressions/mace-subpackage-restructure-07-cleanup.py`：链上各段已建 `regressions/`；本段新增自己的回归脚本。

命名与错误处理沿用最近的既有 pattern：`ensure_graphs` 的 keyword 参数命名跟随 `energy_core(..., num_graphs: int, ...)`；回归脚本的 `RESULT: PASS/FAIL` + `sys.exit(0/1)` 形状跟随 `benchmarks/bench_mace_matpes.py:140-154`。

## Files to create or modify

- `scripts/matpes_port/run_nve.py`
- `benchmarks/bench_mace_matpes.py`
- `src/molpot/derivation/protocol.py`
- `src/molpot/composition/energy_force.py`（删除）
- `src/molpot/composition/__init__.py`
- `tests/test_molpot/test_composition/test_energy_force.py`（删除）
- `tests/test_molpot/test_derivation/test_protocol.py` (new)
- `regressions/mace-subpackage-restructure-07-cleanup.py` (new)
- `src/molzoo/specs/mace.md`
- `src/molzoo/specs/mace_matpes.md`
- `src/molzoo/specs/mace_omol.md`
- `docs/molzoo/specs/mace_omol.md`
- `.claude/notes/architecture.md`
- `.claude/notes/notes.md`

## Tasks

- [ ] Write failing unit tests for `ensure_graphs` / `write_energy` graph shape (tests/test_molpot/test_derivation/test_protocol.py → TestEnsureGraphs, TestWriteEnergyGraphShape)
- [ ] Generalize `ensure_graphs` in src/molpot/derivation/protocol.py to take `num_graphs: int | None = None` and pass it from `write_energy` (Google-style docstring flagging the `[]` fallback as schema-nonconforming)
- [ ] Re-point scripts/matpes_port/run_nve.py `_matpes_energy_forces` onto the public `energy_core` and collapse `build_model` onto `MACEPotential.from_checkpoint` (drop the now-unused `json` import; record the pre/post single-point E and |F|max)
- [ ] Re-point benchmarks/bench_mace_matpes.py:129 compiled-core arm onto the public `energy_core`
- [ ] Remove `EnergyForceModel`: delete src/molpot/composition/energy_force.py and tests/test_molpot/test_composition/test_energy_force.py, drop lines 15 and 34 of src/molpot/composition/__init__.py
- [ ] Refresh anchors and add the restructure §9 pinning row in src/molzoo/specs/{mace,mace_matpes,mace_omol}.md, then re-sync docs/molzoo/specs/mace_omol.md
- [ ] Update .claude/notes/architecture.md layout lines and the EnergyForceModel known-gap, and correct the `EnergyForceModel` claim in .claude/notes/notes.md
- [ ] Add regression example regressions/mace-subpackage-restructure-07-cleanup.py (public API only; hard-coded goldens, no third-party runtime)
- [ ] Run full check + test suite (`ruff check src/ && ruff format --check src/`, `python -m pytest tests/ -v`, `python scripts/check_test_mirror.py --strict-pinet`)

## Testing strategy

**单元测试（新增，镜像 `src/molpot/derivation/protocol.py` → `tests/test_molpot/test_derivation/test_protocol.py`）**，每个用例只打一个函数：

- `TestEnsureGraphs`（目标：`ensure_graphs`）
  - happy：`ensure_graphs(TensorDict(batch_size=[]), num_graphs=3)["graphs"].batch_size == torch.Size([3])`（硬编码期望值）。
  - 向后兼容：不传 `num_graphs` 时为 `torch.Size([])`。
  - 幂等：已存在的 `graphs`（`batch_size=[2]`，含 `num_atoms`）原样返回，不被覆盖、`batch_size` 不被改写。
  - 边界：`num_graphs=0` → `torch.Size([0])`；`num_graphs=1` → `torch.Size([1])`。
- `TestWriteEnergyGraphShape`（目标：`write_energy`）
  - `energy` 形状 `(4,)` → `batch["graphs"].batch_size == torch.Size([4])` 且 `batch["graphs","energy"]` 与传入张量逐元素相等。
  - 0 维 `energy` → 仍为 `torch.Size([])`（不抛 `IndexError`）。
  - 传 `atomic_energy` 时 `("atoms","energy")` 照写（回归 `ATOMIC_ENERGY_KEY` 路径）。

**既有套件（必须不改一行地通过）**：`tests/test_molpot/test_derivation/test_readouts.py`（直接调 `write_energy`）、`tests/test_molzoo/test_mace/`（06 迁移后的全部）、`tests/test_molpot/test_composition/`（余下文件）。

**删除面核验**：`rg -n "EnergyForceModel"` 在 `src/ tests/ docs/ scripts/ benchmarks/ .claude/` 归零；`rg -n "_compute_energy"` 在 `scripts/ benchmarks/` 归零；`python -c "import molpot"` 成功。镜像门 `python scripts/check_test_mirror.py --strict-pinet` 退出 0（删掉一个测试文件后必须复核）。

**CLI 逐字兼容**：`run_nve.py --help` 的 flag 集合仍为 21 个（`--structure --system --dump-system --weights-dir --out --steps --dt --temperature --stride --precision --checkpoint-every --resume --flush-every --seed --device --fallback --threads --rebuild-every --capacity-factor --autocast-bf16 --compile`）且默认值不变；`--dump-system` 分支不受影响；`.graph.pt` 的键恰为 `{Z, cell, edge_index, shifts, num_edges, rebuild_count, energy_scale}`。

**回归样例（仓库根 `regressions/`）**：`regressions/mace-subpackage-restructure-07-cleanup.py`

- 环境：`molix.config.set_precision("fp64")` 在任何构造之前调用；CPU；`use_fallback=True`（无 `cuequivariance-ops-torch` wheel 时唯一可行路径）。
- 体系：字面量写死的 6 原子非周期构型（Å），`shifts=None`，`batch=zeros(6)`，`num_graphs=1`。
- 模型：`torch.manual_seed(0)` 后以公开构造器建一个小 MACE（Z 表 `[1, 8]`、`r_max=4.0`、`num_bessel=8`、`l_max=2`、`num_features=16`、`num_interactions=2`、`correlation=3`、`mlp_dim=8`），随机权重——本样例守的是**重构不变性**，不是物理精度。
- 路径：通过公开 `energy_core` + `molpot.derivation.force.autograd_forces_from_energy` 复刻 `run_nve._matpes_energy_forces` 的闭包形状（leaf `requires_grad_`、`torch.enable_grad()`、grad 在编译区外），断言硬编码 golden：总能量 E（eV）、`max|F|`（eV·Å⁻¹）、`F[0]` 三个分量；容差 1e-9 eV / 1e-9 eV·Å⁻¹。
- Golden 出处：**在 task 3 重指向之前**、于 chain tip 06 的私有核路径上跑同一脚本取得，作为字面量嵌入并在文件头注释记录 `工具=molnex 自身（无第三方 oracle）/ commit / 命令 / 日期`。脚本运行期不得 import 或 subprocess 任何 `mace-torch` / ASE / e3nn。
- 第二段断言（本 spec 另一处公开面变更）：`ensure_graphs(TensorDict(batch_size=[]), num_graphs=3)["graphs"].batch_size == torch.Size([3])`。
- 输出：末行 `RESULT: PASS` / `RESULT: FAIL`，退出码 0/1。

**GPU 侧（人工，非门禁）**：在 GH200 上跑一次 `run_nve.py --compile`（fp64，`wat64_h3o+`，200 步），确认单点 E 与 `E_tot drift` 与 §7.4 既有 run log 口径一致；结果作为 `src/molzoo/specs/mace_matpes.md` §7.4 的一条新 run log。

## Out of scope

- `src/molix/engine/adapter.py:133` 与 `:154` 的 `isinstance(out, dict)` 断言与本链把模型统一为 in-place nested `TensorDict` 的方向直接冲突。**显式推迟到后续 engine spec**；本 spec 不动 `molix/engine/`，指针留在此处以免遗失。
- `MessageAggregation` 的移除（另案）。
- `molpot.composition.pooling` 与 `molpot.pooling` 的双家问题（known-gap 保留原文其余两项）。
- `docs/molzoo/specs/allegro.md` 与 `src/molzoo/specs/allegro.md` 的同类双副本漂移——不在本 spec 的触及面（只同步 mace_omol 的那一对）。已在评审时点名，留给下一次 docs 同步。
- MACE 权重、数值、超参与 `energy_forces` 公共签名的任何改动；性能调优；`tests/regression/` 的任何改动。
- 考虑过但否决的替代方案：(1) 给 `EnergyForceModel` 补文档并留着——否决，零生产子类 + 与单态方向冲突，留着就是 silent debt；(2) 把 `ensure_graphs` 的 `num_graphs` 改为必填——否决，破坏树外适配者且树内零收益；(3) 让 `run_nve.py` 保留 `build_model` 只在内部调 `from_checkpoint`——否决，那正是 CLAUDE.md §Forbid 的工厂函数包装。
