---
title: Polar MD 编译缝落地 — mlip 侧 compile_core、bind_cell 与 eager-only 元素门
status: code-complete
grilled: true
created: 2026-09-03
slug: polar-compile-clean-03-wire
chain: polar-compile-clean
---

# Polar MD 编译缝落地 — mlip 侧 compile_core、bind_cell 与 eager-only 元素门

## Summary

把前两步的成果接到 MD 步上。编译核**装在 mlip 上**，不在容器上：`MacePotential` 获得两个单一职责的公开方法 —— `bind_cell(cell)`（基类是有记录的 no-op，`MacePolar` 覆写为把 cell 交给自己那两个 GTO 网格）与 `compile_core(compiler)`（唯一动作：`self._core = compiler(self.energy_core)`；未做过 eager 前向就抛错），`_write_energy` 于是保持它**唯一**的那次拆包、`num_graphs` 解析（`potential.py:608-615`）、`graphs` 命名空间创建（`:641-642`）与 `protocol.write_energy`（`:643`，写**未 detach** 的能量），只是把 `self.energy_core(...)` 换成 `self._core(...)`。`molpot.forcefield.potentials.Potentials` 只做两层一行委托：`bind_cell` 经 `isinstance(self.mlip, CellBound)` 守卫，`compile_core` 经 `CoreCompilable` 守卫；`calc_energy` 的 mlip 分支**一律**走 `protocol.call_energy`（编译与否由 mlip 自己决定），顺手把 `potentials.py:215` 那个重复实现 `call_energy` 策略的鸭子类型探针删掉。`MacePotential._write_energy` 的一次性元素校验加 `torch.compiler.is_compiling()` 条件，并把"扁平缝不做元素校验、缝的调用方负责先校验"写进 `energy_core` 与 `validate_elements` 的 docstring。

**编译范围要说准**：`mlip.energy_core` 在给了 cell 的周期、B=1 路径上达到 **1 图 / 0 断点**；而整步 `Potentials.calc_energy_forces_virial` **不是** 0 断点，也不该是 —— `autograd_forces_and_virial`（`force.py:167-168` 建叶子、`:177-182` 取 `autograd.grad`）按 Q3 裁定留在编译区外，它就是那条**已命名的 autograd 缝**，owner = molpot derivation。因此本步的断言是"整步 = 编译核 + 且仅 + `force.py` 那条具名缝"。基线 51 图 / 50 断点（`.claude/notes/notes.md:279-293`）里的五个断点源，本链消掉四个（`kspace.py` 的 `.item()` 与掩码、`features.py:204,214`、`hartree.py:281`、`encoder.py:303`），保留第五个（`force.py:167-168`）并把它从"缺陷"改判为"设计"。能量、力与 eager 在 fp64 下一致到 1e-10（tiny CPU 上 virial 同样 1e-10；GH200 PEO 盒上 virial 1e-9，因为该盒上 inductor vs eager 的 max_dW 稳定落在 5e-11..2e-10、相对 |W|~5200 eV 约 2e-14），`regressions/mace-polar-periodic-gto.py` 的 GH200 官方 goldens 一个数不变。顺带修掉 `src/molzoo/mace/potential.py:79,87` 两处指向已不存在的 `PotentialForceField` 的陈旧 docstring、三份仍在教读者用已删除构造函数的用户文档，以及一条跑在验收前面的 `[code-complete]` 标记。

## Domain basis

- 编译缝只覆盖**一阶**推理，且只覆盖**周期（`cell` 已绑定）、B=1** 的路线。`autograd_forces_and_virial`（`src/molpot/derivation/force.py:136-187`）保持为**编译核的调用方**：位置叶子（`:167`）、应变叶子（`:168`，`torch.zeros(..., requires_grad=True)`）、`pos_leaf @ ft`（`:172`）与 `torch.autograd.grad`（`:177-182`）全在编译区之外，反向由 AOTAutograd 提供。二阶反传（力监督训练）在 AOTAutograd 下不可编译，由在建 spec `cuet-force-doublebackward` 记录 —— 训练侧的力损失继续走 eager，本 spec 不改变这一点。
- 应变约定：\(pos_s=pos(I+\varepsilon_{\rm sym})^{\mathsf T}\)、\(cell_s=cell(I+\varepsilon_{\rm sym})^{\mathsf T}\)、\(\varepsilon=0\) 处求导；virial \(W=\partial E/\partial\varepsilon\)（3×3，能量单位），stress = \(W/V\)。k 空间部分的 \(\partial E/\partial\varepsilon\) 在固定盒写法下与掩码写法一致（01 的 D7：半球/非零指示只依赖整数 n，\(\partial w/\partial\varepsilon=0\)）。**Q3 已裁定**：缝保持在 `energy_core`，不把应变包装层一起编译（那会造出应变约定的第二个家）；若日后实测显示 eager 的应变 gather 有分量，就用"应变过的 `shifts`"加宽现有缝，另开 spec。
- **失效规则（invalidation）**：整数半盒对应变是协变的（k 随 \((I+\varepsilon)^{-\mathsf T}\) 走，整数 n 不变），所以一次 barostat 缩放**既不需要重建盒、也不允许重建盒**（01 的决定：`bind` 只在第一次定尺）—— 只需要重新**检查容量**。检查发生在两处：`bind_work` 与每次 barostat 更新（02 的 `rebind_cell` → `Potentials.bind_cell` → `MacePolar.bind_cell` → `KSpaceGrid.bind`，eager 抛 `RuntimeError` 同时点名 `KSpaceGrid` 与 `capacity_factor`），以及编译区内 01 的 in-graph NaN 哨兵。`max_axis_index` 的 fail-closed 只在第一次 bind 时可能触发。
- **"网格已绑定"这件事没有布尔属性**：它由本 spec 强制的调用序列（`bind_cell` → 一次 eager 前向 → `compile_core`）与 01 那条具名的 trace 期 `RuntimeError("KSpaceGrid is not bound — call bind_cell(...)")` 共同保证。不引入 `is_cell_bound`：对 MatPES / OMol 这些没有倒空间网格的变体它没有意义，会变成一个每个 mlip 都要撒谎回答的问题。
- Polar 常数：σ=1.5 Å、multipole \(l_{\max}=1\)、κ=1.0（已核对等于官方 checkpoint 的 `kspace_cutoff` buffer 1.8467166200173746 Å⁻¹），`eV_A = 14.399645478425667` eV·Å/e²。能量 eV、力 eV/Å、位置 Å。参考：Batatia et al., arXiv:2602.19411, §II.2.6b 与 SI V.6；MACE 骨干 Batatia et al., NeurIPS 2022, arXiv:2206.07697。
- 官方 fp64 goldens（PEO 4995 原子 / 46.36 Å 立方胞，GH200 捕获 2026-09-02，已硬编码在 `regressions/mace-polar-periodic-gto.py:40-46`）：`E0+inter = -2829206.9445318216`、`electron = 685.2709477417063`、`electrostatic = 9.143779770475982`、`total = -2828512.52980431` eV，`n_edges = 285546`，`atol = 1e-3` eV。**这些数字不得变化。**

## Design

**编译核装在 mlip 上（`src/molzoo/mace/potential.py`）—— 这是本步最重要的一条。**

- `def compile_core(self, compiler: Callable[[Callable[..., Any]], Callable[..., Any]]) -> None:`
  - 前置条件**一条**：`self._elements_validated` 为 True（即已经发生过至少一次 eager `_write_energy`）。否则抛 `RuntimeError`，消息点名 `compile_core` 与"先跑一次 eager 前向"。不再有第二个计数器（见下）。
  - 唯一动作：`self._core = compiler(self.energy_core)`。`self._core` 在 `__init__` 里初始化为 `None`（与 `_elements_validated` 同一段，`potential.py:240` 附近），它是普通属性、不进 `state_dict`。
  - **重复调用不重编**：`self._core is not None` 时直接返回。（不能拿 `self._core is self.energy_core` 判断 —— 绑定方法每次访问都是新对象。）docstring 写明：安装是**幂等且共享的** —— 这个 mlip 对象被从同一 `ForceField` 建出的每一个 `Potentials` 共用，第二次用**不同** `compiler` 调用会被静默忽略；要回到 eager 核（例如之后拿同一实例做力监督训练，那条路本就不可编译），**构造一个新的势**，本类不提供 `uninstall`。
  - `MacePolar` 继承它即可：`energy_core` 是被覆写的那一个，`compiler(self.energy_core)` 自然编译 Polar 的核。`MacePotential` 因此**结构上**满足 02 的 `CoreCompilable`。
  - **不 import `molix.compile.Compiler`**：参数按结构类型标注，`Compiler.__call__`（02 的 PEP 695 泛型）天然满足。
- `_write_energy`（`potential.py:559-644`）只改一行半：调用点从 `self.energy_core(...)` 换成 `core = self._core if self._core is not None else self.energy_core` 再 `core(positions, Z, edge_index, atom_batch, num_graphs, edge_diff=…, total_charge=…, total_spin=…, cell=…)`。**它保留的东西**（这正是把核装在 mlip 上的理由）：唯一的一次拆包（`:596-599`、`:617-638`）、`num_graphs` 解析（`:608-615`）、`graphs` 命名空间创建（`:641-642`）、以及 `protocol.write_energy(batch, energy)`（`:643`）—— 写回的是**未 detach、仍挂在 autograd 图上**的能量（模块 docstring `:67-80` 已把"不 detach"写成契约）。`molpot` 侧因此**一行 batch→flat 适配都不需要写**，也不需要读 molzoo 的任何私有属性。
- `graphs` 缺失或 `batch_size=[]` 的 batch：**不抛错**，走 `:612-615` 那条有记录的、**唯一**的 host sync（`int(atom_batch.max().item()) + 1`）。这条路现在永远在 eager 里跑（`_write_energy` 本身从不被 trace，只有 `energy_core` 被编），所以它的代价是一次同步而不是一个断点；`graphs` 随后由 `:641-642` 补建。真正会抛的是 `edges.edge_diff` 缺失 —— `:617-621` 的 `ValueError("MacePotential.forward requires edges.edge_diff; edges.shifts is no longer a PBC channel")`。两条行为都写进 `compile_core` 的 docstring，因为它们决定了调用方该往 batch 里放什么。

**`Potentials` 侧只有两层一行委托（`src/molpot/forcefield/potentials.py`）。**

- `def bind_cell(self, cell: torch.Tensor) -> None:` —— `if isinstance(self.mlip, CellBound): self.mlip.bind_cell(cell)`。`CellBound` 从 **`molix.schema`** import（molpot 已经在 `derivation/protocol.py:36` import 该模块，单向包规则保住；`molix.md` 仍然不可 import）。**必须是 isinstance 而不是无条件调用**：`def_pairstyle` 接受任意 `nn.Module`（`container.py:83-95`，`:92` 把对象原样存进 `_mlip`），树内就有一个没有 `bind_cell` 的玩具（`tests/test_molpot/test_forcefield/test_potentials.py:238-252` 的 `_EdgeSpring`），无条件调用会把每一个非 MACE 的 mlip 打挂。没有 mlip 时同样 no-op（经典 Class-I 路径没有随 cell 定尺的状态；邻居表自己管自己的重绑定）。`Potentials` 由此**结构上**满足 `CellBound`。
- `def compile_core(self, compiler) -> None:` —— `if not isinstance(self.mlip, CoreCompilable): raise RuntimeError(...)`（消息点名 `compile_core` 与"这个 mlip 没有编译核"），否则 `self.mlip.compile_core(compiler)`。**`Potentials` 上不存在 `_compiled_core`**，也没有第二条求值路径。
- `calc_energy` 的 mlip 分支（`potentials.py:212-220`）收敛成三行：`pos` 覆盖照旧先落到 batch 上（`:213-214`，编译与否都一样，先写后读），然后 `out = protocol.call_energy(self.mlip, system)`（`protocol.py:83-93`，它已经拥有"优先 `_write_energy`，否则 `model(batch)`"这条策略），`return out[ENERGY_KEY]`。今天 `:215-220` 在自己重复一遍这条策略（`getattr(self.mlip, "_write_energy", None)` 的鸭子类型探针），正是 architect 点名的、本步正在改的那几行里的既有缺陷 —— 按 iron law 一并修掉，不留。**编译与否对这条分支不可见**：mlip 自己决定用不用 `_core`。
- 力仍由既有内核给：`calc_forces`（`:255-257`）与 `calc_energy_forces_virial` 的 `energy_fn` 闭包（`:307-321`）都经 `calc_energy`，因此自动走上编译核；`autograd_forces_and_virial`（`force.py:136`）一行不改。`calc_energy_forces_virial` 末尾仍用 detach 过的副本覆写 `graphs.energy`（`:332-334`），这与 `_write_energy` 写 attached 能量并不矛盾：前者是 MD 侧的存档值，后者是同一步内力路径要用的活张量。

**编译核在 `to_potentials()` 重建后依然生效（architect N6）。** `ForceField.to_potentials()`（`container.py:180-183`）每次调用都 `return Potentials(self, **kwargs)`，而 `MD._wire()`（`driver.py:300-318`，`:305`）在 `set_forcefield` / `set_neighbors` / `set_integrator` / `set_hooks` / `set_barostat` **每一个 setter** 里都被调一次（`:233,253,261,267,283`）。但 `Potentials.__init__` 的 `self.mlip = getattr(forcefield, "_mlip", None)`（`potentials.py:89`）取的是 `def_pairstyle` 存进 `container.py:92` 的**同一个 `nn.Module` 对象**，所以装在 mlip 上的 `_core` **跨所有重建存活**。这是"把核装在 mlip 上"相对于"装在容器上"的决定性优势：后者会在下一次 `set_*` 时被静默丢弃。**规定的调用序列**（写进两个 `compile_core` 的 docstring 并在 regression 里演示）：

```
ff.def_pairstyle(model)
pots = ff.to_potentials()
pots.bind_cell(cell)          # 定尺 + 容量检查
pots.calc_energy(work)        # 一次 eager 前向：元素校验 + 网格已绑定
pots.compile_core(Compiler(dynamic=True, backend=...))
MD(dtype).set_forcefield(ff).set_neighbors(nl).set_integrator(...).run(system, n)
```

这条序列是**调用方**的职责（脚本、docs、regressions），不是库的 —— 库只暴露原语（CLAUDE.md "All-in-one façade APIs" 禁令）。

**MD 路径上用 `dynamic=True`（architect 🟡）。** `_TensorDictForce._expose_live_edges`（`driver.py:99-115`）每步把 `edges` 切成 `[0, n_live)`，而 `n_live` 随邻居表重建变化，**E 是变量**。静态形状会让 dynamo 每次 E 改变都重编。因此 MD 路径规定用 `Compiler(dynamic=True)`：E 走符号维、一张图。固定容量的静态边 + CUDA graph 捕获（那需要把 padding 交给势并靠 dead-edge 中和）是**另一条路**，本步明确不做，见 Out of scope。

**eager-only 元素门（`src/molzoo/mace/potential.py:601-606`）。** 现有一次性门在**第一次**被 trace 时把 `torch.unique(Z[~isin(Z, z_table)])`（`encoder.py:303`）拖进图里。改成

```python
# torch.compiler.is_compiling() is a tracing-time constant: dynamo folds this
# branch away, so the host-syncing element check never enters the graph.
if not self._elements_validated and not torch.compiler.is_compiling():
    self.validate_elements(Z)
    self._elements_validated = True
```

**不增加 `_eager_calls` 计数器**（architect 🔴 N5）：属性自增在 trace 下会被 dynamo 记录并重放，计数会在编译期悄悄前进；而且它与 `_elements_validated` 表达的是同一件事。`_elements_validated`（只在非 trace 分支里被置位）是**唯一**的那个事实，`compile_core` 就读它。

保证不被静默降级，而是变成**调用方可见**的：

- `_write_energy` 的 `Raises:` 段（`potential.py:590-594`）改写为"在第一次 **eager** 调用时抛；编译缝从不校验"。
- `energy_core` 的 docstring（方法 `:363-415` 与模块段 `:56-65`）写明：**扁平缝不做任何元素校验**；缝的调用方（`torch.compile` / `torch.export` / `export_for_lammps`）必须先自己调 `validate_elements`。
- `encoder.py:288-308` 的 `validate_elements` docstring 补上这条 eager-only 契约与它的两个承担者（首次 eager 前向、或 `compile_core` 的前置条件）。

（`torch.compiler.is_compiling()` 在 `src/` 下的首次使用由 01 的 `KSpaceGrid` bind 门引入，此处是同一写法的第二处。）

**编译范围与保留断点（写进 Summary、两处 docstring 与 notes）。** 编译核 = **绑定了 cell 的（周期、B=1、一阶）`energy_core`**，1 图 0 断点。整步 = 编译核 + `force.py` 那条具名 autograd 缝（`:167-168` 建两个叶子、`:177-182` 取梯度），owner = **molpot derivation**，由 Q3 裁定保留。其余已命名断点 —— 孤立分子的批量全对枚举（`hartree.py:96-151`，owner `polar-compile-clean-04-pairs`）与 `multipole_ewald.py:221-245`/`:352-370`（owner molpot elec）—— 住在 `.claude/notes/notes.md` topic `compile-no-graph-breaks`，本步用链末的实测数据与 `ExplainSummary.break_sites` 生成的清单更新它。

**陈旧文档与一条跑在验收前面的标记（iron law: no silent debt）。**

- `src/molzoo/mace/potential.py:79` 与 `:87` 仍写 `molix.md.forcefield.PotentialForceField`；该类已不存在（`molix/md/__init__.py:54-75` 的 `__all__` 里没有它），实际类是 `molpot.forcefield.potentials.Potentials`。两处改正，并在同一段落写明本步新增的编译缝与"只覆盖一阶、周期、B=1"的边界。`tests/test_molzoo/test_mace/test_potential.py:318` 的 docstring 同名同病，一并改。
- `docs/molpot/tutorials/class-i-forcefield.md:96,103` 与 `docs/molix/user-guide/ff-export.md:13,27` 仍教读者 `from molix.md import MD, PotentialForceField`（导入即 `ImportError`）与已删除的 `energy_scale=` 关键字；`docs/molix/user-guide/md.md` 同样还在讲 `energy_scale=`。三处按今天的真实 API 改写：`MD(dtype).set_forcefield(ff).set_neighbors(nl).set_integrator(VelocityVerletIntegrator(dt))`，能量换算由 `molix.units.preset_energy_to_md(ff.units)` 从 `ff.units` 推出（`driver.py:309`），`ff` 是满足 `MDForceField` 的 `molpot.forcefield.ForceField`。
- **并且把账结掉**：`.claude/specs/INDEX.md:22` 把 `md-forcefield-molpy-reexport-05-wire` 标成 `[code-complete]`，而它的 `ac-007`（"`ff-export.md` 与 `class-i-forcefield.md` 里既无 `PotentialForceField` 也无 `energy_scale=`"，`.claude/specs/md-forcefield-molpy-reexport-05-wire.acceptance.md:52-60`）至今 `status: pending` 且事实上未达成。本步把文档债还掉之后，**把那条 `ac-007` 翻成 `verified` 并注明证据是本 spec**，同时在 notes 里记一句"`[code-complete]` ≠ 验收通过"。
- **不改**该 acceptance 文件里其余仍 pending 的条目（ac-005 / ac-006 / ac-008 / ac-009），也不动 INDEX 的标记语义 —— 那是另一次盘点。

### Reuse decision

- **(15) `molpot/forcefield/potentials.py:212-220 Potentials.calc_energy` 的 mlip 分支（mlip 在 `__init__:89` 一次性绑定）— `generalize`。** 收敛为 `protocol.call_energy` 一条策略：既去掉 `:215` 那个重复实现的鸭子类型 `getattr` 探针（原判决是 `reuse`（零改动），被 architect 的 🔴 "pre-existing in the exact lines touched" 推翻），也不在这里新增编译分支（编译核在 mlip 上）。
- **(17) `molpot/derivation/force.py:136 autograd_forces_and_virial` — `reuse`（零改动）。** 保持为调用方；边界是 `potentials.py:307-321` 的 `energy_fn` 闭包。Q3 裁定不把它并进编译区，因此它是本 spec **点名认领**的保留断点，而不是遗漏。
- **(18) `potential.py:601-606` 的 `_elements_validated` 门 + `encoder.py:288-308` docstring — `generalize`。** 见 Tasks 的 "Generalize the _elements_validated gate…"：同一个门加 `is_compiling()` 条件，**不**新建第二套校验、**不**新增 `_eager_calls` 计数器（架构裁定 N5：属性自增在 trace 下会被重放）。
- **(14) 02 的 `CellBound` 协议 — `reuse`（实现者）。** `Potentials.bind_cell` 与 `MacePotential.bind_cell` / `MacePolar.bind_cell` 是它在树内的实现者，结构满足，molpot 从 `molix.schema` import 而非 `molix.md`。
- **(14b) 02 的 `CoreCompilable` 协议 — `reuse`（实现者）。** `MacePotential.compile_core` 结构满足；`Potentials.compile_core` 用它做守卫，不做鸭子类型 `getattr` 探针。
- **(16) `molzoo/pinet/potential.py` 的 `compile` 删除 — `pattern`（照抄意图）。** `compile_core` 只收一个结构类型的 `Callable`，不再抄一份 `torch.compile` kwargs，也不覆盖任何基类方法名。
- **(13) 02 的 `Compiler.explain` / `ExplainSummary`（含 `break_sites`）— `reuse`。** 本步所有 explain 断言与 notes 的断点清单都走它，不再手搓 `torch._dynamo.explain`。
- **(10) `MacePolar.energy_core` 的 `num_graphs` 下传 — 已在 01-kgrid 落地。** 本步不重复；`_write_energy` 现有的 `num_graphs` 解析（`potential.py:608-615`）原样复用，编译核不需要第二个来源。
- **(20) `tests/test_molzoo/test_mace/test_potential.py:740-748` 与 `tests/test_molix/test_md/test_compile.py:17 _BACKEND = "aot_eager"` — `pattern`（照抄）。**
- **(1)–(9)(11)(12) → 01-kgrid；(19) → 02-seam。** 本步不碰。

## Files to create or modify

- src/molzoo/mace/potential.py
- src/molzoo/mace/polar.py
- src/molzoo/mace/encoder.py
- src/molpot/forcefield/potentials.py
- tests/test_molzoo/test_mace/test_potential.py
- tests/test_molzoo/test_mace/test_polar.py
- tests/test_molpot/test_forcefield/test_potentials.py
- tests/test_molix/test_md/test_driver.py
- docs/molpot/tutorials/class-i-forcefield.md
- docs/molix/user-guide/ff-export.md
- docs/molix/user-guide/md.md
- .claude/specs/md-forcefield-molpy-reexport-05-wire.acceptance.md
- .claude/notes/notes.md
- regressions/polar-compile-clean-03-wire.py (new)

## Tasks

- [x] Write failing unit tests for the documented no-op MacePotential.bind_cell, the MacePolar override reaching both GTO grids, and Potentials.bind_cell's CellBound guard over a non-CellBound toy mlip (tests/test_molzoo/test_mace/test_potential.py → TestMacePotentialCellBinding; tests/test_molzoo/test_mace/test_polar.py → TestMacePolarBindCell; tests/test_molpot/test_forcefield/test_potentials.py → TestPotentialsBindCell)
- [x] Implement the no-op MacePotential.bind_cell in src/molzoo/mace/potential.py, the MacePolar override in src/molzoo/mace/polar.py, and the isinstance-guarded Potentials.bind_cell in src/molpot/forcefield/potentials.py importing CellBound from molix.schema
- [x] Write failing unit tests for the eager-only element gate without a call counter and for MacePotential.compile_core installing the core exactly once behind its single precondition (tests/test_molzoo/test_mace/test_potential.py → TestMacePotentialElementGate, TestMacePotentialCompileCore)
- [x] Generalize the _elements_validated gate with torch.compiler.is_compiling() and implement MacePotential.compile_core plus the _core dispatch inside _write_energy in src/molzoo/mace/potential.py, restating the eager-only contract in energy_core and in src/molzoo/mace/encoder.py validate_elements
- [x] Write failing unit tests for Potentials.compile_core delegation, for the mlip branch of calc_energy routing through protocol.call_energy, for the compiled/eager explain split, and for a Potentials rebuilt by to_potentials() still dispatching through the installed core (tests/test_molpot/test_forcefield/test_potentials.py → TestPotentialsCompileCore, TestPotentialsCompiledRoute; tests/test_molix/test_md/test_driver.py → TestCompiledMlipStep)
- [x] Implement Potentials.compile_core behind the CoreCompilable guard and collapse the mlip branch of calc_energy onto protocol.call_energy in src/molpot/forcefield/potentials.py
- [x] Fix the stale PotentialForceField docstrings at src/molzoo/mace/potential.py:79 and :87 and tests/test_molzoo/test_mace/test_potential.py:318, rewrite the deleted-API examples in docs/molpot/tutorials/class-i-forcefield.md, docs/molix/user-guide/ff-export.md and docs/molix/user-guide/md.md, and flip ac-007 of .claude/specs/md-forcefield-molpy-reexport-05-wire.acceptance.md to verified with this spec as evidence
- [x] Fix found during GH200 verification (2026-09-04): inductor fused-scatter leak — `molix.compile.register_inductor_fallbacks()` keeps `aten.index_add` an extern ATen kernel (called from `Compiler.__call__` and `Exporter._package`); PEO-box compiled dE 14.24 eV → 0.0 (jobs 1971898 → 1974355); one slow CPU test in tests/test_molix/test_compile.py
- [x] Verify on GH200 that compiled equals eager to 1e-10 in fp64 on energy/forces and to 1e-9 on virial, that the regressions/mace-polar-periodic-gto.py goldens and n_edges are unchanged, and record the compiled fp32 PEO step time, peak memory, dynamo recompile count and the break_sites-derived post-chain inventory in .claude/notes/notes.md (job 1976727: dE=0.0, max_dF=4.97e-14, max_dW=6.29e-11, n_edges=285546, fp32 0.190 s / 25.87 GiB / 0 recompiles; goldens unchanged)
- [x] Add regression example regressions/polar-compile-clean-03-wire.py (public API only; hard-coded goldens, no third-party runtime)
- [x] Run full check + test suite (ruff check+format clean; ty check --exit-zero-on-warning; default unit tier 3216 passed / 11 skipped in 80 s; GH200 job 1974822 645 polar/compile/md tests passed; job 1976727 measure+goldens)

## Testing strategy

单元测试只测单函数/单模块，路径镜像 `src/`（`src/molzoo/mace/potential.py` → `tests/test_molzoo/test_mace/test_potential.py`，`src/molzoo/mace/polar.py` → `tests/test_molzoo/test_mace/test_polar.py`，`src/molpot/forcefield/potentials.py` → `tests/test_molpot/test_forcefield/test_potentials.py`，`src/molix/md/driver.py` → `tests/test_molix/test_md/test_driver.py`），类型镜像。CPU explain 一律 `backend="aot_eager"`，断言经 02 的 `Compiler.explain`。**镜像归属**：基类的 no-op `bind_cell` 与 `compile_core` 住在 `potential.py`，所以测在 `test_potential.py`；`MacePolar.bind_cell` 的覆写住在 `polar.py`，测在 `test_polar.py`；MD 路径上的断言（编译核跨 `to_potentials()` 重建存活、每步 `graphs.energy` 更新）住在 `tests/test_molix/test_md/test_driver.py`。

- `tests/test_molzoo/test_mace/test_potential.py::TestMacePotentialCellBinding`
  - happy path：一个非 Polar 的 `MacePotential`（MatPES 变体）调 `bind_cell(cell)` 不抛、不改任何状态；`MacePotential.bind_cell.__doc__` 说明"非极化变体没有倒空间网格可定尺"。
  - happy path：`isinstance(potential, CellBound)` 为 True（协议从 `molix.schema` import）。
- `tests/test_molzoo/test_mace/test_polar.py::TestMacePolarBindCell`
  - happy path：给两个 GTO 子模块的 `bind_cell` 打桩，`MacePolar.bind_cell(cell)` 后两者各被调用一次、实参与传入 cell 同存储（`.data_ptr()` 相等）；构造点是 `polar.py:157`（描述子）与 `:165`（coulomb）。
  - edge case：容量溢出的 cell 让 `bind_cell` 冒泡 `RuntimeError`，消息**同时含** `"KSpaceGrid"` 与 `"capacity_factor"`（01 的 ac-004 对齐）。
  - edge case：`(1,3,3)` 的 cell 与 `(3,3)` 走同一条路（`_as_cell` 在 `KSpaceGrid` 入口，01 的 ac-014）。
- `tests/test_molzoo/test_mace/test_potential.py::TestMacePotentialElementGate`
  - happy path：eager 首调用对越界 Z 抛 `ValueError` 并列出越界原子序数（行为不变）；成功后 `_elements_validated` 为 True，第二次 eager 调用不再校验。
  - edge case：`Compiler.explain` 下（`is_compiling()` 为真）门被折掉 —— 断言 `graph_break_count == 0`，此路径**不**因越界 Z 抛错。
  - 反向守卫：`MacePotential` 上**不存在** `_eager_calls` 属性（`hasattr` 为 False）—— 计数器在 trace 下会被重放，本步明确不引入。
  - docs 守卫：`MacePotential.energy_core.__doc__` 含"no element validation"字样并点名 `validate_elements` 是缝调用方的职责，`MACEEncoder.validate_elements.__doc__` 含 eager-only 契约与它的两个承担者。
- `tests/test_molzoo/test_mace/test_potential.py::TestMacePotentialCompileCore`
  - edge case（唯一前置条件）：全新实例（`_elements_validated is False`）调 `compile_core(stub)` ⇒ `RuntimeError`，消息点名 `compile_core` 与"先跑一次 eager 前向"；`stub` 零调用。
  - happy path：一次 eager `_write_energy` 之后 `compile_core(stub)` 恰调用 `stub` 一次，`self._core` 变成 `stub` 的返回值；**二次** `compile_core(stub)` 不再调用（计数仍为 1）。
  - happy path（派发）：装核后 `_write_energy` 每次调用命中 `_core`（对返回的 callable 计数），且 `batch["graphs","energy"]` 仍由 `protocol.write_energy` 写、**未 detach**（`.requires_grad` 在 enable_grad 下为 True、`grad_fn` 非 None）。
  - edge case（拆包不变）：装核前后 `_write_energy` 对同一 batch 的 `num_graphs` 解析结果一致；缺 `graphs` 命名空间时不抛而走 `:612-615` 的单次 host sync，随后 `:641-642` 补建 `batch_size=[B]`；缺 `edges.edge_diff` 时抛既有 `ValueError`（消息含 `"requires edges.edge_diff"`）。
  - compile：`Compiler.explain(mlip.energy_core, *flat_args, cell=cell)` 在 `bind_cell` 之后给 `(graph_count == 1, graph_break_count == 0)`。
- `tests/test_molpot/test_forcefield/test_potentials.py::TestPotentialsBindCell`
  - happy path：`bind_cell((3,3))` 一层委托到 mlip（对 mlip 的 `bind_cell` 计数为 1，实参同存储）；无 mlip 时 no-op。
  - edge case（架构裁定 N2）：mlip 是 `tests/test_molpot/test_forcefield/test_potentials.py:238-252` 那个没有 `bind_cell` 的 `_EdgeSpring` 玩具时，`bind_cell` 是 no-op、**不抛 AttributeError**，随后 `calc_forces` 的结果与改动前逐位相同。
  - 静态：`isinstance(pots, CellBound)` 为 True（从 `molix.schema` import）；`grep -nE "^\s*(from|import)\s+molix\.md" src/molpot/` 无输出。
- `tests/test_molpot/test_forcefield/test_potentials.py::TestPotentialsCompileCore`
  - edge case：mlip 不满足 `CoreCompilable`（`_EdgeSpring`）⇒ `compile_core` 抛 `RuntimeError` 点名 `compile_core`；无 mlip ⇒ 同样抛。
  - happy path：满足时恰把同一个 `compiler` 转交给 `self.mlip.compile_core` 一次；`Potentials` 实例上**不存在** `_compiled_core` 属性。
  - 静态：参数按结构类型标注为 `Callable[[Callable[..., Any]], Callable[..., Any]]`；`grep -nE "^\s*(from|import)\s+molix\.compile" src/molpot/` 无输出。
- `tests/test_molpot/test_forcefield/test_potentials.py::TestPotentialsCompiledRoute`
  - **eager 分支走 `protocol.call_energy`**：monkeypatch `molpot.derivation.protocol.call_energy` 计数，`calc_energy` 恰调用它一次并返回 `system[ENERGY_KEY]`；`grep -n "_write_energy" src/molpot/forcefield/potentials.py` 无输出；`pos=` 覆盖仍在读扁平张量**之前**落到 `system["atoms","pos"]`（改 `pos` 会改能量）。
  - **`graphs.energy` 每步刷新**：装核后连续两次求值、两次 `atoms.pos` 不同 ⇒ `work[("graphs","energy")]` 两次值不同（`not torch.equal`），且各自等于当次编译核的返回值。
  - compile（**范围断言，架构裁定 N1**）：tiny Polar + `graphs.cell`，规定序列走完后 (a) `Compiler.explain(pots.mlip.energy_core, …)` 给 `(1, 0)`；(b) `Compiler.explain(pots.calc_energy_forces_virial, work)` 的 `graph_break_count > 0`，且**每一个** `break_sites` 项都指向 `src/molpot/derivation/force.py` —— 出现任何其他文件即失败（断言消息里写明 owner = molpot derivation、依据 Q3）。无 cell 的孤立路径另断言 `graph_break_count > 0`（01 已命名的 04-pairs 断点）。
  - domain validation：tiny Polar 上 compiled 与 eager 的 `energy` / `forces` / `virial` 在 fp64 下 `atol=1e-10`（只差归约顺序，非逐位）。GH200 PEO 盒（4995 原子）上 energy / forces 仍是 1e-10，virial 是 1e-9（作业 1974822：dE = 0.0、max_dF = 6.4e-14、max_dW = 1.087e-10，对照 |W| ~ 5200 eV）。
  - **失效规则**：装核之后把 cell 各向同性放大 1.5×（越过 `capacity_factor=1.2`），(a) 经 `MD.run` 的 barostat 路径（02 的 `rebind_cell`）抛 `RuntimeError`，消息含 `"KSpaceGrid"` 与 `"capacity_factor"`，并断言**邻居表没有先抛**（`NeighborList.scale_isotropic` 的 `ValueError` 不在其中：先构造一个 r_build 足够小的列表让它不触发）；(b) 绕过重绑定直接调编译核，得到 `isnan` 的能量与力（01 的 in-graph 哨兵），**不是**被截断的有限值；1.1× 既不抛也不出 NaN。
- `tests/test_molix/test_md/test_driver.py::TestCompiledMlipStep`
  - **跨重建存活（架构裁定 N6）**：`compile_core` 之后再调一次 `ff.to_potentials()` 得到**新的** `Potentials` 对象（`is not` 前一个），但 `pots2.mlip is pots.mlip`，且 `pots2.calc_energy(work)` 仍命中同一个编译 callable（对它计数 +1）；随后 `MD(...).set_forcefield(ff).set_neighbors(nl).set_integrator(...)`（每个 setter 都触发 `driver.py:300-318` 的 `_wire`）再 `run(system, 2)`，计数继续增长。
  - **MD 路径上的 `graphs.energy`**：走完 `_TensorDictForce._eval_mlip`（`driver.py:143-145`）两步后 `out["energy"]` 随位置变化而变化（读的是 `work[ENERGY_KEY]`，不是热身那一步的陈值）。
  - **dynamic 形状**：邻居表重建让 `n_live` 变化两次，`Compiler(dynamic=True)` 下 `torch._dynamo.utils.counters` 记录的 recompile 次数在热身后 `<= 1`。
- **GH200 验证（唯一能跑 fused cueq 与 PEO 盒的环境）**：登录节点是 x86，必须用作业模板 `.scratch/convfusion/sbatch_polar_fix_verify.sh`（它把 aarch64 molrs wheel 建进 shim；**禁止** `maturin develop`）。该作业一次产出五项读数：(a) compiled vs eager fp64：energy / forces 1e-10，virial 1e-9（作业 1974822 已测 max_dW = 1.087e-10，tiny-box 的 1e-10 virial 在该盒上略紧于归约噪声）；(b) `python regressions/mace-polar-periodic-gto.py` 的四个官方 golden 与 `n_edges=285546` 全部不变（`atol=1e-3` eV）；(c) compiled fp32 的 PEO 每步耗时，对照 0.38 s eager 基线（`.claude/notes/notes.md:231-236`）**记录**；(d) compiled fp32 峰值显存，对照 28.3 GiB 基线 —— 01 的固定半盒把 M 从 5318 抬到 21437，`structure_factor` 的三张 (N,M) fp32 张量各 408.5 MiB，预计 +~1 GiB，必须实测确认没有把 28.3 GiB 顶到 OOM；(e) 热身后的 dynamo recompile 计数。
- **Regression example**：`regressions/polar-compile-clean-03-wire.py` —— 只用公开 API（`MacePolar`、`molpot.forcefield.ForceField` / `Potentials`、`molix.compile.Compiler`），CPU 秒级 tiny 周期构型，并**演示规定的调用序列** `def_pairstyle → to_potentials → bind_cell → 一次 eager → compile_core(Compiler(dynamic=True))`：(a) 序列走完后 `calc_energy_forces_virial` 的 fp64 能量/力/virial 与硬编码字面量一致到 1e-10、(b) 与未编译路径一致到 1e-10、(c) `Compiler.explain(mlip.energy_core, …)` 为 `(1, 0)` 且 `Compiler.explain(pots.calc_energy_forces_virial, work)` 的所有 `break_sites` 都落在 `src/molpot/derivation/force.py`、(d) 连续两个不同位置的步给出**不同**的 `graphs.energy`、(e) `ff.to_potentials()` 重建出的新 `Potentials` 仍命中同一个编译核、(f) 跳过那次 eager 前向直接 `compile_core` 抛 `RuntimeError`、(g) 断言 `"PotentialForceField"` 不再出现在 `molzoo.mace.potential.__doc__` 与 `MacePotential.__doc__` 里。另外用 CPU 上的 wall-clock 做一次**无阈值**的步时冒烟打印（只记录，不判定）。goldens 是本仓改动前 eager 捕获的字面量（注释注明来源、日期 2026-09-03、命令）；不 import、不 subprocess `mace` / `e3nn` / `graph_longrange` / ASE。
- `regressions/mace-polar-periodic-gto.py:40-46` 的既有 goldens **只被重跑，不被改写**；任何数值变动都是回归而不是更新。`regressions/mace-polar-official-water.py:28-33`（孤立路径）同样重跑并断言不变。

## Out of scope

- **整步 0 断点** —— `force.py:167-168` / `:177-182` 的 autograd 缝按 Q3 留在编译区外，owner = molpot derivation。本步只断言"断点全部且仅在该文件内"，不消除它。
- 二阶反传编译 —— AOTAutograd 不支持 double backward（spec `cuet-force-doublebackward`）。本 spec 的编译核**只对一阶、周期（cell 已绑定）、B=1 的 MD 推理成立**；训练侧力损失继续走 eager，模块 docstring 必须写明这条边界。
- 把应变包装层并进编译区（Q3 已裁定不做：会造出应变约定的第二个家）。若实测显示 eager 的应变 gather 有分量，用"应变过的 `shifts`"加宽现有缝，另开 spec。
- **容量定尺的静态边 + 整步 CUDA graph 捕获** —— 本步 MD 路径用 `Compiler(dynamic=True)` 吃掉 `_expose_live_edges`（`driver.py:99-115`）带来的可变 E。把 padding 边交给势并靠 dead-edge 中和、进而捕获 CUDA graph，是**后续 spec** 的事，点名于此。
- `is_cell_bound` 之类的"网格已绑定"布尔属性 —— 对 MatPES / OMol 无意义，已由调用序列 + 01 的具名 trace 期 `RuntimeError` 覆盖。
- 孤立分子的批量全对枚举（`hartree.py:96-151`，owner `polar-compile-clean-04-pairs`）与 `multipole_ewald.py:221-245`/`:352-370`（owner molpot elec）—— 本步只在 notes 的清单里记录它们的最终状态。
- 邻居表 build/update 的编译（另 spec）、OMol/MatPES（已 0 断点）。
- spec `mace-neighbor-graph-correctness` 覆盖的邻居表正确性。
- **点名的依赖面债务（本步只报，不修）**：
  - `src/molix/ff_export/{compiler,adapter,lammps_adapter,openmm_adapter,lammps_writer}.py` 各 import molpot（`compiler.py:18`、`adapter.py:20`、`lammps_adapter.py:27`、`openmm_adapter.py:28`、`lammps_writer.py:26`）—— 违反单向包规则 `molix ← molrep ← molzoo/molpot` 的反向边 → 独立的 refactor spec。本步的 `compile_core` 按结构类型收 `Callable`、协议从 `molix.schema` 取，**不**因为"反正已经有反向边"而再加一条。
  - `.claude/notes/architecture.md:84` 的 `molix.md` 导出清单已陈旧（还列着 `CallableForceField` / `PotentialForceField` / `PeriodicPotentialForceField` / `HarmonicForceField` / `LennardJones*ForceField` / `ForceOutput`，而 `molix/md/__init__.py:54-75` 里没有它们；反过来缺 `BerendsenBarostat` / `MDForceField` / `instantaneous_pressure` / `pressure_bar_to_md`）→ 路由 `/mol:map`（blueprint 重生成），不在本步手改。
  - `.claude/specs/md-forcefield-molpy-reexport-05-wire.acceptance.md` 里除 ac-007 之外仍 pending 的 ac-005 / ac-006 / ac-008 / ac-009 —— 本步不碰，属另一次盘点。
  - Polar 目前没有树内 bench guard，而它承载 GH200 的头条数字（notes topic `bench-guard-encoders` 要求 `benchmarks/bench_<encoder>.py`）。本步只**记录**步时与显存，不建 guard；`benchmarks/bench_polar_compile.py` 作为后续 spec 的待办点名于此，不静默。
- blueprint 重生成（`/mol:map`）。
