---
title: molix.md ForceField 再导出 molpy.ForceField
status: done
last-updated: 2026-09-04
created: 2026-08-16
slug: md-forcefield-molpy-reexport-01-reexport
chain: md-forcefield-molpy-reexport
---

# molix.md ForceField 再导出 molpy.ForceField

## Summary

`molix.md` 的公开力场类型面收束为 molpy 的那一份：`from molix.md import ForceField` 与 `from molpy import ForceField` 是同一对象。自制的 `ForceField` / `CallableForceField` / `HarmonicForceField` / `LennardJonesForceField` / `LennardJonesCutForceField` / `PotentialForceField` / `PeriodicPotentialForceField` 全部删除，不再出现在 `molix.md` 的 `__all__` 或 `forcefield` 模块。用户只通过 `MD.set_forcefield(ff)` 挂载力场（`ClassIForceField` / `OPLSForceField` 因是 `molpy.ForceField` 子类而合法），并通过新的 `MD.set_neighbors(nl)` 挂载邻表；`MD` 内部执行 `pots = ff.to_potentials()`，对工作 `TensorDict` 调用 `calc_energy` / `calc_forces`，用返回值组装 `ForceOutput`。`src/molix/md` 不 import `molpot`，不读 `ff.units`，也不再提供 `energy_scale`。积分器单测改走无名 `PotentialLike` 辅助（或 `ForceField.def_pairstyle`），禁止再写 `ForceField` 子类。`NeighborList.build` 仍只写入 `edge_index` 与 `shifts`。本子规格可单独合并。

## Domain basis

本规格不声明、不引入新物理：能量与力完全委托 `ff.to_potentials()` 的既有实现，MD 侧不做单位换算、不改 PES。周期边界与邻表完整性（含 `edge_diff` 写入、最小镜像契约）属于链上 `02-nblist`，本规格不改 `NeighborList` 的写表面。

## Design

**实体与所有权。** `src/molix/md/forcefield.py` 变为薄再导出：`from molpy import ForceField` 后放入 `__all__`，无自制 `nn.Module`、无工厂、无 façade。`MD` 仍是 composer，沿用 `MaxwellBoltzmann` / 现有 `set_forcefield` / `set_integrator` / `set_hooks` 的 fluent `set_*`：`set_forcefield` 只接受 `molpy.ForceField`（`isinstance` 门禁，拒绝 callable / `nn.Module` / 任意对象），`set_neighbors` 接受 `NeighborList` 并返回 `self`。`ClassIForceField` 与 `OPLSForceField` 是 `molpy.ForceField` 子类，门禁自然放行。`MD.force` 保存这份容器；`MD.neighbors` 保存邻表或 `None`。

**生命周期。** `_wire` 在 `force` 与 `integrator` 都齐时：`pots = ff.to_potentials()`，构造 **私有** `PotentialLike` 适配器（非公开类型、非 `ForceField` 子类、非 `make_*` 工厂）并 `integrator.bind_force(adapter)`。`run(system)` 先按现有契约 `_cast_system`（`POS_KEY` / `VEL_KEY` / `MASS_KEY`），若已 `set_neighbors` 则 `neighbors.build(system)`（复用现有写表面：只写 `edges.edge_index` `(capacity, 2)` 与 `edges.shifts` `(capacity, 3)` Å，不写 `edge_diff` / `edge_dist`），再把该 `TensorDict` 绑给适配器。每次 `eval_force(pos)`：若 `rebuild` 则 `neighbors.update(pos)`；`work[POS_KEY] = pos`；`energy = pots.calc_energy(work)`、`forces = pots.calc_forces(work)`（返回值，同时写入 `ENERGY_KEY` / `FORCES_KEY`）；`ForceOutput(energy.detach().sum(), forces.detach())`。无 `energy_scale` 乘法，不读 `ff.units`；调用方若需把 kcal/mol 等换成积分器的 amu·Å²/fs²，只用 `molix.units.units` 在参数侧完成。`MD.to(device)` 移动 integrator、neighbors，以及 `pots`（若其暴露 `.to`）；**禁止** 对 molpy 容器调用 `.to`。

**求值协议与单位（molpy `UnitSystem`）。** `ff.to_potentials()` 的返回值必须提供吃 TensorDict 的 `calc_energy` / `calc_forces`（molpot 覆写）。裸 molpy `ForceField` 返回的是吃 Frame 的 molrs 核：第一次调用时若不能吃 TensorDict，抛 `TypeError`，点名 `calc_energy` 与 TensorDict。`src/molix` 仍不 import `molpot`。

能量/力换成积分器的 (amu, Å, fs) **只**走 `molix.units`（`from molpy import UnitSystem` 的那份注册表）。禁止 `energy_scale=`，禁止在 `driver.py` 里写 metal→eV 对照表。LAMMPS 风格名由 `UnitSystem.preset(style)` 解析（`metal` → `electron_volt`，`real` → `kilocalorie_per_mole`）；换算是 pint `.to("amu * angstrom ** 2 / femtosecond ** 2")`。辅助函数若需要，只加在 `src/molix/units.py`。`ForceOutput` 在组装时经该模块换算。

**积分器内部协议。** `Integrator.eval_force` 暂时仍通过内部 `PotentialLike` 得到 `ForceOutput`：`__call__(pos) -> ForceOutput`，只读 `rebuilds_neighbors`，以及 `rebuild_neighbors(pos)`（有邻表则 `NeighborList.update`，否则 no-op）。`bind_force` 的注解从自制 `ForceField` 改为该协议。公开挂载只有 `MD.set_forcefield`；`bind_force` 留给积分器单测与 `_wire`。单测用 **无名** `PotentialLike`（测试文件内局部对象，例如谐振子 `E=½k‖x‖²`、`F=-kx`）或 `ForceField.def_pairstyle`，**禁止** 再写 `ForceField` 子类。molpy.optimize 的 `PotentialLike` 吃 `Frame`，不能复用。

**被删类型的测试债。** `test_forcefield.py` 改为再导出 / 删除面 / 挂载契约。`test_integrators.py` / `test_runner.py` / `test_compile.py` 的谐振子与 compile 路径改绑无名 `PotentialLike`。`test_driver.py` 与 `test_neighbors.py` 里依赖 `LennardJonesCutForceField` 的氩晶格能量守恒改为 `set_neighbors` 接线断言（`rebuild` 开关、`update` 被问一次）；LJ/cut PES 数值属 `03-potentials`。`NeighborStrategy` 文档去掉已删类型名，不改 `build`/`update` 行为。

**Reuse decision**

- reuse `molpy.ForceField`（`from molpy import ForceField` 再导出；`MD.set_forcefield` 的唯一公开类型）
- reuse `molix.units.units` / `molpy.UnitSystem.preset`（唯一单位入口；E/F→MD 换算只写在 `molix.units`，MD 不自建对照表）
- reuse `ENERGY_KEY` / `FORCES_KEY` / `POS_KEY`（写活位置、记录 `pots` 返回的 E/F）
- reuse `NeighborList.build` / `NeighborList.update`（本规格不改写表面；`set_neighbors` 只挂载并在 `run`/`eval_force` 调用）
- reuse `ForceOutput`（`eval_force` 与适配器的返回契约）
- generalize `MD.set_forcefield`（类型改为 `molpy.ForceField`；内部 `pots = ff.to_potentials()` 后对 TensorDict 调 `calc_energy`/`calc_forces`）
- pattern `MaxwellBoltzmann` fluent `set_*` — `set_neighbors` / 泛化后的 `set_forcefield` 返回 `MD`
- delete `molix.md.forcefield` 内全部自制力场类型（`ForceField` 模块类、`CallableForceField`、`HarmonicForceField`、`LennardJonesForceField`、`LennardJonesCutForceField`、`PotentialForceField`、`PeriodicPotentialForceField`）
- new — 内部 `PotentialLike` 协议：公开面没有可复用的 TensorDict/`ForceOutput` 符号（molpy.optimize 的 `PotentialLike` 吃 Frame）

## Files to create or modify

- src/molix/md/forcefield.py
- src/molix/md/__init__.py
- src/molix/md/driver.py
- src/molix/md/integrators.py
- src/molix/md/types.py
- src/molix/md/neighbors.py
- src/molix/units.py
- tests/test_molix/test_md/test_forcefield.py
- tests/test_molix/test_md/test_driver.py
- tests/test_molix/test_md/test_integrators.py
- tests/test_molix/test_md/test_compile.py
- tests/test_molix/test_md/test_runner.py
- tests/test_molix/test_md/test_neighbors.py
- regressions/md-forcefield-molpy-reexport-01-reexport.py (new)

## Tasks

- [x] Write failing unit tests for ForceField re-export and deleted names (tests/test_molix/test_md/test_forcefield.py → TestForceFieldReexport)
- [x] Write failing unit tests for MD.set_forcefield and MD.set_neighbors (tests/test_molix/test_md/test_driver.py → TestMD)
- [x] Write failing unit tests for Integrator PotentialLike helpers (tests/test_molix/test_md/test_integrators.py → TestIntegratorRebuildSwitch)
- [x] Re-export molpy.ForceField and drop fake types in src/molix/md/forcefield.py and src/molix/md/__init__.py (__all__ alphabetized)
- [x] Generalize MD.set_forcefield in src/molix/md/driver.py to accept molpy.ForceField, bind pots=ff.to_potentials(), TypeError if calc_energy does not take TensorDict, add MD.set_neighbors, and scale ForceOutput only via molix.units / UnitSystem.preset
- [x] Implement internal PotentialLike in src/molix/md/integrators.py (bind_force annotation + eval_force) and retarget ForceOutput docstring in src/molix/md/types.py
- [x] Rewrite tests/test_molix/test_md/test_compile.py, test_runner.py, and test_neighbors.py off deleted ForceField subclasses; drop deleted-type names from NeighborStrategy docstring in src/molix/md/neighbors.py (build still writes only edge_index+shifts)
- [x] Add docstring per google with units on MD.set_forcefield, MD.set_neighbors, PotentialLike, and Integrator.bind_force
- [x] Add regression example regressions/md-forcefield-molpy-reexport-01-reexport.py (public API only; hard-coded goldens, no third-party runtime)
- [x] Run full check + test suite

## Testing strategy

- **单测路径镜像**：`src/molix/md/<mod>.py` → `tests/test_molix/test_md/test_<mod>.py`；类型镜像 `MD` → `TestMD`，再导出面 → `TestForceFieldReexport`。每个测试只打一个函数/方法。`tests/` 下不做 e2e。
- **Happy path**：`from molix.md import ForceField` 与 `from molpy import ForceField` 身份相同；`MD.set_forcefield(ClassIForceField().def_bondstyle("harmonic").with_params(...))` 后 `run` 对双原子谐振子键写出 `ForceOutput`；`set_neighbors(nl)` 返回 `self` 且 `integrator.rebuild is True`；无名 `PotentialLike` 谐振子上 `VelocityVerletIntegrator.step` 仍一步一力。
- **边界**：`set_forcefield(object())` / callable / `nn.Module` 抛 `TypeError` 并点名 `ForceField`；未 `set_neighbors` 时 `rebuild is False`；`set_forcefield`+`set_integrator` 缺一则 `run` 仍抛既有 `RuntimeError`；`src/molix/md/` 源文本无 `import molpot` / `from molpot`；`NeighborList.build` 后 `edges` 键集恰为 `{edge_index, shifts}`。
- **域验证（无新物理）**：回归与 `TestMD` 使用硬编码谐振子键 `E = ½ k (r − r0)²`：`k = 2`、`r0 = 1` Å、`r = 1.5` Å → `E = 0.25`（`ClassIForceField` 默认 `units="real"` 的 kcal/mol，MD 不换算），`atol = 1e-10`。不是新势函数，不引 DOI。
- **回归示例**：`regressions/md-forcefield-molpy-reexport-01-reexport.py`（仓库根，公共 API）。硬编码：`ForceField is molpy.ForceField`；六个已删名不在 `molix.md.__all__`；双原子谐振子能量字面量 `0.25`（注释写明公式、无第三方）；`set_neighbors` 可链式调用。禁止运行时 import/subprocess 第三方 oracle。

## Out of scope

- `02-nblist`：`NeighborList` 写入 `edge_diff` / 最小镜像契约变更；本规格 `build` 仍只写 `edge_index`+`shifts`。
- `03-potentials`：molpot 经典势（lj/cut、谐振子键以外的项）接入 MD；氩晶格 NVE 漂移数值。
- `04-mace`：MACE / MLIP pair 作为 `def_pairstyle` 模块的 MD 接线。
- `05-wire`：`scripts/`、`benchmarks/`、既有 `regressions/md-neighborlist-skin-07-wire.py`、用户文档（`docs/molix/user-guide/md.md` 等）的迁移。
- 恢复 `energy_scale=`，或在 `driver.py` 手写 metal→eV 对照表（风格名只交给 `UnitSystem.preset`）。
- `src/molix/ff_export/` 既有的 molpot 依赖（本规格门禁只覆盖 `src/molix/md/`）。
- 公开 `CallableForceField` 替代物、公开 `make_*` 适配器、对 `ForceField` 再做子类。
- 改 `molpot` / `molzoo`、改积分器公式、改 `TrainState` / collate schema。
