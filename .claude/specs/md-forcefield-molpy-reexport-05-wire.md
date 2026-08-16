---
title: Wire leftover MD callers off homemade ForceField adapters
status: code-complete
created: 2026-08-16
slug: md-forcefield-molpy-reexport-05-wire
chain: md-forcefield-molpy-reexport
---

# Wire leftover MD callers off homemade ForceField adapters

## Summary

树内所有 MD 调用方改为 01–04 已落地的公开组合：`ForceField.def_pairstyle` 注册经典或 MLIP 对势，`NeighborList` 写 `edge_diff` / `edge_dist`，`MD.set_forcefield(ff).set_neighbors(nl)` 接线，单位桥只走 `molix.units`。`CallableForceField`、`energy_scale=` 构造参数，以及自制 ForceField 子类从 scripts、benchmarks、docs、遗留 `tests/test_molix/test_md/*` 与 `regressions/md-neighborlist-skin-07-wire.py` 中消失。`/home/jicli594/work/mace-nve` 不动。

## Domain basis

本规范只接线，不引入新物理。PBC 与边约定已由 02–04 固定。MD 内部单位仍是 (amu, Å, fs)；任何 eV 或 kcal/mol 换算必须经 `from molix.units import units` 的 pint `.to(...)`，禁止 `energy_scale=` 或硬编码换算浮点。

## Design

唯一故事：

```python
ff = ForceField().def_pairstyle("lj/cut", cutoff=r_cut).with_params(...)
nl = NeighborList(cell=cell, cutoff=r_cut, positions=pos, skin=skin)
md = MD(torch.float64).set_forcefield(ff).set_neighbors(nl).set_integrator(VelocityVerletIntegrator(dt))
md.run(system, n_steps)
```

MLIP（`run_nve.py`）：`ForceField().def_pairstyle(model)`。经典 parity 脚本删掉 `bind_forcefield` / `CallableForceField`。`run_nve.py` 的 `.graph.pt["energy_scale"]` 是树外元数据键，保留。

**Reuse decision**

- reuse `molix.units.units`、`MD.set_forcefield`、`MD.set_neighbors`、`ForceField.def_pairstyle`、`to_potentials`
- new — 不新造 MD 引擎或适配器

## Files to create or modify

- scripts/matpes_port/run_nve.py
- scripts/mm_param_learning/parity_project_md.py
- scripts/mm_param_learning/parity_lammps_classical_md.py
- benchmarks/verify_md_lj_nve.py
- benchmarks/verify_md_ljcut_nve.py
- docs/molix/user-guide/md.md
- docs/molix/user-guide/ff-export.md
- docs/molpot/tutorials/class-i-forcefield.md
- tests/test_molix/test_md/conftest.py
- tests/test_molix/test_md/test_forcefield.py
- tests/test_molix/test_md/test_driver.py
- tests/test_molix/test_md/test_integrators.py
- tests/test_molix/test_md/test_neighbors.py
- tests/test_molix/test_md/test_runner.py
- tests/test_molix/test_md/test_compile.py
- regressions/md-neighborlist-skin-07-wire.py
- regressions/md-forcefield-molpy-reexport-05-wire.py (new)

## Tasks

- [ ] Write failing unit tests for leftover MD export and composer surface (tests/test_molix/test_md/test_forcefield.py → TestForceField; tests/test_molix/test_md/test_driver.py → TestMD)
- [ ] Implement leftover test rewrites in tests/test_molix/test_md/conftest.py, test_forcefield.py, test_driver.py, test_integrators.py, test_neighbors.py, test_runner.py, and test_compile.py
- [ ] Implement script rewrites in scripts/matpes_port/run_nve.py, scripts/mm_param_learning/parity_project_md.py, and scripts/mm_param_learning/parity_lammps_classical_md.py
- [ ] Implement benchmark rewrites in benchmarks/verify_md_lj_nve.py and benchmarks/verify_md_ljcut_nve.py
- [ ] Update docs/molix/user-guide/md.md, docs/molix/user-guide/ff-export.md, and docs/molpot/tutorials/class-i-forcefield.md
- [ ] Implement regressions/md-neighborlist-skin-07-wire.py rewrite onto ForceField.def_pairstyle + MD.set_neighbors
- [ ] Add regression example regressions/md-forcefield-molpy-reexport-05-wire.py (public API only; hard-coded goldens, no third-party runtime)
- [ ] Run full check + test suite

## Testing strategy

`molix.md` 不再导出已删适配器。双原子 lj/cut `r=3.4` Å、`ε=0.15`、`σ=3.0`、`cutoff=10` → \(V=4\varepsilon[(\sigma/r)^{12}-(\sigma/r)^{6}]\) 字面量。skin-07 只换构造，金值不变。回归脚本公开组合 + 有限 `md.run(system, 1)`。

## Out of scope

- `/home/jicli594/work/mace-nve`
- `src/molix/md` 引擎（01–04）
- sbatch 文件（已核对不引用旧 API）
- CHANGELOG / architecture 历史叙述
- `molpot` 项内 `LJ126.energy_scale`
- 公开 `bind_*` / `make_*` façade
