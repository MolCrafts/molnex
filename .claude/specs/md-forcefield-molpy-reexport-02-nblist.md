---
title: MD 邻表写入 live MIC 几何并由 MD 调度
status: code-complete
created: 2026-08-16
slug: md-forcefield-molpy-reexport-02-nblist
chain: md-forcefield-molpy-reexport
---

# MD 邻表写入 live MIC 几何并由 MD 调度

## Summary

`molix.md.NeighborList` 在 `build` / `update(batch)` 时除既有 `edges.edge_index` 与常数晶格余项 `edges.shifts` 外，还写入由**当前** `atoms.pos` 算出、且**不** `detach` 的 `edges.edge_diff` 与 `edges.edge_dist`。MD 在每次力评估处先 `nl.update(batch)` 再 `pots.calc_*`；`Integrator.eval_force` 只消费 `ForceOutput` 并做状态精度转换，不再询问 `force.rebuild_neighbors`。公开符号 `rebuilds_neighbors` 从 `molix.md` 删除。边矢量遵循 molnex 约定 `edge_diff = MIC(pos[target] − pos[source])`，不是 LAMMPS `delx`。立方盒 `L = 10 Å`、源 `(0.5, 5, 5) Å`、宿 `(9.5, 5, 5) Å` 时，裸差 `(9, 0, 0) Å`，写入 `edge_diff = (−1, 0, 0) Å`、`edge_dist = 1 Å`、`rsq = 1 Å²`；该 `edge_diff` 留在 `pos` 的 autograd 图上，否则 `F = 0`。

## Domain basis

LAMMPS 把邻表构建与 pair 循环拆开：邻表先给出已包络的位移，pair 循环不再做 MIC。GROMACS 要求 `r_c < L/2`，使每个矢量只有唯一最小镜像。

对正交盒，分数坐标四舍五入给出最小镜像。令源、宿笛卡尔坐标为 `r_s`、`r_t`（Å），边为 `source → target`：

```
raw = r_t − r_s
S   = MIC(raw) − raw          # 常数晶格余项，Å；不进入 pos 图
d   = r_t − r_s + S           # = MIC(raw) = edges.edge_diff，Å
r   = ‖d‖                     # = edges.edge_dist，Å
rsq = d · d                   # Å²
```

正交立方盒 `L = 10 Å` 的硬编码算例：`r_s = (0.5, 5, 5)`，`r_t = (9.5, 5, 5)`，`raw = (9, 0, 0)`，`S = (−10, 0, 0)`，`edge_diff = (−1, 0, 0)`，`edge_dist = 1`，`rsq = 1`。`9 − L · round(9/L) = −1`。反向边 `d → −d`。

**符号不是 LAMMPS `delx`。** molnex：`edge_index[:,0] = source`，`edge_diff = pos[target] − pos[source] + S`。

**Autograd。** `S`（`shifts`）可保持为常数余项。`d` 必须由未 `detach` 的 `pos` 算出。探针 `E = ½ ‖d‖²`：`F_t = −d = (1, 0, 0)`，`F_s = d = (−1, 0, 0)`。若 `edge_diff` 被 `detach`，则 `F = 0`。

长度一律 Å。单位桥接走 `molix.units`。

参考文献：

- LAMMPS *Notes for writing a new pair style* — https://docs.lammps.org/Developer_write_pair.html
- LAMMPS *neighbor* — https://docs.lammps.org/neighbor.html
- LAMMPS *Communication* — https://docs.lammps.org/Developer_par_comm.html
- K. Nordlund, lecture 2 — https://www.mv.helsinki.fi/home/knordlun/moldyn/lecture02.pdf
- GROMACS：`r_c < L/2` 保证最小镜像唯一

## Design

前序 `01-reexport` 已落地。本规格只改邻表写入面与 MD 调度。

**1. 泛化 `NeighborList.build` / `update` 的写入面。** 列表仍独占 `edge_index` 与 `shifts`。`build(batch)` 用当前 `batch["atoms", "pos"]`（禁止 `detach`）写入：

```
src, tgt = edge_index[:, 0], edge_index[:, 1]
edge_diff = pos[tgt] - pos[src] + shifts
edge_dist = ‖edge_diff‖
```

`update(batch)`：无论是否触发连通性重建，都重写 `edge_diff` / `edge_dist`。MD 调度一律走 `update(batch)`。`build` 与 `update(batch)` 共享私有 `_write_geometry(batch, pos)`。禁止合并两套 `NeighborList`。

**2. MD 调度。** 新增 `MD.compute_force(pos) -> ForceOutput`：写 `pos` 进工作 batch；若已 `set_neighbors` 则 `nl.update(batch)`；`pots.calc_energy` / `calc_forces`；返回 `ForceOutput`。`Integrator.eval_force` 只做 `out = self.force(pos)` 与 dtype 转换。删除 `rebuild` / `rebuilds_neighbors` / `rebuild_neighbors`。冻结列表 = 不 `set_neighbors`。

**Reuse decision**

- generalize `NeighborList.build` / `NeighborList.update`
- generalize `Integrator.eval_force`
- reuse `ForceOutput`、Edge Convention（PATTERN）、`molix.units`
- new — 不新增第三套邻表，不合并两套 `NeighborList`

## Files to create or modify

- src/molix/md/neighbors.py
- src/molix/md/integrators.py
- src/molix/md/driver.py
- src/molix/md/__init__.py
- src/molix/md/runner.py
- src/molix/md/forcefield.py
- tests/test_molix/test_md/test_neighbors.py
- tests/test_molix/test_md/test_integrators.py
- tests/test_molix/test_md/test_driver.py
- tests/test_molix/test_md/test_forcefield.py
- tests/test_molix/test_md/test_compile.py
- tests/test_molix/test_md/test_runner.py
- regressions/md-forcefield-molpy-reexport-02-nblist.py (new)

## Tasks

- [ ] Write failing unit tests for NeighborList live MIC write (tests/test_molix/test_md/test_neighbors.py → TestNeighborList)
- [ ] Generalize NeighborList.build/update in src/molix/md/neighbors.py to write live edges.edge_diff and edges.edge_dist
- [ ] Write failing unit tests for MD.compute_force dispatch and Integrator.eval_force (tests/test_molix/test_md/test_driver.py → TestMD, tests/test_molix/test_md/test_integrators.py → TestIntegrator)
- [ ] Generalize Integrator.eval_force in src/molix/md/integrators.py to consume only ForceOutput (delete rebuild / rebuilds_neighbors derivation)
- [ ] Implement MD.compute_force in src/molix/md/driver.py (nl.update(batch) then pots.calc_*) and wire it from MD._wire
- [ ] Delete rebuilds_neighbors / rebuild_neighbors / _STALE_EDGE_KEYS from src/molix/md/forcefield.py, src/molix/md/__init__.py, src/molix/md/runner.py and leftover tests
- [ ] Add Google-style docstrings with Å units on NeighborList._write_geometry, NeighborList.build, NeighborList.update, MD.compute_force, Integrator.eval_force
- [ ] Add regression example regressions/md-forcefield-molpy-reexport-02-nblist.py (public API only; hard-coded goldens, no third-party runtime)
- [ ] Verify against the L=10 Å MIC pair (raw=(9,0,0), edge_diff=(-1,0,0), edge_dist=1, rsq=1; F≠0 through live edge_diff)
- [ ] Run full check + test suite

## Testing strategy

单元测试只放在 `tests/`，路径镜像 `src/`。硬编码 MIC 对：`L=10`，`edge_diff=(-1,0,0)`，`edge_dist=1`，`rsq=1`（`atol=1e-12`）。`pos.requires_grad_(True)` 时探针 `E=½‖d‖²` 给出 `F_s=(-1,0,0)`、`F_t=(1,0,0)`；detach 对照 `F=0`。回归脚本复现这组字面量。

## Out of scope

- 势 / pair 内核自己算 PBC（03/04）
- 合并两套 NeighborList
- 修改 Verlet skin 政策算术
- 把 edge_diff 改成 LAMMPS delx 符号
- 恢复 Integrator.rebuild=
- 前端
