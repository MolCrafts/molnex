---
title: 经典 pair 项读取 batch 已放置几何（禁止二次 PBC）
status: approved
created: 2026-08-16
slug: md-forcefield-molpy-reexport-03-potentials
chain: md-forcefield-molpy-reexport
---

# 经典 pair 项读取 batch 已放置几何（禁止二次 PBC）

## Summary

在 molpot 的 Class-I 非键路径上，当 MD / 已整理 batch 已经带有 `edges.edge_diff` 或 `edges.edge_dist` 时，`Potentials` 直接使用这份已放置几何计算 Lennard-Jones / 开边界 Coulomb 的实空间对距离，不再用 `cell` 做第二次周期最小镜像。`LJCutPotential.energy` 继续只吃 `distances`。没有位移字段的训练 / 单点路径保留现有 `NeighborListBuilder`。跨周期面的 LJ 对使用最小镜像距离（盒长 10 Å、原子在 0 与 9 时用 r = 1 Å），而不是裸坐标差 r = 9 Å。

## Domain basis

势函数本身不负责周期边界。LAMMPS `pair_lj_cut` 从邻表已放好的邻居坐标取位移再算 `rsq`（`Developer_write_pair`、`Developer_par_comm`）。molnex：

\[
\texttt{edge\_index}[:,0]=\text{source},\quad
\mathbf{r}_{ij}=\texttt{edge\_diff},\quad
r=\texttt{edge\_dist}.
\]

LJ 12-6（截断 \(r_c\)）：\(E_{ij}=4\varepsilon[(\sigma/r)^{12}-(\sigma/r)^{6}]\)，\(r<r_c\)。Lorentz–Berthelot。

硬编码夹具：\(L=10\) Å，原子 \((0,0,0)\) 与 \((9,0,0)\)，已放置 \(r=1\) Å，\(\varepsilon=1\) kcal/mol、\(\sigma=2\) Å、\(r_c=5\) Å → \(E=16128\) kcal/mol。错误用 \(r=9\) 得 \(\approx-4.816\times10^{-4}\)。力 \(F_0=+195072\,\hat{x}\)，\(F_1=-195072\,\hat{x}\)。

参考文献：Jones 1924 DOI 10.1098/rspa.1924.0082；https://docs.lammps.org/pair_lj.html；https://docs.lammps.org/Developer_write_pair.html；https://docs.lammps.org/Developer_par_comm.html

## Design

只改 molpot。`def_pairstyle` / `to_potentials` / `calc_energy` / `calc_forces` 调用面不动。`molpot` 不 import `molix`。

`NeighborListBuilder.build` 新增可选 `edges`：含 `edge_index` 且含 `edge_diff` 或 `edge_dist` 时禁止进入 `_periodic`、禁止用 `cell` 做 MIC。`pos` 是新叶子时用 `pos[t]-pos[s]+shifts` 或 implied shift 重建 \(r\)。无位移字段时回退今日 `_open` / `_periodic`。

`Potentials._scaled_pairs` 把 `edges` 交给 builder。`LJCutPotential.energy` 签名不变，文档写明 distances 已放置。

**Reuse decision**

- reuse `LJCutPotential.energy`、`ForceField.def_pairstyle`、`to_potentials`、`calc_energy` / `calc_forces`
- generalize `NeighborListBuilder.build`、`Potentials._scaled_pairs`

## Files to create or modify

- src/molpot/forcefield/neighbor_list.py
- src/molpot/forcefield/potentials.py
- src/molpot/forcefield/pair_terms.py
- tests/test_molpot/test_forcefield/test_neighbor_list.py (new)
- tests/test_molpot/test_forcefield/test_pair_terms.py (new)
- tests/test_molpot/test_forcefield/test_potentials.py
- tests/test_molpot/test_forcefield/test_lj.py
- regressions/md-forcefield-molpy-reexport-03-potentials.py (new)

## Tasks

- [ ] Write failing unit tests for NeighborListBuilder (tests/test_molpot/test_forcefield/test_neighbor_list.py → TestNeighborListBuilder)
- [ ] Generalize NeighborListBuilder.build in src/molpot/forcefield/neighbor_list.py to cover precomputed batch geometry
- [ ] Write failing unit tests for LJCutPotential.energy and Potentials._scaled_pairs (tests/test_molpot/test_forcefield/test_pair_terms.py → TestLJCutPotential; tests/test_molpot/test_forcefield/test_potentials.py → TestPotentials; tests/test_molpot/test_forcefield/test_lj.py → TestLJEnergy)
- [ ] Generalize Potentials._scaled_pairs in src/molpot/forcefield/potentials.py to prefer batch edges
- [ ] Add docstring per google with units on NeighborListBuilder.build, Potentials._scaled_pairs, and LJCutPotential.energy
- [ ] Add regression example regressions/md-forcefield-molpy-reexport-03-potentials.py (public API only; hard-coded goldens, no third-party runtime)
- [ ] Verify against periodic-face LJ textbook case (L=10 Å, r=1 Å, E=16128 kcal/mol)
- [ ] Run full check + test suite

## Testing strategy

有 `edge_dist=1` 或 `edge_diff=(-1,0,0)` 时距离为 1 不是 9；poisoned `edge_dist=2` 返回 2。`_periodic` 不被调用。无 edges 时回退原路径。`calc_energy` → `16128` kcal/mol；`calc_forces` 绑在 pos 叶子上。回归脚本公开 API，金值字面量。

## Out of scope

- 修改 molix / molzoo
- 改 LJCutPotential.energy 签名
- 改周期 Coulomb k 空间
- 新工厂或第二套 Potentials
