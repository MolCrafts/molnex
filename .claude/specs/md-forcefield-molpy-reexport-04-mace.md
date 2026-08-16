---
title: MACE 消费已成像的 edge_diff，不再用 shifts 做最小镜像
status: code-complete
created: 2026-08-16
slug: md-forcefield-molpy-reexport-04-mace
chain: md-forcefield-molpy-reexport
---

# MACE 消费已成像的 edge_diff，不再用 shifts 做最小镜像

## Summary

`molzoo.mace` 的 `MACEPotential.forward` 与编译缝 `energy_core` 改为消费 batch 上已经写好的 `edges.edge_diff`（必要时用它对 `pos[target]-pos[source]` 的常数余项保住 `∂d/∂pos`），不再把 `edges.shifts` 当成周期边界通道，也不再在势函数内部做最小镜像。周期几何的值来自上游已成像的 `edge_diff`；力仍对 `pos` 反传。符号遵循 molnex 边约定。只改 `molzoo.mace`，不动 mace-nve。

## Domain basis

MACE 作用在邻接位移及其模长上（Batatia et al., NeurIPS 2022, https://arxiv.org/abs/2206.07697）。成像由 NeighborList / collate 完成。常数余项：

\[
\mathbf{r}^{\text{raw}}=\mathbf{r}_t-\mathbf{r}_s,\quad
\mathbf{S}=\operatorname{detach}(\texttt{edge\_diff}-\mathbf{r}^{\text{raw}}),\quad
\mathbf{r}=\mathbf{r}^{\text{raw}}+\mathbf{S}.
\]

硬编码：\(L=10\) Å，source \((0.5,0,0)\)，target \((9.5,0,0)\) → `edge_diff=(-1,0,0)`，\(d=1\)，\(\partial d/\partial r_s=(+1,0,0)\)，\(\partial d/\partial r_t=(-1,0,0)\)。反例：传入 `edge_diff=(+5,0,0)` 必须吐出 `(+5,0,0)`，不得包回 `(-1,0,0)`。

## Design

`energy_core(positions, Z, edge_index, batch, num_graphs, *, edge_diff=None, total_charge=None, total_spin=None)`。删除 `shifts` 作为 PBC 通道：位置第六参与 `shifts=` 响亮失败。`edge_vectors(..., *, edge_diff=None)` 同一契约。`forward` / `_write_energy` 必须读 `edges.edge_diff`；缺键 `ValueError`；忽略残留 `edges.shifts`。不读 `edge_dist` 作为能量图输入。`MACEMatpes.energy_forces` / `MACEOMol.energy_forces` 改为 `edge_diff=`。

**Reuse decision**

- generalize `MACEPotential.energy_core`、`molzoo.mace.geometry.edge_vectors`
- reuse Edge Convention、NeighborList（不重建）
- new — 不 import pinet `edge_bond_diff`

## Files to create or modify

- src/molzoo/mace/geometry.py
- src/molzoo/mace/potential.py
- src/molzoo/mace/variants.py
- tests/test_molzoo/test_mace/test_geometry.py
- tests/test_molzoo/test_mace/test_potential.py
- tests/test_molzoo/test_mace/test_variants.py
- regressions/md-forcefield-molpy-reexport-04-mace.py (new)

## Tasks

- [ ] Write failing unit tests for edge_vectors / edge_lengths (tests/test_molzoo/test_mace/test_geometry.py → TestEdgeVectors, TestEdgeLengths)
- [ ] Write failing unit tests for MACEPotential (tests/test_molzoo/test_mace/test_potential.py → TestMACEPotential)
- [ ] Write failing unit tests for MACEMatpes.energy_forces / MACEOMol.energy_forces (tests/test_molzoo/test_mace/test_variants.py → TestMACEMatpes, TestMACEOMol)
- [ ] Generalize edge_vectors in src/molzoo/mace/geometry.py to consume edge_diff as a detached remainder
- [ ] Generalize MACEPotential.energy_core and _write_energy in src/molzoo/mace/potential.py
- [ ] Implement edge_diff on MACEMatpes.energy_forces and MACEOMol.energy_forces in src/molzoo/mace/variants.py
- [ ] Add docstring per google with units (Å, eV, eV/Å)
- [ ] Add regression example regressions/md-forcefield-molpy-reexport-04-mace.py (public API only; hard-coded goldens, no third-party runtime)
- [ ] Verify against the L=10 MIC pair edge_diff=(-1,0,0)
- [ ] Run full check + test suite

## Testing strategy

MIC 字面量 `(-1,0,0)`；反 MIC `(+5,0,0)` 不被再包；`shifts=` 与第六位置参失败；缺 `edge_diff` 的 batch 失败。回归脚本公开 API，无 ASE / e3nn / mace-torch / mace-nve。

## Out of scope

- 修改 molix / molpot
- 改动 mace-nve、scripts/matpes_port、benchmarks（由 05 改调用方）
- 在势内恢复 pos+shifts 或重做 MIC
- 改 energy_core 前五个位置参数
