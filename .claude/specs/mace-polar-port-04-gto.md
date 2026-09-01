---
title: MACE-Polar 原生 GTO Hartree 与 Fukui 约束
status: code-complete
created: 2026-08-30
slug: mace-polar-port-04-gto
chain: mace-polar-port
---

# MACE-Polar 原生 GTO Hartree 与 Fukui 约束

## Summary

在 `molpot.potentials.gto` 落地孤立分子 GTO Hartree（官方核 `erf(r/(2σ))/r`，不是论文 Eq.37）和 Fukui 总电荷/自旋约束。无 `graph_longrange`、无 e3nn、无能量对照。k 空间 PBC 与外场留 05。

## Domain basis

Batatia et al., arXiv:2602.19411. Polar-1-M dump 2026-08-30：σ=1.5 Å，multipole \(l_{\max}=1\)，`include_electrostatic_self_interaction=True`。`total_spin` 是 ASE 多重度 \(M=2S+1\)，\(S_{\mathrm{paper}}=M-1\)。Coulomb 前因子走 `molpot.potentials.elec.prefactors.eV_A`（eV·Å·e⁻²），禁止复述换算浮点。

## Design

- `GTOCoulombKernel`：`erf(r / (2σ)) / r`，σ 单位 Å。
- `FukuiEquilibration`：`fukui` `(N, 2)` 按图归一；\(Q_\uparrow=Q+(M-1)\)，\(Q_\downarrow=Q-(M-1)\)；加到两自旋通道的 0e。
- `GTOHartreeEnergy`：完全图（无自环）成对求和；有向边各带 1/2，使 \(E=k_e\sum_{i<j}q_i q_j\,\mathrm{erf}(r_{ij}/(2σ))/r_{ij}\)。\(l=1\) 用官方有限差分位移标量（offset 0.02 Å）。自能可选。
- 新包 `molpot.potentials.gto/`。不复用 `Polarization` / LES / QEq。

## Files

- src/molpot/potentials/gto/{__init__,kernel,fukui,hartree}.py (new)
- tests/test_molpot/test_potentials/test_gto/{test_kernel,test_fukui,test_hartree}.py (new)
- regressions/mace-polar-port-04-gto.py (new)

## Tasks

- [x] Write failing unit tests for GTOCoulombKernel
- [x] Implement GTOCoulombKernel erf(r/(2σ))/r
- [x] Write failing unit tests for FukuiEquilibration
- [x] Implement FukuiEquilibration (multiplicity M=2S+1)
- [x] Write failing unit tests for GTOHartreeEnergy l=0
- [x] Implement GTOHartreeEnergy isolated-molecule pair sum
- [x] Add google docstrings with units
- [x] Add regression regressions/mace-polar-port-04-gto.py
- [x] Run tests + ruff

## Out of scope

k-space PBC、外场、field features、molzoo Polar 接线、checkpoint remap、官方能量对照。
