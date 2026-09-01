---
title: MACE-POLAR molrep 表示块（out_mul 读出、自旋源、层混合）
status: code-complete
created: 2026-08-30
slug: mace-polar-port-02-blocks
chain: mace-polar-port
---

# MACE-POLAR molrep 表示块

## Summary

泛化 `NonLinearBiasReadout.out_mul`（默认 1，Fukui 用 2），新增 `EnvironmentDependentSpinSourceBlock` 与 `MultiLayerFeatureMixer`。无能量、无 molpot。

## Design

- `NonLinearBiasReadout(..., out_mul=1)` 保持 OMOL 键名。
- 自旋源：裸 `cuet.Linear`，`ir_mul`，两份拼接 `Y_{lmax}`，`view (N, 2, (l+1)^2)`。
- 混合器：`ModuleList` 个 `cuet.Linear`，对层轴求和。

## Files

- src/molrep/readout/mace.py
- src/molrep/interaction/polar/{__init__,source,mixer}.py (new)
- tests/test_molrep/test_readout/test_mace.py
- tests/test_molrep/test_interaction/test_polar/ (new)
- regressions/mace-polar-port-02-blocks.py (new)

## Tasks

- [x] Write failing unit tests for out_mul / spin source / mixer
- [x] Implement out_mul, EnvironmentDependentSpinSourceBlock, MultiLayerFeatureMixer
- [x] Add google docstrings with units
- [x] Add regression regressions/mace-polar-port-02-blocks.py
- [x] Run unit tests + ruff

## Out of scope

SparseUvu / 场更新（03）；GTO/Fukui 能量（04）；molzoo 接线（05）。
