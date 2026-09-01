---
title: MACE-Polar 场耦合原语（SparseUvu / 单体场更新 / 场读出）
status: code-complete
created: 2026-08-30
slug: mace-polar-port-03-field
chain: mace-polar-port
---

# MACE-Polar 场耦合原语

## Summary

在 `molrep.interaction.polar` 落地 SparseUvu、单体场更新与场读出。原生 `cue.ir_mul`，无 e3nn、无运行时 layout 转置、无能量。

## Design

- `SparseUvuTensorProduct`：只实现 `(ℓ×ℓ→0)` 与 `(ℓ×0→ℓ)`，`path_weight=1`，除以 `√(2ℓ+1)`。
- `AgnosticEmbeddedOneBodyVariableUpdate`：三路 `cuet.Linear` 势嵌入 + SparseUvu + `RadialMLP([2 n_0e, 64, 64, 64, n_0e])` + SparseUvu 调制 + `GeneralNonLinearBiasReadout` → `charges ⊕ 2x0e`。子模块名对齐官方 remap。
- `OneBodyMLPFieldReadout`：`linear_up_q/v` + SparseUvu + `RadialMLP([2 n_0e, 128, 128, 128, 1])`。
- 放置：`molrep.interaction.polar/`（与 02 的源/混合器同包），不新建顶层。

## Files

- src/molrep/interaction/polar/uvu.py (new)
- src/molrep/interaction/polar/field.py (new)
- src/molrep/interaction/polar/__init__.py
- tests/test_molrep/test_interaction/test_polar/test_uvu.py (new)
- tests/test_molrep/test_interaction/test_polar/test_field.py (new)
- regressions/mace-polar-port-03-field.py (new)

## Tasks

- [x] Write failing unit tests for SparseUvuTensorProduct
- [x] Implement SparseUvuTensorProduct
- [x] Write failing unit tests for AgnosticEmbeddedOneBodyVariableUpdate and OneBodyMLPFieldReadout
- [x] Implement field update and field readout
- [x] Export names; google docstrings with units
- [x] Add regression regressions/mace-polar-port-03-field.py
- [x] Verify l=0 contraction 6.0 and z-rotation invariance
- [x] Run tests + ruff
- [x] Hygiene simplify on polar field files

## Out of scope

能量、molpot、checkpoint remap、e3nn、molzoo 接线。
