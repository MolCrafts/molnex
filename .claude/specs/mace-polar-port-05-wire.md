---
title: MACE-Polar 势能接线、POLAR_REMAP 与孤立分子前向
status: code-complete
created: 2026-08-30
slug: mace-polar-port-05-wire
chain: mace-polar-port
---

# MACE-Polar 势能接线

## Summary

`PolarPotential(MACEEncoder)` 接上 02–04 的块：局部骨干能量 + Fukui + U=2 场更新 + 场读出 + 孤立分子 GTO Hartree。`POLAR_REMAP` 翻译官方 cueq 键名。无 e3nn/mace-torch/graph_longrange。官方能量对照与 k 空间 PBC 不在本步。

## Domain basis

Batatia et al. arXiv:2602.19411；dump 2026-08-30：`num_recursion_steps=2`（不是 Hubbard U），σ=1.5 Å，multipole \(l_{\max}=1\)，field widths `[1.5, 3.0]`，norms `[20,20,0.5,0.5]`，`add_local_electron_energy=True`。`total_spin` 是多重度 \(M=2S+1\)。能量 \(E_0+E_{\mathrm{local}}+E_{\mathrm{electron}}+E_{\mathrm{Hartree}}\)（外场为 0）。checkpoint 仍是 e3nn；本步加载 **已 cueq 转换** 的 `state_dict`（`mace.cli.convert_e3nn_cueq` 在树外）。

## Design

- **不**把 Polar 塞进 `MACEPotential`。新 `PolarPotential` 继承 `MACEEncoder`，子模块名对齐官方（`lr_source_maps`、`layer_feature_mixer`、`fukui_source_map`、`field_dependent_charges_maps`、`local_electron_energy`）。
- `MACEPolarSpec` 补 dump GTO 旋钮（01 故意没写）。
- `GTOFieldFeatures`：孤立分子实空间有限差分，核 `erf(r/(2σ_{\mathrm{comb}}))/r`，σ_comb=`sqrt((σ_ρ²+σ_f²)/2)`。
- `POLAR_REMAP`：骨干键同 MatPES；`on_unexpected="return"`（k 空间/转置/e3nn 描述符）；`_LinearWithScalarBias` 的 `linear_mid.weight` → `linear_mid.linear.weight`。
- 未填 `nn.Parameter` 仍 raise。

## Files

- src/molpot/potentials/gto/features.py (new)
- src/molpot/potentials/gto/__init__.py
- src/molzoo/mace/spec.py
- src/molzoo/mace/polar.py (new)
- src/molzoo/mace/checkpoint.py
- src/molzoo/mace/__init__.py
- tests/test_molpot/test_potentials/test_gto/test_features.py (new)
- tests/test_molzoo/test_mace/test_polar.py (new)
- tests/test_molzoo/test_mace/test_spec.py
- tests/test_molzoo/test_mace/test_checkpoint.py (or polar tests)
- regressions/mace-polar-port-05-wire.py (new)

## Tasks

- [x] Write failing unit tests for GTOFieldFeatures and PolarPotential / POLAR_REMAP
- [x] Implement GTOFieldFeatures
- [x] Add dump GTO fields to MACEPolarSpec
- [x] Implement PolarPotential
- [x] Add POLAR_KEY_REMAP / POLAR_REMAP; export PolarPotential
- [x] Add regression regressions/mace-polar-port-05-wire.py
- [x] Run tests + ruff

## Out of scope

k 空间 PBC、外场块、官方 `.model` 能量对照、树内 e3nn→cueq 转换器。
