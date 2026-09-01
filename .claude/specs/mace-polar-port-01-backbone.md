---
title: 泛化 MACE 骨干使 Polar-1-M 走同一套 MACEEncoder
status: code-complete
created: 2026-08-30
slug: mace-polar-port-01-backbone
chain: mace-polar-port
---

# 泛化 MACE 骨干使 Polar-1-M 走同一套 MACEEncoder

## Summary

把 MACE-POLAR-1-M 的局部骨干接到现有 `MACEEncoder`：残差栈不再绑定 `MACEOMolSpec`，新 `MACEPolarSpec` 钉住 2026-08-30 官方 dump 的骨干超参。Polar 走 `interaction="residual"` + `conditioning="none"` + `keep_last_layer_irreps=True`（末层完整 `hidden`）+ `readout="per_layer"`（`LinearReadout` 再 `NonLinearBiasReadout`）。禁止另起 Polar 编码器。本步不计算静电极化、不加载官方能量。

## Domain basis

MACE 等变消息传递（Batatia et al., NeurIPS 2022, arXiv:2206.07697）。单位：长度 Å，能量 eV。

**Polar-1-M 骨干以 2026-08-30 官方 `MACE-POLAR-1-M.model` dump 为准**（stub unpickler；覆盖 PolarMACE 类默认与论文 SI 若冲突）：

- `r_max=6` Å，`num_interactions=2`，83 元素 \(Z=1\ldots 83\)
- `hidden = 512x0e+512x1o`（dim 2048），**不是** OMOL 的 `_irreps(l_max-1, C)`（\(l_{\max}=3\) 会给出 dim 4608）
- 第 0 层 `linear_up`/`skip_tp` numel `512²`；第 1 层 `linear_up` numel `131072 = 512·128 + 512·128` → mid-edge **`128x0e+128x1o`**（`max_hidden_l=1`，不是 `l_max-1=2`）
- `keep_last_layer_irreps=True`，`apply_cutoff=False`，无 pair_repulsion、无 joint_embedding
- 读出：`readouts.0` `LinearReadoutBlock`（weight 512），`readouts.1` **`NonLinearBiasReadoutBlock`**（`mlp_dim=16`，`linear_2.bias` 标量）
- `cueq_config=None`（e3nn checkpoint；转换属 05）
- `E0[H]=-15.99000358581543` eV（fp32）

静电 dump（`U=2`，σ=1.5 Å，multipole \(l_{\max}=1\)，field widths `[1.5, 3.0]` Å，`add_local_electron_energy=True`，`include_electrostatic_self_interaction=True`）只记录，01 不进 spec 字段。

## Design

**不新增 PolarEncoder。** 骨干仍是 `MACEEncoder`。`MACEPolarSpec` 是 `MACESpec` 子类，**不是** `MACEOMolSpec` 子类。

**残差与 OMOL 解耦。** 未导出 mixin `MACEResidualMixin`：`edge_channels` + `l_max >= 1`。`MACEOMolSpec` 与 `MACEPolarSpec` 混入它。`MACEEncoder` 残差分支：`edge_channels = getattr(spec, "edge_channels", None)`，缺则 `TypeError` 点名 `edge_channels`（不得把残差等同于 OMOL）。

**残差 irreps（dump 修正）。** `hidden_l = getattr(spec, "max_hidden_l", l_max - 1)`。OMOL 无该字段 → 仍为 `l_max-1`。Polar `max_hidden_l=1`：

```
hidden = _irreps(hidden_l, C)
edge_irr = [feat0] + [_irreps(hidden_l, edge_channels)] * (n-1)
```

hidden 与 mid-edge **共用** `hidden_l`（建筑师 🔴：禁止留下无名的 `l_max-1` 边日程）。

**末层。** `keep_last_layer_irreps` **只写在 `MACEPolarSpec`**（默认 `True`），不进 `MACESpec`。Encoder：`last = hidden if getattr(spec, "keep_last_layer_irreps", False) else feat0`。

**读出。** `per_layer` 非末层 `LinearReadout`。末层：`NonLinearBiasReadout` 当 `getattr(spec, "last_readout_bias", False)`（Polar dump），否则现有 `NonLinearReadout`（MatPES）。OMOL `readout="final"` 仍是单个 `NonLinearBiasReadout(feat0)`。

**`MACEPolarSpec` dump 默认。** `r_max=6.0`，`num_bessel=8`，`l_max=3`，`num_features=512`，`max_hidden_l=1`，`num_interactions=2`，`correlation=3`，`mlp_dim=16`，`edge_channels=128`，`keep_last_layer_irreps=True`，`last_readout_bias=True`，`interaction="residual"`，`readout="per_layer"`，`distance_transform="none"`，`pair_repulsion="none"`，`conditioning="none"`。`atomic_numbers`/`atomic_energies` 必填。`num_interactions < 2` 拒绝。不写 `avg_num_neighbors` / GTO 旋钮。

**Reuse decision**

- generalize `MACEEncoder` residual `TypeError` 与 `hidden_sched` / mid-edge 日程
- generalize 末层读出类选择（`last_readout_bias`）
- reuse `ResidualInteraction`、`EquivariantProductBasis`、`LinearReadout`、`NonLinearBiasReadout`、`NonLinearReadout`
- new `MACEPolarSpec` + private `MACEResidualMixin`（第二份残差 spec，提取共享 `edge_channels`）
- deferred GTO / Fukui / remap / `from_checkpoint` / 能量对照 → 链 02–05

## Files to create or modify

- src/molzoo/mace/spec.py
- src/molzoo/mace/encoder.py
- src/molzoo/mace/__init__.py
- tests/test_molzoo/test_mace/test_spec.py
- tests/test_molzoo/test_mace/test_encoder.py
- tests/test_molzoo/test_mace/conftest.py
- regressions/mace-polar-port-01-backbone.py (new)

## Tasks

- [x] Write failing unit tests for MACEPolarSpec dump defaults (tests/test_molzoo/test_mace/test_spec.py → TestMACEPolarSpec)
- [x] Write failing unit tests for residual decoupling and Polar last-layer irreps (tests/test_molzoo/test_mace/test_encoder.py → TestMACEEncoder); tiny Polar fixtures in conftest.py
- [x] Implement MACEResidualMixin and MACEPolarSpec in src/molzoo/mace/spec.py; export MACEPolarSpec from src/molzoo/mace/__init__.py
- [x] Generalize MACEEncoder residual hidden_l/edge schedule, keep_last_layer_irreps, and last_readout_bias in src/molzoo/mace/encoder.py
- [x] Add docstring per google with units on MACEPolarSpec
- [x] Add regression example regressions/mace-polar-port-01-backbone.py (public API; hard-coded goldens; no third-party runtime)
- [x] Verify MatPES/OMOL last-layer scalars still hold
- [x] Run full check + test suite

## Testing strategy

- 镜像：`test_spec.py` / `test_encoder.py`。单函数。`test_variants.py::test_last_layer_keeps_scalars_only` 不改断言。
- tiny Polar：`l_max=2`，`num_features=16`，`max_hidden_l=1`，`num_interactions=2`，`edge_channels=8`，`use_fallback=True`。末层宽 **64**（`16×(1+3)`）。MatPES/OMOL tiny 末层宽 **16**。
- `MACESpec(interaction="residual")` 无 `edge_channels` → `TypeError` 匹配 `edge_channels`。`isinstance(MACEPolarSpec(...), MACEOMolSpec) is False`。
- Polar `readouts[0]` `LinearReadout`，`readouts[1]` `NonLinearBiasReadout`；`joint_embedding is None`。
- 回归硬编码 dump：`r_max=6.0`，`num_features=512`，`max_hidden_l=1`，`E0[H]=-15.99000358581543`，`83*512=42496`，tiny Polar 64 / tiny MatPES 16。注释 dump 日期 2026-08-30。无能量对照。

## Out of scope

- Polar 势能 / 力 / 官方能量 parity（05）
- GTO、Fukui、checkpoint remap、e3nn→cueq
- 把 `keep_last_layer_irreps` 放进 `MACESpec` 基类
- 另起 `PolarEncoder` 或让 Polar 继承 `MACEOMolSpec`
- 改变 MatPES/OMOL 默认末层为非标量
