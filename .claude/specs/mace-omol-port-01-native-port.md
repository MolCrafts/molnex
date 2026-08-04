---
title: MACE-OMOL native port into molnex (weight-loadable, E/F faithful)
status: done
created: 2026-06-21
chain: mace-omol-port
---

# MACE-OMOL native port into molnex

## Summary

把官方 **MACE-omol-0** 基础模型（`ScaleShiftMACE`，1024 通道，r_max=6.0，l_max=3，3 层
interaction，product correlation=2，83 元素，单 head `omol`，带 total_charge/total_spin
条件化，52.7M 参数）**原生重写进 molnex**（molrep/molpot/molzoo，cuEquivariance 后端），
使其能加载官方权重并复现官方的能量/力。每个构件作为 molrep/molpot 的通用模块（**不带 MACE
前缀**），在 CPU(float64) 上逐位对照官方实现验证（0 误差），整模型加载官方权重后对官方能量/力达
**7e-7 eV / 4e-6 eV·Å**（亚微 eV，PASS）。

策略：官方 e3nn 模型经 `mace.cli.convert_e3nn_cueq` 转成 cueq 版（cuet 原语、ir_mul 布局——与
molnex 同栈，已验证复现 e3nn OMOL 至 1.5e-8 eV），以此 cueq 双胞胎作**块级参照**；molnex 每个原生
块对照对应 cueq 子模块逐位验证，权重经直接 `load_state_dict` 迁入。

## Domain basis

MACE 等变消息传递（Batatia et al., NeurIPS 2022, arXiv:2206.07697）。OMOL 变体相对标准 MACE 的
关键差异（均已忠实移植）：

- 截断 `PolynomialCutoff`（非 cosine）；Bessel 基**可学习且不归一化**（`B_n(r)=√(2/r_max)·sin(w_n r)/r`，`w_n=nπ/r_max`，无 eps）。
- interaction 为 `RealAgnosticResidualNonLinearInteractionBlock`：source/target 节点属性嵌入拼进
  radial 特征（8+1024+1024=2056）→ 带 LayerNorm 的 RadialMLP → channel-wise TP（`node ⊗ Y_{l≤3}`）→
  邻居 scatter；密度归一 `message/(ρ·β+α)`，`ρ=Σ tanh(density_mlp(e))²·cutoff`，α/β 可学习标量；
  残差 `linear_res` + 门控非线性（scalar SiLU、l>0 sigmoid 门，带 e3nn normalize2mom）+ `linear_2`；
  独立 `skip_tp`（Linear）供 product 块。
- product 为 `EquivariantProductBasisBlock`：对称收缩（**degree=2、num_elements=1 元素无关**、
  `original_mace=True`）+ Linear + skip add。
- 能量头：`NonLinearBiasReadoutBlock`（Linear→SiLU→o3.Linear(bias)→SiLU→o3.Linear(bias)，仅末层读出）+
  `AtomicEnergiesBlock`(per-element E0) + `ScaleShiftBlock`；charge/spin 经 `GenericJointEmbedding`
  加到 node_feats，`embedding_readout` 贡献加进 E0。
- edge 向量约定 `v = pos[receiver]-pos[sender]`（receiver=edge_index[1]）。

数值等价基线：官方 e3nn↔cueq 转换本身误差 1.5e-8 eV，是本对照的地板。cue 默认 `"O3"` 群与官方
`O3_e3nn` 群的 Clebsch-Gordan 基在 conv_tp / 对称收缩上差约 1e-14/op。

## Design

新增/扩展的 molrep/molpot 模块（全部纯 cuEquivariance，无 e3nn 依赖；命名无 MACE 前缀）：

| 模块 | 文件 | 说明 |
|---|---|---|
| `BesselRBF`(+`trainable`,`eps=0`,`normalize=False`) | `molrep/embedding/radial.py` | 加可学习/不归一化模式；并修 dtype 契约 float32→config.ftype |
| `PolynomialCutoff` | `molrep/embedding/cutoff.py` | 已存在，与官方一致 |
| `JointFeatureEmbedding`(+`JointFeatureSpec`) | `molrep/embedding/node.py` | charge/spin 类别嵌入+SiLU 投影，per-graph 经 batch 广播 |
| `RadialMLP` | `molrep/interaction/radial.py` | Linear→LayerNorm→SiLU 堆叠（ESEN/FairChem 风格） |
| `GatedNonlinearity` | `molrep/interaction/gate.py` | e3nn `Gate` 等价，原生 `cue.Irreps`，无参 |
| `ResidualInteraction` | `molrep/interaction/residual.py` | 完整非线性残差 interaction，`group` 可选 |
| `EquivariantProductBasis` | `molrep/interaction/product_basis.py` | 对称收缩+linear+skip，`group` 可选 |
| `NonLinearBiasReadout`(+`_ScalarO3Linear`) | `molrep/readout/scalar.py` | 复刻 o3.Linear 标量约定 `(x@W/√in)+b` |
| `AtomicReferenceEnergy` | `molpot/heads/energy.py` | per-element E0，Z 索引查表 |
| `GlobalRescale` | `molpot/heads/rescale.py` | 已存在，单 head scale/shift |

整模型 `MACEOMol`（`molzoo/mace_omol.py`）按 `ScaleShiftMACE.forward` 接线：embeddings →
3×(interaction→product) → 末层 readout → scale_shift → +E0；力经 autograd。转换器
`load_omol_state_dict` 把 cueq state_dict 映射进 `MACEOMol`（直接拷贝 + 少量 key 重命名）。

权重路径：OMOL(e3nn) → `mace.cli.convert_e3nn_cueq` → cueq state_dict → `load_omol_state_dict` →
molnex。e3nn `o3.Linear` ↔ `cuet.Linear` 权重为直接拷贝（两种布局均 0 误差）；对称收缩权重由 mace
转换器按 `["_max",".0"]` 拼接 + CG 投影完成。

## Files

- `src/molrep/embedding/{radial,cutoff,node}.py`、`src/molrep/interaction/{radial,gate,residual,product_basis}.py`、
  `src/molrep/readout/scalar.py`、`src/molpot/heads/{energy,rescale}.py`（含各 `__init__.py` 导出）
- `src/molzoo/mace_omol.py`（`MACEOMol` + `load_omol_state_dict`）
- 验证脚本 + 实现说明：`scripts/omol_port/`（`SPEC.md`、`verify_*.py`、`README.md`）
- 参照资产（非仓库代码）：`work/.mace-ref`（CPU 参照环境）、`work/mace_models/`（checkpoint、`OMOL-cueq.model`、`OMOL_REFERENCE.md`、`omol_inventory.json`）

## Tasks

- [x] CPU 参照环境 + 下载 OMOL + 架构兼容性分析
- [x] 移植 PolynomialCutoff + 可学习 Bessel（CPU 验证）
- [x] charge/spin GenericJointEmbedding + embedding_readout
- [x] RealAgnosticResidualNonLinearInteractionBlock（3 层 0 误差）
- [x] NonLinearBiasReadout + ScaleShift + E0
- [x] product 块（对称收缩 degree=2/num_elements=1）
- [x] 组装 MACEOMol
- [x] e3nn→cueq→molnex 权重转换器
- [x] 端到端能量/力对照官方（CPU）

## Testing

`scripts/omol_port/verify_*.py`（以 `.mace-ref` 的 python 跑，env `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`，
float64）：radial 7e-15 / radial_mlp 0 / e0+scaleshift 0 / joint_embed 0 / interaction(3 层) 0 /
product(3 层) 0 / readout 0 / **e2e 7e-7 eV、4e-6 eV·Å**。`verify_omol_cueq_equiv.py` 证 cueq 双胞胎
对 e3nn OMOL 达 1.5e-8 eV。

## Out of scope（follow-up，详见 `scripts/omol_port/SPEC.md`）

承接到链式后续 **`mace-omol-port-02-pipeline-integration`**：

- 逐位精确（~1e-8）：需 `O3_e3nn` 群（cuequivariance 0.10 的 `cue.Irreps.sort()` 在该群崩溃；`MACEOMol(group=)` 接口已留）。→ 02 ac-003（上游阻塞）。
- molnex TensorDict / `molpot.PotentialComposer` / `ForceDerivation` / `NeighborList` 集成；`molzoo/__init__` 惰性接线。→ 02 ac-001/ac-002。
- GPU/aarch64 + `cuequivariance-ops-cu12` 性能验证。
- 多 head、pair_repulsion、距离变换（Agnesi/Soft）、stress/virial；直连 e3nn→molnex 转换器（去 mace 依赖）。
