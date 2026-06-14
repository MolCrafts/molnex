---
title: PiNet quantization thermal-noise — final verdict synthesis (Phase B, part 3)
status: approved
created: 2026-06-01
chain: pinet-quantization-thermal-noise
---

# PiNet quantization thermal-noise — final verdict synthesis (Phase B, part 3)

## Summary

本子规格构建整个 `pinet-quantization-thermal-noise` 链路的**综合判定层**。它消费 -01 的静态诊断聚合表（判据 a、b 与静态 T_eff）和 -03 的轨迹诊断表（判据 c、d、e、f、g 以及动力学判据 h），并针对变量矩阵中的**每一个条件单元**（量化方案 × 训练精度 × 数据集 × MD 条件）渲染最终的热噪声判定。对每个单元，依据显式的通过/失败阈值评估全部 8 条判据，并将该单元分类为「可近似为热噪声」（全部 8 条在容差内通过）或「结构化非热扰动」（任一系统性违背）。当判定为非热时，**报告其实际形式**——即哪些判据失败以及由此得到的物理表征（例如「有偏 + 空间相关 ⇒ PES 畸变而非噪声」；「无偏 + 高斯但有色 τ_c≫Δt ⇒ 淬火/构型锁定噪声」；「能量漂移>0 ⇒ 注入不平衡/需要恒温器摩擦」）。输出一份人类可读报告与一张机器可读表。判定逻辑本身必须在**已知答案的合成 fixture** 上独立验证，不依赖任何真实运行。这是整个链路对「PiNet 量化误差是否为热噪声?」这一问题的最终交付答案。

## Domain basis

ΔF(x)=F_quant(x)−F_ref(x) 在给定构型下是确定性的；我们要证伪的命题是：沿轨迹 x(t) 的 ΔF(t) 表现为 Langevin 随机力。

- Eq1 Langevin: m v̇_i=F^phys_i−γ m v_i+ξ_i。
- Eq2 ⟨ξ⟩=0。
- Eq3 平稳高斯。
- Eq4 白噪声且逐自由度独立 ⟨ξ_i(t)ξ_j(t')⟩=2γ m k_B T δ_ij δ(t−t')。
- Eq5 FDT：同一 γ 同时设定耗散与噪声幅度 2γmk_BT → 正则采样。
- Eq6 仅有力而无配对的 −γv ⇒ 注入能量 d⟨½mv²⟩/dt=A d/(2m)>0 → 失控加热，**不是**恒温器。
- Eq7 ΔF 在无耗散下加入 ⇒ FDT 违背 ⇒ 长期能量漂移。
- Eq8 k_B T_eff=⟨|ΔF|²⟩·Δt/(2γ m d)，d=3N；有色情形 ⟨|ΔF|²⟩Δt→∫C(τ)dτ；报告 T_eff(γ,Δt)/T_target（∝1/γ）。
- Eq9 C(τ)=⟨ΔF(t)·ΔF(t+τ)⟩，相关时间 τ_c。
- Eq10 Einstein 扩散 D。
- Eq11 Green–Kubo D（VACF 积分）。

8 条判定判据及其失败→形式映射：

- (a) 无偏 ⟨ΔF⟩→0 [失败 ⇒ 系统性力场畸变]。
- (b) 高斯 skew/exkurt≈0 [失败 ⇒ 非高斯，到 Langevin 的映射无效]。
- (c) 白噪声 τ_c≲数个 Δt [失败 ⇒ 有色/淬火构型锁定]。
- (d) 空间/跨自由度独立，ΔF 协方差非对角≈0 [失败 ⇒ 空间相关，即 shared-weight-PTQ 的可能结果；热噪声禁止此项]。
- (e) Σ_iΔF_i=0 [失败 ⇒ 净 COM 动量注入，无恒温器可移除]。
- (f) 能量漂移 d⟨E⟩/dt≈0 [失败 ⇒ 注入不平衡/FDT 违背，按 Eq6 失控加热]。
- (g) 平稳性 [失败 ⇒ 构型锁定畸变]。
- (h) FDT 一致的 T_eff(γ,Δt) 且观测量 g(r)/VACF/D 与参考+T_eff 修正吻合 [失败 ⇒ 结构/动力学被改变]。

全部通过 → 可近似为热噪声（报告 T_eff(γ,Δt)/T_target 与所处区间）；任一系统性违背 → 带有命名形式的结构化非热扰动。

参考文献：Bussi & Parrinello PRE 75 056707 (2007) DOI 10.1103/PhysRevE.75.056707；Wu et al. JCP 2024 DOI 10.1063/5.0213811 / arXiv:2401.11427（ML 力误差作为 Langevin 噪声——先例）；Frenkel & Smit, Understanding Molecular Simulation, 2nd ed；PiNN Li et al. JCTC 2025 DOI 10.1021/acs.jctc.4c01570。

## Design

本层**不引入新的物理 kernel**，只增加阈值逻辑、分类与报告渲染。保持纯函数风格（`verb_noun`，docstring 标注单位）。

### 输入契约

- 来自 -01 的聚合表：每条件含 `F_bias`（⟨ΔF⟩，eV/Å）、`F_skew`、`F_exkurt`、`T_eff_ratio_static`、`unbiased`/`gaussian` 布尔列。
- 来自 -03 的轨迹诊断：每条件含 `tau_c`（fs）、`cov_offdiag_max`（无量纲，归一化协方差）、`sum_dF`（ΔF 之和的范数，eV/Å）、`energy_drift_slope`（eV/fs）、`stationary`（布尔）、`T_eff_gamma_dt_ratio`、`gr_delta`/`vacf_delta`/`D_delta`（相对参考的偏差）。

两表通过条件键 `(quant_scheme, trained_precision, dataset, md_condition)` 连接（inner join；缺失键报告为不完整单元，不静默丢弃）。

### 8 判据阈值表（命名常量 + 单位 + 依据）

每个阈值是显式策略决策，作为带 docstring 依据的命名常量：

| 判据 | 常量 | 含义与依据 |
|------|------|------------|
| a | `BIAS_TOL_SIGMA = 2.0` | ⟨ΔF⟩ 在 2σ 统计误差内视为无偏；σ 由每条件样本量估计。 |
| b | `SKEW_TOL = 0.2`, `EXKURT_TOL = 0.5` | \|skew\|<0.2 且 \|exkurt\|<0.5 视为近高斯（有限样本下的常用宽容带）。 |
| c | `TAU_C_DT_FACTOR = 3.0` | τ_c ≲ 3·Δt 视为白；超过则有色。 |
| d | `COV_OFFDIAG_TOL = 0.1` | 归一化协方差非对角 max<0.1 视为跨自由度独立。 |
| e | `SUM_DF_TOL_SIGMA = 2.0` | ‖Σ ΔF‖ 在 2σ 内视为动量守恒（COM 力为零）。 |
| f | `ENERGY_DRIFT_TOL = 1e-4` | \|d⟨E⟩/dt\|<1e-4 eV/fs 视为无漂移（区分 FDT 平衡与 Eq6 失控加热）。 |
| g | `STATIONARITY_P = 0.05` | -03 平稳性检验 p≥0.05 视为平稳。 |
| h | `OBSERVABLE_DELTA_TOL = 0.05` | g(r)/VACF/D 相对偏差<5% 且 T_eff(γ,Δt) 自洽视为 FDT 一致。 |

阈值常量集中在单一 `VERDICT_THRESHOLDS` 不可变结构中，便于审计与 fixture 注入（分类器接受可选的 thresholds 参数以支持测试）。

### 分类与表征逻辑

- `evaluate_criteria(cell, thresholds) -> dict[str,bool]`：返回 8 条判据的逐项 pass/fail。
- `classify_cell(criteria) -> Verdict`：全部通过 ⇒ `"thermal-noise-approximable"`；任一失败 ⇒ `"structured-non-thermal"`。
- `characterize_failure(criteria) -> str`：依据失败判据组合返回命名形式，覆盖映射表中的所有形式（a 失败⇒force-field distortion；d 失败⇒spatially-correlated PES distortion；c 失败而 a/b 通过⇒quenched/colored；e 失败⇒COM-momentum injection；f 失败⇒unbalanced injection/needs thermostat friction；g 失败⇒configuration-locked distortion；b 失败⇒non-Gaussian mapping invalid；h 失败⇒altered structure/dynamics）。当多条失败时，按物理优先级（a/d > c/g > e/f > b/h）组合命名形式（例如「biased + spatially-correlated ⇒ PES distortion, not noise」）。
- thermal 情形附带 `T_eff_gamma_dt_ratio` 与所处区间（∝1/γ）。

### 输出

- `render_report(verdicts) -> str`：人类可读 Markdown 报告，按条件分组，列出每单元的判定、失败判据、命名形式、T_eff 比值。
- `build_machine_table(verdicts) -> list[dict]`：稳定 schema 的机器可读表（每行：条件键 + 8 个布尔判据列 + `verdict` + `form` + `T_eff_ratio`），供下游消费。

### Lifecycle / ownership

无持久状态；全部纯函数。`run_verdict(static_table, traj_table, thresholds=VERDICT_THRESHOLDS) -> (report_str, machine_table)` 为顶层入口，组合上述步骤。

## Files to create or modify

- `examples/molzoo/verdict.py` (new) — 阈值常量、`evaluate_criteria`、`classify_cell`、`characterize_failure`、`render_report`、`build_machine_table`、`run_verdict`。
- `examples/molzoo/tests/test_verdict.py` (new) — 合成 fixture 上的分类器正确性测试与报告/表 schema 测试。

（注：`examples/` 在 molnex 中被 gitignore；这些文件按约定置于该工作目录但不纳入版本控制。）

## Tasks

- [ ] Write failing tests for classifier on synthetic fixtures (examples/molzoo/tests/test_verdict.py)
- [ ] Implement threshold constants VERDICT_THRESHOLDS with documented rationale and units in examples/molzoo/verdict.py
- [ ] Implement evaluate_criteria and classify_cell in examples/molzoo/verdict.py
- [ ] Implement characterize_failure mapping all 8 failure forms in examples/molzoo/verdict.py
- [ ] Write failing tests for report + machine-table schema (examples/molzoo/tests/test_verdict.py)
- [ ] Implement render_report and build_machine_table in examples/molzoo/verdict.py
- [ ] Implement run_verdict entrypoint joining -01 static and -03 trajectory tables in examples/molzoo/verdict.py
- [ ] Add docstrings per project doc style with units on every public function in examples/molzoo/verdict.py
- [ ] Verify per-condition classification across the full variable matrix on a multi-cell synthetic fixture
- [ ] Run full check + test suite

## Testing strategy

- **分类器正确性（scientific）**：clean white-Gaussian-unbiased-momentum-conserving fixture ⇒ `"thermal-noise-approximable"`；针对每条判据单独构造单一违背 fixture（a 有偏、b 非高斯、c 有色 τ_c≫Δt、d 空间相关、e Σ ΔF≠0、f 能量漂移>0、g 非平稳、h 观测量偏离）⇒ 分类为 `"structured-non-thermal"` 且 `characterize_failure` 返回正确的命名形式。
- **多重违背组合（scientific）**：biased+spatially-correlated fixture ⇒ 「PES distortion, not noise」组合形式；验证物理优先级排序。
- **报告/表 schema（code）**：`build_machine_table` 输出固定列集合且类型稳定；`render_report` 含每单元判定、失败判据与 T_eff 比值；缺失条件键报告为不完整而非静默丢弃。
- **全矩阵分类（scientific）**：多单元合成矩阵 fixture（混合 thermal 与各类 non-thermal 单元）⇒ 每单元判定与命名形式逐一正确。
- **域验证**：阈值常量与 Domain basis 的 Eq8/Eq9 及失败→形式映射一致（T_eff 比值 ∝1/γ 报告；f 失败映射到 Eq6 失控加热表述）。

## Out of scope

- MD 引擎与采样（-02）。
- 轨迹诊断 kernel：τ_c、协方差、能量漂移、平稳性、g(r)/VACF/D（-03）。
- 静态聚合：F_bias/F_skew/F_exkurt/static T_eff（-01）。
- 激活量化（activation quantization）。
- 量化感知训练（QAT）。
- 任何真实 PiNet 运行或数据生成——本层仅在合成 fixture 上验证判定逻辑。
