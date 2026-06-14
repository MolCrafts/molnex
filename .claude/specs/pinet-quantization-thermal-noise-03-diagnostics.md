---
title: PiNet quantization thermal-noise — trajectory diagnostics suite (Phase B.2)
status: approved
created: 2026-06-01
chain: pinet-quantization-thermal-noise
---

# PiNet quantization thermal-noise — trajectory diagnostics suite (Phase B.2)

## Summary

本子规格构建轨迹诊断模块 `md_diagnostics.py`，消费 -02 阶段产出的配对轨迹工件（逐步 pos/vel/E/F_ref/F_quant/ΔF 加上元数据 T,γ,Δt,mass,N,dof），计算量化误差作为 Langevin 热噪声假设的动力学判据。本模块拥有 8 项判据中的 (c) 时间自相关 / τ_c、(d) 跨原子与跨 x/y/z 分量协方差独立性、(e) 逐帧动量守恒、(f) 能量漂移斜率、(g) 平稳性，以及 (h) 的动力学部分（colored-noise 形式的 T_eff、g(r)、VACF、Einstein/Green–Kubo 扩散系数 D 的量化-参考对比）。每个核函数都必须在具有已知解析答案的合成 fixture 上验证（白噪声 vs AR(1)、独立 vs 相关、零和 vs 净力、平直 vs 漂移、已知 D 的随机游走），使整套诊断的正确性独立于任何真实 PiNet 轨迹即可证明。

## Domain basis

ΔF(x)=F_quant(x)−F_ref(x)，在给定构型下确定性；待证伪命题：沿轨迹 x(t) 的 ΔF(t) 表现为 Langevin 随机力。

- Eq1 朗之万方程：m v̇_i = F^phys_i − γ m v_i + ξ_i。
- Eq2 零均值：⟨ξ⟩ = 0。
- Eq3 平稳高斯过程。
- Eq4 白噪声且逐自由度独立：⟨ξ_i(t) ξ_j(t')⟩ = 2 γ m k_B T δ_ij δ(t−t')。
- Eq5 涨落耗散定理（FDT）：同一 γ 同时设定耗散与噪声幅 2γmk_BT → 正则采样。
- Eq6 有随机力但无配对 −γv → 泵入能量 d⟨½mv²⟩/dt = A d/(2m) > 0 → 加热而非控温。
- Eq7 有 ΔF 但无耗散 → FDT 违反 → 长期能量漂移。
- Eq8 k_B T_eff = ⟨|ΔF|²⟩ · Δt / (2 γ m d)，d = 3N；colored-noise 修正：⟨|ΔF|²⟩Δt → ∫C(τ)dτ；报告 T_eff(γ,Δt)/T_target（∝ 1/γ）。
- Eq9 C(τ) = ⟨ΔF(t)·ΔF(t+τ)⟩，τ_c = (1/C(0)) ∫C(τ)dτ。
- Eq10 Einstein：D = lim_{t→∞} 1/(2 d t) ⟨|r(t)−r(0)|²⟩。
- Eq11 Green–Kubo：D = (1/d) ∫⟨v(0)·v(t)⟩dt。

8 项判据：(a) 无偏 ⟨ΔF⟩→0；(b) 高斯 skew/exkurt≈0；(c) 白噪声 τ_c ≲ few Δt；(d) 空间/跨自由度独立 — ΔF 协方差非对角≈0【最可能失败：共享权重 PTQ → 空间相关残差】；(e) Σ_i ΔF_i = 0；(f) d⟨E⟩/dt ≈ 0；(g) 平稳性；(h) FDT 一致的 T_eff 与观测量 g(r)/VACF/D 匹配参考。全部满足 → 可近似为热噪声；任一系统性违反 → 结构化的非热扰动，报告其形式。

参考文献：Bussi & Parrinello, PRE 75, 056707 (2007), DOI 10.1103/PhysRevE.75.056707；Wu et al., JCP 2024, DOI 10.1063/5.0213811 / arXiv:2401.11427；Frenkel & Smit, Understanding Molecular Simulation 2nd ed.（Green–Kubo/Einstein）；PiNN, Li et al., JCTC 2025, DOI 10.1021/acs.jctc.4c01570。

## Design

模块 `md_diagnostics.py` 沿用 `quant_study.py` 的纯函数风格：公开函数 `verb_noun`，内部辅助 `_prefixed`，全部约简用 double 精度，docstring 标注单位（ΔF eV/Å，D Å²/fs，T_eff K，Δt fs，γ 1/fs）。输入统一来自 -02 轨迹工件（视为一个映射：`pos`/`vel`/`E` 形状 (T_steps,N,3) 或 (T_steps,)、`F_ref`/`F_quant`/`dF` 形状 (T_steps,N,3)，元数据标量 T,γ,Δt,mass,N,dof）。本模块不重新计算 ΔF，也不运行 MD。静态判据 (a)(b) 复用 `quant_study.summarize_delta`，Eq8 标量入口复用 -01 的 `t_eff_estimate`；时间序列/协方差/RDF/VACF/MSD/扩散核全部新建。

每个核（公式 → -02 工件输入 → 合成 fixture 验证）：

- **(c) `autocorr_dF` (Eq9)**：对 ΔF 时间序列计算 C(τ)=⟨ΔF(t)·ΔF(t+τ)⟩（按原子·分量求点积再平均），归一化 τ_c=(1/C(0))∫C(τ)dτ（梯形积分 × Δt）。白噪声 ⇒ τ_c ≲ Δt；有色 ⇒ τ_c ≫ Δt。Fixture：时间上 i.i.d. 的 ΔF → τ_c ≈ 0 量级的 Δt；AR(1) 序列 ΔF_t = φ ΔF_{t-1}+ε → τ_c ≈ Δt·(1+φ)/(1−φ) 的解析值。
- **(d) `crossdof_covariance`（δ_ij 检验）**：把每帧 ΔF 展平为长度 3N 的向量，构造跨原子且跨 x/y/z 分量的协方差矩阵，按时间平均；返回非对角能量占比（off-diag Frobenius / total Frobenius）。Fixture：各自由度独立 ΔF → 非对角 ≈ 0；注入跨原子相关 → 被标记。
- **(e) `momentum_residual` (Newton 3rd)**：逐帧 Σ_i ΔF_i，返回净力幅的时间统计（应为 0）。Fixture：零和构造 → pass；注入净力 → 标记。
- **(f) `energy_drift_slope`**：对 E(t) 线性最小二乘拟合，返回斜率 d⟨E⟩/dt（单位 eV/fs）及其标准误。Fixture：平直 E → 斜率 0；注入线性漂移 → 准确恢复。
- **(g) `stationarity_blocks`**：把轨迹按时间三等分，逐块计算 ⟨ΔF⟩、var、τ_c，返回块间相对漂移；若漂移超阈则标记非平稳。Fixture：平稳序列 → 块间一致；分段方差递增 → 标记。
- **(h, 动力学部分)**：
  - `t_eff_colored` — 当 τ_c ≫ Δt 时，按 Eq8 的 colored-noise 分支把 ⟨|ΔF|²⟩Δt 替换为 ∫C(τ)dτ，复用 -01 的 `t_eff_estimate` 作为白噪声标量入口，返回 T_eff(γ,Δt)/T_target。
  - `rdf` (g(r)) — 对一条轨迹的 pos 直方图化得到径向分布函数；量化运行 vs 参考运行对比。Fixture：解析可算的稀疏构型做 sanity。
  - `vacf` & `diffusion_einstein` (Eq10) & `diffusion_green_kubo` (Eq11) — 由 vel 计算速度自相关，由 pos 计算 MSD；D 经两条独立路径估计。Fixture：已知 D 的随机游走（MSD 路径）与理想气体玩具（两路径应一致）。

公开聚合入口 `diagnose_trajectory(artifact, *, gamma, dt, ...) -> dict[str, float | dict]`，把上述核的结果汇成判据 c–h 的诊断标量字典（不做最终裁决——裁决在 -04）。

## Files to create or modify

- examples/molzoo/md_diagnostics.py (new)
- examples/molzoo/tests/test_md_diagnostics.py (new)

注意：`examples/` 在本仓库 gitignored —— 这两个文件不会被 git 跟踪，需在合并工作流中显式说明（不会出现在 PR diff 中）。

## Tasks

- [ ] Write failing tests for time-series + covariance kernels on synthetic fixtures: white/AR(1) for autocorr_dF, independent/cross-correlated for crossdof_covariance (examples/molzoo/tests/test_md_diagnostics.py)
- [ ] Implement autocorr_dF (Eq9 C(τ)/τ_c) and crossdof_covariance (δ_ij off-diagonal) in examples/molzoo/md_diagnostics.py
- [ ] Write failing tests for momentum_residual (zero-sum vs net), energy_drift_slope (flat vs drift), stationarity_blocks (stationary vs drifting) on synthetic fixtures
- [ ] Implement momentum_residual (e), energy_drift_slope (f), and stationarity_blocks (g) in examples/molzoo/md_diagnostics.py
- [ ] Write failing tests for t_eff_colored (Eq8 colored branch) and diffusion (random-walk known-D, ideal-gas Einstein-vs-Green–Kubo agreement) plus rdf/vacf shape sanity
- [ ] Implement t_eff_colored (reusing -01 t_eff_estimate), rdf, vacf, diffusion_einstein (Eq10), diffusion_green_kubo (Eq11) in examples/molzoo/md_diagnostics.py
- [ ] Implement diagnose_trajectory aggregator wiring kernels + reused summarize_delta/t_eff_estimate into a criteria c–h diagnostic dict, with docstring units
- [ ] Run full check + test suite

## Testing strategy

- 合成 fixture（核心 — 每个核独立可证）：
  - (c) 时间 i.i.d. ΔF → τ_c ≈ Δt 量级；AR(1) 序列 → τ_c 命中解析值 Δt·(1+φ)/(1−φ)（相对容差）。
  - (d) 跨自由度独立 ΔF → 协方差非对角占比 ≈ 0；注入跨原子相关 → 占比显著 > 0 被标记。
  - (e) 零和帧 → 净力幅 ≈ 0；注入恒定净力 → 准确恢复其幅值。
  - (f) 平直 E(t) → 斜率 ≈ 0；注入已知斜率 → 拟合恢复斜率与符号。
  - (g) 平稳序列 → 三块统计一致（漂移 ≈ 0）；分段方差递增 → 块间漂移被标记。
  - (h) 已知 D 的随机游走 → diffusion_einstein 命中 D（相对容差）；理想气体玩具 → Einstein 与 Green–Kubo 估计在容差内一致；rdf/vacf 形状与归一化 sanity。
- Happy path：完整 -02 形状的合成工件经 diagnose_trajectory → 返回含 c–h 全部键的字典。
- Edge cases：单帧轨迹 / 单原子 / τ_c 积分窗超出轨迹长度（截断而非越界）。
- 领域验证：上述每条合成-fixture 断言即对应 Eq8–Eq11 的数值正确性检验。

## Out of scope

- MD 引擎、轨迹生成与配对工件的产出 —— 属 -02。
- 跨条件（γ、scheme、温度）的最终裁决/verdict 合成 —— 属 -04。
- 静态判据 (a) 无偏与 (b) 高斯性的实现 —— 已在 -01 经 `summarize_delta` 完成（本模块仅复用）。
- 激活量化与量化感知训练（QAT）。
