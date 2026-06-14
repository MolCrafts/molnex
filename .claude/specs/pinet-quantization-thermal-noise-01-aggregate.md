---
title: PiNet Quantization-as-Thermal-Noise — Phase A Aggregation + T_eff Correction
status: approved
created: 2026-06-01
chain: pinet-quantization-thermal-noise
---

# PiNet Quantization-as-Thermal-Noise — Phase A Aggregation + T_eff Correction

## Summary
本子实验完成 Phase A 的收尾：运行已有的量化扫描（`quant_study.py` / `train_pinet_sweep.py`），把跨条件（量化方案 × 训练精度 × 数据集）的单构型系综统计 ΔF/ΔE 聚合成一张诊断表，并据此回答 Phase A 的三个静态问题——量化残差是否无偏？是否高斯？其量级相对 k_BT 如何？同时修正 `quant_study.py` 中量纲错误的有效温度启发式公式，使其符合 Eq8。Phase A 使用静态探针系综（非轨迹），因此明确不做时间自相关——那是 Phase B 的工作。本子实验的产物是一张可被 `-04-verdict` 直接消费的跨条件诊断表与一个量纲正确的 `T_eff(γ,Δt)` 估计器。

## Domain basis
量化误差作为朗之万噪声的可证伪假设。沿轨迹 x(t)，残差 ΔF(x)=F_quant(x)−F_ref(x) 是构型的确定性函数；待证伪的命题是 ΔF(t) 表现得像朗之万随机力。

朗之万运动方程 (Eq1): m v̇_i = F_i^phys − γ m v_i + ξ_i(t)。
随机力定义性质：(Eq2) ⟨ξ⟩=0；(Eq3) 平稳高斯；(Eq4) 白噪声且逐自由度独立 ⟨ξ_i(t)ξ_j(t')⟩=2γ m k_B T δ_ij δ(t−t')。
涨落-耗散定理 (Eq5): 同一个 γ 同时设定耗散与噪声幅度 2γmk_BT → 保证正则采样。
(Eq6) 注入一个没有配对 −γv 拖曳的力会泵入能量：d⟨½mv²⟩/dt = A·d/(2m) > 0 → 失控加热，而非恒温器。
(Eq7) 因此量化 ΔF 在无耗散配对的情况下加入 → 违反 FDT → 除非真实恒温器的摩擦吸收它，否则预期出现长期能量漂移。

有效温度（关键修正）：`quant_study.py` 文档字符串中现有的启发式 `k_B T_eff ~ ⟨|ΔF|²⟩/(2γ·dim)` 量纲错误（得到 N²·s，而非能量——缺 Δt 与 m）。正确形式 (Eq8): k_B T_eff = ⟨|ΔF|²⟩·Δt / (2 γ m d)，其中 d = 总自由度（3N），逐自由度语义须显式说明；对有色噪声将 ⟨|ΔF|²⟩Δt 替换为 ∫C(τ)dτ（零频谱密度）。T_eff 随 1/γ 标度 → 报告 T_eff(γ,Δt)/T_target，而非裸数值。

诊断量：(Eq9) 力噪声自相关 C(τ)=⟨ΔF(t)·ΔF(t+τ)⟩，相关时间 τ_c=(1/C(0))∫C(τ)dτ；白噪声 ⇒ τ_c≲Δt，有色 ⇒ τ_c≫Δt。(Eq10) Einstein D=lim 1/(2d t)⟨|r(t)−r(0)|²⟩。(Eq11) Green–Kubo D=(1/d)∫⟨v(0)·v(t)⟩dt（VACF）。

判定准则（完整 8 条，本子实验仅覆盖静态可判定的 a/b/部分 h；c–g 需轨迹，归 `-03`/`-04`）：(a) 无偏 ⟨ΔF⟩→0 在统计误差内【首要硬门槛】；(b) 高斯 skew≈0、超额峰度≈0；(c) 白 τ_c≲几个 Δt；(d) 空间/跨自由度独立——ΔF 协方差在原子间及 x/y/z 间的非对角≈0【最可能的失败模式：PTQ 扰动共享 PiNet 权重 → 预期空间相关残差】；(e) 动量守恒 Σ_i ΔF_i = 0；(f) 能量漂移斜率 d⟨E⟩/dt ≈ 0；(g) ⟨ΔF⟩/var/C(τ) 沿轨迹平稳；(h) FDT 一致的 T_eff(γ,Δt) 与可观测量（g(r)/VACF/D）匹配参考。全部满足 → 可近似为热噪声；任一系统性违背 → 归类为结构化的非热扰动并报告其实际形式。

参考文献：Bussi & Parrinello, Phys. Rev. E 75, 056707 (2007), DOI 10.1103/PhysRevE.75.056707；Wu et al., J. Chem. Phys. (2024), DOI 10.1063/5.0213811 / arXiv:2401.11427（将 ML 力误差作为朗之万噪声处理——核心先例）；Frenkel & Smit, Understanding Molecular Simulation 2nd ed.（Green–Kubo/Einstein）；PiNN: Li et al. JCTC 2025, DOI 10.1021/acs.jctc.4c01570。

## Design
Phase A 已基本构建：`quant_study.py` 提供 `fake_quantize_tensor` / `quantize_state_dict`（6 个方案 fp16/bf16/int8/int8_pc/int4/int4_pc）、`_build_loaded`（warmup-lazy → load_state_dict，含 `_strip_compile_prefix` 处理编译 checkpoint）、`_forward_energy_forces`（fresh detached pos ⇒ 干净 F=−dE/dx）、`paired_delta`、`summarize_delta`（已产出 F_bias/F_std/F_rms/F_skew/F_exkurt/F_cos/E_bias 等标量，覆盖准则 a、b）。`train_pinet_sweep.py` 提供跨条件 checkpoint。

本子实验新增三项：
1. **T_eff 修正**：把 `quant_study.py:19` 文档字符串中的量纲错误公式替换为 Eq8，并新增一个纯函数 `t_eff_estimate(f_rms_sq, dt, gamma, mass, dof)`（逐自由度语义显式），返回 T_eff(γ,Δt)/T_target。该函数不依赖轨迹，可用 Phase-A 的 ⟨|ΔF|²⟩ 估计静态 T_eff 下界（白噪声假设下，Δt/γ 作为参数扫描）。
2. **跨条件聚合**：新增 `aggregate_phase_a.py`，读取 `train_pinet_sweep.py` 经 `src/molix` Checkpoint/TorchSaveBackend 写出的 checkpoint，对变量矩阵 {方案 6 × 训练精度 (fp32/fp64/bf16-mixed) × 数据集 (QM9/revMD17-aspirin)} 的每个单元调用 `evaluate_quantization`，汇总成一张长表（CSV + 控制台），每行一条件、每列一诊断标量，并附 Phase-A 判定列（unbiased / gaussian / T_eff_ratio）。
3. **复用映射**：模型构造一律走 `benchmarks/molzoo/bm_pinet.py` 的 `build_encoder`/`build_model`/`build_datamodule`（model_factory 契约），不另起 PiNet 构造器；力学经 `PiNetPotential.forward(td, compute_forces=True)` → {"energy","forces"}（双反向 ⇒ eager ⇒ 真 int8 与 fake-quant 一致）。

变量矩阵（本子实验静态部分）：方案 ∈ {fp16, bf16, int8, int8_pc, int4, int4_pc}；训练精度 ∈ {fp32, fp64, bf16-mixed}；数据集 ∈ {QM9, revMD17-aspirin}。MD 条件 (T/γ/Δt/length) 不在本子实验，归 `-02`。参考（ground-truth）势 = 全精度 fp64 模型；量化 = 同构型上的 fake-quant 拷贝。

放置说明：`examples/molzoo/` 与 Phase A 既有脚本同列。注意 `examples/` 被 gitignore（实验代码未纳入版本控制）——Phase A 已接受此约定；若该聚合能力需版本控制/复用，应迁至受跟踪的 `src/` 位置，否则默认留在 `examples/` 与 Phase A 保持一致。

## Files to create or modify
- /Users/roykid/work/molcrafts/molnex/examples/molzoo/quant_study.py
- /Users/roykid/work/molcrafts/molnex/examples/molzoo/aggregate_phase_a.py (new)
- /Users/roykid/work/molcrafts/molnex/examples/molzoo/tests/test_t_eff.py (new)
- /Users/roykid/work/molcrafts/molnex/examples/molzoo/tests/test_aggregate_phase_a.py (new)

## Tasks
- [ ] Write failing tests for t_eff_estimate dimensional correctness per Eq8 (examples/molzoo/tests/test_t_eff.py)
- [ ] Implement t_eff_estimate(f_rms_sq, dt, gamma, mass, dof) -> T_eff/T_target in examples/molzoo/quant_study.py
- [ ] Replace dimensionally-wrong T_eff heuristic in quant_study.py module docstring (line ~19) with Eq8
- [ ] Write failing tests for cross-condition aggregation table schema (examples/molzoo/tests/test_aggregate_phase_a.py)
- [ ] Implement aggregate_phase_a.py: load sweep checkpoints, run evaluate_quantization over the variable matrix, emit long-form CSV + console table with unbiased/gaussian/T_eff_ratio verdict columns
- [ ] Add docstrings per Python style with units (ΔF in eV/Å, T_eff in K, Δt in fs) on t_eff_estimate and aggregate entrypoint
- [ ] Verify aggregated table against a known fp64-vs-fp64 control (ΔF≡0 ⇒ F_bias and T_eff_ratio collapse to 0)
- [ ] Run full check + test suite
