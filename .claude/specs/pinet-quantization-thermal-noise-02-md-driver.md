---
title: PiNet quantization thermal noise — Phase B.1 in-process MD driver
status: approved
created: 2026-06-01
chain: pinet-quantization-thermal-noise
---

# PiNet quantization thermal noise — Phase B.1 in-process MD driver

## Summary

构建一个纯 Python/PyTorch 的进程内（in-process）分子动力学驱动器，采用 velocity-Verlet 积分 + Bussi–Parrinello 风格的 Langevin 恒温器，并附带一个可选的轻量 ASE-Calculator 外壳，包裹既有的力计算接缝 `PiNetPotential.forward(td, compute_forces=True)`。驱动器执行成对轨迹协议：先用全精度（fp64）参考势跑出参考轨迹 x(t)，在每一帧上同时评估 F_ref 与 F_quant（fake-quant 副本）得到沿轨迹的时间序列 ΔF(t)；再用量化势独立跑一条相同初始/恒温器设置的轨迹供 -03 做可观测量对比。两条轨迹都产出逐步轨迹工件（pos、vel、E、F_ref、F_quant、ΔF 及元数据），作为 -03 分析（自相关、能量漂移、T_eff）的输入。本子规范只负责积分器的正确性与工件协议，不下任何"热噪声"结论。

## Domain basis

判定核心：ΔF(x)=F_quant(x)−F_ref(x) 在给定构型下确定，证伪/证实其沿 x(t) 的时间序列 ΔF(t) 是否表现为 Langevin 随机力。

- Eq1 Langevin: m v̇_i = F^phys_i − γ m v_i + ξ_i(t)。
- Eq2 ⟨ξ⟩=0。
- Eq3 ξ 为平稳高斯过程。
- Eq4 白噪声、逐自由度独立：⟨ξ_i(t)ξ_j(t')⟩ = 2γ m k_B T δ_ij δ(t−t')。
- Eq5 涨落-耗散定理（FDT）：同一 γ 同时设定耗散与噪声幅度 2γmk_BT → 保证正则（NVT）采样。
- Eq6 注入力若无配对的 −γv 阻尼则泵入能量 d⟨½mv²⟩/dt = A·d/(2m) > 0 → 失控升温，非恒温器。
- Eq7 量化 ΔF 在无耗散伙伴时加入 → 违反 FDT → 除非恒温器摩擦吸收，否则长期能量漂移。
- Eq8（修正 T_eff，于 -01 引入）：k_B T_eff = ⟨|ΔF|²⟩·Δt/(2γ m d)，d=3N；有色噪声时 ⟨|ΔF|²⟩Δt → ∫C(τ)dτ；报告 T_eff(γ,Δt)/T_target（∝1/γ）。
- Eq9 C(τ)=⟨ΔF(t)·ΔF(t+τ)⟩，τ_c=(1/C(0))∫C(τ)dτ。
- Eq10 Einstein D = lim 1/(2dt)⟨|r(t)−r(0)|²⟩。
- Eq11 Green–Kubo D = (1/d)∫⟨v(0)·v(t)⟩dt。

8 项判定准则：(a) 无偏 (b) 高斯 (c) 白 τ_c≲few Δt (d) 空间/跨自由度独立 [最可能失效：共享权重 PTQ → 空间相关残差] (e) Σ_iΔF_i=0 (f) 能量漂移≈0 (g) 平稳 (h) FDT 一致的 T_eff 与可观测量。本子规范不拥有任何热噪声判定准则（它们需要 -03 的诊断），本子规范拥有积分器正确性，以保证轨迹可信。

参考文献：Bussi & Parrinello PRE 75 056707 (2007) DOI 10.1103/PhysRevE.75.056707（Langevin 积分器、FDT）；Wu et al. JCP 2024 DOI 10.1063/5.0213811 / arXiv:2401.11427（ML 力误差作为 Langevin 噪声——先例）；Frenkel & Smit, Understanding Molecular Simulation 2nd ed（velocity-Verlet、Green–Kubo）；PiNN Li et al. JCTC 2025 DOI 10.1021/acs.jctc.4c01570。

## Design

新增模块 `examples/molzoo/md_quant_study.py`，包含三层：

**1. 力接缝适配（force seam）。** 复用 `quant_study.py` 的 `_build_loaded`（warmup-lazy → `load_state_dict`）、`_strip_compile_prefix`、`quantize_state_dict`/`SCHEMES` 构建 ref（fp64）与 quant（fake-quant 副本）两个 `PiNetPotential`。**关键区别于 `_forward_energy_forces`**：MD 一步是"推进位置 → 重算 (E,F)"，但积分需要 LIVE 力，因此不能像 `_forward_energy_forces` 那样 `detach` 输出。适配为：以当前 `pos` 构造叶子张量（`requires_grad_(True)`）写入 TensorDict 的 `("atoms","pos")`，调用 `model(td, compute_forces=True)` 取 `out["forces"]`，仅在写入轨迹工件时再 detach。每步重建干净的输入叶子以保持 autograd 图清洁并支持 PiNet 的 double-backward。

**2. 进程内积分器（in-process integrator）。** 纯 PyTorch，与模型无关，接受任意 `force_fn: pos(N,3) → (E, F(N,3))`：
- velocity-Verlet 半步更新：v += ½ a Δt；x += v Δt；重算 F；v += ½ a Δt（a = F/m）。
- Bussi–Parrinello 风格 Langevin：在每个全步的速度更新两侧施加确定性阻尼 + 随机踢动算子 v ← c₁ v + c₂ R，其中 c₁ = exp(−γΔt)，c₂ = sqrt((1−c₁²) k_B T / m)，R 为单位方差高斯，逐自由度独立（满足 Eq4）。RNG 由显式 `torch.Generator(seed)` 控制以保证可复现。
- 力函数注入式设计：积分器对解析玩具势（谐振子 F=−k x）与 `PiNetPotential` 一视同仁，使其可脱离模型单元测试（interaction_points 风险点）。

**3. 成对轨迹协议（paired-trajectory protocol）。**
- (1) REFERENCE 运行：用 fp64 参考势 force_fn 驱动 velocity-Verlet + Langevin，得到参考轨迹 x(t)。
- (2) 沿 x(t) 每帧同时评估 F_ref 与 F_quant → 逐步 ΔF(t)=F_quant−F_ref 时间序列（这使 ΔF(t) 成为 -03 自相关的输入）。
- (3) QUANTIZED 运行：用量化势独立跑一条相同初始/恒温器（同 seed、同 T、γ、Δt、length）的轨迹，供 -03 可观测量对比。

**4. ASE-Calculator 外壳（可选）。** `PiNetCalculator(Calculator)` 实现 `get_potential_energy`/`get_forces`，内部委托同一力接缝。ASE 为可选依赖：模块顶层 try-import，缺失时进程内积分器仍是主路径；外壳仅作薄包装。

**5. 工件 schema。** 每条轨迹发出逐步记录：`pos (T,N,3)`、`vel (T,N,3)`、`E (T,)`、`F_ref (T,N,3)`、`F_quant (T,N,3)`、`dF (T,N,3)`，加元数据 `{T, gamma, dt, mass, N, dof=3N, scheme, ckpt_precision, dataset, molecule, seed, integrator="velocity-verlet+langevin-BP"}`。以单个 `.pt`（torch.save 张量字典）落盘，供 -03 直接加载。

新符号：`LangevinVerletIntegrator`、`run_reference_trajectory`、`run_quantized_trajectory`、`evaluate_delta_along_trajectory`、`TrajectoryArtifact`（frozen dataclass）、`PiNetCalculator`、`build_force_fn`。

## Files to create or modify

- examples/molzoo/md_quant_study.py (new)
- examples/molzoo/tests/test_md_driver.py (new)

## Tasks

- [ ] Write failing tests for LangevinVerletIntegrator on a harmonic oscillator (examples/molzoo/tests/test_md_driver.py)
- [ ] Implement LangevinVerletIntegrator (velocity-Verlet + Bussi–Parrinello Langevin) in examples/molzoo/md_quant_study.py
- [ ] Implement build_force_fn wrapping PiNetPotential.forward with live-leaf pos (reuse _build_loaded / quantize_state_dict) in examples/molzoo/md_quant_study.py
- [ ] Write failing tests for TrajectoryArtifact schema and paired ΔF time series (examples/molzoo/tests/test_md_driver.py)
- [ ] Implement run_reference_trajectory, run_quantized_trajectory, evaluate_delta_along_trajectory and TrajectoryArtifact in examples/molzoo/md_quant_study.py
- [ ] Implement optional PiNetCalculator ASE shim (try-import, get_potential_energy/get_forces) in examples/molzoo/md_quant_study.py
- [ ] Write failing skip-if-no-ASE test that PiNetCalculator.get_forces matches PiNetPotential.forward (examples/molzoo/tests/test_md_driver.py)
- [ ] Add module + function docstrings with units (forces eV/Å, T in K, γ in 1/fs, Δt in fs) per project doc style
- [ ] Run full check + test suite

## Testing strategy

- 积分器正确性（脱离 PiNet，解析谐振子势 F=−k x）：
  - NVE 守恒（γ=0）：总能 E=½mv²+½kx² 在 N 步内相对漂移 ≤ tol（type code/scientific）。
  - Langevin 平衡（γ>0）：长程平均动能 ⟨½mv²⟩ 收敛到目标 ½ d k_B T，等分温度落在 target ± tol（type scientific）。
  - 高斯踢动统计：随机力按 seed 复现，逐自由度方差 ≈ 2γ m k_B T / Δt 标度（type code/scientific）。
- 工件 schema（type code）：两条轨迹工件含全部键，形状 (T,N,3)/(T,)，dof=3N，ΔF=F_quant−F_ref 逐元素一致，元数据字段齐全。
- 成对协议（type code）：沿参考轨迹评估的 F_ref 与参考运行内部用的力数值一致；ΔF(t) 长度等于步数。
- ASE 外壳（type code，skip-if-no-ASE）：`PiNetCalculator.get_forces` 与 `PiNetPotential.forward(...,compute_forces=True)["forces"]` 在同一构型上逐元素一致（容差内）。
- 完整 check + 测试套件（type runtime）。

## Out of scope

- 热噪声诊断与最终判定（无偏/高斯/白/独立/T_eff/能量漂移/平稳/FDT 8 准则的 verdict）—— 属于 -03/-04。
- 外部 LAMMPS 引擎（v1 已决策为进程内积分器 + ASE 外壳；LAMMPS 仅作为已记录的未来生产选项，因 PiNet 力经 PyTorch autograd double-backward，外部引擎需重型 plugin-force 回调）。
- 激活量化（activation quantization）、量化感知训练（QAT）。
- NPT / 应力（stress）系综。
- 跨 scheme×ckpt×dataset×MD 参数的完整变量矩阵扫描运行（驱动器需支持参数化，但批量扫描编排不在本子规范）。
