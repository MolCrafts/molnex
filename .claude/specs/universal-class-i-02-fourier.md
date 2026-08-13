---
title: Universal Class-I：真二面角笛卡尔傅里叶坐标 (a_n, b_n)
status: approved
created: 2026-08-13
grilled: true
---

# Universal Class-I：真二面角笛卡尔傅里叶坐标 (a_n, b_n)

## Summary

将 Class-I 真二面角的可训练坐标从极坐标（softplus 势垒 \(k\) + 自由相位 \(\gamma\)）改为笛卡尔傅里叶振幅 \((a_n, b_n)\)（默认 \(n=1\ldots 6\)），只在 IR / `to_potentials` / `project` 导出时映回既有 \((k,\gamma,\mathrm{periodicity},\mathrm{idivf})\) 表，使 `ProperTorsionPeriodic` 能量公式与 `ClassicalIMM.project(OPLSForceField)` 的 fail-loud 契约保持不变。Improper 的 Universal-v0 默认改为周期项 \(n=1,2\)（中心仍为 `atomi`）；谐波 improper 仍可显式打开，但不再是默认。Bond / Angle 头的物理约束（\(k>0\)、\(r_0>0\)、\(0<\theta_0<\pi\)）原样复用。单位仍为 `CLASS_I_CANONICAL`（kcal/mol、Å、rad）。本 spec 不改 ChemEncoder，头只消费特征张量，因此与今日的 ChemEncoder 以及即将落地的 `universal-class-i-01-encoder` 都能接线。

## Domain basis

训练坐标（无恒定项、无生成热 \(c_0\)）

\[
E_\varphi=\sum_{n=1}^{6}\bigl[a_n\cos(n\varphi)+b_n\sin(n\varphi)\bigr].
\]

导出 / IR / 既有核（OpenMM §19.4 / SMIRNOFF / AMBER）

\[
E=\sum_n\frac{k_n}{s}\bigl[1+\cos(n\varphi-\gamma_n)\bigr],
\qquad
k=\sqrt{a^2+b^2},\quad
\gamma=\mathrm{atan2}(b,a),\quad
s=\mathrm{idivf}.
\]

三角恒等式（核相对 \(E_\varphi\) 多一个与 \(\varphi\) 无关的 \(\sum k_n/s\)，力不受影响；Class-I **不能**用表参数拟合生成热，故 \(c_0\) **不是**表字段）

\[
k\bigl[1+\cos(n\varphi-\gamma)\bigr]
=k+(k\cos\gamma)\cos(n\varphi)+(k\sin\gamma)\sin(n\varphi)
=k+a\cos(n\varphi)+b\sin(n\varphi).
\]

正变换：\(a_n=k\cos\gamma\)，\(b_n=k\sin\gamma\)。`idivf` 缺省为 1，已有值则原样保留。

极坐标 \((k,\gamma)\) 在 \(\gamma\approx\pm\pi\) 处有 \(2\pi\) 割缝：插值相位会凭空造出势垒。Wang 2022 把 \(\gamma\) 钉在 0 并用有符号 \(K\) 规避割缝，但无法表示 \(\gamma\notin\{0,\pi\}\)。笛卡尔 \((a,b)\) 连续，且能表示任意相位。

OPLS / LAMMPS `dihedral_style opls`

\[
E=\tfrac12\sum_{n=1}^{4}K_n\bigl[1+(-1)^{n+1}\cos(n\varphi)\bigr]
\]

**仅当** \(b_n\approx 0\) 且 \(n\le 4\) 时可投影：\(K_{\mathrm{odd}}=2a_n\)，\(K_{\mathrm{even}}=-2a_n\)。否则 `project` fail-loud。**不得**声称 \((a_n,b_n)\)「就是 OPLS」。

Improper 没有单一典范形式。本仓库保持 molnex **center-first**（`atomi` = 中心）。Universal-v0 训练周期项 \(n=1,2\)（Takaba 2023）。若仍使用谐波 improper，角度差必须走 \(\mathrm{atan2}(\sin\Delta,\cos\Delta)\) 包裹（既有 `_dihedral_phi` 已如此）；本 spec 不改谐波核能公式。

单位：`CLASS_I_CANONICAL` — \(a_n,b_n,k\) 为 kcal/mol，\(\gamma\) 为 rad，长度 Å。经 `from molix.units import units` 使用 pint，不重写换算浮点。

参考文献：arXiv:2010.01196；arXiv:2307.07085；OpenMM User Guide §19.4；LAMMPS `dihedral_style opls`。

硬编码对照（float64；回归与单测共用）

| 量 | 值 |
|---|---|
| \((k,\gamma)=(2,0)\to(a,b)\) | \((2.0,\ 0.0)\) |
| \((k,\gamma)=(1.5,\pi)\to(a,b)\) | \((-1.5,\ 0.0)\) |
| \((a,b)=(0.6,0.8)\to(k,\gamma)\) | \((1.0,\ \mathrm{atan2}(0.8,0.6)=0.9272952180016123)\) |
| \(\varphi=\pi/3,n=2,k=2,\gamma=\pi/4\) 的核能量 | \(2.5176380902050415\) |
| 映射 atol | \(10^{-12}\) |
| 能量 atol | \(10^{-10}\) |

## Design

本 spec 只推广两个已有类型，并在三条导出缝上把 \((a,b)\) 收成既有余弦表。不新增 ForceField 门面、不新增 `to_molpy`、不从 head 写引擎记录、不改键能公式。

**训练坐标（`ProperTorsionParamHead`）**

- 默认 `n_terms=6`，`periodicity=(1,2,3,4,5,6)`。调用方可缩短，但 Universal-v0 默认覆盖到 \(n=6\)。
- MLP 输出 \(2T\) 个**无约束**通道，切成 `a`、`b`，形状 `(N, T)`。禁止 softplus-\(k\) + 自由相位作为训练坐标；删除 `min_k`（对笛卡尔振幅无意义）。
- `forward` 返回 `{"a", "b", "periodicity", "idivf"}`。`idivf` 为 `default_idivf`（默认 1.0）的 `(N,)` 向量。不返回 `k` / `phase`，不返回 `c0`。
- 头仍是纯 MLP：端点对称由调用方的序无关特征保证（与 Bond/Angle 相同）。只吃 `(N, D)` 特征张量，不读 batch / 拓扑 / ChemEncoder。
- 轻 L2 正则若需要，属于后续 loss spec，不写进 head。

**IR 存储（`ProperTorsionBag`）**

- 字段仍仅为 `k, phase, periodicity, idivf`。**不加 `c0`，不加 `a`/`b` 存盘字段。**
- 增加语义不同的交替构造子 `ProperTorsionBag.from_cartesian(a, b, periodicity, idivf=None)`：`k = hypot(a, b)`，`phase = atan2(b, a)`；`idivf` 缺省为全 1。零振幅约定 \(\mathrm{atan2}(0,0)=0\)。
- 修订同文件中 `ImproperPeriodicBag` 过时的「SMIRNOFF trefoil / center at row 1」文档句，使其与 molnex center-first（`atomi` / 核 `improper_index` 第 0 行 = 中心）一致。不改 improper bag 字段，不改核。

**导出缝（转换发生在这里，而不是训练坐标里）**

`ForceField.params("propers")` 在训练期可以持有 `{a, b, periodicity, idivf}`。下列三条路径把笛卡尔表收成余弦表后再走旧逻辑：

1. `ForceField._propers_to_ir`：若同时给出 `(a,b)` 与 `(k,phase)` → `ValueError`；若有 `(a,b)` → `ProperTorsionBag.from_cartesian`；既有 `k1..k4` 与 `(k,phase)` 分支不变。
2. `Potentials._make_proper`：同样先收成 bag，再构造**未改公式**的 `ProperTorsionPeriodic`。
3. `ClassicalIMM._propers_to_opls`：先 `from_cartesian`（若源是 `(a,b)`），再走**现有** fail-loud：\(n\notin\{1,2,3,4\}\) 且 \(|k|>\mathrm{atol}\) → `ValueError`（匹配 `n=`）；\(\lvert k\sin\gamma\rvert>\mathrm{atol}\) → `ValueError`（匹配 sin / phase）。OPLS 流形上 \(K_{\mathrm{odd}}=2a_n\)、\(K_{\mathrm{even}}=-2a_n\)。

手工写入的 `(k,phase)` 表（现有 `_OplsLikeProperHead`、回归 `learnable-classical-ff-project-opls.py`）继续有效。

**Improper Universal-v0 默认**

- `ImproperParamHead`：**复用类型**，只改默认值 `include_harmonic=False`、`include_periodic=True`、`n_terms=2`、`periodicity=(1, 2)`。周期支路仍是 softplus-\(k\) + 自由相位（**不当**笛卡尔）。谐波支路仍可通过关键字打开。
- `ClassIForceField`：**不**在 `__init__` 自动 `def_improperstyle`，以免 `project(OPLSForceField())` 在无 improper 表时被「缺参」打掉。Universal-v0 调用方在挂 improper 头时显式 `def_improperstyle("periodic")`；谐波则 `def_improperstyle("harmonic")`。
- 拓扑：center-first（`atomi`）。OpenFF trefoil 平均是后续 export-only，本 spec 不做。

**不动的物理约束**

`BondParamHead` / `AngleParamHead` 保持 softplus \(k>0\)、\(r_0>0\)、\(\theta_0\in(0,\pi)\)。本 spec 不得改它们的构造子、`forward` 或单测断言。

**生命周期**

`ClassicalIMM.forward` 仍把 head 字典写入 `ForceField.with_params(propers=...)`；能量仍是 `ff.to_potentials().calc_energy`；投影仍是 `model.project(OPLSForceField())`。Head 不写 LAMMPS/OpenMM 记录。

### Reuse decision

- generalize `ProperTorsionParamHead` — 训练坐标改为无约束 \((a_n,b_n)\)，默认 \(n=1\ldots 6\)
- generalize `ProperTorsionBag` — 增加 `from_cartesian`；存储仍为 \((k,\gamma,\mathrm{periodicity},\mathrm{idivf})\)
- reuse `BondParamHead`
- reuse `AngleParamHead`
- reuse `ImproperParamHead` — 仅改 Universal-v0 默认（periodic \(n=1,2\)）；不改谐波/周期实现
- reuse `ClassicalIMM` — 只在 `_propers_to_opls` 入口收 \((a,b)\)，OPLS 代数与 fail-loud 不重写
- reuse `ClassIForceField` — 不新增门面，不改 `__init__` style 集合
- reuse `PotentialIR` — bag 字段集不变
- reuse `BondHarmonic`
- reuse `ProperTorsionPeriodic` — 不改 \(E=\sum(k/s)[1+\cos(n\varphi-\gamma)]\)
- reuse `LJParameterHead`
- reuse `AtomicChargeModel`
- reuse `EEM`
- reuse `ChemEncoder` / contexts — 本 spec 不改；头只吃特征张量
- reuse `Condenser` / `ForceFieldCompiler` / losses — 本 spec 不改
- GCBlock pattern — N/A（本 spec 不引入、不调用）

## Files to create or modify

- `src/molpot/composition/mm_heads.py`
- `src/molpot/ir/bags.py`
- `src/molpot/forcefield/container.py`
- `src/molpot/forcefield/classical_imm.py`
- `src/molpot/forcefield/potentials.py`
- `tests/test_molpot/test_composition/test_mm_heads.py`
- `tests/test_molpot/test_ir/test_bags.py`
- `tests/test_molpot/test_forcefield/test_to_ir.py`
- `tests/test_molpot/test_forcefield/test_classical_imm.py`
- `regressions/learnable-classical-ff-03-mm-heads.py`
- `regressions/universal-class-i-02-fourier.py` (new)

## Tasks

- [ ] Write failing unit tests for ProperTorsionParamHead Cartesian output and ImproperParamHead Universal-v0 defaults (tests/test_molpot/test_composition/test_mm_heads.py → TestProperTorsionParamHead, TestImproperParamHead)
- [ ] Write failing unit tests for ProperTorsionBag.from_cartesian mapping and shape guards (tests/test_molpot/test_ir/test_bags.py → TestProperTorsionBag)
- [ ] Write failing unit tests for Cartesian to_ir / to_potentials / project fail-loud (tests/test_molpot/test_forcefield/test_to_ir.py → TestForceFieldToIR; tests/test_molpot/test_forcefield/test_classical_imm.py → TestProject)
- [ ] Generalize ProperTorsionParamHead in src/molpot/composition/mm_heads.py to unconstrained (a, b) with default n=1..6; set ImproperParamHead defaults to periodic n=1,2; update regressions/learnable-classical-ff-03-mm-heads.py key asserts
- [ ] Generalize ProperTorsionBag in src/molpot/ir/bags.py with from_cartesian (k=hypot(a,b), phase=atan2(b,a), default idivf=1, no c0) and fix ImproperPeriodicBag center-first docstring
- [ ] Wire a,b → (k, γ) at export in src/molpot/forcefield/container.py `_propers_to_ir`, src/molpot/forcefield/potentials.py `_make_proper`, and src/molpot/forcefield/classical_imm.py `_propers_to_opls` (convert then existing fail-loud)
- [ ] Add Google-style docstrings with CLASS_I_CANONICAL units on ProperTorsionParamHead and ProperTorsionBag.from_cartesian
- [ ] Add regression example regressions/universal-class-i-02-fourier.py (public API only; hard-coded (a,b)↔(k,γ) goldens + one fail-loud OPLS sine residual)
- [ ] Verify against Domain-basis identities (mapping atol 1e-12; φ=π/3 energy 2.5176380902050415 atol 1e-10)
- [ ] Run full check + test suite

## Testing strategy

单测只放 `tests/`，路径镜像 `src/`，类型镜像（`ProperTorsionParamHead` → `TestProperTorsionParamHead`）。每个测试只打一个方法。不在 `tests/` 写 e2e。

**`TestProperTorsionParamHead`（改写既有 softplus-\(k\) 断言）**

- Happy：默认头 `forward(randn(N,D))` 给出 `a,b` 形状 `(N,6)`，`periodicity.tolist()==[1,2,3,4,5,6]`，`idivf` 全为 1；`k`/`phase`/`c0` 不在字典中；`a,b` 可为负。
- Edge：`n_terms=3, periodicity=(1,2,3)` 时形状 `(N,3)`；大负特征仍产出有限无约束振幅（不再要求 \(k\ge 0\)）。
- 不改 `TestBondParamHead` / `TestAngleParamHead` 的正性 / \(\theta_0\in(0,\pi)\) 断言。

**`TestImproperParamHead`**

- Happy：无关键字构造 → 周期支路，`periodicity==[1,2]`，有 `k,phase,idivf`，无 `chi0`。
- Edge：显式 `include_harmonic=True, include_periodic=False` 仍给出谐波 `k,chi0`（谐波仍可用）。

**`TestProperTorsionBag`**

- Domain：`from_cartesian` 对上表三组映射，atol \(10^{-12}\)；往返 \(a,b\to k,\gamma\to a',b'\) 同容差。
- Edge：`a.shape!=b.shape` 或非 2D → `ValueError`；零振幅 \(k=0,\gamma=0\)；未传 `idivf` → 全 1。
- 既有 `(k,phase)` 构造与 periodicity 校验保持。
- 字段集合不含 `c0`。

**`TestForceFieldToIR` / `TestProject`**

- Happy：`ClassIForceField.with_params(propers={a,b,...}).to_ir().propers` 是 `ProperTorsionBag`，字段为 `(k,phase,periodicity,idivf)`。
- Domain：`to_potentials().calc_energy` 与 \(k[1+\cos(n\varphi-\gamma)]\) 在硬编码 \(\varphi=\pi/3,n=2,k=2,\gamma=\pi/4\) 上等于 `2.5176380902050415`（atol \(10^{-10}\)）；或与 \(k+a\cos(n\varphi)+b\sin(n\varphi)\) 一致。
- Happy OPLS 流形：\(b=0\)、\(n=1..4\) 时 `project(OPLSForceField)` 给出 \(K_{\mathrm{odd}}=2a\)、\(K_{\mathrm{even}}=-2a\)，不抛。
- Edge fail-loud：\(n=6\) 且 \(|a|\) 或 \(|b|\) 非零 → `ValueError` 匹配 `n=6`；\((a,b)=(0,0.4)\)、\(n=1\)（\(\gamma=\pi/2\)）→ `ValueError` 匹配 sin/phase。
- Edge：同时给 `(a,b)` 与 `(k,phase)` → `ValueError`。
- 既有 `_OplsLikeProperHead`（直接写 `k,phase`）能量/力投影单测保持绿。

**回归** `regressions/universal-class-i-02-fourier.py`：公共 API（`ProperTorsionBag.from_cartesian`、`ClassIForceField.to_ir`、`ClassicalIMM.project`）。硬编码上表映射与能量恒等式；另含一个 fail-loud OPLS 正弦残差（\((a,b)=(0,0.4)\)）。注释写明解析来源（OpenMM §19.4 恒等式 + `math.cos` / `math.atan2`，2026-08-13），运行时不调用第三方。容差：映射 \(10^{-12}\)、能量 \(10^{-10}\)。

## Out of scope

- ChemEncoder 消息传递 / `universal-class-i-01-encoder`（前驱；本 spec 不重做，头只吃特征张量）
- 损失（含 \((a,b)\) 上的轻 L2）、PES bundle、THEMol
- LJ 解冻、live QEq、OpenFF trefoil 平均（export-only，后续）
- `scripts/mm_param_learning/`（`espaloma_mm_protocol_train.py` 仍读 `k/phase`，本 spec 不改脚本）
- 表参数 \(c_0\) / 生成热偏移
- 新的 ForceField 门面、`to_molpy`、从 head 写引擎记录
- 修改 `BondHarmonic` / `ProperTorsionPeriodic` / `ImproperHarmonic` / `ImproperPeriodic` 能量公式
- 重写 `BondParamHead` / `AngleParamHead`
- Condenser、ForceFieldCompiler、molix 导出适配器（它们继续读 IR 的 \(k,\gamma\)）
