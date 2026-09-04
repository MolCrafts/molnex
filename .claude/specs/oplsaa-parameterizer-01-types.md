---
title: OPLSAAForceField 类型面与 PotentialMeta improper schema
status: done
last-updated: 2026-09-04
created: 2026-08-20
revised: 2026-08-20
grilled: true
slug: oplsaa-parameterizer-01-types
chain: oplsaa-parameterizer
---

# OPLSAAForceField 类型面与 PotentialMeta improper schema

## Summary

本阶段在 `molpot.forcefield` 落地唯一的 OPLS-AA 容器类型 `OPLSAAForceField(ForceField)`：构造时注册 harmonic 键/角、`dihedral:opls` proper、优先的 `improper:periodic`（ImproperPeriodic）、默认 `pair:lj/cut` 与 `pair:coulomb`，并以 `NonbondedScaling(scale_q_14=0.5, scale_lj_14=0.5)` 绑定 OPLS-AA 的 \(f_{14}=1/2\)（禁止落入 `NonbondedScaling` 的 AMBER `scale_q_14=5/6` 默认）。同步 generalize `PotentialMeta` 补齐 improper schema 条目。硬删 `ClassIForceField` 与 `OPLSForceField`（无别名）；所有因删除而断裂的 Python 构造/导入点迁到 `OPLSAAForceField` 或裸 `ForceField` + `def_*style`。**`ClassicalIMM` 默认 template 不得改为 `OPLSAAForceField()`**：在 `02-heads` 提供 OPLS `k1..k4` 之前，Cartesian Fourier heads 与 `dihedral:opls` 不兼容；默认改用裸 `ForceField` + Fourier dihedral（或显式 `def_dihedralstyle("fourier")`），完整 `Parameterizer`+`OPLSAAForceField` 装配属链上 03。能量路径继续只走既有 `to_potentials()` → `Potentials`。

## Domain basis

本阶段不引入新物理；能量评估仍是 OPLS-AA Class-I 对角形式。总能量：

\[
E = E_{\mathrm{bond}} + E_{\mathrm{angle}} + E_{\mathrm{proper}} + E_{\mathrm{improper}}
  + E_{\mathrm{LJ}} + E_{\mathrm{Coul}}
\]

键/角（仓库显式 ½；与文献 \(K_r(\Delta r)^2\) 差因子 2，当且仅当 \(k=2K_r\)）：

\[
E_{\mathrm{bond}}=\sum\tfrac12 k\,(r-r_0)^2,\quad
E_{\mathrm{angle}}=\sum\tfrac12 k\,(\theta-\theta_0)^2
\]

OPLS proper（LAMMPS `dihedral_style opls`；表存 \(K_n\)，进入周期核时 \(k_m=K_m/2\)）：

\[
E=\tfrac12 K_1[1+\cos\phi]+\tfrac12 K_2[1-\cos 2\phi]
 +\tfrac12 K_3[1+\cos 3\phi]+\tfrac12 K_4[1-\cos 4\phi]
\]

Improper：1996 主文非核心项；本仓库工程优先 **周期余弦** `ImproperPeriodic`（不得把 harmonic improper 默认为 OPLS-AA 标准）。非键 LJ 12-6 + 库仑；**1–4（OPLS-AA）** \(f_{ij}=1/2\) 同时作用于库仑与 LJ；1–2/1–3 为 0。单位：LAMMPS `units real`（kcal/mol，Å，\(e\)）。

主文献：Jorgensen, Maxwell, Tirado-Rives (1996) DOI [10.1021/ja9621760](https://doi.org/10.1021/ja9621760)；Watkins & Jorgensen (2001) DOI [10.1021/jp004071w](https://doi.org/10.1021/jp004071w)；键/角谱系 Cornell et al. (1995) DOI [10.1021/ja00124a002](https://doi.org/10.1021/ja00124a002)。**实现陷阱：** `NonbondedScaling` 全局默认是 AMBER（`q_14=5/6`）；声称 OPLS-AA 时必须 `scale_q_14=scale_lj_14=0.5`。

## Design

**实体与所有权。** 新增 `OPLSAAForceField(ForceField)`，直接继承通用容器（不经已删的 Class-I 中间类）。`__init__` 内一次性：`def_atomstyle("full")`、`def_bondstyle("harmonic")`、`def_anglestyle("harmonic")`、`def_dihedralstyle("opls")`、`def_improperstyle("periodic")`、`def_pairstyle("lj/cut")`、`def_pairstyle("coulomb")`；当调用方未传入 `nonbonded_scaling` 时构造 `NonbondedScaling(scale_q_14=0.5, scale_lj_14=0.5)`（1–2/1–3 仍为 0）并交给 `ForceField.__init__`。禁止再提供 `ForceField.oplsaa()` 类方法 façade。硬删 `ClassIForceField` / `OPLSForceField` 类定义及 `__all__` / `molpot` 顶层导出；无 deprecated 别名。

**PotentialMeta。** 在 `PotentialMeta._REGISTRY` 补 `improper:periodic`（与 `ImproperPeriodic` / `ImproperPeriodicBag` 字段一致：`k`、`periodicity`、`phase`、`idivf`）及 `improper:harmonic`（既有谐波路径的 schema 对称项），使 `schema_for(OPLSAAForceField().style_names())` 对 improper 不再回落为 `(unregistered style)`。

**迁移边界。** 仅因符号删除而必须改的 Python 构造/导入点：MD / IR / ClassicalIMM 测试、LAMMPS export 测试、`parity_lammps_classical_md` / `parity_project_md` / `espaloma_mm_protocol_train` 中的 `ClassIForceField`/`OPLSForceField` 引用、以及相关 regressions。Fourier-only / ClassicalIMM 默认路径改用裸 `ForceField` + `def_*style`（含 `def_dihedralstyle("fourier")`），**不得**在 01 把默认 template 钉成 `OPLSAAForceField`。需要 OPLS-AA 产品容器的调用点显式构造 `OPLSAAForceField()`。**不**改名 ClassicalIMM、**不**删 `project()`、**不**动 `ClassIPredictions` / `ProperTorsionParamHead`（分属 03 / 04 / 02）。

**Reuse decision**

- generalize `PotentialMeta`（`src/molpot/forcefield/schema.py`）— 补 `improper:periodic` / `improper:harmonic` schema 条目
- pattern `ClassIForceField` / `OPLSForceField` → 以同形 `def_*style` 构造写出 `OPLSAAForceField`，然后硬删二者（无别名）
- reuse `ForceField.to_potentials` / `Potentials` — 能量唯一入口不变；本阶段不新写 evaluator
- deferred `ClassicalIMM` → 链上 `03-parameterizer`（本阶段仅保证默认 template 与 Fourier heads 兼容，不挂 OPLSAA）
- deferred `ProperTorsionParamHead` → 链上 `02-heads`
- deferred `ClassIPredictions` → 链上 `03`/`04`（本阶段不删名）

## Files to create or modify

- `src/molpot/forcefield/container.py`
- `src/molpot/forcefield/schema.py`
- `src/molpot/forcefield/__init__.py`
- `src/molpot/__init__.py`
- `src/molpot/forcefield/classical_imm.py` (default template only: Fourier `ForceField`, not OPLSAA)
- `tests/test_molpot/test_forcefield/test_container.py`
- `tests/test_molpot/test_forcefield/test_schema.py` (new)
- `tests/test_molpot/test_forcefield/test_to_ir.py`
- `tests/test_molpot/test_forcefield/test_classical_imm.py`
- `tests/test_molix/test_md/test_driver.py`
- `tests/test_molix/test_ff_export/test_lammps_writer.py`
- `scripts/mm_param_learning/parity_lammps_classical_md.py`
- `scripts/mm_param_learning/parity_project_md.py`
- `scripts/mm_param_learning/espaloma_mm_protocol_train.py`
- `regressions/md-forcefield-molpy-reexport-01-reexport.py`
- `regressions/learnable-classical-ff-project-opls.py`
- `regressions/universal-class-i-02-fourier.py`
- `regressions/universal-class-i-08-adapter.py`
- `regressions/oplsaa-parameterizer-01-types.py` (new)

## Tasks

- [x] Write failing unit tests for OPLSAAForceField styles + OPLS-AA NonbondedScaling (tests/test_molpot/test_forcefield/test_container.py → TestOPLSAAForceField) and for PotentialMeta improper registry (tests/test_molpot/test_forcefield/test_schema.py → TestPotentialMeta)
- [x] Generalize PotentialMeta in src/molpot/forcefield/schema.py to register improper:periodic and improper:harmonic StyleSchema entries
- [x] Implement OPLSAAForceField in src/molpot/forcefield/container.py (harmonic B/A, dihedral:opls, improper:periodic, lj/cut+coulomb, scale_q_14=scale_lj_14=0.5); hard-delete ClassIForceField and OPLSForceField with no aliases
- [x] Update public exports in src/molpot/forcefield/__init__.py and src/molpot/__init__.py to export OPLSAAForceField and drop ClassIForceField/OPLSForceField
- [x] Retarget ClassicalIMM default template to a Fourier ForceField (def_dihedralstyle fourier), not OPLSAAForceField, in src/molpot/forcefield/classical_imm.py (keep ClassicalIMM name and project() for later phases)
- [x] Migrate Python call sites that imported/constructed ClassIForceField or OPLSForceField to OPLSAAForceField (when OPLS-AA form is intended) or ForceField+def_*style (Fourier / harmonic-only cases) (listed tests/, scripts/mm_param_learning/, and regressions above)
- [x] Add docstring per google style on OPLSAAForceField with units and f14=0.5 contract
- [x] Add regression example regressions/oplsaa-parameterizer-01-types.py (public API only; hard-coded goldens for styles, f14 scales, and 1–4 Coulomb energy pin)
- [x] Verify against hard-coded four-atom 1–4 Coulomb case: E = 0.5 * kcalmol_A * q_i * q_j / r (atol 1e-10)
- [x] Run full check + test suite

## Testing strategy

- **Unit（镜像路径）**
  - `tests/test_molpot/test_forcefield/test_container.py` → `TestOPLSAAForceField`：默认 styles 集合；`nonbonded_scaling.scale_q_14 == scale_lj_14 == 0.5`；`clone()` 保型与缩放；显式传入自定义 `NonbondedScaling` 时不被静默覆写。
  - `tests/test_molpot/test_forcefield/test_schema.py` → `TestPotentialMeta`：`schema_for(["improper:periodic"])` / `["improper:harmonic"]` 返回已注册 `StyleSchema`（非 `(unregistered style)`）；`OPLSAAForceField().to_potentials().get_schema()` 含 `improper:periodic`。
  - 既有 `TestClassIForceField` / `TestOPLSForceField` 删除或改写为 `TestOPLSAAForceField`；断言 `ClassIForceField`/`OPLSForceField` 不可从 `molpot` / `molpot.forcefield` 导入。
- **Happy path**：`OPLSAAForceField().with_params(...).to_potentials().calc_energy(batch)` 对仅键谐振子仍得 \(E=\tfrac12 k(r-r_0)^2\)。
- **Edge**：裸 `ForceField()` 仍可用 AMBER 默认缩放（本阶段不改全局 `NonbondedScaling` 默认）；仅 OPLSAA 构造绑 \(f_{14}=0.5\)。
- **域验证（scientific pin）**：四原子链拓扑（键 0–1、1–2、2–3；proper 0–1–2–3），\(q_0=q_3=1\)、中间电荷 0，\(r_{03}=2\) Å，仅 coulomb 非键；期望 \(E=0.5\times\texttt{kcalmol_A}/2\)（即 \(0.25\times\texttt{kcalmol_A}\)），`atol=1e-10`。对照：若误用 `scale_q_14=5/6` 则失败。
- **Regression example**：`regressions/oplsaa-parameterizer-01-types.py` — 公开 API 构造 `OPLSAAForceField`，断言 styles、\(f_{14}\) 属性与上述硬编码能量字面量（注释写明 `kcalmol_A` 来源与日期）；禁止运行时依赖第三方力场引擎。

## Out of scope

- `ClassicalIMM` → `Parameterizer` 改名与 `classical_imm.py` → `parameterizer.py`（03）
- `ProperTorsionParamHead` 的 OPLS `k1..k4` 输出重写（02）
- 删除 `project()` / `ClassIPredictions` 全仓字符串清零 / notes 与 docs 硬切（04）
- WaterLES 训练、GAFF / AMBER 产品力场、PME 库仑策略变更
- 改写全局 `NonbondedScaling` 默认值（仍保持 AMBER 5/6，供非 OPLS 路径）
