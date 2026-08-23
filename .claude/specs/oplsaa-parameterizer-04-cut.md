---
title: oplsaa-parameterizer-04-cut — 公开面硬切清零与链回归
status: code-complete
grilled: true
slug: oplsaa-parameterizer-04-cut
chain: oplsaa-parameterizer
created: 2026-08-20
---

# oplsaa-parameterizer-04-cut — 公开面硬切清零与链回归

## Summary

在 `01`–`03` 已落地 `OPLSAAForceField`、`Parameterizer` 与 `ForceFieldPredictions` 的前提下，本阶段对 molnex 做硬切清零：公开 API、`scripts/mm_param_learning` 主路径、对应单测与 `regressions/`、以及 `.claude/notes/learnable-classical-ff.md`（及 `architecture.md` 中相关公开名）中不得再出现 `ClassIForceField`、`OPLSForceField`、`ClassicalIMM`、`ClassIPredictions` 字符串；训练/协议脚本（含 `espaloma_mm_protocol_train`）一律改走 `Parameterizer` + `OPLSAAForceField`；并新增链级回归 `regressions/oplsaa-parameterizer.py`，以硬编码小体系钉死 OPLS-AA `f14=0.5` 与 `OPLSAAForceField → to_potentials()` 能量路径。

## Domain basis

OPLS-AA 非键 1–4 缩放（相对全强度库仑 / LJ）为

\[
f_{14}^{(q)} = f_{14}^{(\mathrm{LJ})} = \tfrac{1}{2},
\]

即 `NonbondedScaling.scale_q_14 = scale_lj_14 = 0.5`。这与 AMBER/GAFF 默认 `scale_q_14 = 5/6`、`scale_lj_14 = 0.5` 不同；本链禁止把 AMBER 默认写进 `OPLSAAForceField` 构造路径。

参考文献：Jorgensen, Maxwell & Tirado-Rives, *J. Am. Chem. Soc.* **118**, 11225–11236 (1996), DOI [10.1021/ja9621760](https://doi.org/10.1021/ja9621760)（OPLS-AA）。

价键能量钉（回归硬编码，单位 `real` → kcal/mol）：谐振子键

\[
E_{\mathrm{bond}} = \tfrac{1}{2}\,k\,(r-r_0)^2.
\]

取 \(k=2\)、\(r_0=1\) Å、\(r=1.5\) Å → \(E=0.25\) kcal/mol（`atol=1e-10`）。能量只经填参后的 `OPLSAAForceField.to_potentials().calc_energy(batch)`，不经已删除的 `project()`。

## Design

**前提（reuse，不在本阶段重造）。** `01` 已提供 `OPLSAAForceField(ForceField)`（谐波键角、OPLS proper、`ImproperPeriodic`、默认 `lj/cut`+`coulomb`、构造时 `scale_q_14=scale_lj_14=0.5`）；`02` 已 generalize `ProperTorsionParamHead` 的 `k1..k4` 路径；`03` 已将 `ClassicalIMM` → `Parameterizer(encoder, template: ForceField)`、`ClassIPredictions` → `ForceFieldPredictions`，并删除 `project()`。本阶段不新增力场/参数化类型，只做字符串清零、调用方迁移与链回归。

**实体与所有权。**

| 符号 | 本阶段职责 |
|------|------------|
| `OPLSAAForceField` | 唯一具名 Class-I 产品模板；脚本/回归默认 `template` |
| `Parameterizer` | 唯一可训练参数化模块；替换所有 `ClassicalIMM` 构造点 |
| `ForceFieldPredictions` | model→task 映射；替换 `ClassIPredictions` |
| `ForceField` + `def_*style` | MD/极简谐振子用例的无产品名容器路径 |
| `MMEnergyModel` / `build_espaloma_like_model`（`espaloma_mm_protocol_train`） | 内部改挂 `Parameterizer`+`OPLSAAForceField`；对外类型注解与 docstring 同步 |
| notes / architecture | 绑定用户路径改写为 Parameterizer 叙事；废除 ClassI/ClassicalIMM/project 产品叙述 |

**清零范围（硬）。** 下列四处标识符在目标树中出现次数必须为 **0**（含 `__all__`、import、类型注解、docstring、注释）：

`ClassIForceField` · `OPLSForceField` · `ClassicalIMM` · `ClassIPredictions`

目标树：`src/molpot/` 公开再导出、`scripts/mm_param_learning/`、`tests/`（与上述相关的镜像单测）、`regressions/`、`.claude/notes/learnable-classical-ff.md`、`.claude/notes/architecture.md`，以及 `CLAUDE.md` 中仍点名旧公开符号的自由段落。已删除的 `project()` **不得恢复**；依赖 `project` 的脚本/回归（如 `parity_project_md.py`、`learnable-classical-ff-project-opls.py`）删除或改写为不经 `project` 的 OPLSAA 直填路径。

**用户路径（绑定）。**

```python
from molpot import Parameterizer, OPLSAAForceField, ForceFieldPredictions

ff = OPLSAAForceField()
model = Parameterizer(encoder, template=ff, bond_head=..., ...)
ff_t = model(batch)                          # 填参 OPLSAAForceField
E = ff_t.to_potentials().calc_energy(batch)
pred = ForceFieldPredictions(model, keys=("energy", "forces"))
```

**Reuse decision**

- reuse `Parameterizer`（03）— 全部可训练装配入口
- reuse `OPLSAAForceField`（01）— 唯一具名模板与 `f14=0.5` 契约
- reuse `ForceFieldPredictions`（03）— 替换 `ClassIPredictions`
- reuse `ChemEncoder`、`BondParamHead` / `AngleParamHead` / `ProperTorsionParamHead` / `ImproperParamHead` / `LJParameterHead`、`AtomicChargeModel` — 协议脚本继续挂接，不重写
- reuse `ForceField.to_potentials` / `Potentials` — 唯一能量/力路径
- reuse `NonbondedScaling` — 回归断言 `scale_q_14`/`scale_lj_14`
- new — 无新领域类型；本阶段仅 cut + migrate + 链回归

## Files to create or modify

- `src/molpot/__init__.py`
- `src/molpot/forcefield/__init__.py`
- `scripts/mm_param_learning/espaloma_mm_protocol_train.py`
- `scripts/mm_param_learning/class_i_fit.py`
- `scripts/mm_param_learning/parity_lammps_classical_md.py`
- `scripts/mm_param_learning/parity_project_md.py`（删除或改写为无 `project`）
- `tests/test_molpot/test_forcefield/test_exports_cut.py` (new)
- `tests/test_scripts/test_mm_param_learning/test_espaloma_mm_protocol_train.py` (new)
- `tests/test_molpot/test_forcefield/test_parameterizer.py`（由 03 落地；本阶段扫尽旧名）
- `tests/test_molpot/test_forcefield/test_predictions.py`
- `tests/test_molpot/test_forcefield/test_container.py`
- `tests/test_molix/test_md/test_driver.py`
- `regressions/oplsaa-parameterizer.py` (new)
- `regressions/learnable-classical-ff-05-neural-parameterizer.py`
- `regressions/learnable-classical-ff-project-opls.py`（删除或改写）
- `regressions/universal-class-i-08-adapter.py`
- `regressions/md-forcefield-molpy-reexport-01-reexport.py`
- `.claude/notes/learnable-classical-ff.md`
- `.claude/notes/architecture.md`
- `CLAUDE.md`（自由段中旧公开名）

## Tasks

- [x] Write failing unit tests for banned-name absence and Parameterizer/OPLSAAForceField/ForceFieldPredictions exports (`tests/test_molpot/test_forcefield/test_exports_cut.py` → `TestBannedNameAbsence`, `TestPublicExports`)
- [x] Write failing unit tests for `build_espaloma_like_model` / `MMEnergyModel` using Parameterizer+OPLSAAForceField (`tests/test_scripts/test_mm_param_learning/test_espaloma_mm_protocol_train.py` → `TestBuildEspalomaLikeModel`, `TestMMEnergyModel`)
- [x] Migrate `scripts/mm_param_learning/espaloma_mm_protocol_train.py` and `scripts/mm_param_learning/class_i_fit.py` to Parameterizer+OPLSAAForceField+ForceFieldPredictions
- [x] Purge banned names from `src/molpot/__init__.py` and `src/molpot/forcefield/__init__.py`; migrate or remove `parity_lammps_classical_md.py` / `parity_project_md.py` without restoring `project()`
- [x] Migrate remaining listed unit tests and regressions to Parameterizer/OPLSAAForceField/ForceFieldPredictions or `ForceField`+`def_*style`; delete or rewrite project-only regressions
- [x] Sync `.claude/notes/learnable-classical-ff.md`, `.claude/notes/architecture.md`, and `CLAUDE.md` free-form ForceField bullets to Parameterizer+OPLSAA user path
- [x] Add regression example `regressions/oplsaa-parameterizer.py` (public API only; hard-coded `f14=0.5` and \(E=0.25\) goldens, no third-party runtime)
- [x] Run full check + test suite

## Testing strategy

- **单元（`tests/`，单函数/单方法，无 e2e）**
  - `TestBannedNameAbsence`：对 `src/molpot/__init__.py`、`src/molpot/forcefield/__init__.py`、`scripts/mm_param_learning/` 做源码/导出扫描，四禁用名出现次数为 0；`from molpot import ClassIForceField`（等）均 `ImportError`。
  - `TestPublicExports`：`Parameterizer`、`OPLSAAForceField`、`ForceFieldPredictions` 在 `molpot` / `molpot.forcefield` 的 `__all__` 中；`ClassicalIMM` 等不在。
  - `TestBuildEspalomaLikeModel`：返回实例为 `Parameterizer`，`template`/产出力场类型为 `OPLSAAForceField`。
  - `TestMMEnergyModel`：`forward` 经 `to_potentials().calc_energy` 给出 `(B,)` energy；不引用旧类名。
  - 既有 `TestParameterizer` / `TestForceFieldPredictions` / `TestOPLSAAForceField` / MD driver 镜像：断言与构造路径无禁用名。
- **边角**
  - `OPLSAAForceField().nonbonded_scaling.scale_q_14 == 0.5` 且 `scale_lj_14 == 0.5`（禁止静默回到 `5/6`）。
  - 删除/改写后，仓库内不再有可执行的 `project(` 调用于 `scripts/mm_param_learning/` 与 `regressions/`。
- **域验证（硬编码）**
  - 回归：双原子谐振子 \(k=2,r_0=1,r=1.5\) → \(E=0.25\)；同脚本断言 `f14` 双通道 `0.5`。
- **回归示例**：`regressions/oplsaa-parameterizer.py` — 公开 API：`OPLSAAForceField`（或 `Parameterizer`+常量 head）→ `to_potentials().calc_energy`；字面量黄金值；注释写明公式与 DOI/日期；禁止第三方运行时。

## Out of scope

- WaterLES 数据集训练与对比实验（独立未来链）
- freeze-encoder / head-swap 精度探针
- `GAFFForceField` 或任何非 OPLS-AA 具名力场
- 恢复 `Parameterizer.project` / `ClassicalIMM.project`
- 重写 ChemEncoder 或 MM heads 数值实现（属 01–03）
- `docs/` 叙事长文全面改写（公开代码与 notes 优先；文档可后续 `/mol:docs`）
- 已关闭历史 spec 正文（`.claude/specs/md-forcefield-*` 等）的考古式改名
