---
title: ProperTorsionParamHead 支持 OPLS-AA k1..k4 直接输出
status: done
last-updated: 2026-09-04
grilled: true
created: 2026-08-20
slug: oplsaa-parameterizer-02-heads
chain: oplsaa-parameterizer
---

# ProperTorsionParamHead 支持 OPLS-AA k1..k4 直接输出

## Summary

在已有 `OPLSAAForceField`（01）与后续 `Parameterizer`（03）之间，本阶段只改 composition 侧：泛化 `ProperTorsionParamHead`，新增与 `dihedral:opls` schema 对齐的 OPLS 输出路径，使 proper chem features 直接映射为可训练的 LAMMPS `Kn` 约定参数 `k1..k4`（各 shape `(N,)`、可微）。默认 Cartesian `(a,b)` 行为保持不变。`BondParamHead` / `AngleParamHead` / `ImproperParamHead`（periodic）/ `LJParameterHead` / `AtomicChargeModel.write_forcefield` 本阶段不重写，仅以契约测试钉死其输出键，供 03 按 schema 挂接。不引入 `Parameterizer`、不删 `ClassI*`、不碰 `project()` 或 WaterLES。

## Domain basis

OPLS-AA proper 扭转（与 LAMMPS `dihedral_style opls` 一致）能量为：

\[
\begin{aligned}
E = &\ \tfrac12 K_1\bigl[1+\cos\phi\bigr]
  + \tfrac12 K_2\bigl[1-\cos 2\phi\bigr] \\
  &+ \tfrac12 K_3\bigl[1+\cos 3\phi\bigr]
  + \tfrac12 K_4\bigl[1-\cos 4\phi\bigr]
\end{aligned}
\]

其中 \(\phi\) 为 i–j–k–l 二面角（rad），\(K_n\)（本仓库键名 `k1..k4`）单位为 kcal/mol。**关键陷阱：** 能量式里的 \(\tfrac12\) 已写在公式中；参数表里的 `Kn` / `k1..k4` **就是** LAMMPS `dihedral_coeff` 的系数，**不要**再乘或再除 \(\tfrac12\)。仓库内 Class-I 多周期余弦形式

\[
E=\sum_m (k_m/s)\bigl[1+\cos(n_m\phi-\gamma_m)\bigr]
\]

在 OPLS 流形上满足 \(k_m = K_m/2\)、\(n=(1,2,3,4)\)、\(\gamma=(0,\pi,0,\pi)\)（见 `ForceField._propers_to_ir` / `_opls_to_periodic`）。本 head 的 OPLS 路径必须输出 **\(K_n\)**（`k1..k4`），而不是 Class-I 的半幅 \(k_m\)。

单位：`CLASS_I_CANONICAL`（kcal/mol）。长度/角度不进入本 head。

参考文献：

- Jorgensen, Maxwell, Tirado-Rives, *J. Am. Chem. Soc.* **118**, 11225–11236 (1996). DOI [10.1021/ja9621760](https://doi.org/10.1021/ja9621760)（OPLS-AA）
- LAMMPS `dihedral_style opls` — https://docs.lammps.org/dihedral_opls.html（明确 “usual 1/2 factor is not included in the K values”）
- Watkins & Jorgensen, *J. Phys. Chem. A* **105**, 4118–4125 (2001). DOI [10.1021/jp004071w](https://doi.org/10.1021/jp004071w)

## Design

### 实体与符号

- **泛化** `ProperTorsionParamHead`（`molpot.composition.mm_heads`）：在现有 Cartesian Fourier 路径之外增加 OPLS 输出模式。
  - 构造参数（建议名）`form: Literal["cartesian", "opls"] = "cartesian"`。
  - `form="cartesian"`：保持现状 — MLP → unconstrained `a`/`b` `(N,T)`，附带 `periodicity` buffer 与 `idivf`；默认 `T=6`、`n=1…T`。
  - `form="opls"`：MLP 输出维固定为 4；`forward(features)` → `{"k1","k2","k3","k4"}`，各 `(N,)`、dtype/device 跟随输入、**无符号约束**（与 Cartesian `a`/`b` 同为自由振幅）。不发射 `a`/`b`/`k`/`phase`/`periodicity`/`idivf`（对齐 `PotentialMeta` 的 `dihedral:opls` ParamSpec：仅 `k1..k4`）。
  - `form="opls"` 时若调用方仍传入与 Cartesian 冲突的 `n_terms`/`periodicity` 且无法忽略，应在 `__init__` **显式报错**（loud fail），避免 silent ignore。
- **生命周期**：纯 `nn.Module` MLP；无 ForceField 所有权；endpoint 对称仍由调用方做 order-invariant pooling（文档合同不变）。
- **挂接契约（给 03，本阶段不实现 Parameterizer）**：
  | Head / API | Schema 键 | 形状 |
  |---|---|---|
  | `BondParamHead` | `k`, `r0` | `(N,)` |
  | `AngleParamHead` | `k`, `theta0` | `(N,)` |
  | `ProperTorsionParamHead(form="opls")` | `k1..k4` | `(N,)` |
  | `ImproperParamHead`（默认 periodic） | `k`, `phase`, `periodicity`, `idivf` | 与现实现一致 |
  | `LJParameterHead` | `epsilon`, `sigma` | `(N,)` |
  | `AtomicChargeModel.write_forcefield` | 写入 `atoms["q"]`（及 alias `charge`） | `(N,)` |

### Reuse decision

- **generalize** `ProperTorsionParamHead` — 增加 `form="opls"` 输出路径，默认 Cartesian 不变。
- **reuse** `BondParamHead` — 不改实现；契约测试钉死键与形状。
- **reuse** `AngleParamHead` — 同上。
- **reuse** `ImproperParamHead`（periodic 默认）— 同上。
- **reuse** `LJParameterHead` — 同上。
- **reuse** `AtomicChargeModel.write_forcefield` — 不改 `charge/`；契约测试仅断言可调用且写出 `q`。
- **new — deferred**：`Parameterizer` / `OPLSAAForceField` 装配逻辑、`project()` 删除、公开面硬切 → 03 / 04；WaterLES → 独立链。

## Files to create or modify

- `src/molpot/composition/mm_heads.py` — 泛化 `ProperTorsionParamHead`
- `tests/test_molpot/test_composition/test_mm_heads.py` — OPLS 路径 + reuse 契约测试
- `regressions/oplsaa-parameterizer-02-heads.py` (new) — 公开 API：features → `k1..k4` 形状 / 零初值金标 / 可微

## Tasks

- [x] Write failing unit tests for ProperTorsionParamHead OPLS form (tests/test_molpot/test_composition/test_mm_heads.py → TestProperTorsionParamHead)
- [x] Write failing schema-contract tests for reused Bond/Angle/Improper/LJ/charge write_forcefield keys (tests/test_molpot/test_composition/test_mm_heads.py → TestOPLSAAHeadSchemaContracts)
- [x] Generalize ProperTorsionParamHead in src/molpot/composition/mm_heads.py with form="cartesian"|"opls"
- [x] Add docstring per google style with units and Kn vs 1/2 pitfall on ProperTorsionParamHead
- [x] Add regression example regressions/oplsaa-parameterizer-02-heads.py (public API only; hard-coded goldens, no third-party runtime)
- [x] Verify OPLS path emits LAMMPS Kn (k1..k4) not Class-I half-amplitudes against hard-coded zero-init case
- [x] Run full check + test suite

## Testing strategy

单元测试（`tests/test_molpot/test_composition/test_mm_heads.py`，单方法）：

- **Happy path（OPLS）**：`ProperTorsionParamHead(feature_dim=D, form="opls")(feats)` → 键集恰好 `{k1,k2,k3,k4}`，各 shape `(N,)`，有限。
- **Default 回归**：`form` 缺省 / `"cartesian"` 仍输出 `{a,b,periodicity,idivf}`，`a`/`b` shape `(N,6)`。
- **可微**：`feats.requires_grad_(True)`；对 `out["k1"].sum()`（或四键之和）`backward()`；`feats.grad` 非 `None` 且有限。
- **零初值金标**：MLP 权重与偏置全零时，任意有限 features → `k1=k2=k3=k4=0`（atol 1e-12）——证明输出的是裸线性 `Kn`，没有 softplus/½ 缩放。
- **构造失败**：`form="opls"` 且传入不兼容的 `n_terms`/`periodicity` → `ValueError`。
- **Reuse 契约（不改实现）**：`BondParamHead`→`k,r0`；`AngleParamHead`→`k,theta0`；`ImproperParamHead()` 默认 periodic 键；`LJParameterHead`→`epsilon,sigma`；`AtomicChargeModel.write_forcefield` 在最小 ForceField+batch 上写出 `q`。

**回归示例** `regressions/oplsaa-parameterizer-02-heads.py`：

- 仅公开 API（`from molpot… import ProperTorsionParamHead` 等）。
- 固定零权重：断言 `k1..k4` 全零字面量；再对随机 features 断言 shape 与 `backward` 成功。
- 金标注释写明：tool=手算/恒等线性层，date=2026-08-20；无第三方 runtime。

## Out of scope

- `Parameterizer` 类落地 / `ClassicalIMM` 改名（→ 03）
- 删除 `ClassIForceField` / `OPLSForceField` / `ClassIPredictions` 公开名（→ 01/04）
- 删除或重写 `project()`（→ 03/04）
- WaterLES 数据集与训练脚本
- 重写 Bond/Angle/Improper/LJ/charge 头内部实现
- GAFF 或其他非 OPLS proper 形式
- 在本阶段把 OPLS head 接入 ClassicalIMM 训练主路径
