---
slug: oplsaa-parameterizer-02-heads
created: 2026-08-20
criteria:
  - id: ac-001
    summary: ProperTorsionParamHead form=opls emits k1..k4 shapes
    type: code
    pass_when: |
      ProperTorsionParamHead(feature_dim=D, form="opls").forward on
      features (N,D) returns a dict whose keys are exactly
      {"k1","k2","k3","k4"} and each value has shape (N,) on the same
      device/dtype as features; keys a/b/k/phase/periodicity/idivf are absent.
    status: verified
  - id: ac-002
    summary: Default cartesian ProperTorsionParamHead unchanged
    type: code
    pass_when: |
      ProperTorsionParamHead(feature_dim=D) with default or
      form="cartesian" still returns keys {a,b,periodicity,idivf} with
      a/b shape (N,6) and periodicity [1,2,3,4,5,6].
    status: verified
  - id: ac-003
    summary: OPLS k1..k4 are unconstrained Kn (zero-init golden)
    type: runtime
    pass_when: |
      With all MLP parameters zeroed, form="opls" forward on any finite
      (N,D) features yields k1=k2=k3=k4=0 (atol 1e-12), proving outputs
      are LAMMPS Kn not softplus-scaled or Class-I half-amplitudes.
    status: verified
  - id: ac-004
    summary: OPLS proper head path is differentiable wrt features
    type: runtime
    pass_when: |
      features.requires_grad_(True); loss = sum of k1..k4; loss.backward()
      succeeds; features.grad is non-None and finite with shape (N,D).
    status: verified
  - id: ac-005
    summary: Reused bond/angle/improper/LJ/charge keys stay schema-ready
    type: code
    pass_when: |
      TestOPLSAAHeadSchemaContracts asserts BondParamHead→{k,r0},
      AngleParamHead→{k,theta0}, ImproperParamHead default periodic keys,
      LJParameterHead→{epsilon,sigma}, and AtomicChargeModel.write_forcefield
      writes atoms q — without modifying those implementations in this spec.
    status: verified
  - id: ac-006
    summary: Regression script hard-codes OPLS head goldens
    type: runtime
    pass_when: |
      regressions/oplsaa-parameterizer-02-heads.py exits 0 and asserts
      zero-init k1..k4 all equal the literal 0.0 (atol 1e-12) plus
      (N,) shapes and a successful backward; no third-party import or
      subprocess at runtime.
    status: verified
  - id: ac-007
    summary: Google docstring documents Kn vs 1/2 pitfall
    type: docs
    pass_when: |
      ProperTorsionParamHead class or form="opls" docstring (google style)
      states units kcal/mol and that k1..k4 are LAMMPS Kn (½ already in
      the energy formula), not Class-I k_m = Kn/2.
    status: verified
out_of_scope:
  - Parameterizer / ClassicalIMM rename
  - Deleting ClassI* / OPLSForceField public names
  - Deleting project()
  - WaterLES training
  - Rewriting Bond/Angle/Improper/LJ/charge heads
---

# Acceptance criteria

**Done** 表示 composition 侧可向 `OPLSAAForceField`（01）直接供给 OPLS proper 参数，且其它 head 的 schema 键仍可被 03 `Parameterizer` 挂接；不要求本阶段完成 Parameterizer 或公开面硬切。

## AC-001 — OPLS 键集与形状

`form="opls"` 的输出必须精确匹配 `PotentialMeta` 的 `dihedral:opls` ParamSpec。

## AC-002 — Cartesian 默认不变

既有 `learnable-classical-ff-03` / `TestProperTorsionParamHead` Cartesian 行为不得被破坏。

## AC-003 — Kn 而非 k_m/2

零初值全零是可观察的合同：若错误地做了 softplus 或乘了 ½，金标会失败。

## AC-004 — 可微

保证 03 训练路径上 proper chem features → `k1..k4` 可反传。

## AC-005 — Reuse 契约

钉死未改动的 head / `write_forcefield` 输出键，避免 03 装配时 silent mismatch。

## AC-006 — 回归脚本

公开 API 最小脚本，硬编码字面量金标，无第三方 oracle。

## AC-007 — 文档陷阱

必须写明 LAMMPS `Kn` 与 Class-I 半幅换算，防止后续实现把 `k1..k4` 再除以 2。
