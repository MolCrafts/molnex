---
spec: oplsaa-parameterizer-01-types
slug: oplsaa-parameterizer-01-types
created: 2026-08-20
criteria:
  - id: ac-001
    summary: OPLSAAForceField registers full OPLS-AA style set
    type: code
    pass_when: |
      TestOPLSAAForceField asserts bond:harmonic, angle:harmonic,
      dihedral:opls, improper:periodic, pair:lj/cut, and pair:coulomb
      are present on a default-constructed OPLSAAForceField.
    status: verified
  - id: ac-002
    summary: Default NonbondedScaling uses OPLS-AA f14=0.5
    type: code
    pass_when: |
      Default OPLSAAForceField().nonbonded_scaling has
      scale_q_14 == 0.5 and scale_lj_14 == 0.5 (not 5/6 for q).
    status: verified
  - id: ac-003
    summary: PotentialMeta registers improper schemas
    type: code
    pass_when: |
      TestPotentialMeta shows schema_for(["improper:periodic"]) and
      schema_for(["improper:harmonic"]) return registered StyleSchema
      entries whose energy_formula is not "(unregistered style)".
    status: verified
  - id: ac-004
    summary: ClassIForceField and OPLSForceField hard-deleted
    type: code
    pass_when: |
      ClassIForceField and OPLSForceField are absent from
      molpot.forcefield.__all__ and molpot.__all__; importing either
      name from those packages raises ImportError/AttributeError;
      no alias symbols exist.
    status: verified
  - id: ac-005
    summary: Public exports expose OPLSAAForceField
    type: code
    pass_when: |
      from molpot.forcefield import OPLSAAForceField and
      from molpot import OPLSAAForceField succeed; symbol is in both
      packages' __all__.
    status: verified
  - id: ac-006
    summary: ClassicalIMM default template stays Fourier-compatible
    type: code
    pass_when: |
      ClassicalIMM constructed without forcefield= uses a template that is
      NOT OPLSAAForceField (Fourier ForceField with dihedral:fourier or
      equivalent); OPLSAAForceField remains available for explicit construction.
    status: verified
  - id: ac-007
    summary: Former ClassI/OPLS Python call sites compile against new types
    type: runtime
    pass_when: |
      Listed migrated tests/scripts/regressions import and construct
      OPLSAAForceField or ForceField+def_*style only; pytest collection
      for those paths does not reference ClassIForceField/OPLSForceField.
    status: verified
  - id: ac-008
    summary: Regression pins styles, f14, and 1-4 Coulomb energy
    type: runtime
    pass_when: |
      regressions/oplsaa-parameterizer-01-types.py exits 0 and asserts
      OPLSAAForceField default f14 scales are 0.5/0.5 and the hard-coded
      four-atom 1-4 Coulomb energy equals 0.25 * kcalmol_A within atol
      1e-10 (literal golden embedded in the script).
    status: verified
  - id: ac-009
    summary: Scientific pin matches OPLS-AA f14=1/2 Coulomb
    type: scientific
    evaluator_hint: marker: oplsaa-f14
    pass_when: |
      Hard-coded four-atom chain (bonds 0-1,1-2,2-3; proper 0-1-2-3),
      q=[1,0,0,1], r_03=2 Å, coulomb-only nonbonded via
      OPLSAAForceField.to_potentials().calc_energy yields
      E = 0.25 * kcalmol_A (i.e. 0.5 * kcalmol_A * 1 * 1 / 2) within
      atol 1e-10; citation DOI 10.1021/ja9621760 for f14=1/2.
    status: verified
  - id: ac-010
    summary: Full check and test suite pass
    type: runtime
    pass_when: |
      `ruff check src/ tests/ scripts/ regressions/ && ruff format
      --check src/ tests/ scripts/ regressions/ && ty check src/
      --exit-zero-on-warning` and `python -m pytest tests/ -q -n 12
      --dist worksteal` exit 0.
    status: verified
out_of_scope:
  - Parameterizer rename / classical_imm.py → parameterizer.py
  - ProperTorsionParamHead OPLS k1..k4 rewrite
  - project() removal and ClassIPredictions wipe
  - Full scripts/tests/docs/notes ClassI* string cut (04)
  - WaterLES, GAFF, changing global NonbondedScaling defaults
---

# Acceptance criteria

本阶段「完成」= 公开类型面只剩 `ForceField` + `OPLSAAForceField`（Class-I/OPLS 容器已死），OPLS-AA 默认 styles 与 \(f_{14}=0.5\) 可测，`PotentialMeta` 能描述 improper，既有 `to_potentials` 能量路径在硬编码 1–4 库仑钉上给出文献约定的半缩放能量；Parameterizer / head / 全仓清零留给后续子规格。

## AC-001 — styles

默认构造即齐备 OPLS-AA 对角形式所需的 bond/angle/dihedral/improper/pair 风格名。

## AC-002 — f14 缩放属性

构造路径不得静默使用 AMBER `5/6` 库仑 1–4 默认。

## AC-003 — PotentialMeta improper

`improper:periodic` / `improper:harmonic` 进入 registry，供后续 Parameterizer schema 装配（03）消费。

## AC-004 / AC-005 — 硬切与导出

删除无 shim；新名进入公开 `__all__`。

## AC-006 — ClassicalIMM 过渡

默认 template 保持与 Cartesian/Fourier heads 兼容；不得在 01 挂上 `OPLSAAForceField`。改名与删 `project` 不在本阶段。

## AC-007 — 迁移面

所有因符号删除会 ImportError 的树内 Python 调用点已改写。

## AC-008 / AC-009 — 回归与科学钉

同一硬编码四原子 1–4 库仑算例：runtime 脚本可执行，scientific 准则绑定 \(f_{14}=1/2\) 与 DOI 10.1021/ja9621760。

## AC-010 — 门禁

仓库标准 check + pytest 全绿。
