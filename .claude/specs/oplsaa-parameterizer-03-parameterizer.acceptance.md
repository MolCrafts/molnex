---
slug: oplsaa-parameterizer-03-parameterizer
criteria:
  - id: ac-001
    summary: Parameterizer forward fills ForceField params from heads
    type: code
    pass_when: |
      Parameterizer(encoder, template=OPLSAAForceField(), bond_head=...,
      angle_head=..., proper_head=..., improper_head=..., atom_head=...,
      charge_model=...)(batch) returns a ForceField copy whose params tables
      for bonds/angles/propers/impropers/atoms are populated; propers contain
      k1..k4 tensors; the original template object is not mutated in place.
    status: verified
  - id: ac-002
    summary: Parameterizer has no project attribute or method
    type: code
    pass_when: |
      After import, hasattr(Parameterizer(...), "project") is False and
      "project" is absent from Parameterizer.__dict__ / public API docs
      surface; no Fourier↔OPLS projection helpers remain on the type.
    status: verified
  - id: ac-003
    summary: ForceFieldPredictions is importable from molpot.forcefield
    type: code
    pass_when: |
      `from molpot.forcefield import ForceFieldPredictions` and
      `from molpot import ForceFieldPredictions` succeed; constructing
      ForceFieldPredictions(parameterizer, keys=("energy","forces")) then
      calling it yields energy/forces via filled.to_potentials().
    status: verified
  - id: ac-004
    summary: ClassicalIMM import fails with no alias
    type: code
    pass_when: |
      `from molpot.forcefield import ClassicalIMM` and
      `from molpot import ClassicalIMM` raise ImportError; ClassicalIMM is
      not listed in molpot.forcefield.__all__ or molpot.__all__.
    status: verified
  - id: ac-005
    summary: ClassIPredictions import fails with no alias
    type: code
    pass_when: |
      `from molpot.forcefield import ClassIPredictions` and
      `from molpot import ClassIPredictions` raise ImportError;
      ClassIPredictions is not in either package __all__.
    status: verified
  - id: ac-006
    summary: Parameterizer exposes no energy evaluation API
    type: code
    pass_when: |
      Parameterizer has no public calc_energy/calc_forces/energy method;
      unit tests obtain E/F only through returned ForceField.to_potentials().
    status: verified
  - id: ac-007
    summary: Schema gate requires OPLS proper and NB/charge heads
    type: code
    pass_when: |
      Building Parameterizer with OPLSAAForceField template but missing
      proper_head (or proper output lacking k1..k4), atom_head, or
      charge_model raises ValueError/TypeError before or on first forward.
    status: verified
  - id: ac-008
    summary: Docstrings name Parameterizer and ForceFieldPredictions with units
    type: docs
    pass_when: |
      Public docstrings on Parameterizer and ForceFieldPredictions use Google
      style and state kcal/mol (energy), Å (length), e (charge); module
      examples show Parameterizer + to_potentials, not ClassicalIMM/project.
    status: verified
  - id: ac-009
    summary: Regression script nails Parameterizer+OPLSAA public path
    type: runtime
    pass_when: |
      `python regressions/oplsaa-parameterizer-03-parameterizer.py` exits 0;
      constructs Parameterizer(ChemEncoder, OPLSAAForceField), asserts forward
      filled params, reproduces hard-coded energy via to_potentials within
      stated tolerance, asserts no project attr, and demonstrates
      ImportError for ClassicalIMM and ClassIPredictions (literals only;
      no live third-party).
    status: verified
---

# Acceptance criteria

- **ac-001 / ac-007**：`03` 的核心交付是「按 `OPLSAAForceField` schema 挂头并填参」，proper 必须是 OPLS `k1..k4`，并含 LJ + charge。
- **ac-002 / ac-004 / ac-005**：硬切——无 `project`、无 `ClassicalIMM` / `ClassIPredictions` 别名。
- **ac-003 / ac-006**：任务预测面改名后仍只经 `to_potentials()` 取 \(E/F\)。
- **ac-009**：公开 API 回归钉；脚本清扫留给 `04-cut`。
