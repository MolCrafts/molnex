---
slug: oplsaa-parameterizer-04-cut
criteria:
  - id: ac-001
    summary: Banned Class-I product names absent from molpot public exports
    type: code
    pass_when: |
      `src/molpot/__init__.py` and `src/molpot/forcefield/__init__.py` contain
      zero occurrences of the identifiers ClassIForceField, OPLSForceField,
      ClassicalIMM, and ClassIPredictions (including __all__ and imports);
      `from molpot import ClassIForceField` (and the other three) each raise
      ImportError; Parameterizer, OPLSAAForceField, and ForceFieldPredictions
      are present in both modules' __all__.
    status: verified

  - id: ac-002
    summary: mm_param_learning main path has zero banned-name strings
    type: code
    pass_when: |
      Under `scripts/mm_param_learning/`, ripgrep/count of ClassIForceField,
      OPLSForceField, ClassicalIMM, and ClassIPredictions is zero in every
      .py file; espaloma_mm_protocol_train.build_espaloma_like_model returns a
      Parameterizer whose force-field template is OPLSAAForceField;
      class_i_fit prediction path types ForceFieldPredictions (or equivalent
      Parameterizer→to_potentials composition) with no ClassicalIMM import.
    status: verified

  - id: ac-003
    summary: Export-cut unit tests pass for banned names and public symbols
    type: runtime
    pass_when: |
      `python -m pytest tests/test_molpot/test_forcefield/test_exports_cut.py -v`
      exits 0; TestBannedNameAbsence and TestPublicExports green.
    status: verified

  - id: ac-004
    summary: Espaloma protocol builder unit tests pass on Parameterizer path
    type: runtime
    pass_when: |
      `python -m pytest tests/test_scripts/test_mm_param_learning/test_espaloma_mm_protocol_train.py -v`
      exits 0; TestBuildEspalomaLikeModel and TestMMEnergyModel green.
    status: verified

  - id: ac-005
    summary: Migrated forcefield/MD unit mirrors pass without banned imports
    type: runtime
    pass_when: |
      `python -m pytest tests/test_molpot/test_forcefield/test_parameterizer.py
      tests/test_molpot/test_forcefield/test_predictions.py
      tests/test_molpot/test_forcefield/test_container.py
      tests/test_molix/test_md/test_driver.py -q` exits 0; none of those files
      import or mention ClassIForceField, OPLSForceField, ClassicalIMM, or
      ClassIPredictions.
    status: verified

  - id: ac-006
    summary: Chain regression pins f14=0.5 and OPLSAA harmonic energy 0.25
    type: scientific
    evaluator_hint: "marker: oplsaa-parameterizer"
    pass_when: |
      `regressions/oplsaa-parameterizer.py` embeds literal goldens
      scale_q_14=0.5, scale_lj_14=0.5 on OPLSAAForceField nonbonded_scaling
      and bond energy E=0.25 kcal/mol for k=2, r0=1 Å, r=1.5 Å via
      to_potentials().calc_energy (atol 1e-10); script documents OPLS-AA
      DOI 10.1021/ja9621760; no live third-party oracle.
    status: verified

  - id: ac-007
    summary: regressions/oplsaa-parameterizer.py exits 0 on public API path
    type: runtime
    pass_when: |
      `python regressions/oplsaa-parameterizer.py` exits 0 using only public
      molpot APIs (OPLSAAForceField and/or Parameterizer + to_potentials);
      no ClassIForceField/OPLSForceField/ClassicalIMM/ClassIPredictions
      strings in that file; no project() call.
    status: verified

  - id: ac-008
    summary: Listed legacy regressions migrated or removed without project()
    type: runtime
    pass_when: |
      `regressions/learnable-classical-ff-project-opls.py` is deleted or
      rewritten with zero project() and zero banned names;
      `regressions/learnable-classical-ff-05-neural-parameterizer.py`,
      `regressions/universal-class-i-08-adapter.py`, and
      `regressions/md-forcefield-molpy-reexport-01-reexport.py` contain zero
      banned-name strings and still exit 0 when executed (or are explicitly
      removed if obsolete); `scripts/mm_param_learning/parity_project_md.py`
      is deleted or rewritten without project().
    status: verified

  - id: ac-009
    summary: Notes and architecture document Parameterizer+OPLSAA user path
    type: docs
    pass_when: |
      `.claude/notes/learnable-classical-ff.md` describes user path with
      Parameterizer + OPLSAAForceField + to_potentials and contains zero
      ClassIForceField, OPLSForceField, ClassicalIMM, ClassIPredictions
      product-name endorsements (no project() contract section);
      `.claude/notes/architecture.md` molpot.forcefield / public-surface
      bullets name Parameterizer, OPLSAAForceField, ForceFieldPredictions
      instead of ClassicalIMM/ClassIForceField/OPLSForceField/ClassIPredictions;
      CLAUDE.md free-form ForceField reuse bullet matches the same names.
    status: verified

  - id: ac-010
    summary: Full molnex check and test suite pass after cut
    type: runtime
    pass_when: |
      From molnex root, `ruff check src/ tests/ scripts/ regressions/ &&
      ruff format --check src/ tests/ scripts/ regressions/ &&
      ty check src/ --exit-zero-on-warning` and
      `python -m pytest tests/ -q -n 12 --dist worksteal` both exit 0.
    status: verified
---

# Acceptance criteria

- **ac-001 / ac-002**：代码面硬切——公开再导出与 `mm_param_learning` 主路径对四个禁用标识符为零命中，并改挂 `Parameterizer`+`OPLSAAForceField`。
- **ac-003 / ac-004 / ac-005 / ac-008 / ac-010**：运行时——新单测、迁移后的镜像单测/回归、以及全量 check+pytest 全绿。
- **ac-006 / ac-007**：科学钉 + 回归脚本——`f14=0.5` 与谐振子 \(E=0.25\) 以字面量落在 `regressions/oplsaa-parameterizer.py` 且脚本可独立跑通。
- **ac-009**：文档/notes——绑定笔记与 architecture 公开名与 Parameterizer 叙事一致，且不恢复 `project()`。
