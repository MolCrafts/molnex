---
slug: pinet-quantization-thermal-noise-04-verdict
criteria:
  - id: ac-001
    summary: Clean fixture classified as thermal-noise-approximable
    type: scientific
    pass_when: |
      A synthetic white-Gaussian-unbiased-momentum-conserving fixture
      passes all 8 criteria and classify_cell returns
      "thermal-noise-approximable" with a reported T_eff_gamma_dt_ratio.
    status: pending
  - id: ac-002
    summary: Each single-criterion violation yields correct named form
    type: scientific
    pass_when: |
      For each of the 8 criteria, a fixture violating only that criterion
      classifies as "structured-non-thermal" and characterize_failure
      returns the form named in the Domain basis failure-form mapping
      (a=force-field distortion, b=non-Gaussian, c=colored/quenched,
      d=spatially-correlated, e=COM-momentum injection,
      f=unbalanced injection/needs thermostat friction,
      g=configuration-locked distortion, h=altered structure/dynamics).
    status: verified
    last_checked: 2026-06-21
    note: |
      Relocated to src/molix/analysis/verdict.py (_FAILURE_FORM:27-36);
      all 8 a-h forms covered by test_each_single_violation_is_structured_with_named_form
      (tests/test_molix/test_verdict.py:55, PASS).
  - id: ac-003
    summary: Combined biased+correlated fixture reports PES-distortion form
    type: scientific
    pass_when: |
      A fixture violating criteria a and d together is characterized as
      "biased + spatially-correlated => PES distortion, not noise"
      with criteria ordered by the documented physical priority.
    status: verified
    last_checked: 2026-06-21
    note: |
      characterize_failure (verdict.py:99-100) + _PRIORITY:38; asserted by
      test_biased_plus_correlated_is_pes_distortion
      (tests/test_molix/test_verdict.py:66, PASS).
  - id: ac-004
    summary: Threshold constants are explicit, named, and documented
    type: code
    pass_when: |
      VERDICT_THRESHOLDS exposes named constants BIAS_TOL_SIGMA,
      SKEW_TOL, EXKURT_TOL, TAU_C_DT_FACTOR, COV_OFFDIAG_TOL,
      SUM_DF_TOL_SIGMA, ENERGY_DRIFT_TOL, STATIONARITY_P,
      OBSERVABLE_DELTA_TOL, each with a docstring stating units/rationale,
      and classify_cell accepts an injected thresholds argument.
    status: pending
  - id: ac-005
    summary: Machine table has stable schema and no silent drops
    type: code
    pass_when: |
      build_machine_table returns rows with the condition key,
      8 boolean criterion columns, verdict, form, and T_eff_ratio;
      a missing condition key from either input table appears as an
      incomplete-cell row rather than being silently dropped.
    status: verified
    last_checked: 2026-06-21
    note: |
      build_machine_table (verdict.py:118-129) + incomplete-row path
      (run_verdict:165-174); asserted by test_machine_table_schema_is_stable
      and test_run_verdict_joins_and_flags_incomplete
      (tests/test_molix/test_verdict.py:73,91, PASS).
  - id: ac-006
    summary: Human-readable report renders per-cell verdict and T_eff
    type: code
    pass_when: |
      render_report output (Markdown) lists, per condition cell, the
      verdict, the failed criteria, the named form (when non-thermal),
      and the T_eff_gamma_dt_ratio (when thermal).
    status: pending
  - id: ac-007
    summary: Full variable-matrix fixture classified cell-by-cell
    type: scientific
    pass_when: |
      A multi-cell synthetic matrix mixing thermal and several
      non-thermal cells (quant scheme x trained-precision x dataset x
      MD condition) yields the correct verdict and named form for every
      cell when run through run_verdict.
    status: pending
  - id: ac-008
    summary: Full check and test suite pass
    type: runtime
    pass_when: |
      The project check command and the full test suite (including
      examples/molzoo/tests/test_verdict.py) run to completion with
      zero failures.
    status: pending
---

# Acceptance criteria

- **ac-001 / ac-002 / ac-003 / ac-007** (scientific) — 判定逻辑的正确性核心：clean fixture 必须判为热噪声；每条判据的单一违背与典型组合违背必须映射到 Domain basis 中命名的物理形式；全矩阵 fixture 逐单元正确。这些在已知答案的合成数据上验证，独立于任何真实运行。
- **ac-004 / ac-005 / ac-006** (code) — 阈值作为显式命名常量（带依据）的可审计性，以及报告/机器表的稳定 schema 与不静默丢弃缺失单元的契约。
- **ac-008** (runtime) — 完整 check 与测试套件通过。
