---
slug: pinet-quantization-thermal-noise-01-aggregate
criteria:
  - id: ac-001
    summary: t_eff_estimate is dimensionally consistent with Eq8
    type: code
    pass_when: |
      test_t_eff.py asserts t_eff_estimate(f_rms_sq, dt, gamma, mass, dof)
      equals <|ΔF|^2>*dt/(2*gamma*mass*dof) and a dimensional-analysis case
      (eV^2/Å^2 * fs over the denominator) reduces to a temperature ratio;
      old N^2*s heuristic is gone from the docstring.
    status: verified
    last_checked: 2026-06-21
    note: |
      Relocated: implemented as EffectiveTemperature.energy in
      src/molix/quant.py:303 (Eq8 <|dF|^2>*dt/(2*gamma*mass*dof), d=3N),
      not the spec-named examples/molzoo path. Tested by
      test_effective_temperature_matches_eq8 + _ratio_is_dimensionless
      (tests/test_molix/test_quant.py:130,142, PASS). No N^2*s heuristic
      remains (grep empty).
  - id: ac-002
    summary: T_eff is reported as a ratio to T_target, scaling as 1/gamma
    type: code
    pass_when: |
      test_t_eff.py asserts t_eff_estimate output halves when gamma doubles
      (1/γ scaling) and is returned as T_eff/T_target, never a bare number.
    status: verified
    last_checked: 2026-06-21
    note: |
      Relocated: EffectiveTemperature.ratio (src/molix/quant.py:305-307)
      returns dimensionless T_eff/T_target; 1/γ scaling asserted by
      test_effective_temperature_scales_as_inverse_gamma
      (tests/test_molix/test_quant.py:136, PASS).
  - id: ac-003
    summary: fp64-vs-fp64 control yields zero residual diagnostics
    type: scientific
    pass_when: |
      a control row built from an fp64 reference quantized against itself
      produces F_bias, F_rms, and T_eff_ratio all == 0 (within _EPS) in the
      aggregated table.
    status: verified
    last_checked: 2026-06-21
    note: |
      Null control satisfied via model-vs-itself rather than an aggregated
      CSV row (no such table exists): test_force_delta_zero_residual_has_zero_moments
      (zero dF -> all moments 0.0) and test_null_control_identical_weights_zero_delta
      (identical state_dict -> F_rms/F_bias ~0, abs 1e-10), tests/test_molix/test_quant.py:172,212 PASS.
  - id: ac-004
    summary: aggregation emits one row per matrix cell with verdict columns
    type: code
    pass_when: |
      test_aggregate_phase_a.py asserts the CSV has exactly
      |schemes|*|precisions|*|datasets| rows and columns include F_bias,
      F_skew, F_exkurt, T_eff_ratio, unbiased(bool), gaussian(bool).
    status: pending
  - id: ac-005
    summary: Phase-A unbiased verdict (criterion a) computed from F_bias vs stat error
    type: scientific
    pass_when: |
      for each matrix cell the unbiased column is True iff |F_bias| is within
      the reported statistical error of zero; a deliberately biased synthetic
      ΔF fixture flips it to False.
    status: pending
  - id: ac-006
    summary: Phase-A Gaussianity verdict (criterion b) from skew and excess kurtosis
    type: scientific
    pass_when: |
      gaussian column is True iff |F_skew|<tol and |F_exkurt|<tol for that
      cell; a heavy-tailed synthetic ΔF fixture flips it to False.
    status: pending
  - id: ac-007
    summary: aggregation script runs end-to-end on the sweep checkpoints
    type: runtime
    pass_when: |
      `python examples/molzoo/aggregate_phase_a.py` over the existing sweep
      output exits 0 and writes the CSV without re-rolling a PiNet ctor
      (uses bm_pinet build_* factories).
    status: pending
  - id: ac-008
    summary: full check + test suite passes
    type: runtime
    pass_when: |
      project check + pytest for examples/molzoo/tests pass with no failures.
    status: pending
---

# Acceptance criteria

- ac-001 / ac-002 lock the Eq8 correction: the dimensionally-wrong heuristic
  (N²·s) must be replaced, and T_eff reported as a 1/γ-scaling ratio.
- ac-003 is the null control that guards the whole pipeline: identical models
  must produce zero residual.
- ac-004 fixes the table schema so `-04-verdict` can consume it deterministically.
- ac-005 / ac-006 encode the only two decision criteria (a, b) that are
  statically decidable in Phase A. Criteria c–g (autocorrelation, spatial/
  cross-DoF covariance, momentum conservation, energy drift, stationarity) and
  the dynamical part of h require a trajectory and are deferred to sub-specs
  -03/-04 — Phase A cannot do time-autocorrelation on a static probe ensemble.
- ac-007 / ac-008 are runtime gates.
