---
slug: pinet-quantization-thermal-noise-03-diagnostics
criteria:
  - id: ac-001
    summary: autocorr_dF returns C(tau) and tau_c with documented shapes/units
    type: code
    pass_when: |
      autocorr_dF(synthetic dF artifact) returns a mapping with keys
      "C_tau" (1D array over lags) and "tau_c" (float, fs); C_tau[0]==C(0)
      and len(C_tau)==requested max_lag+1, no exceptions.
    status: pending
  - id: ac-002
    summary: tau_c separates white from AR(1) on synthetic fixtures (Eq9)
    type: scientific
    pass_when: |
      time-iid synthetic dF gives tau_c <= ~few*dt; AR(1) dF with coeff phi
      gives tau_c within 15% rel-tol of analytic dt*(1+phi)/(1-phi).
    status: pending
  - id: ac-003
    summary: crossdof_covariance off-diagonal ratio flags cross-atom correlation (d)
    type: scientific
    pass_when: |
      independent-per-DoF synthetic dF gives off-diagonal Frobenius ratio
      < 0.05; injected cross-atom-correlated dF gives ratio above the same
      threshold and is reported as flagged.
    status: pending
  - id: ac-004
    summary: momentum_residual detects net force per frame (e, Newton 3rd)
    type: scientific
    pass_when: |
      zero-sum synthetic frames give net-force magnitude ~0 (< 1e-8 eV/A);
      frames with injected constant net force recover that magnitude within
      1% rel-tol.
    status: pending
  - id: ac-005
    summary: energy_drift_slope recovers d<E>/dt by linear fit (f)
    type: scientific
    pass_when: |
      flat E(t) gives slope ~0; E(t) with injected slope s recovers s within
      5% rel-tol and correct sign, units eV/fs.
    status: pending
  - id: ac-006
    summary: stationarity_blocks flags drift across trajectory thirds (g)
    type: scientific
    pass_when: |
      stationary synthetic series gives block-to-block drift in <dF>/var/tau_c
      below threshold (not flagged); series with monotonically increasing
      per-block variance is flagged non-stationary.
    status: pending
  - id: ac-007
    summary: t_eff_colored applies Eq8 colored-noise branch reusing -01 t_eff_estimate
    type: scientific
    pass_when: |
      for tau_c >> dt the integral form int C(tau) dtau replaces <|dF|^2>*dt in
      Eq8; on a fixture where tau_c==dt the colored result matches the reused
      t_eff_estimate white-noise scalar within 1e-6 rel-tol; output is
      T_eff/T_target and scales as 1/gamma.
    status: pending
  - id: ac-008
    summary: diffusion via Einstein and Green-Kubo agree on known-D fixtures (Eq10/Eq11)
    type: scientific
    pass_when: |
      random-walk fixture with known D recovers D via diffusion_einstein within
      10% rel-tol; ideal-gas toy gives diffusion_einstein and diffusion_green_kubo
      agreeing within 10% rel-tol; D in A^2/fs.
    status: pending
  - id: ac-009
    summary: rdf and vacf return correctly shaped, normalized arrays
    type: code
    pass_when: |
      rdf(pos artifact) returns (r_centers, g_r) of equal length with g_r->~1
      at large r on a uniform fixture; vacf(vel artifact) returns C_v with
      C_v[0]==<v(0).v(0)> and len==requested lags; no exceptions.
    status: pending
  - id: ac-010
    summary: diagnose_trajectory aggregates criteria c-h into one diagnostic dict
    type: code
    pass_when: |
      diagnose_trajectory(synthetic -02-shaped artifact, gamma=, dt=) returns a
      dict containing keys for criteria c,d,e,f,g and dynamical-h
      (tau_c, crossdof_offdiag, momentum_residual, energy_drift_slope,
      stationarity, T_eff_ratio, D_einstein, D_green_kubo) without raising.
    status: pending
  - id: ac-011
    summary: full check + test suite passes
    type: runtime
    pass_when: |
      the project check + pytest run over examples/molzoo/tests/test_md_diagnostics.py
      completes green with no failures or errors.
    status: pending
---

# Acceptance criteria

- **ac-001 / ac-009 (code)** — kernel API/shape contracts for the time-series and structural kernels; verified by shape/exception assertions independent of physics.
- **ac-002…ac-008 (scientific)** — each binds a kernel to a synthetic fixture with a known analytic answer, so numerical correctness of Eq8–Eq11 is provable without any real PiNet trajectory.
- **ac-010 (code)** — the `diagnose_trajectory` aggregator surfaces every owned criterion (c,d,e,f,g and dynamical h) as a key; presence + no-raise, not value correctness (values are covered by the per-kernel scientific criteria).
- **ac-011 (runtime)** — full check + suite green; subsumes implicit smoke/build checks.

criteria a,b (static unbiasedness/Gaussianity) are intentionally NOT owned here — they belong to -01 via the reused `summarize_delta`. The cross-condition verdict synthesis is -04. Files live under `examples/molzoo/` which is gitignored, so neither file will appear in a git diff.
