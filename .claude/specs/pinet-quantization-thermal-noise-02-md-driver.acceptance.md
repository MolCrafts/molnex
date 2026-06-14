---
slug: pinet-quantization-thermal-noise-02-md-driver
criteria:
  - id: ac-001
    summary: Integrator conserves energy in NVE harmonic-oscillator run
    type: scientific
    pass_when: |
      With gamma=0 on the analytic harmonic potential F=-k x, total energy
      E = 0.5*m*v^2 + 0.5*k*x^2 drifts by <= the tolerance asserted in
      test_md_driver.py over the full step count (no model involved).
    status: pending
  - id: ac-002
    summary: Langevin run reaches target equipartition temperature
    type: scientific
    pass_when: |
      With gamma>0 on the harmonic potential, time-averaged kinetic energy
      converges to 0.5*d*k_B*T and the equipartition temperature lands within
      the asserted tolerance of T_target in test_md_driver.py.
    status: pending
  - id: ac-003
    summary: Random kick is seed-reproducible and FDT-scaled
    type: code
    pass_when: |
      Two integrator runs with the same torch.Generator seed produce identical
      trajectories, and the per-DoF kick variance scales as 2*gamma*m*k_B*T/dt;
      asserted in test_md_driver.py.
    status: pending
  - id: ac-004
    summary: LangevinVerletIntegrator is model-agnostic via injected force_fn
    type: code
    pass_when: |
      LangevinVerletIntegrator drives both the analytic harmonic force_fn and a
      PiNetPotential-backed force_fn through the same interface with no
      model-specific branches; test instantiates it with the toy potential and
      steps without importing molzoo.pinet.
    status: pending
  - id: ac-005
    summary: Paired-trajectory artifact carries full schema
    type: code
    pass_when: |
      Both reference and quantized TrajectoryArtifact instances expose pos, vel,
      E, F_ref, F_quant, dF arrays with shapes (T,N,3)/(T,) and metadata keys
      {T, gamma, dt, mass, N, dof, scheme, ckpt_precision, dataset, molecule,
      seed, integrator}; verified in test_md_driver.py.
    status: pending
  - id: ac-006
    summary: ΔF time series equals F_quant - F_ref along reference trajectory
    type: code
    pass_when: |
      evaluate_delta_along_trajectory returns dF whose length equals the step
      count and equals F_quant - F_ref elementwise at each frame; asserted in
      test_md_driver.py.
    status: pending
  - id: ac-007
    summary: build_force_fn keeps pos a live autograd leaf (no detached output)
    type: code
    pass_when: |
      The force_fn from build_force_fn returns forces obtained from
      model(td, compute_forces=True) where ("atoms","pos") is a requires_grad
      leaf and the output forces are NOT detached before integration; verified
      by a gradient-flow assertion in test_md_driver.py.
    status: pending
  - id: ac-008
    summary: Optional ASE shim matches the force seam when ASE present
    type: code
    evaluator_hint: skip-if-no-ASE (pytest.importorskip("ase"))
    pass_when: |
      When ase is importable, PiNetCalculator.get_forces equals
      PiNetPotential.forward(..., compute_forces=True)["forces"] elementwise
      within tolerance on the same configuration; test is skipped (not failed)
      when ase is absent.
    status: pending
  - id: ac-009
    summary: Full check and test suite pass
    type: runtime
    pass_when: |
      The project check command plus the full test suite (including
      test_md_driver.py) run green.
    status: pending
---

# Acceptance criteria

- ac-001 / ac-002 / ac-003 establish integrator correctness independently of PiNet on the analytic harmonic oscillator (NVE energy conservation, Langevin equipartition temperature, FDT-scaled reproducible noise). These are the science gate for trajectory trustworthiness — this sub-spec owns integrator correctness, not the thermal-noise verdict.
- ac-004 enforces the model-agnostic injected-force design so the integrator is unit-testable without a real PiNet.
- ac-005 / ac-006 bind the paired-trajectory protocol and artifact schema that -03 consumes (per-step ΔF(t) time series for autocorrelation).
- ac-007 guards the load-bearing deviation from `_forward_energy_forces`: MD needs live forces, so `("atoms","pos")` stays a requires_grad leaf and forces are not detached before integration.
- ac-008 covers the optional ASE-Calculator shim, skipped when ASE is absent.
- ac-009 is the full check + suite gate.
