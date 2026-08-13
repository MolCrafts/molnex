---
spec: universal-class-i-02-fourier
slug: universal-class-i-02-fourier
created: 2026-08-13
criteria:
  - id: ac-001
    summary: ProperTorsionParamHead emits unconstrained a,b (default n=1..6)
    type: code
    pass_when: |
      ProperTorsionParamHead(feature_dim=D).forward(x) with x.shape (N, D)
      returns keys a, b, periodicity, idivf only (no k, phase, or c0);
      a and b have shape (N, 6); periodicity is [1, 2, 3, 4, 5, 6];
      idivf is ones (N,); values are unconstrained (may be negative).
      MLP path has no softplus on the torsion coordinates.
    status: pending
  - id: ac-002
    summary: ImproperParamHead defaults to periodic n=1,2
    type: code
    pass_when: |
      ImproperParamHead(feature_dim=D)() with no style kwargs emits periodic
      keys k, phase, periodicity, idivf with periodicity.tolist()==[1, 2]
      and no chi0. Explicit include_harmonic=True, include_periodic=False
      still emits harmonic k, chi0.
    status: pending
  - id: ac-003
    summary: BondParamHead and AngleParamHead constraints stay unchanged
    type: code
    pass_when: |
      TestBondParamHead and TestAngleParamHead in
      tests/test_molpot/test_composition/test_mm_heads.py still assert
      k>0, r0>0, and theta0 in (0, pi); those classes' forward
      implementations are not rewritten.
    status: pending
  - id: ac-004
    summary: from_cartesian maps (a,b) to (k,γ) at 1e-12
    type: scientific
    evaluator_hint: path: tests/test_molpot/test_ir/test_bags.py
    pass_when: |
      ProperTorsionBag.from_cartesian reproduces the hard-coded pairs
      (k,γ)=(2,0)→(a,b)=(2.0, 0.0); (k,γ)=(1.5, π)→(a,b)=(-1.5, 0.0);
      (a,b)=(0.6, 0.8)→(k,γ)=(1.0, 0.9272952180016123) at atol 1e-12;
      bag fields are only k, phase, periodicity, idivf (no c0).
    status: pending
  - id: ac-005
    summary: to_ir stores cosine tables and rejects mixed keys
    type: code
    pass_when: |
      ClassIForceField.with_params(propers={a,b,periodicity,idivf}).to_ir()
      yields ProperTorsionBag with k=hypot(a,b) and phase=atan2(b,a);
      PotentialIR.propers has no c0. Passing both (a,b) and (k,phase)
      raises ValueError. Existing k1..k4 and (k,phase) branches still work.
    status: pending
  - id: ac-006
    summary: Converted kernel matches k[1+cos] identity at 1e-10
    type: scientific
    evaluator_hint: path: tests/test_molpot/test_forcefield/test_to_ir.py
    pass_when: |
      After a,b → (k,γ) conversion, ProperTorsionPeriodic / to_potentials
      energy equals k[1+cos(nφ-γ)] and k + a cos(nφ) + b sin(nφ) for
      φ=π/3, n=2, k=2, γ=π/4; hard-coded value 2.5176380902050415
      at atol 1e-10. Kernel formula itself is unchanged.
    status: pending
  - id: ac-007
    summary: project stays fail-loud off the OPLS manifold
    type: code
    pass_when: |
      ClassicalIMM.project onto an opls dihedral raises ValueError matching
      n=6 when a n=6 Cartesian term is nonzero, and raises ValueError
      matching sin or phase when (a,b)=(0, 0.4) at n=1 (γ=π/2).
      On-manifold b=0, n=1..4 still yields K_odd=2a, K_even=-2a.
    status: pending
  - id: ac-008
    summary: Regression reproduces mapping goldens and sine fail-loud
    type: runtime
    evaluator_hint: path: regressions/universal-class-i-02-fourier.py
    pass_when: |
      regressions/universal-class-i-02-fourier.py exits 0 and reproduces
      hard-coded literals (no third-party at runtime):
      (2,0)↔(2.0,0.0); (1.5,π)↔(-1.5,0.0);
      (0.6,0.8)↔(1.0, 0.9272952180016123) at atol 1e-12;
      φ=π/3 n=2 k=2 γ=π/4 energy 2.5176380902050415 at atol 1e-10;
      project of (a,b)=(0,0.4) n=1 raises ValueError.
    status: pending
  - id: ac-009
    summary: Google docstrings state CLASS_I_CANONICAL units
    type: docs
    pass_when: |
      ProperTorsionParamHead and ProperTorsionBag.from_cartesian have
      Google-style docstrings naming a,b,k in kcal/mol and phase in rad,
      and stating c0 is not a table parameter.
    status: pending
  - id: ac-010
    summary: No new ForceField façade, to_molpy, or engine records
    type: code
    pass_when: |
      No new public ForceField classmethod or to_molpy helper is added;
      ProperTorsionParamHead.forward returns only a param dict;
      ClassIForceField / ClassicalIMM / PotentialIR remain the user path.
    status: pending
out_of_scope:
  - ChemEncoder MP / universal-class-i-01-encoder
  - losses, PES bundle, THEMol
  - LJ unfreeze, live QEq, OpenFF trefoil averaging
  - scripts/mm_param_learning
  - c0 / heat-of-formation table parameter
  - new ForceField façade, to_molpy, engine records from the head
  - BondHarmonic / ProperTorsionPeriodic energy-formula changes
  - BondParamHead / AngleParamHead rewrites
  - Condenser / ForceFieldCompiler
---

# Acceptance — universal-class-i-02-fourier

本 spec「完成」= 真二面角以无约束 \((a_n,b_n)\) 训练，IR/核仍吃 \((k,\gamma,s)\)，OPLS 投影在正弦残差或 \(n>4\) 上 fail-loud，improper 默认周期 \(n=1,2\)，Bond/Angle 不动。
