---
spec: universal-class-i-03-losses
slug: universal-class-i-03-losses
created: 2026-08-13
criteria:
  - id: ac-001
    summary: Project Hessian via Wilson B⁺; never flatten H_x
    type: code
    pass_when: |
      ProjectedInternalHessianLoss.project in
      src/molix/core/losses/molecular.py computes
      H_q = pinv(B).T @ H_x @ pinv(B) for unbatched H_x (3N,3N) and
      B (n_q, 3N). No code path MSE-reduces H_x.reshape(-1) against a
      Cartesian target. Construction is ProjectedInternalHessianLoss(...)
      with no make_*/create_* factory.
    status: pending
  - id: ac-002
    summary: Hessian loss MSE on diag(H_q) or H_q_ref
    type: code
    pass_when: |
      ProjectedInternalHessianLoss.forward reads pred[hessian_key] and
      target[b_matrix_key], projects, then MSE(diag(H_q), k_internal)
      when target_kind is diagonal, or MSE(H_q, H_q_ref) when
      target_kind is hq_ref. Diagonal mode does not penalize off-diagonal
      H_q toward zero. Shape mismatches raise ValueError.
    status: pending
  - id: ac-003
    summary: TorsionProfileLoss reuses center_by_group plus barrier
    type: code
    pass_when: |
      TorsionProfileLoss.forward calls center_by_group on pred and ref
      energies (does not reimplement centering and does not rewrite
      molecule_centered_energy_mse). With barrier_weight=0 the fixture
      E_pred=[1,2,5,7], E_ref=[10,11,0,2], groups=[0,0,1,1] yields 0.
      With E_pred=[0,4], E_ref=[0,2], groups=[0,0], barrier_weight=0.5
      the scalar loss is 3.0. No absolute-energy or HoF term exists.
    status: pending
  - id: ac-004
    summary: ChargeMSELoss is q-only MSE in units of e
    type: code
    pass_when: |
      ChargeMSELoss.forward MSE-reduces pred[q] vs target[q] only.
      Tests/test_molix/test_core/test_losses_molecular.py::TestChargeMSELoss
      asserts MSE([0.2,-0.2],[0.1,-0.1])==0.01 and forward succeeds
      without energy keys. No neutrality penalty and no E(q) coupling.
    status: pending
  - id: ac-005
    summary: ScaleNormalizedLoss is L/s² buffer; WeightedLoss sums it
    type: code
    pass_when: |
      ScaleNormalizedLoss(loss, scale=s) registers s as a buffer (not a
      Parameter) at config.ftype, raises ValueError if s==0, and returns
      loss(pred,target)/s**2. A WeightedLoss of one
      ScaleNormalizedLoss(ChargeMSELoss(), scale=2) on the 0.01 charge
      fixture equals 0.0025. No dataset scan is implemented.
    status: pending
  - id: ac-006
    summary: Export new symbols from existing losses package
    type: code
    pass_when: |
      src/molix/core/losses/__init__.py exports
      ChargeMSELoss, ProjectedInternalHessianLoss, ScaleNormalizedLoss,
      and TorsionProfileLoss; __all__ is alphabetically sorted. No new
      losses package tree is added. combined.py and src/molix/__init__.py
      are unchanged.
    status: pending
  - id: ac-007
    summary: Google docstrings name units on new public symbols
    type: docs
    pass_when: |
      ProjectedInternalHessianLoss, TorsionProfileLoss, ChargeMSELoss,
      and ScaleNormalizedLoss in src/molix/core/losses/molecular.py have
      Google-style docstrings that state units (kcal/mol,
      kcal/mol/Å² or kcal/mol/rad² as labeled, e) and the stationary-point
      assumption on the Hessian path.
    status: pending
  - id: ac-008
    summary: Regression embeds Wilson and scale goldens as literals
    type: runtime
    pass_when: |
      python regressions/universal-class-i-03-losses.py exits 0 and
      asserts ProjectedInternalHessianLoss.project(H_x, B) equals
      [[2.0]] with atol 1e-6, diagonal MSE vs k=2.0 is 0, and
      ScaleNormalizedLoss on a 4.0 inner loss with s=2.0 equals 1.0.
      H_x, B, and the 2.0 / 1.0 references are literals in the script
      (comment: analytic Wilson GF, 2026-08-13); no third-party import
      or subprocess.
    status: pending
  - id: ac-009
    summary: Wilson identity H_q equals k on the 2-atom toy
    type: scientific
    evaluator_hint: path: regressions/universal-class-i-03-losses.py
    pass_when: |
      On the Domain-basis 2-atom stationary harmonic (B=[[-1,0,0,1,0,0]],
      H_x=k BᵀB, k=2.0 kcal/mol/Å²) the projected H_q[0,0] satisfies
      abs(H_q[0,0] - 2.0) <= 1e-6. Adding Cartesian noise on a B-orthogonal
      entry (H_x[1,1]) leaves H_q unchanged within the same tolerance.
    status: pending
out_of_scope:
  - flattened Cartesian Hessian loss
  - live geometry-dependent QEq / E(q)
  - gas-phase LJ residual as a nonbonded loss
  - heats of formation / uncentered absolute QM energy
  - run_everything façade
  - new losses package tree
  - THEMol ingest / condensed-phase observables
---

# Acceptance — universal-class-i-03-losses

本 spec 的“完成”= 四个 `nn.Module` 落在既有 `molecular.py`（无新损失树、无工厂主 API），Wilson 投影在二原子驻点谐振子上给出 \(H_q=k=2.0\)（atol `1e-6`），扭转项复用 `center_by_group`，电荷项不耦合能量，标度项以 buffer \(s\) 做 \(L/s^2\) 并交给 `WeightedLoss`。
