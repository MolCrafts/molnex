---
slug: learnable-classical-ff-03-mm-heads
criteria:
  - id: ac-001
    summary: BondParamHead emits positive k and r0 with correct shapes
    type: code
    pass_when: |
      BondParamHead(feature_dim=D)(features (N,D)) returns dict with k,r0 each
      (N,), both > configured floors for random and large-negative inputs
      (softplus path).
    status: pending
  - id: ac-002
    summary: AngleParamHead emits positive k and valid theta0
    type: code
    pass_when: |
      AngleParamHead outputs k (N,)>0 and theta0 (N,) in a documented valid
      range (e.g. (0, pi] or softplus-constrained); shapes match N angles.
    status: pending
  - id: ac-003
    summary: ProperTorsionParamHead multi-term k/phase shapes
    type: code
    pass_when: |
      With n_terms=T, head returns k with shape (N,T) all >= 0 and phase (N,T)
      unconstrained; periodicity is either fixed buffer or integer output as
      documented in the head docstring.
    status: pending
  - id: ac-004
    summary: ImproperParamHead covers harmonic and/or periodic modes
    type: code
    pass_when: |
      ImproperParamHead config selects harmonic (k, chi0) and/or periodic
      (k, phase, periodicity) outputs; positivity on k holds.
    status: pending
  - id: ac-005
    summary: ChargeHead and LJParameterHead still work via MultiHead
    type: code
    pass_when: |
      MultiHead({"lj": LJParameterHead(...), "q": ChargeHead(...)}) merges
      epsilon, sigma, charge without key collision; charge neutrality per
      molecule still holds when batch index provided.
    status: pending
  - id: ac-006
    summary: ClassicalMMComposer.parameterize builds CLASS_I PotentialIR
    type: code
    pass_when: |
      Given topology batch namespaces + feature tensors, parameterize returns
      PotentialIR with unit_system matching CLASS_I_CANONICAL and bags populated
      for the enabled terms.
    status: pending
  - id: ac-007
    summary: ClassicalMMComposer energy matches analytic bond harmonic golden
    type: scientific
    pass_when: |
      With mocked/constant bond parameters k,r0 and a two-atom geometry,
      composer energy equals 0.5*k*(r-r0)^2 within 1e-5 relative/absolute
      tolerance (kcal/mol, Å).
    status: pending
  - id: ac-008
    summary: Forces via ForceDerivation only (no hand-rolled force in composer)
    type: code
    pass_when: |
      Composer energy path differentiates w.r.t. pos through ForceDerivation
      or BasePotential.calc_forces; source of classical_mm.py does not implement
      a third analytic force formula for Class-I terms.
    status: pending
  - id: ac-009
    summary: classical_mm.py does not import molzoo or molrep.chem
    type: code
    pass_when: |
      Static check: composition/classical_mm.py and mm heads modules have no
      molzoo / molrep.chem imports (features are injected).
    status: pending
  - id: ac-010
    summary: Public heads and ClassicalMMComposer have Google docstrings + units
    type: docs
    pass_when: |
      BondParamHead, AngleParamHead, ProperTorsionParamHead, ImproperParamHead,
      ClassicalMMComposer document shapes and CLASS_I units; ruff clean.
    status: pending
  - id: ac-011
    summary: Full unit check + test suite pass
    type: runtime
    pass_when: |
      ruff check/format --check and pytest tests/ exit 0 on the implementing branch.
    status: pending
---

# Acceptance criteria

- ac-001–ac-005 lock continuous head contracts and MultiHead/ChargeHead/LJ reuse.
- ac-006–ac-008 are the composer IR + energy + force path gates (analytic bond golden).
- ac-009 enforces no-encoder boundary (04/05 own perception/parameterizer).
- ac-010 / ac-011 docs + suite.
