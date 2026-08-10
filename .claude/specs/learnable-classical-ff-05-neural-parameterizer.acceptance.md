---
slug: learnable-classical-ff-05-neural-parameterizer
criteria:
  - id: ac-001
    summary: ClassicalMMParameterizer lives in parameterizer.py not classical_mm.py
    type: code
    pass_when: |
      ClassicalMMParameterizer is defined in
      src/molpot/composition/parameterizer.py. classical_mm.py remains the
      ClassicalMMComposer home from sub-spec 03 (not re-homed here).
    status: pending
  - id: ac-002
    summary: FakeEncoder Protocol drives encode -> parameterize -> energy
    type: code
    pass_when: |
      A FakeEncoder nn.Module providing ChemEmbeddings-like tensors runs through
      ClassicalMMParameterizer and returns finite scalar energy without importing
      molzoo in the parameterizer module.
    status: pending
  - id: ac-003
    summary: PotentialIR from parameterize uses CLASS_I kcal/mol units
    type: code
    pass_when: |
      ir = parameterizer.parameterize(batch) has unit_system / units matching
      CLASS_I_CANONICAL (kcal/mol, Å, e, rad). Test fails if primary unit is eV.
    status: pending
  - id: ac-004
    summary: Optional eV conversion only at explicit boundary helper
    type: code
    pass_when: |
      If an eV conversion helper exists, IR bags remain kcal/mol before
      conversion; helper documents the factor. If no helper ships in this spec,
      document that loss-side conversion is caller's job and IR stays kcal/mol.
    status: pending
  - id: ac-005
    summary: Forces via ForceDerivation only
    type: code
    pass_when: |
      compute_forces=True path uses ForceDerivation (or BasePotential.calc_forces);
      forces shape (N,3); parameterizer source does not hand-roll Class-I force
      formulas.
    status: pending
  - id: ac-006
    summary: Reuses ClassicalMMComposer / MM heads from 03
    type: code
    pass_when: |
      Parameterizer constructs or accepts ClassicalMMComposer (or equivalent head
      bundle from 03); does not duplicate BondParamHead implementations inline.
    status: pending
  - id: ac-007
    summary: molpot composition parameterizer does not import molzoo
    type: code
    pass_when: |
      Grep/AST of src/molpot/composition/parameterizer.py (and classical_mm.py)
      shows no molzoo imports.
    status: pending
  - id: ac-008
    summary: Public API has primitive methods not only a god forward
    type: code
    pass_when: |
      ClassicalMMParameterizer exposes encode / parameterize / energy (names may
      vary slightly but must be separable primitives); forward is a thin chain.
      Documented in class docstring.
    status: pending
  - id: ac-009
    summary: Google docstrings with units and shapes
    type: docs
    pass_when: |
      ClassicalMMParameterizer and ChemEncoderProtocol document tensor shapes and
      CLASS_I units; ruff clean on touched paths.
    status: pending
  - id: ac-010
    summary: Full unit check + test suite pass
    type: runtime
    pass_when: |
      ruff check/format --check and pytest tests/ exit 0 on the implementing branch.
    status: pending
---

# Acceptance criteria

- ac-001 locks file ownership vs 03 (parameterizer.py new; classical_mm.py not reclaimed).
- ac-002 / ac-006 / ac-007 / ac-008 define composition + Protocol + primitive API + import boundary.
- ac-003 / ac-004 enforce IR units kcal/mol with eV only optional at boundary.
- ac-005 forces via ForceDerivation only.
- ac-009 / ac-010 docs + suite.
