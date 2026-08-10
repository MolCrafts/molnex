---
slug: learnable-classical-ff-08-ff-export
criteria:
  - id: ac-001
    summary: Bond k golden 100 kcal/Å² → 41840 kJ/nm²
    type: scientific
    pass_when: |
      Convention helper or OpenMMAdapter translation maps k_ir=100 to
      k_omm=41840 (kJ/mol/nm^2) exactly (or allclose with atol=0). Documented
      factor 4.184 / 0.01. Test in unit or regression suite without openmm.
    status: pending
  - id: ac-002
    summary: Torsion Vn=2 → k_omm=4.184 golden
    type: scientific
    pass_when: |
      IR proper torsion coefficient corresponding to Vn=2 kcal/mol in the
      documented [1+cos] form maps to OpenMM PeriodicTorsion k=4.184 kJ/mol.
      Hard-coded golden; no live openmm.
    status: pending
  - id: ac-003
    summary: TranslationCase four-way enum is used by ConventionTable
    type: code
    pass_when: |
      TranslationCase includes DIRECT_UNIT_SCALE, FORM_REPARAMETERIZE, DECOMPOSE,
      UNSUPPORTED; ConventionTable rows reference these cases for bond, angle,
      proper, and at least one unsupported placeholder path.
    status: pending
  - id: ac-004
    summary: UNSUPPORTED terms raise structured errors (no silent drop)
    type: code
    pass_when: |
      Compiling an IR bag marked unsupported (or unknown term) raises a clear
      exception naming the term and case; ForceSpec does not omit the term
      silently.
    status: pending
  - id: ac-005
    summary: OpenMMAdapter emits serializable ForceSpec for Class-I core terms
    type: code
    pass_when: |
      Given a PotentialIR with bond, angle, proper, lj, charge (+ scaling),
      adapter.translate returns ForceSpec whose to_dict is JSON-serializable and
      contains the expected force groups with OpenMM units.
    status: pending
  - id: ac-006
    summary: Nonbonded 1-4 scales flow from IR NonbondedScaling
    type: code
    pass_when: |
      ForceSpec nonbonded exceptions / scale fields reflect scale_q_14=5/6 and
      scale_lj_14=0.5 from Class-I defaults when IR carries NonbondedScaling.
    status: pending
  - id: ac-007
    summary: Default tests never require live openmm import
    type: code
    pass_when: |
      tests/test_molix/test_ff_export/* and regression goldens do not import
      openmm at module level; any live openmm test uses importorskip and is
      optional.
    status: pending
  - id: ac-008
    summary: BackendAdapter registry resolves "openmm" to OpenMMAdapter
    type: code
    pass_when: |
      ForceFieldCompiler(adapter="openmm") or registry lookup returns
      OpenMMAdapter; unknown backend name raises KeyError/ValueError.
    status: pending
  - id: ac-009
    summary: Google docstrings cite OpenMM §19 and unit goldens
    type: docs
    pass_when: |
      ConventionTable / OpenMMAdapter / ForceFieldCompiler document units and
      reference OpenMM §19; ruff clean.
    status: pending
  - id: ac-010
    summary: Full unit check + test suite pass
    type: runtime
    pass_when: |
      ruff check/format --check and pytest tests/ exit 0 on the implementing branch.
    status: pending
---

# Acceptance criteria

- ac-001 / ac-002 are the scientific unit goldens (bond k and torsion Vn).
- ac-003 / ac-004 lock the 4-way TranslationCase contract including hard fail on unsupported.
- ac-005 / ac-006 / ac-008 cover OpenMM force-spec emission, 1-4 scales, and registry.
- ac-007 forbids live OpenMM as a CI dependency.
- ac-009 / ac-010 docs + suite.
