---
slug: learnable-classical-ff-09-provenance
criteria:
  - id: ac-001
    summary: ChemicalSupportIndex membership and coverage_fraction
    type: code
    pass_when: |
      Index built with known type_id sets: contains returns True/False correctly;
      coverage_fraction equals (# predicted ids in support) / (# predicted) for a
      synthetic tensor of ids.
    status: pending
  - id: ac-002
    summary: CoverageRegime enum has in/near/extrapolating/unknown
    type: code
    pass_when: |
      CoverageRegime exposes IN_SUPPORT, NEAR_SUPPORT, EXTRAPOLATING, UNKNOWN
      (names may be equivalent Enum members); documented in regime.py.
    status: pending
  - id: ac-003
    summary: SupportClassifier maps confidence bands to regimes
    type: code
    pass_when: |
      With conf_in=0.8, conf_near=0.5 and ids inside support: conf>=0.8 ->
      IN_SUPPORT; 0.5<=conf<0.8 -> NEAR_SUPPORT; conf<0.5 -> EXTRAPOLATING or
      UNKNOWN per documented policy; out-of-support ids never IN_SUPPORT.
    status: pending
  - id: ac-004
    summary: Reuses TypeHead.decode_with_confidence
    type: code
    pass_when: |
      A unit test obtains (indices, confidence) from TypeHead.decode_with_confidence
      on synthetic logits and feeds them into SupportClassifier without
      reimplementing softmax-max in provenance code.
    status: pending
  - id: ac-005
    summary: ParameterProvenance is a frozen record with regime and source
    type: code
    pass_when: |
      ParameterProvenance(...) stores interaction, type_id, confidence, regime,
      source, optional pattern; mutation of fields raises / is disallowed
      (frozen dataclass).
    status: pending
  - id: ac-006
    summary: molrep.provenance does not import molpot
    type: code
    pass_when: |
      Grep/static check on src/molrep/provenance finds zero molpot imports.
    status: pending
  - id: ac-007
    summary: No active-learning loop APIs in provenance packages
    type: code
    pass_when: |
      Public provenance modules do not define acquisition functions, query
      selectors, or online retrain loops (support/classifier/record only).
    status: pending
  - id: ac-008
    summary: Google docstrings on support, classifier, provenance types
    type: docs
    pass_when: |
      ChemicalSupportIndex, SupportClassifier, CoverageRegime,
      ParameterProvenance have Google-style docstrings; ruff clean.
    status: pending
  - id: ac-009
    summary: Full unit check + test suite pass
    type: runtime
    pass_when: |
      ruff check/format --check and pytest tests/ exit 0 on the implementing branch.
    status: pending
---

# Acceptance criteria

- ac-001–ac-003 define chemical support and coverage regimes.
- ac-004 reuses TypeHead.decode_with_confidence (no parallel confidence math).
- ac-005 locks ParameterProvenance as a frozen metadata record.
- ac-006 / ac-007 enforce package deps and AL-out-of-scope.
- ac-008 / ac-009 docs + suite.
