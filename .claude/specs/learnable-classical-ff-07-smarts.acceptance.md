---
slug: learnable-classical-ff-07-smarts
criteria:
  - id: ac-001
    summary: SymbolicPattern validates non-empty pattern and arity in {1,2,3,4}
    type: code
    pass_when: |
      Constructing SymbolicPattern with empty string or arity 0/5 raises
      ValueError; valid SMARTS string + arity=2 succeeds.
    status: pending
  - id: ac-002
    summary: FakeSmartsMatcher returns configured [arity, K] matches
    type: code
    pass_when: |
      FakeSmartsMatcher preloaded with a pattern -> tensor returns that tensor
      from match(...); unknown pattern returns empty [arity, 0] or raises as
      documented.
    status: pending
  - id: ac-003
    summary: ClassPatternRegistry bind/get and conflict detection
    type: code
    pass_when: |
      bind then get returns the same pattern; binding a different pattern to the
      same (InteractionClass, type_id) raises; reverse lookup works if exposed.
    status: pending
  - id: ac-004
    summary: SymbolicForceField.records pairs prototypes with patterns
    type: code
    pass_when: |
      Given TypeSystems + registry, records() yields DiscreteClassRecord entries
      with type_id, prototype params, and smarts/smirks when bound.
    status: pending
  - id: ac-005
    summary: match_molecule assigns types via matcher without energy
    type: code
    pass_when: |
      Using FakeSmartsMatcher, match_molecule returns expected type ids for
      synthetic hits; SymbolicForceField module does not import molpot energy
      kernels.
    status: pending
  - id: ac-006
    summary: MolpySmartsMatcher imports molpy only (never molrs)
    type: code
    pass_when: |
      MolpySmartsMatcher source and molrep.perception package contain
      `from molpy` / `import molpy` as needed and zero `molrs` imports; a test
      greps or inspects imports to enforce.
    status: pending
  - id: ac-007
    summary: SmartsMatcher is a Protocol satisfiable by Fake and Molpy matchers
    type: code
    pass_when: |
      isinstance(FakeSmartsMatcher(...), SmartsMatcher) if runtime_checkable;
      both expose the documented match API.
    status: pending
  - id: ac-008
    summary: Google docstrings on public perception symbols
    type: docs
    pass_when: |
      DiscreteClassRecord, SymbolicPattern, SmartsMatcher, FakeSmartsMatcher,
      MolpySmartsMatcher, ClassPatternRegistry, SymbolicForceField have Google
      docstrings; ruff clean.
    status: pending
  - id: ac-009
    summary: Full unit check + test suite pass
    type: runtime
    pass_when: |
      ruff check/format --check and pytest tests/ exit 0 on the implementing branch.
    status: pending
---

# Acceptance criteria

- ac-001–ac-005 cover symbolic records, fake matching, registry, and force-field assembly.
- ac-006 / ac-007 enforce molpy-only matching Protocol design.
- ac-008 / ac-009 docs + suite.
