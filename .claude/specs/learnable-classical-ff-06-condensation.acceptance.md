---
slug: learnable-classical-ff-06-condensation
criteria:
  - id: ac-001
    summary: MergeCriterion accepts within budget and rejects outside
    type: scientific
    pass_when: |
      For bond class, two param rows with Δr0 and Δk inside defaults return
      True; a row exceeding either threshold returns False. Units match
      CLASS_I (Å, kcal/mol/Å²) as documented on MergeCriterion.
    status: pending
  - id: ac-002
    summary: Condenser merges identical params to a single type
    type: code
    pass_when: |
      N copies of the same parameter vector produce TypeSystem with n_types==1
      and member_count==N.
    status: pending
  - id: ac-003
    summary: Condenser keeps far params as distinct types
    type: code
    pass_when: |
      Two param vectors outside MergeCriterion yield n_types==2 with stable
      type_ids under the documented sort key.
    status: pending
  - id: ac-004
    summary: Multi-system merge shares global type ids
    type: code
    pass_when: |
      params_by_system with near-duplicate bonds across two systems collapses
      to one global type; assignment tables map both systems' rows to that id.
    status: pending
  - id: ac-005
    summary: TypeSystem.assign maps new params to nearest acceptable type or fails soft
    type: code
    pass_when: |
      assign(prototype-equal params) returns existing type_id; behavior for
      out-of-budget params is documented (new type vs reject) and tested.
    status: pending
  - id: ac-006
    summary: TypeSystemLabeler satisfies Labeler Protocol surface
    type: code
    pass_when: |
      TypeSystemLabeler exposes num_types, type_map, and label(...)->LongTensor
      ids in range; isinstance(..., Labeler) if runtime_checkable.
    status: pending
  - id: ac-007
    summary: TypeHead multi-system / multi-class generalization does not break atom API
    type: code
    pass_when: |
      Existing TypeHead(hidden_dim, num_types) tests still pass; multi-class
      usage (if added) is additive.
    status: pending
  - id: ac-008
    summary: Condensation package emits no SMARTS/SMIRKS strings
    type: code
    pass_when: |
      Public condensation APIs return integer type ids and numeric prototypes
      only; no function returns SMARTS pattern text. Grep for smarts/smirks
      emitters in molrep/condensation is empty (imports of 07 later OK only
      outside this package).
    status: pending
  - id: ac-009
    summary: Google docstrings on Condenser, TypeSystem, MergeCriterion
    type: docs
    pass_when: |
      Public condensation types have Google-style docstrings with parameter
      units and InteractionClass semantics; ruff clean.
    status: pending
  - id: ac-010
    summary: Full unit check + test suite pass
    type: runtime
    pass_when: |
      ruff check/format --check and pytest tests/ exit 0 on the implementing branch.
    status: pending
---

# Acceptance criteria

- ac-001–ac-005 are the physics-aware merge contract (budgets, greedy merge, multi-system).
- ac-006 / ac-007 wire labeling / TypeHead without breaking existing atom typing.
- ac-008 keeps SMARTS out of condensation (07 owns symbols).
- ac-009 / ac-010 docs + suite.
