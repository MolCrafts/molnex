---
slug: learnable-classical-ff-04-chem-encoder
criteria:
  - id: ac-001
    summary: AtomChemEmbedding maps Z to (N, D_a) features
    type: code
    pass_when: |
      AtomChemEmbedding forward on Z (N,) yields features (N, D_a) with
      configured D_a; reuses JointEmbedding or nn.Embedding path as documented.
    status: pending
  - id: ac-002
    summary: BondChemEmbedding is endpoint-symmetric
    type: code
    pass_when: |
      For bond_index columns (i,j) vs swapped (j,i) with the same atom
      embeddings, h_bond allclose (rtol/atol documented in test).
    status: pending
  - id: ac-003
    summary: Angle context invariant under (i,j,k) -> (k,j,i)
    type: code
    pass_when: |
      AngleContext features for reversed endpoint order allclose to original.
    status: pending
  - id: ac-004
    summary: Proper context invariant under (i,j,k,l) -> (l,k,j,i)
    type: code
    pass_when: |
      ProperContext features for reversed torsion order allclose to original.
    status: pending
  - id: ac-005
    summary: ChemEmbeddings holds atom/bond/angle/proper/improper tensors
    type: code
    pass_when: |
      ChemEmbeddings (dataclass or fixed-key mapping) exposes the five feature
      tensors with counts matching topology; missing optional impropers may be
      empty (0, D) rather than absent if encoder always writes all keys —
      document and test one policy.
    status: pending
  - id: ac-006
    summary: ChemEncoder reads valence namespaces and writes features
    type: code
    pass_when: |
      Synthetic TensorDict with atoms.Z, bonds, angles, propers, impropers runs
      through ChemEncoder; output batch or ChemEmbeddings has non-null features
      aligned to counts.
    status: pending
  - id: ac-007
    summary: molzoo ChemPerception recipe constructs and forwards
    type: code
    pass_when: |
      ChemPerception (or ChemPerceptionConfig + module) imports from molzoo.chem,
      builds ChemEncoder, forward succeeds on mini batch without energy keys.
    status: pending
  - id: ac-008
    summary: molrep.chem and molzoo.chem do not import molpot
    type: code
    pass_when: |
      Grep/static test over src/molrep/chem and src/molzoo/chem finds zero
      molpot imports.
    status: pending
  - id: ac-009
    summary: Google docstrings on public chem symbols
    type: docs
    pass_when: |
      AtomChemEmbedding, BondChemEmbedding, ChemEncoder, ChemEmbeddings,
      ChemPerception have Google-style docstrings with tensor shapes; ruff clean.
    status: pending
  - id: ac-010
    summary: Full unit check + test suite pass
    type: runtime
    pass_when: |
      ruff check/format --check and pytest tests/ exit 0 on the implementing branch.
    status: pending
---

# Acceptance criteria

- ac-001–ac-005 define continuous chemical embeddings and symmetry contracts.
- ac-006 / ac-007 bind encoder + molzoo recipe I/O to valence topology (02).
- ac-008 enforces package dependency (no molpot).
- ac-009 / ac-010 docs + suite.
