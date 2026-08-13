---
spec: universal-class-i-01-encoder
slug: universal-class-i-01-encoder
created: 2026-08-13
criteria:
  - id: ac-001
    summary: TestChemGraphBlock covers shape, symmetry, and locality
    type: code
    pass_when: |
      tests/test_molrep/test_chem/test_blocks.py defines TestChemGraphBlock
      with single-method tests that assert forward shapes (N, atom_dim) and
      (E, bond_dim), reverse-endpoint bond-feature allclose at rtol=1e-5
      atol=1e-6, one-block locality (non-neighbor Z change leaves h_i
      unchanged), and local_attention default False.
    status: verified
    last_checked: 2026-08-13
  - id: ac-002
    summary: RED tests exist for iterable h_bond and stacked ChemEncoder
    type: code
    pass_when: |
      tests/test_molrep/test_chem/test_embed.py::TestBondChemEmbedding,
      tests/test_molrep/test_chem/test_context.py::TestBondContext, and
      tests/test_molrep/test_chem/test_encoder.py::TestChemEncoder contain
      single-method tests for optional h_bond, n_blocks in {4,5,6},
      n_blocks rejection outside that set, pos invariance, and reverse
      bond-index bond features.
    status: verified
    last_checked: 2026-08-13
  - id: ac-003
    summary: BondChemEmbedding accepts optional previous h_bond
    type: code
    pass_when: |
      BondChemEmbedding.forward and BondContext.forward accept optional
      h_bond and return (E, bond_dim); h_bond=None keeps the pre-spec
      seed formula so existing TestBondChemEmbedding shape/empty/symmetry
      tests still pass; no new bond-latent type is introduced.
    status: verified
    last_checked: 2026-08-13
    verified_by: agent-auto
  - id: ac-004
    summary: ChemGraphBlock lives in molrep.chem and reuses bond symmetry
    type: code
    pass_when: |
      src/molrep/chem/blocks.py defines ChemGraphBlock; it does not live
      under molrep/interaction/pinet/; its bond update/messages call
      BondContext or BondChemEmbedding; ast scan of src/molrep/chem has
      no molpot or molrep.interaction.pinet imports.
    status: verified
    last_checked: 2026-08-13
    verified_by: agent-auto
  - id: ac-005
    summary: ChemEncoder stacks 4/5/6 blocks and keeps chem_features I/O
    type: code
    pass_when: |
      ChemEncoder(n_blocks=5, local_attention=False) is the default;
      n_blocks in {4,5,6} constructs; other n_blocks raises ValueError;
      compose/forward still write atoms/bonds/angles/propers/impropers
      chem_features; AngleContext/ProperContext/ImproperContext run after
      the block stack; atoms.pos is never read.
    status: verified
    last_checked: 2026-08-13
    verified_by: agent-auto
  - id: ac-006
    summary: Export ChemGraphBlock and add Google-style docstrings
    type: docs
    pass_when: |
      molrep.chem.__all__ is alphabetized and includes ChemGraphBlock;
      ChemGraphBlock and the changed ChemEncoder/BondChemEmbedding/BondContext
      public methods have Google-style docstrings naming tensor shapes and
      dimensionless embeddings.
    status: verified
    last_checked: 2026-08-13
    verified_by: agent-auto
  - id: ac-007
    summary: Regression script pins conf-invariance and reverse-bond goldens
    type: runtime
    pass_when: |
      PYTHONPATH=src python regressions/universal-class-i-01-encoder.py
      exits 0 and reproduces hard-coded goldens: same covalent graph with
      two atoms.pos yields identical ChemEmbeddings (allclose atol=1e-7,
      expected max abs diff 0.0); reversing bonds.atomi/atomj leaves
      bonds.chem_features unchanged (rtol=1e-5, atol=1e-6); feature
      shapes are (4,8), (3,8), (2,8), (1,8), (1,8) at the script widths;
      no energy/force keys. No third-party import or subprocess.
    status: verified
    last_checked: 2026-08-13
    verified_by: agent-auto
  - id: ac-008
    summary: Reverse bond embeddings are equal (i,j)≡(j,i)
    type: scientific
    evaluator_hint: path: tests/test_molrep/test_chem/test_encoder.py
    pass_when: |
      After stacked ChemGraphBlock updates, ChemEncoder bond chem_features
      (and ChemGraphBlock / BondChemEmbedding h_bond) with atomi/atomj
      swapped match the unreversed features at rtol=1e-5, atol=1e-6.
    status: verified
    last_checked: 2026-08-13
    verified_by: agent-auto
  - id: ac-009
    summary: Same covalent graph, two conformations, identical embeddings
    type: scientific
    evaluator_hint: path: tests/test_molrep/test_chem/test_encoder.py
    pass_when: |
      Two batches that share Z and valence indices but differ in atoms.pos
      produce ChemEmbeddings that match leaf-for-leaf at atol=1e-7
      (topology-only; pos is not an encoder input).
    status: verified
    last_checked: 2026-08-13
    verified_by: agent-auto
out_of_scope:
  - Fourier heads, losses, PES bundle, THEMol, LJ, charges
  - molzoo ChemPerception / ChemPerceptionSpec / any molzoo recipe file
  - scripts/, training, datasets
  - energy, force, molpot imports
  - global whole-molecule attention
  - new discrete-channel feature store
  - ClassicalIMM, IR, Condenser, ff_export
  - implementing blocks under molrep.interaction.pinet
---

# Acceptance — universal-class-i-01-encoder

本规格「完成」= 原地泛化后的 `ChemEncoder` 在共价图上产出与构象无关的 `ChemEmbeddings`，键潜变量保持 \((i,j)\equiv(j,i)\)，I/O 仍是五路 `*.chem_features`，从而 `ClassicalIMM` 无需改动。
