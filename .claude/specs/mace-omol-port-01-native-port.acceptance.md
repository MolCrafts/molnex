---
slug: mace-omol-port-01-native-port
criteria:
  - id: ac-001
    summary: radial + cutoff blocks match official MACE bit-for-bit
    type: code
    pass_when: |
      molrep.embedding.BesselRBF(trainable=True, normalize=False, eps=0) matches
      mace.modules.radial.BesselBasis and molrep.embedding.PolynomialCutoff matches
      mace PolynomialCutoff on float64 to < 1e-12 (scripts/omol_port/verify_radial.py).
    status: verified
    last_checked: 2026-06-21
  - id: ac-002
    summary: E0 + scale/shift + charge/spin embedding match official
    type: code
    pass_when: |
      molpot.heads.AtomicReferenceEnergy vs mace AtomicEnergiesBlock = 0;
      molpot.heads.GlobalRescale vs mace ScaleShiftBlock = 0;
      molrep.embedding.JointFeatureEmbedding vs the real OMOL joint_embedding = 0
      (verify_e0_scaleshift.py, verify_joint_embed.py).
    status: verified
    last_checked: 2026-06-21
  - id: ac-003
    summary: residual non-linear interaction matches all 3 cueq layers
    type: code
    pass_when: |
      molrep.interaction.ResidualInteraction, loaded with cueq interactions[i]
      weights, reproduces (message, skip) for i=0,1,2 to 0 diff incl. higher-l
      inputs (verify_interaction.py). RadialMLP and GatedNonlinearity each match
      their mace counterparts to 0 (verify_radial_mlp.py, gate inline).
    status: verified
    last_checked: 2026-06-21
  - id: ac-004
    summary: product basis + readout match cueq blocks
    type: code
    pass_when: |
      molrep.interaction.EquivariantProductBasis (degree=2, num_elements=1) vs
      cueq products[0..2] = 0; molrep.readout.NonLinearBiasReadout vs cueq
      readouts[0] = 0 (verify_product.py, verify_readout.py).
    status: verified
    last_checked: 2026-06-21
  - id: ac-005
    summary: assembled model loads official weights with no missing learnable keys
    type: code
    pass_when: |
      load_omol_state_dict(MACEOMol, cueq_state) leaves 0 missing learnable
      parameters (weights/biases/alpha/beta/scale/shift) (verify_e2e.py prints
      "missing learnable: []").
    status: verified
    last_checked: 2026-06-21
  - id: ac-006
    summary: full model reproduces official OMOL energy and forces
    type: scientific
    pass_when: |
      MACEOMol with official weights vs the cueq OMOL twin on a charged molecule:
      |dE| < 1e-5 eV and max|dF| < 1e-4 eV/Ang. Achieved 7.0e-7 eV / 4.3e-6 eV/Ang
      (verify_e2e.py RESULT: PASS). The cueq twin itself matches e3nn OMOL to
      1.5e-8 eV / 3.2e-8 eV/Ang (verify_omol_cueq_equiv.py).
    status: verified
    last_checked: 2026-06-21
# ac-007 (bit-exact O3_e3nn, upstream-blocked) and ac-008 (molnex pipeline
# integration, deferred) moved to chain step mace-omol-port-02-pipeline-integration.
---
