---
slug: mace-subpackage-restructure-05-checkpoint
criteria:
  - id: ac-001
    summary: one CheckpointRemap replaces both loaders with tables unchanged
    type: code
    pass_when: |
      src/molzoo/mace/checkpoint.py defines exactly one class, CheckpointRemap,
      with .rename and .load; MATPES_KEY_REMAP and OMOL_KEY_REMAP are key-for-key
      identical to the pre-restructure mace_matpes.py:387-399 and
      mace_omol.py:355-370 tables; no `_KEY_REMAP` / `_KEY_REMAP_ORDER` remains in
      src/molzoo/mace_matpes.py or src/molzoo/mace_omol.py (grep returns nothing).
    status: pending
  - id: ac-002
    summary: round-trip load restores every parameter bit-for-bit, both families
    type: code
    pass_when: |
      tests/test_molzoo/test_mace/test_checkpoint.py::TestCheckpointRemap
      test_roundtrip_restores_every_parameter_matpes and
      ..._omol pass: a model's own state_dict renamed into official cueq names and
      loaded back yields torch.equal on every named_parameters() tensor.
    status: pending
  - id: ac-003
    summary: on_unexpected knob raises for MatPES, returns for OMol
    type: code
    pass_when: |
      With an injected key "interactions.0.mystery_layer.weight":
      on_unexpected="raise" raises RuntimeError matching "no home" and leaves the
      model's parameters unchanged (no partial load); on_unexpected="return"
      does not raise and returns that key in the second element of its tuple.
      (test_raise_policy_rejects_unexpected_key, test_return_policy_reports_unexpected_key)
    status: pending
  - id: ac-004
    summary: strict doctrine holds under both policies
    type: code
    pass_when: |
      Under both on_unexpected values: deleting interactions.0.linear_up.weight
      raises RuntimeError matching "not covered"; deleting bessel.freqs (an
      nn.Parameter ending in neither .weight nor .bias) also raises; a genuine
      shape mismatch raises matching "shape mismatch"; a (1,)-shaped
      scale_shift.scale loads into the 0-d buffer with value 0.75.
    status: pending
  - id: ac-005
    summary: MACEPotential.from_checkpoint derives dims from irreps, no hardcodes
    type: code
    pass_when: |
      MACEPotential.from_checkpoint(config_path, weights_path) on a tmp_path
      config with hidden_irreps="32x0e+32x1o" and MLP_irreps="8x0e" builds a model
      with num_features=32, max_hidden_l=1, mlp_dim=8 and parameters torch.equal to
      the source model; the literals 128, 1, 16 appear nowhere as dimension
      defaults in src/molzoo/mace/potential.py; a config missing hidden_irreps
      raises with the missing key named.
    status: pending
  - id: ac-006
    summary: legacy loader names keep working as thin wrappers
    type: code
    pass_when: |
      tests/test_molzoo/test_mace_matpes.py::TestLoadMatpesStateDict passes with
      the test file unmodified; load_omol_state_dict still returns
      (missing_buffers, unexpected) as a tuple of lists; both functions are one
      delegating line over MATPES_REMAP / OMOL_REMAP.
    status: pending
  - id: ac-007
    summary: official MatPES checkpoint loads bit-identically via from_checkpoint
    type: code
    pass_when: |
      On a host with MOLNEX_MACE_WEIGHTS_DIR pointing at the real weights,
      test_official_checkpoint_bit_parity passes: from_checkpoint on
      matpes_r2scan_config.json + matpes_r2scan_cueq_state.pt and the manual
      "construct + MATPES_REMAP.load" path give state_dicts with identical key
      sets and torch.equal on every tensor. A skip (weights absent) is not a pass.
    status: pending
  - id: ac-008
    summary: regression example reproduces hard-coded E/F goldens
    type: runtime
    pass_when: |
      `python regressions/mace-subpackage-restructure-05-checkpoint.py` exits 0 on
      CPU/fp64 with no third-party oracle imported (no mace-torch, e3nn, ASE): it
      writes a synthetic official-named checkpoint + config json to a temp dir,
      reloads it via MACEPotential.from_checkpoint, and reproduces the script's
      hard-coded energy and forces for its fixed 5-atom cluster within
      |dE| <= 1e-9 eV and max|dF| <= 1e-8 eV/Ang. Parameters are filled by a
      shape-derived deterministic rule, never RNG.
    status: pending
  - id: ac-009
    summary: checkpoint module documents units and both prior incidents
    type: docs
    pass_when: |
      src/molzoo/mace/checkpoint.py module + CheckpointRemap + .load docstrings are
      Google-style, cite https://arxiv.org/abs/2206.07697, state eV / eV·Ang units,
      state that "learnable" means exactly nn.Parameter, and reference the
      mace_matpes.py:405-410 doctrine and the mace_omol.py bessel.freqs incident.
    status: pending
---

# Acceptance criteria

- **ac-001 / ac-006** are the "one implementation, no second copy" pair: the class
  exists *and* the duplicated tables and loader bodies are actually gone, while the
  old public names keep their exact call contracts until 06/07.
- **ac-002** is the semantic-equivalence gate for the merge: the remap tables mean
  the same thing after the move, proven by a bijective round-trip on both families.
  The OMol arm is new coverage — that loader shipped with no unit test at all.
- **ac-003** isolates the single intended difference between the two families to
  the one policy knob, and additionally pins "raise leaves no half-loaded model".
- **ac-004** is the non-negotiable strictness doctrine, verified under *both* knob
  values so the knob can never be read as a general laxness dial. The
  `bessel.freqs` case is the regression lock for the name-suffix-heuristic bug.
- **ac-005** rejects the `run_nve.py` hardcoded `128 / 1 / 16` dims. If the real
  official config json turns out to lack `hidden_irreps` / `MLP_irreps`, this
  criterion (and ac-007) must be reported blocked rather than satisfied by
  reinstating the hardcodes.
- **ac-007** is the chain gate: bit-parity, not tolerance parity. It requires a
  host with the offline weights; a skipped run leaves it `pending`.
- **ac-008** must never skip — it is self-contained, hard-coded, and third-party
  free by construction.
