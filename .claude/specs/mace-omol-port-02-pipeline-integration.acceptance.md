---
slug: mace-omol-port-02-pipeline-integration
criteria:
  - id: ac-001
    summary: molnex pipeline integration (TensorDict / molpot / neighbor list)
    type: code
    pass_when: |
      MACEOMol is exposed as a molzoo encoder + molpot potential consuming the
      post-collate atoms/edges/graphs TensorDict, with forces via
      molpot.derivation.ForceDerivation and edges via molpot.graph/NeighborList;
      molzoo/__init__ imports it lazily. An integration test shows the
      TensorDict path agrees with the raw energy_forces path to machine
      precision on a small charged molecule.
    status: verified
    last_checked: 2026-06-21
  - id: ac-002
    summary: molzoo-spec backfilled for mace_omol
    type: docs
    pass_when: |
      src/molzoo/specs/mace_omol.md exists with §2/§3.1/§5 filled from the paper
      + reference impl, status at least `partial`, per the CLAUDE.md molzoo-spec
      workflow.
    status: verified
    last_checked: 2026-06-21
  - id: ac-003
    summary: model-level E/F within 1e-4 of official OMOL (default cue O3 group)
    type: scientific
    pass_when: |
      MACEOMol with official weights matches official OMOL energy/forces to
      |dE| < 1e-4 eV and max|dF| < 1e-4 eV/Ang. Achieved 7.0e-7 eV / 4.3e-6
      eV/Ang with the default cue "O3" group (mace-omol-port-01 ac-006,
      scripts/omol_port/verify_e2e.py: PASS) — well inside the 1e-4 bar.
      Bit-exact ~1e-8 via the e3nn-convention CG group is an unneeded stretch
      goal and is unavailable in cuequivariance 0.10 (no O3_e3nn group); the
      MACEOMol(group=) hook remains for if it returns. The 1e-4 tolerance is
      the accepted requirement (operator decision, 2026-06-21).
    status: verified
    last_checked: 2026-06-21
---
