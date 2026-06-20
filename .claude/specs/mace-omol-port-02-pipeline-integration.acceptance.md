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
    status: pending
  - id: ac-002
    summary: molzoo-spec backfilled for mace_omol
    type: docs
    pass_when: |
      src/molzoo/specs/mace_omol.md exists with §2/§3.1/§5 filled from the paper
      + reference impl, status at least `partial`, per the CLAUDE.md molzoo-spec
      workflow.
    status: pending
  - id: ac-003
    summary: bit-exact (~1e-8) model-level match via e3nn-convention CG group
    type: scientific
    pass_when: |
      Building the equivariant ops with the e3nn-convention O3 group drops the
      model-level residual to ~1e-8 eV (scripts/omol_port/verify_e2e.py under
      group=O3_e3nn). BLOCKED: O3_e3nn crashes cue.Irreps.sort() in
      cuequivariance 0.10; MACEOMol(group=) plumbing exists but is unusable
      until the upstream bug is fixed or O3_e3nn is vendored.
    evaluator_hint: manual
    status: pending
---
