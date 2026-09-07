---
slug: composable-a0-06-gates
status: approved
criteria:
  - id: ac-001
    summary: the assembled A0 energy is invariant under translation, rotation, reflection and permutation
    type: scientific
    pass_when: |
      fp64, OPLS butane pair in a 20 A PME box: |E(g x) - E(x)| <= 1e-10
      kcal/mol for a random translation, a random rotation, pos -> -pos, a
      one-axis mirror, and a random atom permutation (with bond/angle/torsion
      columns permuted consistently); improper:harmonic with chi0 = 0.3 is
      NOT invariant under pos -> -pos (test_potentials.py).
    status: pending
    last_checked: null
  - id: ac-002
    summary: NVE energy drift is within the declared criterion
    type: scientific
    pass_when: |
      200 steps, dt = 0.5 fs, fp64, velocity-Verlet on the same box:
      EnergyDriftHook reports rel_drift < 1e-3 and all sampled E_tot finite
      (tests/test_molix/test_md/test_runner.py, slow tier).
    status: pending
    last_checked: null
  - id: ac-003
    summary: EnergyDriftHook recovers a synthetic slope exactly
    type: code
    pass_when: |
      Feeding the hook a synthetic series E(t) = E0 + a t yields
      rel_drift == |a T| / |E0| to 1e-12 and rms_rel == 0 for a constant series
      (tests/test_molix/test_md/test_runner.py::TestEnergyDriftHook).
    status: pending
    last_checked: null
  - id: ac-004
    summary: the A0 gate tests pass and the chain closes
    type: code
    pass_when: |
      The A0 gate tests pass — fast tier: symmetries, charge sum, FD force + strain,
      cutoff continuity, manifest round trip against sha literals, contribution
      partition; `-m slow`: the 200-step NVE drift test;
      INDEX.md lists composable-a0-01..06 as closed.
    status: pending
    last_checked: null
---
