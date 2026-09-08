---
slug: composable-a0-06-gates
status: approved
criteria:
  - id: ac-001
    summary: the assembled A0 energy is invariant under translation, rotation, reflection and permutation
    type: scientific
    pass_when: |
      fp64, OPLS butane pair in a 20 A periodic box under the mesh-free Ewald
      sum (a PME mesh is anchored to the box, so its interpolation error is not
      translation-invariant and no 1e-10 gate is reachable on it; PME-vs-Ewald
      agreement is a0-03's fact): |E(g x) - E(x)| <= 1e-10
      kcal/mol for a random translation, a random rotation, pos -> -pos, a
      one-axis mirror, and a random atom permutation (with bond/angle/torsion
      columns permuted consistently); improper:harmonic with chi0 = 0.3 is
      NOT invariant under pos -> -pos (test_potentials.py).
    status: verified
    last_checked: 2026-09-08
    verified_by: impl
    evidence: "tests/test_molpot/test_forcefield/test_potentials.py (fast, fp64, OPLS butane pair in a 20 A Ewald box) - translation 7.1e-14, 90-degree z rotation 7.1e-15, permutation 0.0, inversion 6.4e-14, mirror-x 0.0 (tol 1e-10); improper:harmonic chi0=0.3 changes by 13.1 kcal/mol under inversion; fixture in the forcefield conftest"
  - id: ac-002
    summary: NVE energy drift is within the declared criterion
    type: scientific
    pass_when: |
      200 steps, dt = 0.5 fs, fp64, velocity-Verlet on the same box:
      EnergyDriftHook reports rel_drift < 1e-3 and all sampled E_tot finite
      (tests/test_molix/test_md/test_runner.py, slow tier).
    status: verified
    last_checked: 2026-09-08
    verified_by: impl
    evidence: "tests/test_molix/test_md/test_runner.py::TestEnergyDriftHook slow NVE case - 200 steps, dt 0.5 fs, fp64 velocity-Verlet through molix.md on the same box: EnergyDrift(e0=0.0226896, duration_fs=99.5, slope=1.70e-09, rel_drift=7.465e-06, rms_rel=7.68e-05), all totals finite (bound 1e-3); Ewald lr_wavelength 1.0 A for the rollout (identical drift to 7 digits vs 0.5 A), 5.3 s wall; the box has no wrapped pair, so F27 is gated by the boundary FD test, not by this rollout"
  - id: ac-003
    summary: EnergyDriftHook recovers a synthetic slope exactly
    type: code
    pass_when: |
      Feeding the hook a synthetic series E(t) = E0 + a t yields
      rel_drift == |a T| / |E0| to 1e-12 and rms_rel == 0 for a constant series
      (tests/test_molix/test_md/test_runner.py::TestEnergyDriftHook).
    status: verified
    last_checked: 2026-09-08
    verified_by: impl
    evidence: "tests/test_molix/test_md/test_runner.py::TestEnergyDriftHook - synthetic E0 + a t recovers slope and rel_drift = |a T|/|E0| to 1e-12; constant series rms_rel == rel_drift == 0; one sample raises ValueError; on_step_end reads integrator.dt (duration = dt (n-1))"
  - id: ac-004
    summary: the A0 gate tests pass and the chain closes
    type: code
    pass_when: |
      The A0 gate tests pass — fast tier: symmetries, charge sum, FD force + strain,
      cutoff continuity, manifest round trip against sha literals, contribution
      partition; `-m slow`: the 200-step NVE drift test;
      INDEX.md lists composable-a0-01..06 as closed.
    status: verified
    last_checked: 2026-09-08
    verified_by: impl
    evidence: "fast tier 3084 passed (symmetries, charge-sum rejection, FD force + strain, cutoff continuity, manifest round trip with sha literals, contribution partition all pinned in their mirrored modules), -m slow: the NVE drift gate 1 passed; boundary FD gate for F27 green; INDEX.md lists composable-a0-01..06 (and 03b) under Closed at chain close-out"
---
