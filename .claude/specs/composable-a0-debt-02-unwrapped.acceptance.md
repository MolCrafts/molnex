---
slug: composable-a0-debt-02-unwrapped
status: approved
criteria:
  - id: ac-001
    summary: a wrapped bond is refused at bind
    type: code
    pass_when: |
      bind_topology on a periodic system with a bond longer than half the cell's minimum perpendicular width raises ValueError naming the bond, the length, the limit and F29; the unwrapped copy binds; an open box binds (tests/test_molpot/test_forcefield/test_potentials.py).
    status: verified
    last_checked: 2026-09-08
    verified_by: impl
    evidence: "tests/test_molpot/test_forcefield/test_potentials.py - a 10 A cube with a bond at x=1/x=9 raises ValueError naming bond 0 (atoms 0-1), 8 A over the 5 A half-width, F29 and the remedy; the unwrapped copy (x=9/x=11) binds and prices the 2 A bond; the open box binds; a sheared cell refuses a 3.5 A bond against its 3 A perpendicular half-width that the cube's edge rule would accept"
  - id: ac-002
    summary: the check costs one host sync at bind and nothing on the force path
    type: performance
    pass_when: |
      The slow dynamo trace gates for the step (traces_clean / compiled_route) stay at their recorded break sites; no new .item()/.any() reachable from calc_energy.
    status: verified
    last_checked: 2026-09-08
    verified_by: impl
    evidence: "slow trace gates after the check landed: traces_clean / compiled_route / classical_core / test_energy_forces_has_no_graph_breaks 11 passed (1 skipped) - break sites unchanged (derivation/force.py only); _check_geometry is host-side, runs before the compiled core and returns early under torch.compiler.is_compiling(); nothing new on calc_energy's traced region"
---
