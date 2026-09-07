---
slug: composable-a0-03b-neighbor-list-to-molix
status: approved
criteria:
  - id: ac-001
    summary: molpot no longer enumerates neighbour pairs
    type: code
    pass_when: |
      src/molpot/forcefield/neighbor_list.py does not exist; no module under
      src/molpot imports molpy.NeighborQuery or enumerates all-pairs edges;
      Potentials.calc_energy on a batch without an edges namespace raises
      ValueError whose message names molix.batch.NeighborList and
      molix.md.NeighborList (tests/test_molpot/test_forcefield/test_potentials.py).
    status: verified
    last_checked: 2026-09-07
    verified_by: impl
    evidence: "src/molpot/forcefield/neighbor_list.py deleted (git rm); grep of src/molpot finds no NeighborQuery / all-pairs enumeration; tests/test_molpot/test_forcefield/test_potentials.py test_a_pair_style_on_a_batch_without_edges_names_both_molix_owners green (message names molix.batch.NeighborList and molix.md.NeighborList)"
  - id: ac-002
    summary: the placed-list contract is preserved on Potentials
    type: code
    pass_when: |
      Potentials._placed_pairs returns the capacity-shaped (edge_index, dist,
      live_mask) triple with the a0-02 liveness rule, prefers edges.shifts over
      the edge_diff reconstruction, and the moved placed-list tests pass; the
      a0-02 butane goldens and the a0-03 exclusion / Madelung / FD tests are
      unchanged (mirrored run green).
    status: verified
    last_checked: 2026-09-07
    verified_by: impl
    evidence: "Potentials._placed_pairs tests green (capacity shape + live mask, shifts preferred over edge_diff, shift kept on the autograd graph, empty list is empty not absent, edge_dist-only list refused); a0-02 butane golden and a0-03 exclusion / Madelung / FD tests unchanged - tests/test_molpot/test_forcefield 167 passed, mirrored set 690 + 295 passed"
  - id: ac-003
    summary: fixtures go through molix's NeighborList and F4 is closed
    type: code
    pass_when: |
      Every fixture the sweep found builds its edges via
      molix.batch.NeighborList (or molix.md.NeighborList); the F4 row in
      .claude/notes/composable-potentials.md records closure by deletion; the
      a0-02/a0-03 slow explains still report 0 breaks.
    status: verified
    last_checked: 2026-09-07
    verified_by: impl
    evidence: "every pair-style fixture rebuilt through molix.batch.NeighborList(symmetry=True) with pair sets proven identical (enumerator monkeypatched to raise never fired); F4 row in .claude/notes/composable-potentials.md closed by deletion; slow explains (exclusions lookup, LJ energy_forces, periodic Coulomb ewald and pme) 4 passed at 0 breaks"
---
