---
slug: composable-a0-03-periodic-coulomb
status: code-complete
criteria:
  - id: ac-001
    summary: excluded bonded pairs are corrected by the full Coulomb term under Ewald/PME (F1)
    type: scientific
    pass_when: |
      Two ions bonded (scale 0) in a 20 A cubic box: E_periodic(excluded) ==
      E_periodic(unexcluded) - k_e q_i q_j / r_mic to <= 1e-8 kcal/mol (fp64),
      for both calculator="ewald" and "pme"; a 1-4 pair with s = 5/6 obeys the
      same identity with (1 - s) (tests/test_molpot/test_forcefield/test_coulomb.py).
    status: pending
    last_checked: null
  - id: ac-002
    summary: Madelung energy of a rocksalt supercell matches the closed-form golden
    type: scientific
    pass_when: |
      tests/test_molpot/test_forcefield/test_coulomb.py reproduces the closed-form
      rocksalt lattice energy E = -(N/2) * M * k_e / d with M = 1.74756459463 on a
      2x2x2 supercell (64 ions, d = 2.82 A) to rtol 1e-6 with Ewald and with PME.
      (The 8-ion conventional cube is ill-posed under minimum image - its nearest
      neighbours sit at exactly L/2 - see finding F14.)
    status: pending
    last_checked: null
  - id: ac-003
    summary: forces and strain derivative of the assembled Potentials match finite differences
    type: scientific
    pass_when: |
      fp64, PME, 4-ion periodic box, frozen k-space shape: max|F_auto - F_fd| <
      1e-6 kcal/mol/A (central differences, dh = 1e-4); |W_auto - W_fd| < 1e-5
      kcal/mol for the symmetric-strain virial from calc_energy_forces_virial
      (test_potentials.py).
    status: pending
    last_checked: null
  - id: ac-004
    summary: periodic Coulomb compiles with zero breaks; the whole step breaks only at the autograd seam
    type: performance
    pass_when: |
      pytest -m slow: Compiler.explain(coulomb_periodic.energy) after bind_cell
      reports graph_break_count == 0; Compiler.explain(calc_energy_forces_virial)
      reports breaks whose sites are all in molpot/derivation/force.py.
    status: pending
    last_checked: null
  - id: ac-005
    summary: k-space capacity is fail-closed
    type: code
    pass_when: |
      Growing the cell past the bound capacity raises RuntimeError naming
      capacity_factor on the eager path; the compiled path returns NaN energy
      rather than a wrong finite value (test_coulomb.py).
    status: verified
    last_checked: 2026-09-07
    verified_by: impl
    evidence: "tests/test_molpot/test_forcefield/test_coulomb.py TestCoulombPeriodicPotential - eager RuntimeError naming capacity_factor on an over-capacity rebind, NaN energy in-graph on a cell that outgrew the bound shape, bind_cell idempotent"
---
