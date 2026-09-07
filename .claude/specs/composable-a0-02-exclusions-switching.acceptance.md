---
slug: composable-a0-02-exclusions-switching
status: approved
criteria:
  - id: ac-001
    summary: exclusion scaling is keyed per interaction class
    type: code
    pass_when: |
      NonbondedScaling().kinds() == ("q", "lj"); triple("q") == (0, 0, 5/6)
      and triple("lj") == (0, 0, 0.5) by default; ExclusionTable.lookup(edge_index,
      "induced") raises KeyError whose message lists the declared kinds
      (tests/test_molpot/test_ir/test_scaling.py, test_forcefield/test_exclusions.py).
    status: pending
    last_checked: null
  - id: ac-002
    summary: the table reproduces the pre-step pair scales exactly
    type: scientific
    pass_when: |
      On OPLS butane (bonds, angles, propers), ExclusionTable.lookup for kinds
      "q" and "lj" equals the scale vectors produced by the deleted
      Potentials._pair_scales at commit 97c36852, captured as literals in
      tests/test_molpot/test_forcefield/test_exclusions.py (exact equality).
    status: pending
    last_checked: null
  - id: ac-003
    summary: LJ switch and exclusion lookup compile with zero graph breaks
    type: performance
    pass_when: |
      pytest -m slow: molix.compile.Compiler.explain over
      LJCutPotential.energy_forces and over ExclusionTable.lookup on a padded
      fixed-capacity list reports graph_break_count == 0.
    status: pending
    last_checked: null
  - id: ac-004
    summary: energy and force are continuous at the cutoff; analytic force matches autograd through the switch
    type: scientific
    pass_when: |
      fp64: |E(r_c - 1e-6) - E(r_c + 1e-6)| < 1e-10 kcal/mol and
      |F(r_c - 1e-6) - F(r_c + 1e-6)| < 1e-8 kcal/mol/A for a two-atom LJ pair
      with switch_start = 0.9 r_c; analytic LJCutPotential.energy_forces vs
      autograd of energy agree to <= 1e-8 for r inside the switching window
      (test_lj.py).
    status: pending
    last_checked: null
  - id: ac-005
    summary: total charge is fail-closed
    type: code
    pass_when: |
      Parameterizer(charge_model=...) called on a batch without
      graphs.total_charge raises ValueError; Potentials.bind_topology raises
      ValueError when |sum_g q - total_charge_g| > 1e-6 e for any graph
      (test_parameterizer.py, test_potentials.py).
    status: pending
    last_checked: null
---
