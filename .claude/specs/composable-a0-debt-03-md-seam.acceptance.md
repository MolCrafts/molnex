---
slug: composable-a0-debt-03-md-seam
status: approved
criteria:
  - id: ac-001
    summary: MD.set_compiler installs and re-installs the classical core
    type: code
    pass_when: |
      A recording compiler is applied once at set_forcefield and again after any later set_* rebuild; energies and forces through the installed core equal eager to 1e-12 (tests/test_molix/test_md/test_driver.py, tests/test_molpot/test_forcefield/test_potentials.py).
    status: verified
    last_checked: 2026-09-08
    verified_by: impl
    evidence: "tests/test_molix/test_md/test_driver.py::TestMD - a recording compiler is applied at set_forcefield and again after set_integrator (each install on the rebuilt container); identity-compiled forces equal eager to 1e-12; tests/test_molpot/test_forcefield/test_potentials.py::TestPotentialsCompiledCore - classical compile_core called once with the bound core, routed once per calc_energy_forces, values equal eager"
  - id: ac-002
    summary: potential_dtype evaluates the potential in a second dtype
    type: code
    pass_when: |
      MD(torch.float64, potential_dtype=torch.float32) keeps pos/vel/mass fp64, the potential's parameter tables fp32, and returns fp64 forces equal to the fp32 evaluation cast up; potential_dtype=None is byte-identical to today.
    status: verified
    last_checked: 2026-09-08
    verified_by: impl
    evidence: "test_driver.py::test_potential_dtype_evaluates_the_potential_at_the_second_precision - MD(float64, potential_dtype=float32): pos/vel/mass fp64, parameter tables fp32 via ForceField.to_dtype, fp64 forces within 0.25 x the fixture's own fp32/fp64 gap of the fp32 hand reference; potential_dtype=None bitwise identical to the fp64 path"
  - id: ac-003
    summary: the classical core compiles with no new graph break
    type: performance
    pass_when: |
      Compiler.explain on the installed classical core: LJ-only fixture 0 breaks; periodic Coulomb fixture breaks only at the derivation/force.py seam (slow tier).
    status: verified
    last_checked: 2026-09-08
    verified_by: impl
    evidence: "test_potentials.py::test_the_installed_classical_core_traces_with_no_break_outside_force_py (slow) - LJ-only core 1 graph / 0 breaks; periodic-Coulomb core break sites only in derivation/force.py (position_leaf); -m slow traces_clean / compiled_route / classical_core 4 passed"
---
