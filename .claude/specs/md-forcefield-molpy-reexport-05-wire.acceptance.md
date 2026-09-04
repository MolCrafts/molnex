---
spec: md-forcefield-molpy-reexport-05-wire
created: 2026-08-16
criteria:
  - id: ac-001
    summary: Leftover MD tests compose via def_pairstyle and MD setters
    type: code
    pass_when: |
      TestForceField and TestMD construct ForceField.def_pairstyle and
      attach via MD.set_forcefield / set_neighbors; no deleted adapter
      names or energy_scale= constructor.
    status: pending
  - id: ac-002
    summary: molix.md no longer exports homemade ForceField adapters
    type: code
    pass_when: |
      CallableForceField, PotentialForceField, PeriodicPotentialForceField,
      HarmonicForceField, LennardJonesForceField, LennardJonesCutForceField
      are absent from molix.md.__all__ and hasattr is False.
    status: pending
  - id: ac-003
    summary: ForceField and MD constructors reject energy_scale
    type: code
    pass_when: |
      ForceField(...) and MD.set_forcefield raise TypeError on energy_scale=;
      leftover test_molix/test_md files contain none of the deleted names.
    status: pending
  - id: ac-004
    summary: run_nve.py uses def_pairstyle plus set_neighbors
    type: code
    pass_when: |
      scripts/matpes_port/run_nve.py imports neither CallableForceField
      nor homemade ForceField subclasses, registers the MACE model via
      ForceField.def_pairstyle, and wires MD.set_forcefield.set_neighbors.
    status: pending
  - id: ac-005
    summary: Parity scripts pass ForceField to MD.set_forcefield
    type: code
    pass_when: |
      parity_project_md.py and parity_lammps_classical_md.py contain no
      CallableForceField, no bind_* wrappers, and no energy_scale=;
      MD.set_forcefield receives the molpy ForceField directly.
    status: pending
  - id: ac-006
    summary: LJ benchmarks use def_pairstyle and molix.units
    type: code
    pass_when: |
      verify_md_lj_nve.py and verify_md_ljcut_nve.py use
      def_pairstyle("lj/cut") + MD composer and molix.units (no 103.64
      literal).
    status: pending
  - id: ac-007
    summary: User docs drop CallableForceField and energy_scale examples
    type: docs
    pass_when: |
      docs/molix/user-guide/md.md has no CallableForceField table row and
      shows def_pairstyle + set_forcefield.set_neighbors; ff-export.md and
      class-i-forcefield.md contain neither PotentialForceField nor
      energy_scale=.
    status: verified
    last_checked: 2026-09-03
    note: |
      verified by polar-compile-clean-03-wire docs rewrite — md.md's quick
      start and compiled-core section now show def_pairstyle +
      set_forcefield/set_neighbors, and `grep -rn "PotentialForceField|
      energy_scale=" docs/` is empty across all three pages. The same rewrite
      also removed the deleted LennardJonesCutForceField / HarmonicForceField /
      LennardJonesForceField / OPLSForceField names those pages still taught.
  - id: ac-008
    summary: Skin-07 regression keeps goldens after the rename
    type: runtime
    pass_when: |
      md-neighborlist-skin-07-wire.py uses def_pairstyle + set_neighbors
      and still asserts rebuild_count {0.0: 100, 1.0: 6}, ndanger
      {0.0: 99, 1.0: 0}, EXPECTED_E_TOT_FINAL = 1.512422884343174e-02.
    status: pending
  - id: ac-009
    summary: 05-wire regression reproduces public MD composition goldens
    type: runtime
    pass_when: |
      regressions/md-forcefield-molpy-reexport-05-wire.py uses only public
      API, asserts deleted names are absent, and reproduces the two-atom
      lj/cut energy (r=3.4, ε=0.15, σ=3.0, cutoff=10, atol 1e-12) plus
      a finite md.run(system, 1). No third-party import.
    status: pending
out_of_scope:
  - /home/jicli594/work/mace-nve
  - src/molix/md engine (01-04)
  - sbatch files
---

# Acceptance — md-forcefield-molpy-reexport-05-wire

全树调用方改走 `def_pairstyle` + `set_forcefield` + `set_neighbors`；自制适配器与 `energy_scale=` 从 scripts / docs / benchmarks / 遗留测试消失。mace-nve 不动。
