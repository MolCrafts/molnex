---
slug: composable-a0-05-export
status: approved
criteria:
  - id: ac-001
    summary: periodic electrostatics, switching and exclusions are emitted from the IR
    type: code
    pass_when: |
      LammpsAdapter.translate on an IR with nonbonded_method.coulomb == "pme"
      and lj_switch_start set yields the golden records "pair_style
      hybrid/overlay lj/mdf <r_s> <r_c> coul/long <r_c>", "kspace_style pppm
      <tol>" and "special_bonds lj 0 0 0.5 coul 0 0 0.8333333"; with
      lj_switch_start None it yields "pair_style lj/cut/coul/long <r_c>";
      OpenMMAdapter.translate yields NonbondedForce with nonbondedMethod PME,
      useSwitchingFunction true and the same cutoffs (text goldens in
      tests/test_molix/test_ff_export/).
    status: pending
    last_checked: null
  - id: ac-002
    summary: LammpsWriter emits the pair block and the manifest sibling
    type: code
    pass_when: |
      write_pair_block output matches its text golden; write_manifest writes
      manifest.json that ModelManifest.read round-trips; write_settings with
      skip_pair_style=True contains no pair_style line
      (test_lammps_writer.py).
    status: pending
    last_checked: null
  - id: ac-003
    summary: molnex and LAMMPS agree on the exported periodic A0 field
    type: scientific
    pass_when: |
      interface/tests/parity_classical.py --periodic (developer-run with lmp;
      ctest entry skipped when lmp is absent): single-point |dE|/|E| < 1e-6 and
      max|dF| < 1e-5 kcal/mol/A between Potentials and LAMMPS for the ion/water
      box with matched exclusions, switch and pppm 1e-6; --case butane matches
      its captured literals.
    status: pending
    last_checked: null
  - id: ac-004
    summary: AOTI export carries the manifest without changing meta.json
    type: code
    pass_when: |
      export_for_lammps(..., manifest=m.to_dict()) writes <name>.manifest.json;
      <name>.meta.json is byte-identical to an export without manifest;
      LAMMPS_META_SCHEMA == 3 stays pinned by tests/test_molix/test_engine/test_export.py and interface/tests
      (tests/test_molix/test_engine/test_export.py).
    status: pending
    last_checked: null
  - id: ac-005
    summary: no lammps / openmm import anywhere in the Python tree; parity lives under interface/
    type: code
    pass_when: |
      tests/test_repository_boundaries.py asserts no "import lammps", "from
      lammps", "import openmm", "from openmm" under src/, tests/, scripts/;
      scripts/mm_param_learning/parity_lammps_classical_md.py no longer exists
      and interface/tests/parity_classical.py does.
    status: pending
    last_checked: null
---
