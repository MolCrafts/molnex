---
slug: composable-a0-05-export
status: approved
criteria:
  - id: ac-001
    summary: periodic electrostatics, switching and exclusions are emitted from the IR
    type: code
    pass_when: |
      LammpsAdapter.translate on an IR whose nonbonded_method has
      coulomb_calculator "pme" and lj_switch_start set yields the golden records
      "pair_style hybrid/overlay lj/mdf <r_s> <r_c> coul/long <r_real>",
      "kspace_style pppm <kspace_accuracy>" and "special_bonds lj 0 0 0.5 coul
      0 0 0.8333333" (numbers formatted .7g); with lj_switch_start None it
      yields "pair_style lj/cut/coul/long <r_c> <r_real>"; with coulomb_cutoff
      only, "pair_style lj/cut/coul/cut <r_c> <r_q>" and no kspace record;
      OpenMMAdapter.translate yields NonbondedForce with nonbondedMethod "PME",
      useSwitchingFunction true, cutoffDistance / switchingDistance in nm
      (text goldens in tests/test_molix/test_ff_export/).
    status: verified
    last_checked: 2026-09-08
    verified_by: impl
    evidence: "tests/test_molix/test_ff_export/test_lammps_adapter.py (11 passed) - hybrid/overlay lj/mdf 10 12 coul/long 15 + kspace_style pppm 1e-06 + special_bonds lj 0 0 0.5 coul 0 0 0.8333333 goldens, lj/cut/coul/long 12 15, lj/cut/coul/cut 12 12 without kspace, lj/mdf 10 12, UnsupportedTermError, kspace_accuracy ctor; test_openmm_adapter.py (14 passed) - PME / CutoffNonPeriodic / NoCutoff, cutoffDistance 1.2 nm, switchingDistance 1.0 nm, ewaldErrorTolerance"
  - id: ac-002
    summary: LammpsWriter emits the pair block and the manifest sibling
    type: code
    pass_when: |
      write_pair_block output matches its text golden (pair_style line, one
      Lorentz-Berthelot-mixed pair_coeff line per i <= j type pair,
      sub-style-prefixed under hybrid/overlay plus pair_coeff * * for the
      Coulomb sub-style, kspace_style, special_bonds); write_settings with skip_pair_style=True on a force field
      that declares pair styles contains no pair_style / pair_coeff /
      special_bonds line but still its bond_coeff lines (test_lammps_writer.py).
      write_manifest is dropped - ModelManifest.write (a0-04) is the writer.
    status: verified
    last_checked: 2026-09-08
    verified_by: impl
    evidence: "tests/test_molix/test_ff_export/test_lammps_writer.py (11 passed) - exact pair-block goldens (hybrid/overlay and single style, explicit LB-mixed i<=j lines, pair_coeff * * coul/long, kspace_style, special_bonds), skip_pair_style=True include has bond_coeff but no pair_style / pair_coeff / special_bonds / units lines, box from graphs.cell, LAMMPS-legal type labels, special_bonds follow nonbonded_scaling; write_manifest dropped (ModelManifest.write)"
  - id: ac-003
    summary: molnex and LAMMPS agree on the exported periodic A0 field
    type: scientific
    pass_when: |
      interface/tests/parity_classical.py --case periodic (run with the local
      lmp; ctest entry SKIP_RETURN_CODE 77 when lmp is absent): single-point
      |dE|/|E| < 1e-6 and max|dF| < 1e-5 kcal/mol/A between Potentials and
      LAMMPS for the water/ion box with matched exclusions, switch and a
      converged lattice sum on both sides; --case butane matches its captured
      literals. Evidence is the recorded run (Goldens block + provenance).
    status: verified
    last_checked: 2026-09-08
    verified_by: impl
    evidence: "interface/tests/parity_classical.py run 2026-09-08 with /home/jicli594/.local/x86_64/bin/lmp (LAMMPS 30 Mar 2026, patch_30Mar2026-1074): --case butane |dE|/|E| 2.583e-12, max|dF| 3.506e-12; --case periodic (8 water + Na+ + Cl-, L 12 A, hybrid/overlay lj/mdf 4.5 5.5 coul/long 5.5, pppm 1e-10, gewald pinned) |dE|/|E| 9.913e-09, max|dF| 6.876e-06 kcal/mol/A with the writer's deck consumed unmodified; molnex ewald vs pme mesh diagnostic printed (2.5e-07, above the 1e-08 target, reported not hidden); GOLDENS block + provenance in the tool and interface/lammps/README.md"
  - id: ac-004
    summary: AOTI export carries the manifest without changing meta.json
    type: code
    pass_when: |
      export_for_lammps(..., manifest=m.to_dict()) writes <name>.manifest.json
      whose JSON equals the mapping; <name>.meta.json is byte-identical to an
      export without manifest (slow test in
      tests/test_molix/test_engine/test_export.py); LAMMPS_META_SCHEMA == 3
      stays pinned by interface/tests.
    status: verified
    last_checked: 2026-09-08
    verified_by: impl
    evidence: "tests/test_molix/test_engine/test_export.py (fast tier, AOTI compile stubbed behind the Exporter seam, eager schema-v3 checks run) - model.manifest.json written only with manifest=, json equals the mapping, model.meta.json bytes identical across the two exports; LAMMPS_META_SCHEMA == 3 untouched (export.py:130), interface/tests contract case unchanged"
  - id: ac-005
    summary: no lammps / openmm import anywhere in the Python tree (lint gate); parity lives under interface/
    type: code
    pass_when: |
      pyproject.toml selects ruff TID251 and bans "lammps" and "openmm" under
      [tool.ruff.lint.flake8-tidy-imports.banned-api]; ruff check on a probe
      file containing "import lammps", "from openmm import app" and
      "import lammps.core" reports TID251 three times, and ruff check src/
      tests/ scripts/ is clean; scripts/mm_param_learning/parity_lammps_classical_md.py
      no longer exists and interface/tests/parity_classical.py does. No test
      asserts on source text (operator rule 2026-09-07).
    status: verified
    last_checked: 2026-09-08
    verified_by: impl
    evidence: "pyproject.toml selects TID251 and bans lammps / openmm under [tool.ruff.lint.flake8-tidy-imports.banned-api]; ruff check on a probe file with import lammps / from lammps import / import lammps.core / from openmm import app / import openmm.unit reports TID251 5 times; ruff check src/ tests/ scripts/ clean; scripts/mm_param_learning/parity_lammps_classical_md.py deleted, interface/tests/parity_classical.py present; no source-scanning test added"
---
