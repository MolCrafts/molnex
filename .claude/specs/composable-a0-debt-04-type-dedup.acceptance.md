---
slug: composable-a0-debt-04-type-dedup
status: approved
criteria:
  - id: ac-001
    summary: types are condensed on parameter equality with one owner
    type: code
    pass_when: |
      ForceField.assign_types on the water/ion box yields 4 atom types, 1 bond type, 1 angle type with zero-padded first-occurrence labels; realize and LammpsWriter both consume it; identity_type_label is gone.
    status: verified
    last_checked: 2026-09-08
    verified_by: impl
    evidence: "tests/test_molpot/test_forcefield/test_container.py - water/ion box (2 H2O + Na + Cl) assign_types yields 4 atom types t1..t4 in first-occurrence order, 1 bond type b1, 1 angle type a1; two bonds between equal atom types with different r0 are 2 bond types; 10 distinct atoms give t01..t10 sorted; realize registers one molpy type per class; test_to_ir realize names == ['b1']; identity_type_label deleted; LammpsWriter takes types= (test_lammps_writer.py: 3 types -> 6 pair lines; disagreeing parameters within a type refused)"
  - id: ac-002
    summary: the exported deck shrinks and still agrees with LAMMPS
    type: scientific
    pass_when: |
      write_pair_block emits T(T+1)/2 lines; interface/tests/parity_classical.py --case periodic passes with |dE|/|E| < 1e-6 and max|dF| < 1e-5 on the condensed deck (recorded run).
    status: verified
    last_checked: 2026-09-08
    verified_by: impl
    evidence: "interface/tests/parity_classical.py re-run 2026-09-08 with the condensed deck (periodic 545 -> 116 lines: 4 atom / 1 bond / 1 angle types, 10 pair_coeff lines): --case periodic |dE|/|E| 9.912e-09, max|dF| 6.876e-06; --case butane 2.583e-12 / 3.506e-12; GOLDENS untouched; README status RECORDED 2026-09-08 (condensed types)"
---
