---
slug: composable-a0-debt-01-mixing
status: approved
criteria:
  - id: ac-001
    summary: the mixing rule is an explicit style option carried to the IR and manifest
    type: code
    pass_when: |
      def_pairstyle('lj/cut', mixing='geometric') is recorded as NonbondedMethodRecord.lj_mixing == 'geometric' by to_ir() and manifest(); the bare default is 'lorentz-berthelot'; an unknown rule raises ValueError at def_pairstyle (tests mirrored under tests/test_molpot).
    status: verified
    last_checked: 2026-09-08
    verified_by: impl
    evidence: "tests/test_molpot/test_forcefield/{test_to_ir,test_container,test_schema}.py + test_ir/test_nonbonded.py + test_manifest.py - def_pairstyle('lj/cut', mixing='geometric') is recorded as NonbondedMethodRecord.lj_mixing and carried by ForceField.manifest(); bare ForceField reports lorentz-berthelot; unknown rule and unknown option key (mixng=) raise ValueError at def_pairstyle listing the vocabulary (StyleSchema.options); MANIFEST_SCHEMA == 2, schema-1 payloads read back with lj_mixing None"
  - id: ac-002
    summary: LJCutPotential and the exporters honour the declared rule
    type: scientific
    pass_when: |
      With mixing='geometric' the LJ energy of a two-atom pair equals 4 eps_ij[(sig_ij/r)^12 - (sig_ij/r)^6] with eps_ij = sqrt(eps_i eps_j), sig_ij = sqrt(sig_i sig_j) to 1e-12 fp64, and analytic forces match autograd to 1e-10; write_pair_block emits pair_coeff i j lines mixed by the same rule (golden sig_12 2.95804 for sig 3.5/2.5); the OpenMM record carries mixing and marks geometric as DECOMPOSE with a note.
    status: verified
    last_checked: 2026-09-08
    verified_by: impl
    evidence: "tests/test_molpot/test_potentials/test_vdw/test_lj126.py + test_forcefield/test_pair_terms.py + tests/test_molix/test_ff_export/{test_lammps_adapter,test_lammps_writer,test_openmm_adapter}.py - geometric eps_ij 0.04449719 / sig_ij 2.95803989 goldens, LJCutPotential(mixing=GEOMETRIC) energy equals the closed form to 1e-12 and analytic forces match autograd to 1e-10; writer golden pair_coeff 1 2 lj/mdf 0.04449719 2.95804; OpenMM record carries mixing, geometric marked decompose with a note; LJ bag without a rule refused at both exporters; Compiler.explain on energy_forces 1 graph / 0 breaks for both rules (architect-verified)"
  - id: ac-003
    summary: OPLSAAForceField declares geometric mixing
    type: code
    pass_when: |
      OPLSAAForceField().to_ir().nonbonded_method.lj_mixing == 'geometric'; every OPLS-template LJ golden in the tree is re-pinned to the geometric values; full fast suite green.
    status: verified
    last_checked: 2026-09-08
    verified_by: impl
    evidence: "OPLSAAForceField().to_ir().nonbonded_method.lj_mixing == 'geometric' (test_to_ir.py); no OPLS-template LJ golden in the tree changed value (all use exclusion scales or structure); full fast suite 3142 passed / 15 skipped, ruff/format/ty clean (debt-chain gate 2026-09-08)"
---
