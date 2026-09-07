---
slug: composable-a0-04-manifest
status: approved
criteria:
  - id: ac-001
    summary: manifest round-trips through dict and file
    type: code
    pass_when: |
      For an OPLSAAForceField filled by a Parameterizer, m = ff.manifest(profile="A0");
      ModelManifest.from_dict(m.to_dict()) == m and ModelManifest.read(m.write(p)) == m,
      with every enum serialised as its value (tests/test_molpot/test_manifest.py).
    status: pending
    last_checked: null
  - id: ac-002
    summary: schema version is pinned and tables are content-hashed
    type: code
    pass_when: |
      MANIFEST_SCHEMA == 1 is asserted as a literal; changing a single entry of
      any parameter table changes that ParamTableRecord.sha256 and leaves the
      others unchanged; tests/test_molpot/test_manifest.py matches its
      hard-coded key set and sha literals.
    status: pending
    last_checked: null
  - id: ac-003
    summary: PotentialIR carries the nonbonded method (F7)
    type: code
    pass_when: |
      ff.to_ir().nonbonded_method is a NonbondedMethodRecord whose lj_cutoff,
      lj_switch_start, coulomb, real_space_cutoff, smearing and mesh_spacing
      equal the values set through def_pairstyle(...) style options
      (tests/test_molpot/test_forcefield/test_to_ir.py).
    status: pending
    last_checked: null
---
