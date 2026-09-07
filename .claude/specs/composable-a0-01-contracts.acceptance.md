---
slug: composable-a0-01-contracts
status: code-complete
criteria:
  - id: ac-001
    summary: every registry style declares a contract and named terms; every emitted key is registered
    type: code
    pass_when: |
      For every key in PotentialMeta._REGISTRY, contract_for(key).terms is
      non-empty; on a full OPLSAAForceField every key returned by
      Potentials.term_energies() equals some TermSpec.key of a registered
      style, and the set of registered keys for the field's styles equals the
      emitted key set (tests/test_molpot/test_forcefield/test_schema.py,
      test_potentials.py).
    status: verified
    last_checked: 2026-09-07
    verified_by: impl
    evidence: tests/test_molpot/test_forcefield/test_schema.py (registry completeness) + test_potentials.py (emitted keys registered); mirrored run 164 passed
  - id: ac-002
    summary: LJ split and S-decomposition partition are exact
    type: scientific
    pass_when: |
      lj126_pair_energy_split(r, eps, sig) sums to lj126_pair_energy(r, eps, sig)
      to <= 1e-14 (fp64) on random r; at r = 2^(1/6) sigma rep = +eps and
      disp = -2 eps to 1e-12; sum(contribution_energies().values()) ==
      calc_energy() to <= 1e-12 on a Class-I field with LJ + Coulomb declared
      (tests/test_molpot/test_potentials/test_vdw/test_lj126.py, tests/test_molpot/test_forcefield/test_potentials.py).
    status: pending
    last_checked: null
  - id: ac-003
    summary: absent parameter table for a declared style raises; empty topology still zero
    type: code
    pass_when: |
      def_pairstyle("lj/cut") without epsilon/sigma, or pair:coulomb without q,
      raises ValueError naming both the style key and the missing table;
      a field with the style declared, tables present and no pairs within
      cutoff returns 0.0 (test_potentials.py).
    status: verified
    last_checked: 2026-09-07
    verified_by: impl
    evidence: test_potentials.py absent-table raises for pair:lj/cut, pair:coulomb and the four bonded styles; empty-topology guard green
  - id: ac-004
    summary: unsupported combinations and unregistered styles are refused at construction
    type: code
    pass_when: |
      Potentials() over a field declaring two POL styles, or an unregistered
      style key, raises ValueError; Parameterizer(polarizability_head=...)
      raises ValueError whose message contains "composable-a1"; the
      _LES_ELECTROSTATICS_PRECOMPUTED flag no longer exists in
      molpot.potentials.polarization (grep in test_contract.py).
    status: verified
    last_checked: 2026-09-07
    verified_by: impl
    evidence: test_potentials.py two-POL / unregistered raise, test_parameterizer.py polarizability_head raise; `grep -rn _LES_ELECTROSTATICS_PRECOMPUTED src/` = 0 hits
---
