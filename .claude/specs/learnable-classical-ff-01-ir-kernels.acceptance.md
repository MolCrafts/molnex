---
slug: learnable-classical-ff-01-ir-kernels
criteria:
  - id: ac-001
    summary: CLASS_I_CANONICAL units are kcal/mol, Å, e, radian
    type: code
    pass_when: |
      Importing molpot.ir.CLASS_I_CANONICAL (or units.CLASS_I_CANONICAL) yields
      a frozen mapping with energy="kcal/mol", length="angstrom" (or "Å"),
      charge="e", angle="radian" (or "rad"). A unit test asserts exact keys and
      values; mutation of the mapping raises or is a no-op (MappingProxyType /
      frozendict / Final).
    status: verified
    last_checked: 2026-08-10
    verified_by: agent-auto
  - id: ac-002
    summary: NonbondedScaling Class-I defaults for 1-2 / 1-3 / 1-4
    type: scientific
    pass_when: |
      NonbondedScaling() (or .class_i_defaults()) has scale_q_12=0, scale_q_13=0,
      scale_q_14=5/6, scale_lj_12=0, scale_lj_13=0, scale_lj_14=0.5 within float
      tolerance. Test documents AMBER/GAFF / SMIRNOFF Class-I provenance.
    status: verified
    last_checked: 2026-08-10
    verified_by: agent-auto
  - id: ac-003
    summary: PotentialIR holds optional bags and validates unit_system
    type: code
    pass_when: |
      PotentialIR can be constructed with any subset of BondBag, AngleBag,
      ProperTorsionBag, ImproperPeriodicBag, ImproperHarmonicBag, LJBag,
      ChargeBag, and NonbondedScaling. Unknown unit_system raises ValueError.
      Empty PotentialIR is valid (zero contribution).
    status: verified
    last_checked: 2026-08-10
    verified_by: agent-auto
  - id: ac-004
    summary: ProperTorsionPeriodic cis golden E=2.0 kcal/mol
    type: scientific
    pass_when: |
      With k=1.0, s=1, n=1, gamma=0 on a geometry with proper torsion angle
      phi=0 (cis), ProperTorsionPeriodic returns energy allclose to 2.0
      (formula (k/s)[1+cos]=2k with k=1)
      (kcal/mol). Hard-coded golden; no external oracle.
    status: verified
    last_checked: 2026-08-10
    verified_by: agent-auto
  - id: ac-005
    summary: ProperTorsionPeriodic multi-term sums component energies
    type: scientific
    pass_when: |
      Two-term proper with known (k,n,gamma) pairs: total energy equals the
      sum of independently evaluated single-term energies within 1e-6 relative
      or absolute tolerance on a fixed geometry.
    status: verified
    last_checked: 2026-08-10
    verified_by: agent-auto
  - id: ac-006
    summary: ProperTorsionPeriodic rejects non-[4,N] proper_index
    type: code
    pass_when: |
      pytest.raises(ValueError) when proper_index has shape [N,4], [E,2], or
      [2,N]; error message names expected COO-style [4, num_propers].
    status: verified
    last_checked: 2026-08-10
    verified_by: agent-auto
  - id: ac-007
    summary: ImproperPeriodic matches cosine formula on a fixture geometry
    type: scientific
    pass_when: |
      ImproperPeriodic energy equals the analytic sum_n (k_n/s)[1+cos(n*chi-gamma)]
      computed from an independently calculated improper angle on a 4-atom
      fixture (hard-coded positions + expected energy).
    status: verified
    last_checked: 2026-08-10
    verified_by: agent-auto
  - id: ac-008
    summary: ImproperHarmonic matches 1/2 k (chi-chi0)^2
    type: scientific
    pass_when: |
      ImproperHarmonic energy equals 0.5*k*(chi-chi0)^2 for a fixture with
      known chi and parameters; forces via calc_forces / ForceDerivation are
      finite and finite-difference consistent within tolerance.
    status: verified
    last_checked: 2026-08-10
    verified_by: agent-auto
  - id: ac-009
    summary: DihedralHarmonic docstring marks improper-style / not Class-I proper
    type: docs
    pass_when: |
      DihedralHarmonic module or class docstring explicitly states it is the
      harmonic form only and points users at ProperTorsionPeriodic for Class-I
      propers. Grep/test or manual review of src/molpot/potentials/dihedrals/harmonic.py.
    status: verified
    last_checked: 2026-08-10
    verified_by: agent-auto
  - id: ac-010
    summary: Reused kernels BondHarmonic / AngleHarmonic / LJ126 remain importable
    type: code
    pass_when: |
      from molpot import BondHarmonic, AngleHarmonic, LJ126 succeeds; existing
      unit tests for those kernels still pass (no signature break).
    status: verified
    last_checked: 2026-08-10
    verified_by: agent-auto
  - id: ac-011
    summary: New symbols export from molpot with Google docstrings
    type: docs
    pass_when: |
      ProperTorsionPeriodic, ImproperPeriodic, ImproperHarmonic, PotentialIR,
      NonbondedScaling, CLASS_I_CANONICAL are importable from molpot or
      molpot.ir / molpot.potentials; each public class has Google-style docstring
      with tensor shapes and reference pointers (OpenMM / SMIRNOFF / Cornell /
      GAFF as applicable). ruff check clean on touched paths.
    status: verified
    last_checked: 2026-08-10
    verified_by: agent-auto
  - id: ac-012
    summary: Regression suite hard-codes cis E=2.0 golden without OpenMM
    type: scientific
    pass_when: |
      regressions/learnable-classical-ff-01-ir-kernels.py asserts the cis E=2.0
      golden (k=1) and bond/angle 1/2 k sanity values; the file does not import
      openmm; lives under regressions/ (not default unit suite).
    status: verified
    last_checked: 2026-08-10
    verified_by: agent-auto
  - id: ac-013
    summary: Full unit check + test suite pass
    type: runtime
    pass_when: |
      ruff check + format --check on src/tests and pytest tests/ (default
      addopts, regression excluded) exit 0 on the branch implementing this spec.
    status: verified
    last_checked: 2026-08-10
    verified_by: agent-auto
---

# Acceptance criteria

- ac-001 / ac-002 / ac-003 lock the Potential IR contract (units, scaling defaults, bag aggregate) that every later sub-spec imports.
- ac-004 / ac-005 / ac-007 / ac-008 / ac-012 are the scientific gate for Class-I torsion/improper kernels (hard-coded goldens; cis E=2.0 is the load-bearing regression).
- ac-006 is the anti-alias shape guard (topology `[4,N]` vs geometric edges).
- ac-009 / ac-010 prevent silent repurposing of DihedralHarmonic and protect reuse of existing bonded/nonbonded kernels.
- ac-011 / ac-013 are docs + full suite gates.
