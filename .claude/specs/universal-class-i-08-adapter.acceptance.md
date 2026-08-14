---
slug: universal-class-i-08-adapter
criteria:
  - id: ac-001
    summary: Prediction dict contains exactly the declared keys, nothing else
    type: code
    pass_when: |
      Tests in tests/test_molpot/test_forcefield/test_predictions.py assert that
      ClassIPredictions(model, keys=K)(batch) returns a dict whose key set equals
      set(K) for K in {("energy",), ("energy","forces"), ("energy","hessian"),
      ("q",)}; with a call-counting stub Hessian derivation and a call-counting
      charge head, the counters are 0 for every key not in K, and the ("q",)
      case performs no Potentials assembly / energy evaluation (asserted via a
      counter on ForceField.to_potentials or Potentials.calc_energy).
    status: verified
    last_checked: 2026-08-14
    verified_by: mol-impl
  - id: ac-002
    summary: Energy parity with the direct ClassicalIMM path plus shape contract
    type: code
    pass_when: |
      adapter.energy(batch) equals model(batch).to_potentials().calc_energy(batch)
      with torch.allclose(atol=1e-10) in fp64, and its shape is () for a
      single-graph batch and (2,) for a two-molecule batch.
    status: verified
    last_checked: 2026-08-14
    verified_by: mol-impl
  - id: ac-003
    summary: Force parity with ForceDerivation and one-pass energy_forces equivalence
    type: code
    pass_when: |
      adapter.forces(batch) equals ForceDerivation(method="autograd")(
      lambda p: pots.calc_energy(batch, pos=p), pos) within atol 1e-10 (fp64),
      adapter.energy_forces(batch) equals (adapter.energy(batch),
      adapter.forces(batch)) within the same tolerance, and a force-MSE
      .backward() produces non-None grads on the bond-head parameters.
    status: verified
    last_checked: 2026-08-14
    verified_by: mol-impl
  - id: ac-004
    summary: Analytic harmonic-bond goldens for energy, forces and Hessian
    type: code
    pass_when: |
      For the two-atom bond k=100 kcal mol^-1 A^-2, r0=1.0 A, r=1.5 A along x
      (E = 0.5*k*(r-r0)^2), fp64 atol 1e-10: energy == 12.5 kcal/mol,
      forces == [[+50,0,0],[-50,0,0]] kcal mol^-1 A^-1, hessian shape (6,6) with
      H[0,0] == 100.0, H[0,3] == -100.0, H[1,1] == 33.333333333333336
      kcal mol^-1 A^-2, and H symmetric within 1e-10. Values are literals in
      tests/test_molpot/test_forcefield/test_predictions.py with the derivation
      and the Wilson 1941 (https://doi.org/10.1063/1.1750829) citation in the
      test docstring.
    status: verified
    last_checked: 2026-08-14
    verified_by: mol-impl
  - id: ac-005
    summary: q matches the analytic EEM solution and is geometry-independent
    type: code
    pass_when: |
      With chi=(0.0,1.0), eta=(2.0,2.0), total charge 0 and a
      DiagonalHardnessKernel EEM, adapter q == (+0.25,-0.25) e within atol 1e-10
      and equals AtomicChargeModel.charges(features, batch) built from the same
      head and solver; q for two different conformers of the same topology is
      bit-identical under torch.equal (dq/dx = 0).
    status: verified
    last_checked: 2026-08-14
    verified_by: mol-impl
  - id: ac-006
    summary: Bonded-only composition evaluates no LJ or Coulomb term
    type: code
    pass_when: |
      With a ForceField declaring only bond style, a model carrying an
      LJParameterHead atom_head that writes epsilon/sigma into atoms params, the
      adapter energy equals the bond golden 12.5 kcal/mol exactly (atol 1e-10)
      and the keys of Potentials.term_energies(batch) are a subset of
      {"bonds","angles","propers","impropers"} ("lj" and "coulomb" absent).
    status: verified
    last_checked: 2026-08-14
    verified_by: mol-impl
  - id: ac-007
    summary: predictions.py builds no derivative path of its own
    type: code
    pass_when: |
      src/molpot/forcefield/predictions.py contains no occurrence of
      "torch.autograd.grad", "torch.func.", or "autograd.functional.hessian";
      forces are obtained via Potentials/ForceDerivation and the Hessian only via
      the injected molpot.derivation derivation (asserted by a source-text test
      or equivalent static check).
    status: verified
    last_checked: 2026-08-14
    verified_by: mol-impl
  - id: ac-008
    summary: Potentials force methods delegate to the canonical autograd helper
    type: code
    pass_when: |
      src/molpot/forcefield/potentials.py calls
      molpot.derivation.autograd_forces_from_energy in both calc_forces and
      calc_energy_forces (no local torch.autograd.grad remains in those two
      methods), and tests/test_molpot/test_forcefield/test_potentials.py shows
      the returned forces equal the pre-change numerics within atol 1e-12 and
      still support force->parameter double backward.
    status: verified
    last_checked: 2026-08-14
    verified_by: mol-impl
  - id: ac-009
    summary: Fail-loud errors on malformed key sets and missing producers
    type: code
    pass_when: |
      pytest.raises(ValueError) passes for empty keys, duplicate keys, an
      unsupported key (message lists SUPPORTED_KEYS), keys containing "q" while
      model.charge_model is None, and keys containing "hessian" without
      hessian_derivation; pytest.raises(RuntimeError) passes when "q" is declared
      but the filled ForceField has no atoms["q"]. Every message names the
      offending condition.
    status: verified
    last_checked: 2026-08-14
    verified_by: mol-impl
  - id: ac-010
    summary: Public symbols documented with units and shapes; __all__ sorted
    type: docs
    pass_when: |
      ClassIPredictions and each of its public methods carry Google-style
      docstrings with Args/Returns/Raises and unit + shape annotations
      (kcal/mol, kcal mol^-1 A^-1, kcal mol^-1 A^-2, e; (B,), (N,3), (3N,3N),
      (N,)); __all__ in src/molpot/forcefield/__init__.py and
      src/molpot/__init__.py is alphabetically sorted and includes
      "ClassIPredictions"; ruff check and ruff format --check pass on all touched
      files.
    status: pending
  - id: ac-011
    summary: Regression example reproduces the hard-coded Class-I goldens
    type: runtime
    pass_when: |
      `PYTHONPATH=src python regressions/universal-class-i-08-adapter.py` exits 0
      and reproduces, from public API only, E == 12.5 kcal/mol,
      F == [[+50,0,0],[-50,0,0]] kcal mol^-1 A^-1, H[0,0] == 100.0,
      H[1,1] == 33.333333333333336 kcal mol^-1 A^-2 and q == (+0.25,-0.25) e at
      atol 1e-10, and asserts a q-only adapter returns a dict without an
      "energy" key. The script imports no third-party oracle and spawns no
      subprocess; goldens are in-file literals.
    status: verified
    last_checked: 2026-08-14
    verified_by: mol-impl
  - id: ac-012
    summary: Full check and test suite pass
    type: runtime
    pass_when: |
      `ruff check src/ tests/ scripts/ regressions/ && ruff format --check src/
      tests/ scripts/ regressions/ && ty check src/ --exit-zero-on-warning` and
      `python -m pytest tests/ -q -n 12 --dist worksteal` both exit 0 on the
      branch containing this spec.
    status: verified
    last_checked: 2026-08-14
    verified_by: mol-impl
---

# Acceptance criteria

ac-001 is the anti-façade contract: laziness is structural (keys fixed at
construction), so an undeclared prediction must be both absent from the dict and
never computed. ac-002 / ac-003 are the parity gates against the paths this
adapter is allowed to reuse — any numerical divergence from
`ClassicalIMM` + `Potentials` + `ForceDerivation` is a failure. ac-004 / ac-005
are the physical validation: analytic harmonic-bond derivatives and the analytic
EEM solution, plus the `dq/dx = 0` geometry-independence check that makes the
fixed-parameter force exact. ac-006 guards the "bonded-only task path must not
drag LJ in" constraint while `LJParameterHead` stays frozen in v0. ac-007 /
ac-008 keep a single force path in the repo (the adapter derives nothing itself;
`Potentials` stops duplicating the canonical autograd helper). ac-009 is error
quality, ac-010 documentation and the `__all__` ordering rule, ac-011 the
public-API regression example, ac-012 the full gate.

Note on types: the physical validation here is **analytic and in-repo** (no
external oracle, no bench target declared in `$META` for this spec), so ac-004 /
ac-005 are typed `code` — the binding evidence is a pytest assertion on
hard-coded analytic values, verifiable by `/mol:impl` at delivery. No
`scientific`-typed criterion is emitted. No `ui_runtime` criterion exists and the
spec has no UI verification section: nothing here touches a frontend.

Chain note: ac-001 (hessian branch) and ac-004 (Hessian goldens) require the
Hessian derivation from `universal-class-i-06` to be landed and injectable;
`universal-class-i-07` (Wilson `B`) is a loss-side input and is not gated here.
