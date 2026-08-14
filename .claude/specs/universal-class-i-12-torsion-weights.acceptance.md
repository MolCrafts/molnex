---
slug: universal-class-i-12-torsion-weights
criteria:
  - id: ac-001
    summary: ParsleyEnergyWeight reproduces the Parsley three-branch weight table
    type: code
    evaluator_hint: "pytest tests/test_molix/test_core/test_losses_molecular.py -k ParsleyEnergyWeight"
    pass_when: |
      ParsleyEnergyWeight() (denom=1.0, upper=5.0 kcal/mol) maps relative energies
      [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0] to
      [1.0, 1.0, 1.0, 0.7071067811865476, 0.4472135954999579, 0.0, 0.0]
      within atol 1e-12 (float64), and ParsleyEnergyWeight(denom=2.0, upper=6.0) maps
      [1.0, 2.0, 4.0, 6.0, 10.0] to [1.0, 1.0, 0.7071067811865475, 0.0, 0.0].
    status: pending
  - id: ac-002
    summary: weight policy is guarded, continuous, non-negative, and gradient-free
    type: code
    pass_when: |
      In tests/test_molix/test_core/test_losses_molecular.py::TestParsleyEnergyWeight:
      ValueError is raised for denom=0.0, for upper=0.5 with denom=1.0, and for a
      negative relative-energy input; w(1.0 - 1e-9) equals w(1.0) = 1.0 within 1e-6;
      on arange(0, 8, 0.01) every weight is >= 0 and diff(w) <= 0 everywhere; and a
      requires_grad=True input yields weights with requires_grad False.
    status: pending
  - id: ac-003
    summary: center_by_group generalized with keyword-only weights, default unchanged
    type: code
    pass_when: |
      center_by_group(values, group_ids, *, weights=None) subtracts the
      weight-normalized group mean when weights is given — on d = [-0.5, -0.5,
      sqrt(2), 50.0] with weights [1., 1., 1/sqrt(2), 0.] it returns d unchanged within
      atol 1e-12 — and with weights=None reproduces the pre-existing golden
      [-1.0, 1.0, -2.0, 2.0] bit-for-bit. No second weighted-centering helper is added.
    status: pending
  - id: ac-004
    summary: weighted TorsionProfileLoss matches the Sigma-w normalized analytic golden
    type: code
    pass_when: |
      TorsionProfileLoss(energy_weight=ParsleyEnergyWeight()) on fixture P
      (e_true [0.0, 0.5, 2.0, 7.0], e_pred [-0.5, 0.0, 2.0+sqrt(2), 57.0], group
      [0,0,0,0], float64) returns 0.7071067811865476 (= 1/sqrt(2)) within atol 1e-12,
      exercising all three weight branches; and with uniform weights (fixture E:
      e_true [0.0, 0.5], e_pred [0.0, 2.5]) the weighted path equals the default path
      at 1.0.
    status: pending
  - id: ac-005
    summary: one scan contributes O(1) regardless of grid count
    type: code
    pass_when: |
      With group 0 = fixture P (4 grid points) and group 1 = fixture P with every grid
      point duplicated (8 points), TorsionProfileLoss(energy_weight=ParsleyEnergyWeight(),
      reduction="none") returns shape (2,) with both entries 0.7071067811865476 (atol
      1e-12); reduction="sum" returns 1.4142135623730951; reduction="mean" returns
      0.7071067811865476.
    status: pending
  - id: ac-006
    summary: zero-weight configurations are excluded; barrier term stays unweighted
    type: code
    pass_when: |
      On fixture Q (e_true [0.0, 0.5, 2.0, 7.0], e_pred [1.0, 1.5, 3.0, 57.0], group
      [0,0,0,0]) TorsionProfileLoss(energy_weight=ParsleyEnergyWeight()) returns exactly
      0.0 despite the 50 kcal/mol error on the w=0 configuration; the same call with
      barrier_weight=1.0 returns 2401.0; and on fixture P with e_pred.requires_grad_(),
      after backward() e_pred.grad[3] is exactly 0.0 while the other three entries are
      finite and non-zero. A singleton group with weights on returns 0.0.
    status: pending
  - id: ac-007
    summary: default-off preserves current behavior and the 11-step routing contract
    type: code
    pass_when: |
      TorsionProfileLoss.__init__ gains only the keyword-only energy_weight argument
      (default None) with no change to forward's signature, to pred/target key names, or
      to any existing default; TorsionProfileLoss() and
      TorsionProfileLoss(energy_weight=None) return 1.0 bit-for-bit on the pre-existing
      fixture (pred [0., 4.], true [0., 2.], group [0, 0]); and `git diff` on
      tests/test_molix/test_core/test_losses_molecular.py shows additions only — zero
      modified or deleted lines in the pre-existing TestCenterByGroup and
      TestTorsionProfileLoss cases, which all pass.
    status: pending
  - id: ac-008
    summary: ParsleyEnergyWeight exported with alphabetized __all__ and config.ftype buffers
    type: code
    pass_when: |
      `from molix.core.losses import ParsleyEnergyWeight` succeeds; "ParsleyEnergyWeight"
      appears in the alphabetically sorted __all__ of both
      src/molix/core/losses/molecular.py and src/molix/core/losses/__init__.py; denom and
      upper are created with register_buffer(..., dtype=config.ftype) (not nn.Parameter),
      appear in state_dict(), and follow .to(); `ruff check` and `ruff format --check`
      pass on the touched files.
    status: pending
  - id: ac-009
    summary: docstrings carry units, references, and the interpretation note
    type: docs
    pass_when: |
      ParsleyEnergyWeight, the center_by_group weights argument, and
      TorsionProfileLoss.__init__/forward have google-style docstrings with tensor
      shapes and kcal/mol units; ParsleyEnergyWeight's docstring states the three-branch
      formula, contains a Reference block citing Qiu et al. JCTC 17, 6262 (2021)
      https://doi.org/10.1021/acs.jctc.1c00571 (with Sage, Boothroyd et al. JCTC 19,
      3251 (2023) https://doi.org/10.1021/acs.jctc.3c00039), records that the middle
      branch exponent is read as -1/2 (i.e. 1/sqrt(1+(dE-1)^2)) because the literal
      "minus one half" reading yields negative weights above 2 kcal/mol, and notes that
      the eq.-3 1/dE^2 prefactor is composed via ScaleNormalizedLoss rather than
      re-implemented; TorsionProfileLoss.forward documents that with energy_weight set
      the reduction applies over profiles, not configurations, and that the barrier term
      stays unweighted.
    status: pending
  - id: ac-010
    summary: regression script reproduces hard-coded torsion-weight goldens
    type: runtime
    evaluator_hint: "PYTHONPATH=src python regressions/universal-class-i-12-torsion-weights.py"
    pass_when: |
      `PYTHONPATH=src python regressions/universal-class-i-12-torsion-weights.py` prints
      OK and exits 0, asserting the literal goldens 0.7071067811865476 (fixture P), 0.0
      and 2401.0 (fixture Q, the latter with barrier_weight=1.0), 1.0 (default-off), and
      0.7071067811865476 for
      ScaleNormalizedLoss(TorsionProfileLoss(energy_weight=ParsleyEnergyWeight()),
      scale=1.0) on fixture P, each within atol 1e-12 on CPU float64; the script uses
      only molix public imports plus torch, imports no pytest and no third-party oracle,
      spawns no subprocess, and its header records the capture command, commit sha, torch
      version, date, and device/precision per regressions/README.md.
    status: pending
  - id: ac-011
    summary: full check and test suite green
    type: runtime
    pass_when: |
      `ruff check src/ tests/ scripts/ regressions/ && ruff format --check src/ tests/
      scripts/ regressions/ && ty check src/ --exit-zero-on-warning` passes, and
      `python -m pytest tests/ -q -n 12 --dist worksteal` reports zero failures with no
      new skips or xfails.
    status: pending
---

# Acceptance criteria

- **ac-001 / ac-002** pin the physics of the weight policy: the branch table, and the
  three properties (continuity at `dE`, non-negativity, monotone decay) that distinguish
  the accepted exponent reading of the middle branch from the literal "minus one half"
  reading rejected in the spec's Domain basis. ac-002 also pins that weights never carry
  gradient.
- **ac-003** records the reuse verdict: `center_by_group` is generalized, not duplicated,
  and its existing behavior is unchanged when `weights` is omitted.
- **ac-004 / ac-005** are the two headline behaviors: the `Σw`-normalized weighted MSE
  golden, and the profile-level balancing that makes a scan's contribution independent of
  its grid count.
- **ac-006** binds "zero-weight configurations are excluded" at value *and* gradient level,
  and pins that the barrier term is untouched by the weighting.
- **ac-007** is the non-regression contract with the sibling 11-step router: constructor
  extension only, default byte-for-byte, and no edits to pre-existing test cases.
- **ac-010** is the delivery-time public-API regression; every golden is closed-form, so no
  external software is needed at any point.
