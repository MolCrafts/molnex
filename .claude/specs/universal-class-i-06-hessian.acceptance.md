---
slug: universal-class-i-06-hessian
criteria:
  - id: ac-001
    summary: HessianDerivation public surface exists with no third differentiation path
    type: code
    pass_when: |
      src/molpot/derivation/hessian.py defines `class HessianDerivation(nn.Module)`
      with a no-argument constructor, `forward(energy_fn, pos) -> Tensor` returning
      (3N, 3N), and `graph_blocks(energy_core, batch) -> list[Tensor]`. The module
      imports its FIRST derivative from the existing layer
      (`autograd_forces_from_energy` from molpot.derivation.force and/or
      `grad_force_pass` from molpot.derivation.kernels) and contains no `torch.func.`
      call and no finite-difference/central-difference loop. `from molpot import
      HessianDerivation` and `from molpot.derivation import HessianDerivation` both
      succeed and the name appears in both `__all__` lists.
    status: verified
    last_checked: 2026-08-14
    verified_by: mol-impl
  - id: ac-002
    summary: Equilibrium harmonic-dimer Hessian matches the k*B^T*B closed form
    type: scientific
    pass_when: |
      For BondHarmonic(k=[2.0], r0=[1.5]) with atoms at (0,0,0) and (1.5,0,0)
      (float64), HessianDerivation().forward returns a (6, 6) matrix whose only
      non-zero entries are H[0,0]=H[3,3]=+2.0 and H[0,3]=H[3,0]=-2.0, with
      max|H - 2.0 * B.T @ B| <= 1e-10 for B = [[-1,0,0,1,0,0]]. Asserted in
      tests/test_molpot/test_derivation/test_hessian.py::TestHessianDerivation.
    status: pending
  - id: ac-003
    summary: Stretched-dimer Hessian reproduces the transverse k(r-r0)/r term
    type: scientific
    pass_when: |
      For BondHarmonic(k=[2.0], r0=[1.5]) with atoms at (0,0,0) and (2.0,0,0)
      (float64), the returned (6, 6) Hessian matches the hard-coded golden
      H[0,0]=H[3,3]=2.0, H[0,3]=H[3,0]=-2.0,
      H[1,1]=H[2,2]=H[4,4]=H[5,5]=0.5, H[1,4]=H[4,1]=H[2,5]=H[5,2]=-0.5,
      all other entries 0.0, to atol 1e-10. Dropping the (I - u u^T) term fails
      this criterion.
    status: pending
  - id: ac-004
    summary: graph_blocks returns one per-graph block and guards graph contiguity
    type: code
    pass_when: |
      HessianDerivation().graph_blocks(KernelToyPotential(scale=1.5),
      kernel_toy_batch(n_atoms=4, n_graphs=2)) returns a list of length 2 whose
      elements are (6, 6) and equal 3.0 * eye(6) to atol 1e-12, with the energy core
      invoked exactly once (`n_forward == 1`). A batch whose ("atoms","batch") is not
      non-decreasing (e.g. [0,1,0,1]) raises ValueError whose message names
      atoms.batch. No hessian key is written onto the batch by graph_blocks.
    status: verified
    last_checked: 2026-08-14
    verified_by: mol-impl
  - id: ac-005
    summary: Batch Hessian is block-diagonal across independent graphs
    type: scientific
    pass_when: |
      For the same 2-graph toy batch, the dense forward() Hessian of the summed
      energy has cross-graph off-diagonal blocks exactly 0.0 (max abs == 0.0) and its
      two diagonal blocks agree with graph_blocks() output to atol 1e-12, confirming
      per-graph extraction is exact rather than approximate.
    status: pending
  - id: ac-006
    summary: Hessian stays connected to parameters under grad, detached under no_grad
    type: code
    pass_when: |
      With grad enabled, backward() on graph_blocks(...)[0].sum() produces a non-zero
      `.grad` on the toy potential's `scale` parameter (third-derivative path
      d3E/dpos2/dtheta alive). Inside `torch.no_grad()`, the returned Hessian has
      requires_grad False. Both asserted in
      tests/test_molpot/test_derivation/test_hessian.py.
    status: verified
    last_checked: 2026-08-14
    verified_by: mol-impl
  - id: ac-007
    summary: Symmetry and translational invariance hold without symmetrization
    type: scientific
    pass_when: |
      On the stretched-dimer case (ac-003), max|H - H.T| <= 1e-10 and
      max|H.sum(dim=0)| <= 1e-10 (every column sums to zero), while the
      implementation performs no explicit 0.5*(H + H.T) symmetrization — the symmetry
      is measured, not imposed.
    status: pending
  - id: ac-008
    summary: Google docstrings state units, layout, scope and the Wilson reference
    type: docs
    pass_when: |
      src/molpot/derivation/hessian.py module docstring and both public methods use
      Google style with Args/Returns/Raises, tensor shapes, an explicit units line
      (energy kcal/mol, pos Angstrom -> Hessian kcal mol^-1 Angstrom^-2), the
      Cartesian x0,y0,z0,... ordering, the autograd-only / pure-torch scope note
      pointing at the cuet-force-doublebackward spec, and a Reference block citing
      Wilson, J. Chem. Phys. 9, 76 (1941), https://doi.org/10.1063/1.1750829.
    status: pending
  - id: ac-009
    summary: Regression example reproduces hard-coded goldens through the public API
    type: runtime
    pass_when: |
      `PYTHONPATH=src python regressions/universal-class-i-06-hessian.py` exits 0
      using only molnex public API (molpot.BondHarmonic, molpot.HessianDerivation,
      molix.core.losses.ProjectedInternalHessianLoss) and no third-party oracle at
      runtime. It asserts the equilibrium 6x6 Hessian against its hard-coded golden
      (atol 1e-10), then ProjectedInternalHessianLoss.project(H, [[-1,0,0,1,0,0]]) ==
      [[2.0]] (atol 1e-6) and the diagonal-mode loss vs [2.0] <= 1e-12. Goldens are
      analytic (Wilson closed form, 2026-08-14) and documented as such in the module
      docstring.
    status: verified
    last_checked: 2026-08-14
    verified_by: mol-impl
  - id: ac-010
    summary: Full check and test suite pass
    type: runtime
    pass_when: |
      `ruff check src/ tests/ scripts/ regressions/ && ruff format --check src/ tests/
      scripts/ regressions/ && ty check src/ --exit-zero-on-warning` is clean and
      `python -m pytest tests/ -q -n 12 --dist worksteal` is green with
      tests/test_molpot/test_derivation/test_hessian.py included and no pre-existing
      test regressed.
    status: verified
    last_checked: 2026-08-14
    verified_by: mol-impl
---

# Acceptance criteria

- ac-001 pins the shape of the surface: one stateless derivation type, two
  primitive methods, exported next to `ForceDerivation` / `StressDerivation`, and
  — the load-bearing part — the first derivative reused from the existing
  derivation layer with no functorch and no finite-difference path.
- ac-002 / ac-003 are the physics bar. ac-002 ties this spec's golden to the
  `k BᵀB` value already hard-coded in `regressions/universal-class-i-03-losses.py`;
  ac-003 is the discriminating case — a rank-1-only implementation passes ac-002
  and fails ac-003.
- ac-004 / ac-005 are the packed-batch contract: one block per graph, one energy
  forward, exact zero coupling across graphs, and a hard error instead of a
  silently mixed block when atom ordering is not graph-contiguous.
- ac-006 is what makes the producer usable for supervision in spec 08 — the
  Hessian must be differentiable w.r.t. the parameter head under grad, and cheap
  and detached under `no_grad`.
- ac-007 keeps the invariants observed rather than enforced, so a broken energy
  graph surfaces as asymmetry.
- ac-008 is the docstring gate (`doc.style: google`, units + reference required
  by the repo's scientific-correctness rule).
- ac-009 / ac-010 are the delivery gates, verified by `/mol:impl`.

Scope reminders for the evaluator: no criterion here covers Wilson `B`
construction (spec 07) or batch-key writes / loss wiring (spec 08). No
UI-verifiable behaviour exists in this spec.
