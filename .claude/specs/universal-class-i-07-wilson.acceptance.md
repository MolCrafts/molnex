---
slug: universal-class-i-07-wilson
criteria:
  - id: ac-001
    summary: stretch_rows reproduces the analytic diatomic B row
    type: code
    pass_when: |
      In tests/test_molpot/test_forcefield/test_wilson.py,
      WilsonBMatrix().stretch_rows(pos=[[0,0,0],[1,0,0]] float64,
      bond_index=[[0],[1]]) equals torch.tensor([[-1,0,0,1,0,0]]) to atol 1e-12,
      and each 3-component atom block of the row has unit norm (dimensionless
      direction cosines). This is the same golden currently hand-typed at
      tests/test_molix/test_core/test_losses_molecular.py:168.
    status: verified
    last_checked: 2026-08-14
    verified_by: mol-impl
  - id: ac-002
    summary: bend_rows reproduces the analytic right-angle s-vector row
    type: code
    pass_when: |
      WilsonBMatrix().bend_rows(pos=[[1,0,0],[0,0,0],[0,1,0]] float64,
      angle_index=[[0],[1],[2]]) equals torch.tensor([[0,-1,0, 1,1,0, -1,0,0]])
      to atol 1e-12, i.e. apex atom is angle_index row 1 and row units are Å⁻¹.
    status: verified
    last_checked: 2026-08-14
    verified_by: mol-impl
  - id: ac-003
    summary: Rows match central finite differences on water (r=0.9572 Å, θ=104.52°)
    type: code
    pass_when: |
      For the hard-coded water geometry O=(0,0,0), H=(0.9572,0,0),
      H=(-0.23921,0.92695,0) in float64, every entry of stretch_rows and
      bend_rows agrees with a central finite difference (h=1e-5 Å) of
      r=‖x_j−x_i‖ and θ=acos(û_ji·û_jk) to atol 1e-7.
    status: verified
    last_checked: 2026-08-14
    verified_by: mol-impl
  - id: ac-004
    summary: Translational and rotational invariants hold exactly
    type: code
    pass_when: |
      Tests assert (a) the sum of per-atom s-vectors is zero for every bond and
      angle row (atol 1e-12), (b) a global translation of pos leaves B unchanged
      (atol 1e-12), and (c) for a fixed R in SO(3) each per-atom 3-vector block
      of the rotated build equals R @ s of the unrotated build (atol 1e-10).
    status: verified
    last_checked: 2026-08-14
    verified_by: mol-impl
  - id: ac-005
    summary: Degenerate geometry and wrong index shapes fail loud
    type: code
    pass_when: |
      pytest.raises(ValueError) passes for: bond_index passed as (E,2) with a
      message naming the bond_index vs edge_index distinction; angle_index whose
      leading dim is not 3; a bond with r < length_tol; and a bend with
      |sin θ| < sin_tol (θ=179.9999°) whose message names the offending row
      index. θ=179° does not raise and still matches finite differences.
    status: verified
    last_checked: 2026-08-14
    verified_by: mol-impl
  - id: ac-006
    summary: Empty term lists and dtype/device are handled without special-casing
    type: code
    pass_when: |
      stretch_rows with bond_index shape (2,0) and bend_rows with angle_index
      shape (3,0) return tensors of shape (0, 3N) carrying pos's dtype and
      device; a float32 pos yields float32 rows; a graph with no terms yields a
      (0, 3N_g) block from graph_blocks.
    status: verified
    last_checked: 2026-08-14
    verified_by: mol-impl
  - id: ac-007
    summary: graph_blocks emits independent per-molecule blocks in bonds-then-angles order
    type: code
    pass_when: |
      For a two-molecule batch (water then H2, atom_batch=[0,0,0,1,1]),
      graph_blocks returns two tensors of shape (3,9) and (1,6); each equals the
      standalone single-molecule stretch_rows/bend_rows build with locally
      rebased indices (atol 1e-12), rows ordered all bonds then all angles in
      input order. A term whose atoms span two graphs, and a non-monotonic
      atom_batch, each raise ValueError naming the offending term/index.
    status: verified
    last_checked: 2026-08-14
    verified_by: mol-impl
  - id: ac-008
    summary: WilsonBMatrix is exported from molpot and molpot.forcefield
    type: code
    pass_when: |
      `from molpot import WilsonBMatrix` and
      `from molpot.forcefield import WilsonBMatrix` both succeed and bind the
      same class object; the symbol appears in __all__ of both
      src/molpot/__init__.py and src/molpot/forcefield/__init__.py.
    status: verified
    last_checked: 2026-08-14
    verified_by: mol-impl
  - id: ac-009
    summary: Regression example reproduces hard-coded Wilson goldens end to end
    type: runtime
    pass_when: |
      `PYTHONPATH=src python regressions/universal-class-i-07-wilson.py` exits 0,
      asserting against literals embedded in the file (analytic s-vectors,
      hand-derived 2026-08-14, no third-party at runtime): diatomic row
      [[-1,0,0,1,0,0]] (atol 1e-12), right-angle bend row
      [0,-1,0, 1,1,0, -1,0,0] (atol 1e-12),
      ProjectedInternalHessianLoss.project(2·BᵀB, B) == [[2.0]] (atol 1e-6), and
      two-molecule graph_blocks shapes (3,9)/(1,6) equal to the standalone
      single-molecule builds.
    status: verified
    last_checked: 2026-08-14
    verified_by: mol-impl
  - id: ac-010
    summary: Docstrings state shapes, per-row units, refs, and the no-Seminario caveat
    type: docs
    pass_when: |
      WilsonBMatrix and its three public methods in
      src/molpot/forcefield/wilson.py carry Google-style docstrings with
      Args/Returns/Raises and tensor shapes ((N,3), (2,n_b), (3,n_a),
      (n_q,3N)); the module docstring states bond rows are dimensionless and
      angle rows are Å⁻¹, cites Wilson/Decius/Cross 1955 and
      https://doi.org/10.1063/1.1750829, and states that B is supplied at the
      given geometry with no Seminario minimum-geometry projection.
    status: pending
  - id: ac-011
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

ac-001 through ac-003 are the exactness contract: two closed-form s-vector goldens plus a
finite-difference check on a real molecular geometry. ac-004 covers the physical invariants
(translation kills the row sum; rotation is covariant), which are the cheapest guards against a
sign or index-transposition error. ac-005 and ac-006 pin the failure surface: near-linear bends
and mis-shaped index tensors raise instead of returning a plausible-looking matrix. ac-007 is
the packed-batch guarantee that per-molecule blocks never mix graphs. ac-008 is the export
surface, ac-009 the in-repo regression, ac-010 the documentation bar, ac-011 the full gate.

No `scientific`-type criterion is emitted: `$META` declares no bench configuration, and the
domain validation for this spec is fully realizable in-repo as analytic goldens (ac-001, ac-002)
plus finite-difference and invariant checks (ac-003, ac-004). The Seminario restriction is a
*negative* requirement — it is enforced by ac-010 (documented caveat) and by the absence of any
eigen-decomposition in the implementation, not by a numeric threshold.
