---
slug: cache-layers-06-geometry
created: 2026-09-12
criteria:
  - id: ac-001
    summary: F29 on public calc_energy only; energy() uses _sum_energy
    type: code
    pass_when: |
      src/molpot/forcefield/potentials.py no longer assigns _checked_geometry
      and does not skip _check_bonds_unwrapped because _warm,
      torch.compiler.is_compiling(), or a previous Python object is the same.
      The _warm field is gone. energy() does not call calc_energy,
      _check_geometry, or _check_bonds_unwrapped, and does not add an
      is_compiling() / _warm / identity skip. Private Potentials._sum_energy
      includes the mlip call_energy branch and otherwise sums
      term_energies; calc_energy uses it after the classical F29 check;
      energy() uses it after the F13 guard then writes ENERGY_KEY.
      term_energies does not grow an F29 check. bind_topology /
      _check_geometry still run F29 when first reached from _scaled_pairs
      (no skip hatch on that path). _frame_view remains the TensorDict wrap
      memo used by _frame. weakref is imported only to key _frame_view.
      _frame returns a TensorFrame as-is and wraps other TensorDictBase via
      as_tensor_frame. _bound_topology still compares topology shape by
      value and raises on mismatch. No new public type.
    status: verified
    last_checked: 2026-09-12
  - id: ac-002
    summary: Same TensorDict with rewritten pos/cell re-runs F29
    type: scientific
    pass_when: |
      TestPotentials in tests/test_molpot/test_forcefield/test_potentials.py:
      after a passing public calc_energy on the unwrapped 10 Å-cube bond
      (E=50.0 kcal/mol), writing WRAPPED_X=(1,9) Å into the same atoms.pos
      raises ValueError matching F29 and naming 8 Å vs 5 Å; writing the
      sheared cell in place refuses a 4 Å bond against a 3 Å half-width.
    status: verified
    last_checked: 2026-09-12
  - id: ac-003
    summary: warmup/bind does not voucher a later wrapped batch
    type: scientific
    pass_when: |
      TestPotentials: warmup or bind_topology on the unwrapped periodic
      bond, then public calc_energy on a fresh wrapped TensorDict of the
      same shape, and public calc_energy after in-place wrap of the warmed
      object, both raise ValueError matching F29 (8 Å vs 5 Å). The warmup
      batch still prices at 50.0 kcal/mol.
    status: verified
    last_checked: 2026-09-12
  - id: ac-004
    summary: Unvouched compiled TensorFrame calc_energy still raises F29
    type: scientific
    pass_when: |
      The slow TestPotentials compile test in
      tests/test_molpot/test_forcefield/test_potentials.py: torch.compile of
      calc_energy on an unvouched wrapped TensorFrame (and on an unvouched
      wrapped TensorDict) raises ValueError matching F29 (8 Å vs 5 Å).
      No is_compiling() skip at the calc_energy call site. No new
      Compiler.explain task.
    status: pending
  - id: ac-005
    summary: NeighborList, CellBound, and MD driver stay unmodified
    type: code
    pass_when: |
      This spec's diff does not modify src/molix/md/neighbors.py,
      src/molix/neighbor.py, src/molix/schema.py CellBound, or
      src/molix/md/driver.py. Verlet skin / half-skin rebuild and
      CellBound.bind_cell remain the runtime validity owners. op(work)
      is unchanged. bind_work is not treated as an F29 voucher.
    status: verified
    last_checked: 2026-09-12
  - id: ac-006
    summary: Existing F29 public-entry goldens still hold on calc_energy
    type: scientific
    pass_when: |
      Existing TestPotentials F29 tests still refuse WRAPPED_X through
      bind_topology and public calc_energy, accept UNWRAPPED_X at 50.0
      kcal/mol, and refuse the 4 Å bond in the sheared cell (limit 3 Å).
      Fresh same-shape wrapped frames still raise on calc_energy.
      test_a_wrapped_bond_is_refused_through_calc_energy_without_an_explicit_bind
      no longer asserts that energy() or Potentials.__call__ raises F29.
      No golden freezes 2450 kcal/mol through energy().
    status: verified
    last_checked: 2026-09-12
  - id: ac-007
    summary: Fast-tier potentials tests reproduce F29 literals
    type: runtime
    evaluator_hint: "path: tests/test_molpot/test_forcefield/test_potentials.py"
    pass_when: |
      python -m pytest tests/test_molpot/test_forcefield/test_potentials.py -v
      (default fast tier) exits 0. Hard-coded literals reproduced there:
      WRAP_EDGE=10 Å, wrapped bond 8 Å vs half-width 5 Å, unwrapped
      energy 50.0 kcal/mol, sheared width 6 Å / limit 3 Å. No
      regressions/ script and no live third-party oracle.
    status: verified
    last_checked: 2026-09-12
  - id: ac-008
    summary: Polar/LJ/coulomb (1,0) pin wrap identity, not F29 skip
    type: code
    pass_when: |
      In tests/test_molpot/test_forcefield/test_potentials.py the (graph_count,
      graph_break_count)==(1, 0) asserts in
      test_the_classical_op_traces_to_one_graph_in_inference,
      test_the_periodic_coulomb_op_traces_to_one_graph_in_inference,
      test_the_compiled_route_matches_eager_and_breaks_only_at_the_leaf,
      and test_the_grad_enabled_compiled_route_is_one_graph_after_mark_leaves
      are unchanged. Their comments pin _frame_view as wrap identity for
      compiled MD op(work) and do not call that (1, 0) an F29-skip green
      or claim MD bind_work eager-vouches F29. Coulomb comments must not
      claim that _coulomb_batch has no bonds or that _check_bonds_unwrapped
      returns before argmax on it (that fixture has bonds; F29 runs at
      eager bind_topology / calc_energy). No new Compiler.explain
      test is added.
    status: verified
    last_checked: 2026-09-12
out_of_scope:
  - src/molix/md/driver.py op(work) to op(self._frame) (only sound _frame_view deletion); bind_work is not made to call calc_energy / bind_topology
  - F29 re-check inside compiled energy/Derivative without eager public calc_energy or bind_topology on that frame (includes Trainer.compile(step=True) over a stream of fresh batches AND bonded-only MD.run; no public F29 on the initial bonded-only MD frame; coverage is integrator-never-wraps plus the caller unwrapping before run; bind_work's eager op(work) is not an F29 voucher)
  - NeighborList Verlet skin / pipeline cutoff graph / CellBound edits
  - Compiler, Derivative, new explain tests
  - ForceAwareModel retirement
  - Cache, residual labels, AtomicReferenceEnergy, collate, scripts
  - F29 half-width formula, bonded kernels, MD wrap policy
  - term_energies growing an F29 check
  - regressions/ scripts (tier deleted; goldens are unit tests)
---

# Acceptance criteria

Done means public `calc_energy` / `bind_topology` / `warmup` always re-run F29 on the geometry they are handed, `energy()` stays the compile-clean `Derivative` sum through `_sum_energy` that writes `ENERGY_KEY`, `_frame_view` remains wrap identity for `op(work)`, bonded-only `MD.run` is named as compiled-`energy()` residual rather than vouched, and the polar/LJ/coulomb `(1, 0)` explains are not rebranded as F29-skip coverage.

## AC-001 — Split, not a skip

F29 identity / `_warm` / `is_compiling()` are not validity keys. `energy()` does not call `calc_energy` and does not grow a skip hatch. Shared sum is `_sum_energy`. Lazy `_scaled_pairs` → `bind_topology` still F29s once. `_frame_view` wrap identity stays. Exclusion fingerprints stay. No new type.

## AC-002 / AC-003 — Mutated object and post-warmup

Close the F29 residual on the **public scoring** entries (same object mutated; warmup then another batch). Both go through `calc_energy` / `bind_topology`, not `energy()`.

## AC-004 — TensorFrame first-contact

Closes the TensorFrame-first-contact fail-open the `is_compiling()` skip created once wrap no longer graph-breaks. The compiled **kernel** is `energy()`, not `calc_energy`.

## AC-005 — Do not “fix” caches by deleting owners

Forbids deleting Verlet skin, `CellBound`, or the MD `op(work)` seam. Forbids treating `bind_work` as an F29 voucher.

## AC-006 — Landed public-entry refusals

`calc_energy` and `bind_topology` still refuse wrapped bonds. `energy()` / `__call__` is no longer an F29 assertion. No \(2450\) kcal/mol golden through `energy()`.

## AC-007 — Runtime goldens

The mirrored unit module is the golden owner (`regressions/` is gone). Literals are the Domain-basis cube / shear numbers.

## AC-008 — Iron-law explains are wrap identity

Polar never reaches F29 (mlip). LJ has no `bonds`. Coulomb (`_coulomb_batch`) **has** `bonds`; F29 runs at eager `bind_topology` / `calc_energy`. Their `(1, 0)` pins `_frame_view` wrap identity of the explained `energy()` graph, not an F29 skip and not “MD already vouches.”
