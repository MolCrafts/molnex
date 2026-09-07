---
title: Periodic Coulomb — exact exclusions under Ewald/PME, frozen k-space shape, virial (composable-a0-03)
status: approved
created: 2026-09-07
chain: composable-a0
---

# composable-a0-03-periodic-coulomb

## Summary

Fix F1: under Ewald / PME the energy of an excluded or scaled bonded pair
must be corrected by the **full** `(1 − s)·k_e·q_i q_j / r`, not only its
real-space `erfc` part — today the `erf(r/σ√2)/r` (k-space) share of every
1-2 / 1-3 / 1-4 pair survives at full strength
(`src/molpot/forcefield/pair_terms.py:203-249`). Make
`CoulombPeriodicPotential` a `CellBound` with the k-space shape frozen at
`bind_cell`, replace data-dependent branches with `pair_mask`, default to PME
under a cell, and pin the assembled `Potentials` with finite-difference force
and strain-derivative tests. Closes S5 derivatives + virial for A0.

## Domain basis

Ewald identity: `1/r = erfc(r/σ√2)/r + erf(r/σ√2)/r`; the reciprocal sum
contains the `erf` part of *every* pair, including excluded ones, so an
exclusion needs `−(1 − s) q_i q_j [erfc/r + erf/r] = −(1 − s) q_i q_j / r`
(LAMMPS `special_bonds` + `coul/long`; OpenMM `NonbondedForce` exceptions).
`molix.schema.CellBound` (`src/molix/schema.py:105`): idempotent, O(1), reads
the live cell, never truncates silently. Fixed-capacity k-space policy:
`.claude/notes/notes.md` topic `polar-compile-clean-01-kgrid` (over-capacity
→ eager raise, in-graph NaN). Virial convention: symmetric strain, Cauchy
(`src/molpot/derivation/stress.py:25`, commit d362a948).

## Design

- `CoulombPeriodicPotential.energy(...)`: real-space via the calculator with
  `pair_mask` (`Calculator._compute_rspace(pair_mask=)`,
  `src/molpot/potentials/elec/calculators/calculator.py:39-45`) — no `keep` /
  `within` slicing, no `numel()` / `allclose` branches; then subtract
  `(1 − s)·k_e·q_i q_j / r_mic` over `ExclusionTable.pairs` (a0-02;
  topology-fixed count, independent of the neighbour list; minimum-image
  distance from the live cell).
- `bind_cell(cell)`: freezes `ns` (Ewald) or the mesh (PME / P3M) with
  `capacity_factor`; `energy` regenerates k-vectors from the live cell at
  the frozen shape (`generate_kvectors_for_ewald(ns=frozen, cell=live)`);
  over-capacity → eager `RuntimeError` naming `capacity_factor`, in-graph NaN.
  `EwaldCalculator._compute_kspace` takes `kvectors=` from the bound owner
  (its live-cell `ns` path at `elec/calculators/ewald.py:64-71` is the F3 sync).
- `Potentials.bind_cell` fans out to `coulomb_periodic`; `pair:coulomb`
  under a cell defaults to `calculator="pme"` with a power-of-two mesh at
  bind; `"ewald"` stays for small-box FD tests.
- The `KSpaceGrid` promotion trigger (half-box integer grid) is deliberately
  not fired here — `fftfreq` full-box grid stays; record in notes.

## Files

- `src/molpot/forcefield/pair_terms.py`, `src/molpot/forcefield/potentials.py`
- `src/molpot/potentials/elec/calculators/{ewald,pme,p3m,calculator}.py` (kvectors / mesh injection only)
- `tests/test_molpot/test_forcefield/test_coulomb.py`, `test_potentials.py`
- `.claude/notes/notes.md` (topic `compile-no-graph-breaks`: the retained autograd seam)

## Tasks

- [ ] exclusion correction over `ExclusionTable.pairs` + analytic pin test
- [ ] `pair_mask` path; delete slicing/branches
- [ ] `bind_cell` with frozen shape + capacity policy; PME default under a cell
- [ ] FD force + FD strain tests on the assembled `Potentials` (PME, 4-ion box)
- [ ] slow explain test; Madelung regression with capture header

## Testing

Unit: `test_coulomb.py` — bonded ion pair in a box:
`E(excluded) == E(unexcluded) − k_e q_i q_j / r_mic` ≤ 1e-8 (the F1 pin);
frozen-shape k-space equals live-shape; over-capacity raise + NaN.
`test_potentials.py` — FD force and FD strain derivative for PME on a 4-ion
box (pattern `tests/test_molpot/test_composition/test_sonata_periodic.py:294-361`,
frozen grid); slow: `Compiler.explain(calc_energy_forces_virial)` breaks only
at the named `molpot/derivation/force.py` seam (pattern
`test_potentials.py:445`). The 8-ion rocksalt cube Madelung energy is a hard-coded literal in `test_coulomb.py` (rtol 1e-6, Ewald and PME) next to the exclusion pin (no `regressions/` tier — deleted 2026-09-07).

## Out of scope

Dipole / multipole k-space (`composable-a1-04`, `composable-b1-03`); the
`KSpaceGrid` unification; packed periodic training batches (F4).
