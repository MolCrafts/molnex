---
title: Per-class exclusion table, quintic LJ switch, fail-closed total charge (composable-a0-02)
status: approved
created: 2026-09-07
chain: composable-a0
---

# composable-a0-02-exclusions-switching

## Summary

Make exclusions a per-interaction-class object built once per topology
(`ExclusionTable`, fixed-shape lookup, `torch.unique` only at build), turn
`NonbondedScaling` into a keyed set (`q=`, `lj=`) that A1 extends with
`induced=` without changing the contract, replace the hard LJ cutoff with a
branch-free quintic switch, make `NeighborListBuilder._from_placed`
capacity-shaped, and require `graphs.total_charge` whenever a charge model is
present. Closes S5 exclusions + total charge and the S8 cutoff-continuity gate;
removes the F3 graph breaks on the FF pair path.

## Domain basis

S5: "permanent and induced interactions shall retain separately specified
topological exclusions and scaling factors … total charge shall be enforced
explicitly". S8 gate 2: "cutoff continuity". Switch
`S(x) = 1 − x³(10 − 15x + 6x²)`, `x = clamp((r − r_s)/(r_c − r_s), 0, 1)`
(OpenMM `useSwitchingFunction`; algebraically `(1 − x)³(1 + 3x + 6x²)`, the
LAMMPS `lj/mdf` taper — verify in a0-05) has `S(r_s) = 1`, `S(r_c) = 0`,
`S'(r_s) = S'(r_c) = 0`, so E and F are continuous at `r_c`. Zero-graph-break
law: `.claude/notes/notes.md` topic `compile-no-graph-breaks`.

## Design

- `src/molpot/ir/scaling.py`: frozen `ExclusionScales(s12, s13, s14)`;
  `NonbondedScaling(q: ExclusionScales = …, lj: ExclusionScales = …)` with
  `kinds() -> tuple[str, ...]`, `triple(kind) -> tuple[float, float, float]`,
  `as_dict()` nested by kind. Defaults unchanged (q 0/0/5⁄6, lj 0/0/0.5).
- New `src/molpot/forcefield/exclusions.py`: `ExclusionTable` — built from
  bond / angle / proper columns once (`Potentials.bind_topology(system)`),
  holds sorted undirected pair keys per relation; `lookup(edge_index, kind)`
  = `searchsorted` + `torch.where`, output shape = input shape; `pairs`
  (the topology-fixed excluded/scaled pair list, consumed by a0-03).
  `_pair_scales` / `_mask_pairs_in` are deleted.
- `forcefield/pair_terms.py`: `LJCutPotential(cutoff, switch_start=None)`;
  energy and analytic forces over the full-capacity list with an `edge_mask`,
  multiplied by `S(r)` and `dS/dr`; no boolean slicing.
- `forcefield/neighbor_list.py`: `_from_placed` returns capacity-shaped
  `(edge_index, dist, live_mask)`; consumers mask instead of slicing.
- `forcefield/parameterizer.py`: `forward` requires `graphs.total_charge`
  when `charge_model` is set and forwards it to
  `write_forcefield(total_charge=)`; `AtomicChargeModel`'s silent `0.0`
  default is no longer reachable from the Parameterizer.
- `forcefield/potentials.py`: `bind_topology` builds the table and checks
  `Σ_g q == total_charge_g` (eager, `ValueError` beyond 1e-6 e).

## Files

- `src/molpot/ir/scaling.py`; construction sites `src/molpot/forcefield/container.py:52-57,144,269`, `tests/test_molpot/test_forcefield/test_lj.py:77`, `test_coulomb.py:55,86`, `tests/test_molpot/test_ir/test_scaling.py`
- `src/molpot/forcefield/exclusions.py` (new), `src/molpot/forcefield/{potentials,pair_terms,neighbor_list,parameterizer}.py`
- `tests/test_molpot/test_forcefield/test_exclusions.py` (new), `test_{lj,parameterizer,potentials,neighbor_list}.py`, `tests/test_molpot/test_ir/test_scaling.py`

## Tasks

- [ ] `ExclusionScales` / keyed `NonbondedScaling`; update all construction sites
- [ ] `ExclusionTable` + `Potentials.bind_topology`; delete `_pair_scales` / `_mask_pairs_in`
- [ ] quintic switch on `LJCutPotential` (energy + analytic force), mask-based
- [ ] capacity-shaped `_from_placed`
- [ ] fail-closed `graphs.total_charge` + eager Σq check
- [ ] slow explain tests; regression script with capture header

## Testing

Unit: `test_scaling.py` (kinds, triple, as_dict, unknown kind raises);
`test_exclusions.py` (equality with the pre-step `_pair_scales` on OPLS
butane; slow `Compiler.explain(lookup)` 0 breaks on a padded list);
`test_lj.py` (`S(r_s) = 1`, `S(r_c) = 0`, `dS/dr = 0` at both ends, analytic
force vs autograd through the switch, slow explain 0 breaks);
`test_parameterizer.py` / `test_potentials.py` (missing `total_charge`
raises; Σq mismatch raises). The butane per-kind scale vectors are hard-coded literals in `test_exclusions.py`; `S(x = 0.5) = 0.5` and `|E(r_c − δ) − E(r_c + δ)| < 1e-10` in `test_pair_terms.py` (no `regressions/` tier — deleted 2026-09-07).

## Out of scope

Induced scaling values (`composable-a1-01`); periodic Coulomb exclusion
correction (a0-03); the F4 same-batch filter in `_periodic`.
