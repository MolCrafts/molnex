---
title: Fail closed on wrapped molecules at bind (F29)
status: in-progress
created: 2026-09-08
chain: composable-a0-debt
---

# composable-a0-debt-02-unwrapped

## Summary
`Potentials.bind_topology` refuses a periodic system whose bonded molecules are wrapped across a box face, instead of silently pricing the wrong bonded energy. Closes the decision left open in F29 as "reject at bind".

## Domain basis
Bonded terms take plain `pos_j − pos_i` off their index columns while pair terms take minimum-image separations (`_placed_pairs`). If every covalent bond is shorter than half the shortest box width, the molecule is contiguous in the coordinates as given and every bonded displacement (angle, torsion, improper) is right; if any bond is longer, the molecule has been folded across a face and its bonded energy is wrong with no symptom. MD never wraps positions in place, so a bind-time check protects every rollout; a training batch is checked at its bind (the fingerprint is shape-level), so a data pipeline that wraps frames remains the producer's responsibility — the docstring says so.

## Design
- `Potentials.bind_topology`: when `graphs.cell` is present, for every bond compute `|pos[j] − pos[i]|` and compare against half the minimum perpendicular width of that graph's cell (`V / max_k |a_j × a_k|` — for a cube `L/2`); any bond over the limit → `ValueError` naming the bond, its length, the limit and the remedy ("unwrap molecules before evaluation; molpot prices bonded terms on plain displacements — F29"). One host sync at bind, like `_check_pair_orientation`. Open boxes and bond-less systems are unaffected.
- Public entry-point docstrings (`calc_energy`, `calc_energy_forces`, `calc_energy_forces_virial`) point at the check; the F29 ledger row flips to "decided: reject at bind".

## Files
- `src/molpot/forcefield/potentials.py`; `tests/test_molpot/test_forcefield/test_potentials.py`; `.claude/notes/composable-potentials.md`

## Tasks
- [x] RED tests (wrapped bond in a periodic box raises at bind naming the bond; unwrapped passes; open box unaffected; general cell uses the perpendicular width)
- [x] the check in `bind_topology` (`_check_bonds_unwrapped`); docstrings; ledger row F29 decided

## Testing
Unit: a two-atom bonded system in a 10 Å cube at x = 1 and x = 9 (bond 8 Å > 5 Å) raises `ValueError` mentioning "F29"; the same molecule unwrapped (x = 9 and x = 11) binds; the open-box fixture binds; a triclinic cell whose perpendicular width is smaller than its edge length uses the width.

## Out of scope
Minimum-imaging the bonded terms (would change the force/virial path); wrapping frames for the caller.
