---
title: Condensed LAMMPS types by parameter equality (F25)
status: in-progress
created: 2026-09-08
chain: composable-a0-debt
---

# composable-a0-debt-04-type-dedup

## Summary
Replace identity typing (one LAMMPS type per atom / per interaction) by types condensed on parameter equality, owned by one `ForceField.assign_types` primitive that both `realize` and `LammpsWriter` consume. The exported pair block becomes `T(T+1)/2` lines for `T` distinct atom types instead of `N(N+1)/2`. Closes F25.

## Domain basis
LAMMPS types are equivalence classes of parameters: two atoms with the same (mass, q, ε, σ) are one atom type, two bonds with the same (k, r0) between the same atom types are one bond type. Identity typing was the a0 shortcut; with explicit `pair_coeff i j` lines (a0-05) it costs half a million lines at 1000 atoms and a labelmap of 1000 entries.

## Design
- `ForceField.assign_types(system, *, masses=None) -> TypeAssignment` (frozen): per-atom `atom_type: list[int]` (0-based), `atom_labels: list[str]` (`t01`… zero-padded to the type count), and per-interaction `bond_type` / `angle_type` / `proper_type` / `improper_type` index lists with their labels; keys are the exact fp64 parameter tuples (plus mass and charge for atoms; plus the ordered atom-type tuple for bonded types, so a bond between the same types with different parameters is a distinct type). Labels are assigned in first-occurrence order and zero-padded to a common width (the molrs text-sort rule from a0-05).
- `realize` uses the assignment (`def_type` per class, not per atom/interaction). If molpy's `def_type` derives the bonded label from the atom labels alone, bonded types that share atom types but differ in parameters need an explicit name — the implementer probes molpy and either passes a name or reports the API gap; a silent collision is not acceptable.
- `LammpsWriter.to_frame/write_data/write_settings/write_pair_block(path, spec, types)` take the assignment; `write_pair_block` collapses the per-atom `parameters` rows by atom type, asserting equal (ε, σ) within a type, and emits one line per unordered type pair. `identity_type_label` / `_identity_type_width` are deleted. Parity tool and both docs pages use the new call shape; the periodic parity case (8 water + 2 ions → 4 atom types) must reproduce the recorded numbers.

## Files
- `src/molpot/forcefield/container.py`, `src/molix/ff_export/lammps_writer.py`, `interface/tests/parity_classical.py`, `interface/lammps/README.md`, `docs/molix/user-guide/ff-export.md`, `docs/molpot/tutorials/class-i-forcefield.md`
- `tests/test_molpot/test_forcefield/{test_container,test_to_ir}.py`, `tests/test_molix/test_ff_export/test_lammps_writer.py`

## Tasks
- [x] RED tests (assign_types dedup + labels; realize names; writer goldens; pair block line count)
- [x] `assign_types` + `realize`; writer on the assignment; parity re-run (identical numbers, deck 545 -> 116 lines); docs

## Testing
Unit: 8 waters + Na + Cl → atom types O, H, Na, Cl (4) with labels `t1..t4`; the water O–H bonds → one bond type, H–O–H → one angle type; a 4-atom system with two identical atoms → 3 atom types and 6 pair lines; writer goldens updated. Developer-run: `parity_classical.py --case periodic` PASS with the condensed deck, numbers unchanged (GOLDENS untouched).

## Out of scope
Type assignment by chemistry (SMARTS / atom typing) — that is the Parameterizer's world; LAMMPS `pair_modify mix` shortcuts.
