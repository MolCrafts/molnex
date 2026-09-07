---
title: Versioned ModelManifest owned by ForceField (composable-a0-04)
status: approved
created: 2026-09-07
chain: composable-a0
---

# composable-a0-04-manifest

## Summary

Land the S2 manifest as a composition of small frozen records in
`src/molpot/manifest.py`, owned by `ForceField.manifest(*, profile)`, with a
`MANIFEST_SCHEMA` versioned independently of `LAMMPS_META_SCHEMA`, and carry
the nonbonded method (cutoffs, switch, Coulomb solver, smearing, mesh) into
`PotentialIR` so export no longer hard-codes it (F7).

## Domain basis

S2: "a model manifest shall identify the representation and checkpoint,
enabled interaction terms, multipole order, polarization scheme, charge
constraints, damping functions, pair exclusions and scaling, numerical
precision, and solver tolerances … reference electronic-structure method,
energy offsets, training-data version, and software versions". Design
preferences: no god struct — small coherent records; composition is the
caller's job. `ForceSpec.to_dict` (`src/molix/ff_export`) is the
`dataclasses.asdict` precedent; `molix.io.run_record.write_meta(extra=)` is
the training-side sink. Full record table:
`.claude/notes/composable-potentials.md` §"S2 manifest".

## Design

- `src/molpot/manifest.py`: `MANIFEST_SCHEMA = 1`; `RepresentationRecord`,
  `InteractionRecord`, `NonbondedMethodRecord`, `ElectrostaticsRecord`,
  `SolverRecord`, `ParamTableRecord` (sha256 of each fp64 table),
  `NumericsRecord`, `ProvenanceRecord`, `ModelManifest` with
  `to_dict / from_dict / write(path) / read(path)`; enums → `.value`,
  frozenset → sorted list.
- `ForceField.manifest(*, profile: str) -> ModelManifest` (styles from the
  schema view incl. contracts, `NonbondedMethodRecord` from `style_options`,
  `ElectrostaticsRecord` with `polarization_scheme=None` for A0,
  `ParamTableRecord`s from the torch tables, `NumericsRecord` from
  `molix.config`, `representation=None`, `provenance=None`).
- `Parameterizer.representation_record() -> RepresentationRecord`.
- `PotentialIR.nonbonded_method: NonbondedMethodRecord | None`; `to_ir()`
  fills it from `style_options`.
- No `Potentials.manifest()` (would duplicate); no façade that fills
  provenance — the caller uses `dataclasses.replace`.

## Files

- `src/molpot/manifest.py` (new); `src/molpot/__init__.py`, `src/molpot/ir/__init__.py` (alphabetized exports)
- `src/molpot/forcefield/container.py`, `src/molpot/forcefield/parameterizer.py`, `src/molpot/ir/potential_ir.py`
- `tests/test_molpot/test_manifest.py` (new), `tests/test_molpot/test_forcefield/test_{to_ir,container,parameterizer}.py`

## Tasks

- [ ] records + `ModelManifest` (round trip, schema literal)
- [ ] `ForceField.manifest`, `Parameterizer.representation_record`
- [ ] `NonbondedMethodRecord` into `PotentialIR` via `to_ir()`
- [ ] regression with golden key set + sha literals

## Testing

Unit: `test_manifest.py` — `from_dict(to_dict(m)) == m`; `read(write(m)) ==
m`; changing one table entry changes its sha; `MANIFEST_SCHEMA == 1`
literal. `test_to_ir.py` — `nonbonded_method` carries `lj_cutoff`,
`lj_switch_start`, `coulomb`, `smearing`, `mesh_spacing` from
`style_options`. The golden key set + sha256 literals of a fixed fp64 table are hard-coded in `test_manifest.py` (no `regressions/` tier — deleted 2026-09-07).

## Out of scope

Persisting the manifest next to LAMMPS decks / AOTI exports (a0-05);
representation / provenance for Family B (`composable-b1-01`);
`ElectrostaticsRecord.induced_scaling` / `solver` values (`composable-a1-05`).
