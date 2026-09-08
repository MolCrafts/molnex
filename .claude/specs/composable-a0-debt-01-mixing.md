---
title: LJ mixing rule as model identity; OPLS-AA mixes geometrically (F24)
status: in-progress
created: 2026-09-08
chain: composable-a0-debt
---

# composable-a0-debt-01-mixing

## Summary
Make the Lennard-Jones combining rule an explicit, recorded part of the model: a `mixing` option on `pair:lj/cut` (`"lorentz-berthelot"` | `"geometric"`), carried by `NonbondedMethodRecord` (hence `PotentialIR` and `ModelManifest`), honoured by `LJCutPotential` and by both exporters, and declared `geometric` by `OPLSAAForceField`. Closes F24.

## Domain basis
OPLS-AA combines both ε and σ geometrically (Jorgensen, Maxwell, Tirado-Rives, JACS 118, 11225 (1996), DOI 10.1021/ja9621760); AMBER/CHARMM use Lorentz–Berthelot (σ arithmetic, ε geometric). molnex mixed every LJ pair Lorentz–Berthelot regardless of template, and nothing in the IR or manifest said so — a model that was never fully described (S2). A converged comparison against LAMMPS is only meaningful when the deck states the same rule, which `write_pair_block` now does by emitting explicit `pair_coeff i j` lines — mixed with whatever rule the model declares.

## Design
- `src/molpot/potentials/vdw/lj126.py`: `MixingRule(StrEnum)` with `LORENTZ_BERTHELOT = "lorentz-berthelot"`, `GEOMETRIC = "geometric"` and `mix(atom_params, edge_index) -> {"epsilon_ij", "sigma_ij"}`; `geometric_mixing(atom_params, edge_index)` beside `lorentz_berthelot`, ε and σ both `sqrt(p_i p_j)` with the same masked-sqrt zero handling (no additive `1e-12` — the bias `lorentz_berthelot` removed on 2026-09-07; `molpot.potentials.mixing.geometric_arithmetic_mixing` still carries it → F10). Exported from `molpot.potentials.vdw`, `molpot.potentials`, `molpot`.
- `pair:lj/cut` option `mixing` (registry `StyleSchema` options; `PairStyleDefaults.lj_mixing = "lorentz-berthelot"`); `ForceField._nonbonded_method_record` fills `NonbondedMethodRecord.lj_mixing: str | None` (all-None record keeps `None`); `Potentials._assemble_pair_potentials` passes `LJCutPotential(cutoff, switch_start, mixing=MixingRule(...))`; `LJCutPotential.energy` / `energy_forces` call `self.mixing.mix(...)`; an unknown rule string raises `ValueError` at `def_pairstyle` time (fail closed, never at evaluation).
- `OPLSAAForceField.__init__`: `def_pairstyle("lj/cut", mixing="geometric")`; docstring cites the rule and the paper.
- `ModelManifest`: carried by the existing `nonbonded` record (no schema bump needed — the record is nested, `MANIFEST_SCHEMA` stays 1; a manifest written before this change reads back with `lj_mixing=None`, meaning "Lorentz–Berthelot by the then-default", documented on the field).
- Export: `LammpsAdapter._pair` record gains `"mixing": <rule value>`; `LammpsWriter.write_pair_block` mixes the explicit `pair_coeff i j` lines with `MixingRule(record["mixing"]).mix`; `OpenMMAdapter._nonbonded` record gains `"mixing"`, and for `geometric` sets `case = TranslationCase.DECOMPOSE` with a `"mixing_note"` (OpenMM's `NonbondedForce` is Lorentz–Berthelot-only; geometric σ needs per-pair exceptions or a `CustomNonbondedForce`) — text only.
- Goldens on the OPLS template that assumed Lorentz–Berthelot are re-pinned with the geometric values (tester enumerates them).

## Files
- `src/molpot/potentials/vdw/{lj126.py,__init__.py}`, `src/molpot/potentials/__init__.py`, `src/molpot/__init__.py`, `src/molpot/forcefield/{schema,potentials,pair_terms,container}.py`, `src/molpot/ir/nonbonded.py`
- `src/molix/ff_export/{lammps_adapter,lammps_writer,openmm_adapter}.py`
- tests mirrored: `tests/test_molpot/test_potentials/test_vdw/test_lj126.py`, `test_forcefield/{test_schema,test_potentials,test_pair_terms,test_container,test_to_ir}.py`, `test_ir/test_nonbonded.py`, `test_manifest.py`, `tests/test_molix/test_ff_export/{test_lammps_adapter,test_lammps_writer,test_openmm_adapter}.py`

## Tasks
- [x] RED tests (mixing rule kernel goldens, style option + record, OPLS template declares geometric, exporter records; no existing golden changed value)
- [x] `MixingRule` + `geometric_mixing`; `LJCutPotential(mixing=)`; option + record + resolver; OPLS template (+ architect round: `StyleSchema.options` allowlist, schema 2, static dispatch map; F10 `geometric_mean` shared)
- [x] exporters carry and honour `mixing` (fail closed without one, omitted without an LJ bag); docs pages mention the rule

## Testing
Unit only. Kernel: `geometric_mixing` on ε (0.066, 0.03) σ (3.5, 2.5) → ε_ij 0.04449719…, σ_ij 2.95803989… (hard-coded); zero ε masks to 0 with a finite backward. `OPLSAAForceField().to_ir().nonbonded_method.lj_mixing == "geometric"`; a bare `ForceField` reports `"lorentz-berthelot"`. `LJCutPotential(mixing=GEOMETRIC)` energy on a two-atom pair equals the closed form with σ_ij = sqrt(σ_i σ_j). Writer golden: hybrid block with geometric mixing has `pair_coeff 1 2 lj/mdf 0.04449719 2.95804`. Manifest round trip carries the field.

Architect round (2026-09-08): the geometric arm of `LJCutPotential.energy_forces` was traced by the reviewer at 1 graph / 0 breaks — recorded here rather than as a second dynamo test (one trace per seam); unknown option keys are refused per style through `StyleSchema.options`; the export boundary fails closed on an LJ bag without a declared rule and omits `mixing` without one; `MANIFEST_SCHEMA` is 2 (schema-1 payloads read back with `lj_mixing=None`, the forward direction no longer silently drops the field); the composer path's open `mixing_fn` is F30.

## Out of scope
Per-pair (i,j) parameter tables (LAMMPS `pair_coeff i j` from a matrix); Buckingham / other vdW forms.
