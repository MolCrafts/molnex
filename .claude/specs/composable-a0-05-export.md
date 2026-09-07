---
title: Periodic analytical export + parity under interface/ (composable-a0-05)
status: approved
created: 2026-09-07
chain: composable-a0
---

# composable-a0-05-export

## Summary

Export the A0 profile completely: LAMMPS `pair_style hybrid/overlay lj/mdf …
coul/long …` + `kspace_style pppm` + `special_bonds`, OpenMM `NonbondedForce`
with PME and the switching function (text only), a pair block and a sibling
`manifest.json` from `LammpsWriter`, and `export_for_lammps(manifest=)` for
AOTI exports without touching `LAMMPS_META_SCHEMA`. Move the classical
reference-vs-production parity under `interface/tests/` (Decision 4, F5) and
add a periodic case. Closes S7 for A0.

## Domain basis

S7: "Family A should export cached parameters to an analytical MD backend …
shared physics shall mean matched conventions and test results". Decision 4
(`.claude/notes/composable-potentials.md`): no `lammps` / `openmm` imports
under `src/`, `tests/`, `scripts/`; LAMMPS-driven tests only under
`interface/`. LAMMPS `lj/mdf` taper `(1 − x)³(1 + 3x + 6x²)` is
algebraically the OpenMM quintic `1 − x³(10 − 15x + 6x²)` — **verify against
the LAMMPS documentation before emitting**; `special_bonds lj a b c coul d e f`
carries `NonbondedScaling.as_dict()`; `coul/long` + `pppm` subtract the full
excluded-pair Coulomb term (matches a0-03). The C++ meta reader validates only
`doc["lammps"]` (`interface/lammps/lammps_contract.h`), so a sibling file is
invisible to it.

## Design

- `src/molix/ff_export/lammps_adapter.py`: from `ir.nonbonded_method` emit
  `pair_style hybrid/overlay lj/mdf r_s r_c coul/long r_c` + `kspace_style
  pppm <tol>` when `coulomb ∈ {ewald, pme, p3m}`; `lj/cut` (+ `coul/long`)
  when `lj_switch_start is None`; `special_bonds` from the scaling dict;
  per-atom charges unchanged.
- `src/molix/ff_export/openmm_adapter.py`: `NonbondedForce` record gains
  `nonbondedMethod: PME`, `cutoffDistance`, `ewaldErrorTolerance`,
  `useSwitchingFunction`, `switchingDistance` — text goldens only.
- `adapter.py` / `force_spec.py`: scaling via `NonbondedScaling.as_dict()`.
- `src/molix/ff_export/lammps_writer.py`: `write_settings(..., skip_pair_style=True)`
  (molpy `write_lammps_forcefield(skip_pair_style=)`), `write_pair_block(path,
  spec: ForceSpec)`, `write_manifest(path, manifest: ModelManifest)`.
- `src/molix/engine/export.py`: `export_for_lammps(..., manifest: Mapping | None
  = None)` writes `{name}.manifest.json`; `meta.json` byte-identical.
- **Move** `scripts/mm_param_learning/parity_lammps_classical_md.py` →
  `interface/tests/parity_classical.py`; `add_test` in
  `interface/tests/CMakeLists.txt` (skipped when `lmp` is absent; fixtures via
  `PYTHONPATH` to `src/` as `generate_fixtures.py` does); `--periodic` case
  (ion / water box, `boundary p p p`, `pppm 1e-6`); its script-local pair
  block replaced by `write_pair_block`. Delete the `scripts/` copy.
- `tests/test_repository_boundaries.py`: one assertion — no `import lammps`
  / `import openmm` (or `from …`) under `src/`, `tests/`, `scripts/`.

## Files

- `src/molix/ff_export/{lammps_adapter,openmm_adapter,adapter,force_spec,lammps_writer}.py`, `src/molix/engine/export.py`
- `interface/tests/parity_classical.py` (moved), `interface/tests/CMakeLists.txt`, `interface/lammps/README.md` (runbook row)
- `scripts/mm_param_learning/parity_lammps_classical_md.py` (deleted)
- `tests/test_molix/test_ff_export/test_{lammps_adapter,openmm_adapter,lammps_writer}.py`, `tests/test_molix/test_engine/test_export.py`, `tests/test_repository_boundaries.py`

## Tasks

- [ ] verify `lj/mdf` taper form in LAMMPS docs; adapter emission + goldens
- [ ] OpenMM text records; scaling dict
- [ ] `write_pair_block`, `write_manifest`, `export_for_lammps(manifest=)`
- [ ] move parity to `interface/tests/`, ctest entry, `--periodic` case, README row
- [ ] repository-boundary assertion; regression with capture header

## Testing

Unit (text only): adapter goldens for periodic / open / switched / hard
cases; writer output goldens; `export_for_lammps` writes the sibling and
leaves `meta.json` byte-identical; boundary assertion. `ForceSpec` text goldens are hard-coded in the adapter unit tests (no `regressions/` tier — deleted 2026-09-07).
Developer-run under `interface/tests/` (needs `lmp`): `parity_classical.py
--case butane` (captured literals) and `--periodic` single point
`|ΔE|/|E| < 1e-6`, max|ΔF| < 1e-5 kcal/mol/Å vs LAMMPS with matched
exclusions and switch. `LAMMPS_META_SCHEMA == 3` stays pinned by `tests/test_molix/test_engine/test_export.py` and `interface/tests`.

## Out of scope

The neural parity script `scripts/lammps_iface_parity.py` (named debt for
`lammps-iface-post-chain`); cell / charge inputs for AOTI exports
(`composable-iface-01-cell`); any `openmm` execution.
