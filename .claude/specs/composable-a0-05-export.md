---
title: Periodic analytical export + parity under interface/ (composable-a0-05)
status: done
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

Pre-flight amendments (2026-09-08, after a0-04 landed and the operator rule
"no static source-scanning tests" of 2026-09-07 — `.claude/notes/notes.md`):
`LammpsWriter.write_manifest` is dropped (`ModelManifest.write` exists; a
pass-through would be a façade); the boundary *test* is replaced by the lint
gate (ruff `TID251` banned-api); `lj/mdf` taper verified against the LAMMPS
docs — `(1−x)³(1+3x+6x²) = 1 − 10x³ + 15x⁴ − 6x⁵`, identical to molnex's
quintic; a local `lmp` (`/home/jicli594/.local/x86_64/bin/lmp`, KSPACE +
EXTRA-PAIR + MOLECULE) runs the periodic parity for real.
Architect round (2026-09-08): `export_for_lammps(manifest=)` is pinned in the
fast tier with the AOTI compile stubbed at the `Exporter.export_pretraced` seam
(the eager schema-v3 checks still run; a real two-export test needs CUDA, which
this box lacks); the exporter emits `kspace_modify gewald 1/(σ√2)` because
`smearing` is model identity; identity labels are `identity_type_label(i, n_atoms)`
(single derivation site, zero-padded); `NonbondedMethodRecord.is_lattice_sum`
replaces the adapters' own calculator vocabulary; `PotentialIR.nonbonded_method`
defaults to the all-None record.

- `src/molix/ff_export/lammps_adapter.py` — `LammpsAdapter(kspace_accuracy: float = 1e-6)`.
  From `ir.nonbonded_method` (`NonbondedMethodRecord`) and the presence of
  `ir.lj` / `ir.charges` it emits, all numbers formatted `{:.7g}`:
  * LJ only: `pair_style lj/cut <lj_cutoff>`; with `lj_switch_start`:
    `pair_style lj/mdf <lj_switch_start> <lj_cutoff>`.
  * LJ (no switch) + charges under a lattice solver
    (`coulomb_calculator ∈ {ewald, pme, p3m}`): `pair_style lj/cut/coul/long
    <lj_cutoff> <real_space_cutoff>` + a `kspace` record `kspace_style pppm
    <kspace_accuracy>`; open boundary with `coulomb_cutoff`: `pair_style
    lj/cut/coul/cut <lj_cutoff> <coulomb_cutoff>` and no `kspace` record.
  * LJ switched + charges: `pair_style hybrid/overlay lj/mdf <r_s> <r_c>
    coul/long <real_space_cutoff>` (or `coul/cut <coulomb_cutoff>`), plus the
    `kspace` record under a lattice solver.
  * charges only: `coul/long …` / `coul/cut …` likewise.
  * charges with neither a calculator nor a cutoff, or an LJ bag with
    `lj_cutoff None` → `UnsupportedTermError` (no LAMMPS style is that).
  The pair record keeps `category: "pair"`, `parameters` (per-type
  `epsilon` / `sigma`, unchanged) and gains `sub_styles: [{"style", "args"}]`
  so the writer never re-parses the `type` string. `special_bonds` record
  (`category: "special_bonds"`, `type: "special_bonds lj a b c coul d e f"`)
  is emitted iff `ir.scaling is not None`, from `NonbondedScaling.triple`.
- `src/molix/ff_export/openmm_adapter.py` — `OpenMMAdapter(conventions=None,
  ewald_error_tolerance: float = 1e-6)`; the `NonbondedForce` record gains
  `nonbondedMethod` (`"PME"` under a lattice solver, `"CutoffNonPeriodic"`
  with `coulomb_cutoff` only, `"NoCutoff"` otherwise), `cutoffDistance` (nm,
  from `lj_cutoff`; OpenMM ties LJ and real-space Coulomb to one cutoff and the
  converged Ewald sum is split-independent, so this is exact, not a
  reparameterisation), `ewaldErrorTolerance`, `useSwitchingFunction`,
  `switchingDistance` (nm or `None`). Text only — no `openmm` import.
- `adapter.py` — `scaling_dict` iterates `scaling.kinds()` (keys
  `scale_<kind>_<12|13|14>`), so a kind a later model declares cannot be
  silently dropped.
- `src/molix/ff_export/lammps_writer.py` —
  `write_settings(..., skip_pair_style: bool = False)`: when true, every
  `pair:*` style is removed from the realized clone (`ForceField.remove_style`)
  and molpy writes a bonded-only include (no `pair_style`, no `pair_coeff`, no
  `special_bonds`). `write_pair_block(path, spec: ForceSpec) -> Path` renders
  the pair section from the `pair` / `kspace` / `special_bonds` records:
  `pair_style` line, one explicit `pair_coeff i j [<sub-style>] eps_ij sigma_ij`
  line for every `i <= j` over the LJ types, mixed with molnex's own
  `lj126.lorentz_berthelot` (parity by construction; LAMMPS never mixes an LJ
  sub-style onto an `i != j` pair under `hybrid/overlay` — measured
  `E_vdwl = 0` — and its default mixing is geometric in σ, not molnex's
  Lorentz–Berthelot; F24 / F25), `pair_coeff * * <coulomb sub-style>` under
  `hybrid/overlay`, then `kspace_style` and `special_bonds`. `write_settings`
  always passes `skip_units=True` (the include is sourced after `read_data`).
  Identity type labels get a letter prefix (`ForceField.identity_type_label(i)
  == "t{i+1}"`, bonded labels are molpy's join `t1-t2`) because LAMMPS rejects
  type labels that start with a digit; `to_frame` writes `mol_id` so molrs
  emits `Atoms # full`; the box comes from `graphs.cell` when present.
  `ForceField.nonbonded_scaling` is a property whose setter mirrors the policy
  into molpy (`set_special_bonds(lj=triple("lj"), coul=triple("q"))`) so the
  plain include and every molpy consumer carry the same 1-2/1-3/1-4 scales —
  before, molrs wrote `special_bonds … 1 … 1` regardless of the field's policy. Under `hybrid/overlay` the sub-style
  name in `pair_coeff` is mandatory, and molrs's own hybrid path writes plain
  `hybrid` (which would drop Coulomb on every LJ pair) — that is why molnex
  owns this block.
- `src/molix/engine/export.py` — `export_for_lammps(..., manifest: Mapping |
  None = None)` writes `{name}.manifest.json` (`json.dump`, `indent=2`,
  `sort_keys=True`); `meta.json` byte-identical. `molix.engine` takes the dict
  (`ModelManifest.to_dict()`), never imports molpot.
- **Move** `scripts/mm_param_learning/parity_lammps_classical_md.py` →
  `interface/tests/parity_classical.py` (`--case butane|periodic`, `--lmp`,
  exits 77 when no `lmp`): the script-local pair block is replaced by
  `write_settings(skip_pair_style=True)` + `write_pair_block`; `--case
  periodic` builds a small water + ion box (`boundary p p p`), molnex side
  `lj/cut` with a switch + `coulomb` PME under `graphs.cell`, edges placed by
  `molix.batch.NeighborList(symmetry=True)`, LAMMPS side `hybrid/overlay
  lj/mdf … coul/long …` + `kspace_style pppm`; both sides converged (tight
  molnex mesh / smearing checked against the Ewald calculator in-script, LAMMPS
  `pppm` accuracy tightened as needed) so the gate measures convention parity,
  not mesh error. Captured literals live in a `Goldens` block with provenance,
  as in the neural tool. `add_test` entries `parity_classical_butane` /
  `parity_classical_periodic` in `interface/tests/CMakeLists.txt` with
  `SKIP_RETURN_CODE 77`; runbook row in `interface/lammps/README.md`. Delete
  the `scripts/` copy.
- `pyproject.toml` — `[tool.ruff.lint] select` gains `TID251`;
  `[tool.ruff.lint.flake8-tidy-imports.banned-api]` bans `lammps` and `openmm`
  with the rule text. Enforced by `ci-lint` on every commit and by
  `build.check` over `scripts/`. No test asserts on source text.

## Files

- `src/molix/ff_export/{lammps_adapter,openmm_adapter,adapter,lammps_writer}.py`, `src/molix/engine/export.py`, `pyproject.toml` (ruff hunk only)
- `interface/tests/parity_classical.py` (moved), `interface/tests/CMakeLists.txt`, `interface/lammps/README.md` (runbook row)
- `scripts/mm_param_learning/parity_lammps_classical_md.py` (deleted)
- `tests/test_molix/test_ff_export/test_{lammps_adapter,openmm_adapter,adapter,lammps_writer}.py`, `tests/test_molix/test_engine/test_export.py` (new, `slow`)

## Tasks

- [x] verify `lj/mdf` taper form in LAMMPS docs (identical to the quintic; see Design)
- [x] RED unit tests: adapter goldens (periodic / open / switched / hard / unsupported), OpenMM records, `scaling_dict` kinds, writer include + pair block, `export_for_lammps(manifest=)` (slow)
- [x] LAMMPS + OpenMM adapter emission; `scaling_dict` over `kinds()`
- [x] `write_settings(skip_pair_style=True)`, `write_pair_block`, `export_for_lammps(manifest=)`
- [x] move parity to `interface/tests/parity_classical.py`, ctest entries, `--case periodic`, README row; run both cases with the local `lmp` and capture the literals (butane |ΔE|/|E| 2.6e-12; periodic 9.9e-9, max|ΔF| 6.9e-6)
- [x] ruff `TID251` banned-api hunk in `pyproject.toml` (applied 2026-09-08; tree clean under the ban, probe fires 5/5)
- [x] hygiene (2026-09-08): architect round (label owner `identity_type_label(i, n_atoms)`, two-way `special_bonds` sync, `is_lattice_sum`, all-None `nonbonded_method` default, charges-only `pair_coeff * *`, exported `kspace_modify gewald`, public `molpot.lorentz_berthelot` import) + janitor round (`_pair` split, `LammpsAdapter.number` as the one `.7g` owner, shared `PME_SWITCHED_METHOD` conftest, parity-tool `noqa` / `_finalizer` cleanups) applied; janitor manual items recorded in F10 / F25

## Testing

Unit (text only, fast tier): adapter goldens for periodic / open / switched /
hard / unsupported cases; OpenMM `NonbondedForce` fields; `scaling_dict` keys
from `kinds()`; writer include without pair lines and the pair-block text
golden. `slow`: `export_for_lammps(manifest=)` writes the sibling and leaves
`meta.json` byte-identical (two AOTI exports of the conftest stub). No
`regressions/` tier (deleted 2026-09-07); no source-scanning test (rule of
2026-09-07) — the import ban is ruff `TID251`.
Developer-run under `interface/tests/` (needs `lmp`; the local x86 build
qualifies): `parity_classical.py --case butane` (captured literals) and
`--case periodic` single point `|ΔE|/|E| < 1e-6`, max|ΔF| < 1e-5 kcal/mol/Å
vs LAMMPS with matched exclusions, switch and a converged lattice sum.
`LAMMPS_META_SCHEMA == 3` stays pinned by `interface/tests`.

## Out of scope

The neural parity script `scripts/lammps_iface_parity.py` (named debt for
`lammps-iface-post-chain`); cell / charge inputs for AOTI exports
(`composable-iface-01-cell`); any `openmm` execution.
