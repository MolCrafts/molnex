---
title: Interaction contract, S-decomposition labels, LJ split (composable-a0-01)
status: code-complete
created: 2026-09-07
chain: composable-a0
---

# composable-a0-01-contracts

## Summary

Give every registered style an `InteractionContract` (inputs, transformation
law, cutoff kind, exclusion class, supported derivatives, named terms with
their S-decomposition label) and make `Potentials` consume it: group
`term_energies()` into `E_local / E_disp / E_perm / E_pol / E_CT`, split LJ
into `lj_repulsion` (local) and `lj_dispersion` (disp) at the kernel, raise on
a declared style whose table is absent (F2), raise on an unregistered style
(F6), and reject unsupported combinations at construction instead of the
`Polarization` `UserWarning`. Closes S1 and S3 for Family A.

## Domain basis

Spec S1 ("each interaction module shall declare its required inputs, parameter
shapes, units, transformation laws, cutoff or long-range solver, topological
exclusions, and supported derivatives … unsupported module combinations shall
be rejected explicitly"), S3 ("an analytical Lennard–Jones baseline shall
assign its repulsive and attractive parts consistently to the local and
dispersion contributions"). `4ε[(σ/r)¹² − (σ/r)⁶]` splits exactly into
`4ε(σ/r)¹²` (local, repulsive wall) and `−4ε(σ/r)⁶` (dispersion) — at
`r = 2^{1/6}σ` these are `+ε` and `−2ε`. See
`.claude/notes/composable-potentials.md` §"S1 contract".

## Design

- New torch-free leaf `src/molpot/contract.py`: `Contribution`, `CutoffKind`,
  `ExclusionClass`, `Derivative`, `ParamFrame` (`StrEnum`s — not `Frame`, which collides with `molpy.Frame`); frozen `TermSpec(key,
  contribution)`; frozen `InteractionContract(inputs, cutoff, exclusion,
  derivatives, terms, response=None)`.
- `forcefield/schema.py`: `ParamSpec.frame: ParamFrame = SCALAR`;
  `StyleSchema.contract`; `PotentialMeta.contract_for(key)` (KeyError on
  unregistered); `PotentialMeta.check_composition(style_keys)` (ValueError:
  >1 POL; POL without at least one PERM style — the same-cutoff-family
  relation was dropped after the architect review: `pair:coulomb` carries the
  `EWALD` policy label, which made it unsatisfiable for any HARD POL style; it
  returns in `composable-a1-01` as a solver-compatibility relation; INDUCED without
  `induced_scaling`); `schema_for` no longer synthesises entries;
  `ForceFieldSchemaView.as_dict` emits `contract` and `terms`.
- Registry labels: bonded → `LOCAL / NONE / {FORCES, VIRIAL, HESSIAN}`;
  `pair:lj/cut` → `(lj_repulsion: LOCAL, lj_dispersion: DISP)`, `PERMANENT`,
  `SWITCHED`; `pair:coulomb` → `(coulomb: PERM)`, `PERMANENT`, `HARD`/`EWALD`.
- `potentials/vdw/lj126.py`: `lj126_pair_energy_split(distance, eps_ij, sig_ij)
  -> tuple[Tensor, Tensor]`; `lj126_pair_energy` stays as the sum.
- `forcefield/pair_terms.py`: `LJCutPotential.energy` / `energy_forces` return
  the `(rep, disp)` pair (forces unchanged).
- `forcefield/potentials.py`: `term_energies` emits `lj_repulsion` and
  `lj_dispersion` (drops `lj`); new
  `contribution_energies(system, *, pos=None) -> dict[Contribution, Tensor]`;
  absent table for a declared style → `ValueError` naming style and table
  (empty topology still yields zero); `__init__` calls `check_composition`
  and refuses unregistered style keys and induced modules (message names
  `composable-a1`); the `_LES_ELECTROSTATICS_PRECOMPUTED` flag and its
  `UserWarning` in `potentials/polarization.py` are deleted.
- `forcefield/parameterizer.py`: `_validate_schema_heads` reads the registry;
  `polarizability_head` / `induced_*` kwargs raise naming `composable-a1`.

## Files

- `src/molpot/contract.py` (new), `src/molpot/__init__.py` (alphabetized export)
- `src/molpot/forcefield/{schema,potentials,pair_terms,parameterizer}.py`
- `src/molpot/potentials/vdw/lj126.py`, `src/molpot/potentials/polarization.py`
- Consumers of the `"lj"` term key: none outside `potentials.py` itself (verified by grep, 2026-09-07)
- `tests/test_molpot/test_contract.py` (new), `tests/test_molpot/test_forcefield/test_{schema,lj,potentials}.py`

## Tasks

- [x] `contract.py` enums + records; unit tests (frozen, hashable, enum values)
- [x] `ParamSpec.frame`, `StyleSchema.contract`, registry labels, `contract_for`, `check_composition`, `schema_for` raises
- [x] `lj126_pair_energy_split`; `LJCutPotential` returns the pair
- [x] `term_energies` split keys; `contribution_energies`; absent-table raise; `__init__` composition check
- [x] delete LES flag + warning; `Parameterizer` head validation from the registry
- [x] update `"lj"` consumers; scenario goldens live in the mirrored unit tests (regressions tier deleted 2026-09-07) — no consumer outside `potentials.py` existed
- [x] bonded absent-table raise (F2 for `bond/angle/dihedral/improper` styles) + repair the `test_to_ir.py:165` fixture that relied on the silent zero

## Testing

Unit: `tests/test_molpot/test_contract.py`, `test_forcefield/test_schema.py`
(registry completeness, `contract_for` KeyError, `check_composition` cases),
`test_lj.py` (split identity at `2^{1/6}σ` and random r), `test_potentials.py`
(partition identity, absent-table raise, empty-topology zero, refused
combinations). The closed-form goldens (two atoms at `r = 2^{1/6}σ`: `rep = +ε`, `disp = −2ε`) live in `tests/test_molpot/test_potentials/test_vdw/test_lj126.py`; the contribution partition identity in `test_potentials.py` (no `regressions/` tier — deleted 2026-09-07).

## Out of scope

Exclusion tables and switching (a0-02); periodic Coulomb (a0-03); manifest
(a0-04); Family B contract declarations (`composable-b1-01`); the
`pair:induced/thole` entry itself (`composable-a1-01`).
