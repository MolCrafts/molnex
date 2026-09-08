---
title: Neighbour-list and total-charge batch primitives owned by the force field and molix (F12, F21)
status: in-progress
created: 2026-09-08
chain: composable-a0-debt
---

# composable-a0-debt-06-batch-primitives

## Summary
The pair list a Class-I field needs and the `graphs.total_charge` column a charge model needs are force-field semantics on molix infrastructure — not script glue. Two primitives replace the four scripts' private `LJ_CUT_CUTOFF` / `batch_tasks=[NeighborList(...)]` / `_PLACE_EDGES` / manual `total_charge` meta / `_dtype_of` copies: `ForceField.neighbor_list()` (the field declares the `molix.batch.NeighborList` it prices against) and `molix.batch.TotalCharge` (a `BatchTask` that writes the column). Closes F21 and F12 on the molnex side (operator ruling 2026-09-08: these are ours, not the peer session's).

## Domain basis
Since a0-03b molpot consumes a placed `edges` namespace and never enumerates one; the cutoff that list must cover is whatever the field's pair styles run at (`lj_cutoff`, the Ewald `real_space_cutoff`, or an open `coulomb_cutoff`) — model identity already recorded in `NonbondedMethodRecord`, so the field is the only honest owner of that number. `graphs.total_charge` is the S5 constraint the fail-closed `Parameterizer` (a0-02) requires; whether a dataset is neutral is a declaration the pipeline makes once, as a batch task, not a float pasted into every collate.

## Design
- `ForceField.neighbor_list(self, *, pbc: bool = False, max_num_pairs: int = 512) -> molix.batch.NeighborList`: the radius of the term that will run. `NonbondedMethodRecord` describes both Coulomb possibilities (an undeclared calculator resolves to the `pme` default, so `is_lattice_sum` holds for every declared `pair:coulomb`; the data decides — a0-03: PME under a cell, the open sum without), so `pbc` is the switch: radii = `lj_cutoff` if declared; with a `pair:coulomb`, under `pbc=True` its `real_space_cutoff`, under `pbc=False` its declared `coulomb_cutoff` — a bare open Coulomb needs every pair and is refused (`ValueError`; "declare cutoff=… on pair:coulomb or build the list with pbc=True"); no pair style → `ValueError`. `cutoff = max(radii)`, `symmetry=True` always (the half list reads as dead — a0-03b); `max_num_pairs` is a data property. `OPLSAAForceField().neighbor_list(pbc=True)` is 15.0; `.neighbor_list()` on the bare template is refused by design. molpot → molix is the allowed direction (`potentials.py` already imports `molix.schema`).
- `molix.batch.TotalCharge(charge: float = 0.0)` (`src/molix/batch/charge.py`, exported from `molix.batch`): `task_id = "total_charge:<charge>"`; `execute(batch)`: when `("graphs", "total_charge")` is present it is cast to the batch's float dtype (`atoms.pos`'s) — an integral meta scalar collates to int64 and the EEM solver wants the float — and left otherwise untouched; when absent, `graphs.total_charge = full((B,), charge)` in that dtype, `B` from `graphs.batch_size`. Docstring states the neutral-by-declaration semantics and that a dataset with a real charge column keeps it.
- `_dtype_of(model)` in the scripts is replaced by the project's dtype policy: batches go through `molix.core.steps.batch_to(..., dtype=<the config ftype>)` and models are built in the same dtype (`molix.config`), so no probe of `model.parameters()` is needed.
- Scripts `class_i_fit.py`, `align.py`, `espaloma_mm_protocol_train.py`, `scale_audit.py`: `batch_tasks=[ff.neighbor_list(), TotalCharge()]` on the `DataModule`s, and the same two tasks at the collate choke points that bypass it; delete `LJ_CUT_CUTOFF`, `_PLACE_EDGES`, `_dtype_of`, the manual `total_charge` meta emission. These files also carry the (departed) peer session's uncommitted Frame-native port — the cleanup is applied on top of it in the worktree; landing that series is the operator's call, recorded at close.

## Files
- `src/molpot/forcefield/container.py`, `src/molix/batch/charge.py` (new), `src/molix/batch/__init__.py` (export; peer-dirty file → temp-index hunk at commit)
- `tests/test_molpot/test_forcefield/test_container.py`, `tests/test_molix/test_batch/test_charge.py` (new)
- `scripts/mm_param_learning/{class_i_fit,align,espaloma_mm_protocol_train,scale_audit}.py`

## Tasks
- [x] RED tests: `TotalCharge` (absent → filled in the batch float dtype; present int64 → cast; present float → untouched; task_id), `ForceField.neighbor_list` (cutoff = max declared radius; symmetry True; pbc/max_num_pairs forwarded; open coulomb without cutoff and no pair style refused)
- [x] `TotalCharge` + export; `ForceField.neighbor_list` (pbc-aware; + `TypeAssignment.atom_charges` as the writer's charge source, routed from debt-04)
- [x] scripts on the two primitives; glue deleted; import-clean under PYTHONPATH=src; ruff (bare `pair:coulomb` given the explicit open cutoff 10.0 Å the scripts already ran at)

## Testing
Unit only (mirrored). Scripts: import + ruff + a dry task-chain construction.

## Out of scope
Running the training workflows; landing the peer's Frame-native port.
