---
title: The a0 / a0-debt code on the Frame-native collate and the TensorFrame typed view
status: in-progress
created: 2026-09-08
chain: composable-a0-debt
---

# composable-a0-debt-07-frame-native

## Summary
Operator directive (2026-09-08): leave the peer session's uncommitted Frame-native series untouched, and make sure everything the a0 / a0-debt chains implemented sits on its two contracts — data at rest is a molpy `Frame` and reaches the model only through `collate_molecules(frames)`; the batch is read through `molix.batch.TensorFrame` (the `TypedTensorDict` view). This step converts the batch readers the chains added and rebuilds every system the chains construct from scratch as `Frame`s that go through the real collate.

## Domain basis
CLAUDE.md § TensorDict Contract: build batches as plain `TensorDict`, wrap in `TensorFrame` where a typed read is wanted, never a container with storage or behaviour; `TensorFrame.from_tensordict` is zero-copy and traces to the same dynamo graph. `.claude/notes/notes.md` § tensorframe. The peer's `collate_molecules(frames)` (uncommitted) maps `atoms` columns, `meta` scalars → `graphs.<name>`, `bonds` → COO + `type_id`, and every `atomi…atoml` block → its own namespace; it does **not** read `frame.box`, so `graphs.cell` has no Frame-native source yet (F35 — the peer's file, not edited here).

## Design
- **Typed reads, one owner.** The wrap is `molix.batch.as_tensor_frame` (in `molix/batch/frame.py`, beside the view — molpot imports it from molix, as do the MD driver, `TotalCharge` and the LAMMPS writer, so there is one spelling and no twin per package; F39). `Potentials` gets one accessor `_frame(system) -> TensorFrame` (`TensorFrame.from_tensordict` for a `TensorDict`; a `TensorFrame` passes through; a plain nested `Mapping` is wrapped as `TensorDict(mapping, batch_size=[])`), and the duck-typed `_namespace` / `_ns_tensor` readers are replaced by reads on the view (`frame.atoms["pos"]`, `frame.graphs.get("cell")` behind a `"graphs" in frame.keys()` guard — an absent optional namespace raises `AttributeError` on the view). The public entry points wrap once, eagerly, and hand the view to `_classical_core`, so the compiled trace sees the same tensors (slow trace gates must stay at their recorded break sites). `ForceField._system_tensor` / `assign_types` / `realize`, `LammpsWriter._tensor` / `to_frame`, `TotalCharge.execute` and `_TensorDictForce` (`work` reads) read through the same view; signatures keep accepting `TensorDict` (the contract's plain-batch rule) and annotate what they read as `TensorFrame`.
- **Frame-native construction.** Every system the chains build from scratch becomes molpy `Frame`s collated by `collate_molecules`: the butane-pair box (`tests/conftest.py`), the parity tool's butane and water/ion cases (`interface/tests/parity_classical.py`), the argon / LJ-cluster benchmarks, the driver / charge / container / writer fixtures added by the chains. Per-atom force-field tables (`q`, `epsilon`, `sigma`) stay on the field; a frame declares `total_charge` in `meta` when the dataset knows it (`TotalCharge()` then only normalises). Periodic systems set `graphs.cell` from `frame.box` after collate at the two builders, each line naming F35, until the collate carries it; fp64 systems re-cast after collate with `batch_to(batch, dtype=torch.float64)` (F36 — the collate narrows to float32) so the pinned goldens do not move. The two real behaviour deltas of the typed view are the `isinstance(..., TensorDict)` branch points (`calc_energy_forces_virial` refusing a view; `forward` writing nothing for a view; the mlip branches falling through) — those are the RED facts. The scripts already collate Frames (peer's port + debt-06 mounts) — not touched.
- Pre-existing readers outside the two chains (mlip paths, `Parameterizer`, `make_graph_batch`) are out of scope.

## Files
- `src/molix/batch/frame.py` (`as_tensor_frame` beside `TensorFrame`), `src/molix/batch/__init__.py` (its export; docstring repaired), `src/molpot/forcefield/{potentials,container}.py`, `src/molix/ff_export/lammps_writer.py`, `src/molix/batch/charge.py`, `src/molix/batch/task.py` (docstring), `src/molix/md/driver.py`
- `interface/tests/parity_classical.py`, `benchmarks/verify_md_ljcut_nve.py`, `benchmarks/verify_md_lj_nve.py`
- `tests/conftest.py`, `tests/test_molpot/test_forcefield/{test_potentials,test_container}.py`, `tests/test_molix/test_md/test_driver.py`, `tests/test_molix/test_batch/{test_charge,test_frame}.py`, `tests/test_molix/test_ff_export/test_lammps_writer.py`

## Tasks
- [x] RED tests: typed-view reads (a `TensorFrame` input is accepted unchanged and reads the same storage; a nested `Mapping` is wrapped); Frame-native fixtures (butane pair, water/ion) built with the peer's `frame()` helper and `collate_molecules`, `graphs.cell` from `frame.box`
- [x] `Potentials._frame` + readers; `ForceField` / writer / `TotalCharge` / driver on the view; parity tool + benchmarks build `Frame`s; slow trace gates unchanged
- [x] ledger F35 (collate drops `frame.box`) and this step's close-out

## Testing
Unit only; slow tier: the existing trace gates (`traces_clean`, `compiled_route`, `classical_core`, `graph_breaks`) at their recorded break sites with the view in the loop; parity both cases on the Frame-built systems with the recorded numbers.

## Out of scope
Changing what `collate_molecules` carries (F35 / F36 / F38 stay open rows); converting pre-existing readers the chains did not add; the three hand-built compile / FD fixtures in `test_potentials.py` (`_lj_pair_batch`, `_coulomb_batch`, `_polar_batch`) — their neighbour lists are function arguments (edge_index + shifts) so the FD strain derivative and the trace gates see a list that is a function of the positions, and a `Frame` carries no list; they stay hand-built, named here.
