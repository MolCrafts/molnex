---
title: MACE-OMOL molnex pipeline integration + bit-exact (O3_e3nn) follow-ups
status: done
created: 2026-06-21
chain: mace-omol-port
---

# MACE-OMOL molnex pipeline integration (+ bit-exact follow-up)

## Summary

Follow-up chain step to `mace-omol-port-01-native-port` (which delivered the
weight-loadable, E/F-faithful native port — every block bit-exact vs official
on CPU, full-model 7e-7 eV / 4e-6 eV·Å, all of 01's code criteria verified).

This step takes the standalone `MACEOMol` (which today consumes raw tensors via
`MACEOMol.energy_forces(positions, Z, edge_index, batch, total_charge,
total_spin, ...)`) and wires it into the **molnex runtime contract**:

1. **Pipeline integration (ac-008, carried from 01)** — expose `MACEOMol` as a
   first-class molzoo encoder + molpot potential that consumes the post-collate
   `atoms / edges / graphs` `TensorDict`, derives forces via
   `molpot.derivation.ForceDerivation`, sources edges via
   `molpot.graph` / `NeighborList`, and is lazily imported from
   `molzoo/__init__`. (Also: backfill the molzoo-spec at
   `src/molzoo/specs/mace_omol.md` per CLAUDE.md's molzoo-spec workflow.)
2. **Model-level accuracy (ac-003, carried from 01 ac-007)** — RESOLVED. The
   accepted bar is E/F within **1e-4** of official OMOL (operator decision,
   2026-06-21); the cue `"O3"` group already achieves **7e-7 eV / 4.3e-6 eV·Å**
   (01 ac-006), well inside it. Measurement settles the question: O3 vs O3_e3nn
   Clebsch-Gordan differ only ~1.4e-8/op and the cue-O3 twin (into which the
   official e3nn weights are converted) already matches e3nn to 1.5e-8 — so the
   7e-7 is molnex's reimplementation accumulation, **not** a convention diff,
   and `O3_e3nn` would not reduce it (only add an `e3nn` dependency). The
   e3nn-convention pursuit is dropped and the dead `MACEOMol(group=)` hook
   removed; molnex uses cue `"O3"` everywhere.

## Domain basis

Same model as 01 — MACE equivariant message passing (Batatia et al., NeurIPS
2022, arXiv:2206.07697), OMOL variant. No new physics; this step is plumbing +
the CG-basis convention swap. The post-collate batch schema and edge convention
(`edge_index[:,0]=source/sender`, `edge_index[:,1]=target/receiver`,
`bond_diff = pos[target] − pos[source]`) are defined in CLAUDE.md "Architecture".

## Design

- **Encoder adapter**: a thin `forward(td: TensorDict) -> TensorDict` wrapper that
  reads `td["atoms","Z"]`, `td["atoms","pos"]`, `td["atoms","batch"]`,
  `td["edges","edge_index"]`, and per-graph `total_charge` / `total_spin` from
  `td["graphs", ...]`, calls the existing `MACEOMol` core, and writes node /
  energy outputs back into the batch. Reuse the existing `energy_forces` math;
  do not duplicate the forward.
- **Force derivation**: route `F = -dE/dx` through `molpot.derivation.ForceDerivation`
  rather than the encoder's internal autograd, to match the molpot potential contract.
- **Edges**: accept the collate-provided `edges` namespace; optionally build via
  `NeighborList` for standalone inference.
- **Lazy wiring**: add `MACEOMol` (+ a `MACEOMolSpec` pydantic config if the
  other molzoo encoders expose one) to `molzoo/__init__` behind a lazy import so
  importing molzoo does not hard-require the cueq/omol stack.
- **O3_e3nn**: gated entirely on the upstream cuequivariance fix; no molnex-side
  code beyond the already-present `group=` parameter until then.

## Files

- `src/molzoo/mace_omol.py` (TensorDict adapter forward + spec config)
- `src/molzoo/__init__.py` (lazy export)
- `src/molzoo/specs/mace_omol.md` (molzoo-spec backfill)
- integration tests under `tests/test_molzoo/`

## Tasks

- [x] TensorDict adapter: `MACEOMol.forward(td)` reading atoms/edges/graphs, writing outputs back
- [x] Route forces through `molpot.derivation.ForceDerivation`
- [x] Lazy `MACEOMol` export from `molzoo/__init__` (PEP 562 `__getattr__`)
- [x] Integration test: post-collate batch → energy/forces consistency vs the raw `energy_forces` path
- [x] Fix `RadialMLP` dtype contract (honor `config.ftype`) so fp64-via-config works without `.double()`
- [x] Backfill `src/molzoo/specs/mace_omol.md` per the molzoo-spec workflow (ac-002, docs) — status `partial`, mirrored to docs + zensical
- [x] Model-level E/F accuracy ≤ 1e-4 vs official (ac-003) — met at 7e-7 eV / 4.3e-6 eV·Å with default cue O3 (01 ac-006); 1e-4 is the accepted bar (operator decision)
- [x] Drop O3_e3nn pursuit + remove dead `MACEOMol(group=)` hook (CG diff only 1.4e-8/op; 7e-7 is reimplementation accumulation, O3_e3nn would not help)
- [ ] Edge sourcing via `molpot.graph` / `NeighborList` for standalone use (optional; pipeline path sources edges at collate)

## Testing

Integration test asserts the TensorDict path and the raw-tensor `energy_forces`
path agree to machine precision on a small charged molecule. The bit-exact
(ac-007) check re-runs `scripts/omol_port/verify_e2e.py` under `O3_e3nn` once
unblocked, targeting ~1e-8 eV.

## Out of scope

- Multi-head, pair_repulsion, distance transforms (Agnesi/Soft), stress/virial.
- Direct e3nn→molnex converter (removing the mace dependency).
- GPU / aarch64 + `cuequivariance-ops-cu12` performance validation.
