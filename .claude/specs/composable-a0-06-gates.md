---
title: S8 release gates for A0 — reflection, NVE drift hook, chain regression (composable-a0-06)
status: approved
created: 2026-09-07
chain: composable-a0
---

# composable-a0-06-gates

## Summary

Close the S8 gates that A0 still lacks: a reflection-symmetry test (with a
chiral negative control), an `EnergyDriftHook` in `molix.md.runner` that owns
the NVE drift criterion (today a manual script), and the A0 gate tests in the mirrored unit modules — every gate on an OPLS butane pair in a periodic PME box (the MD rollout `slow`-marked) plus the manifest pin; the `regressions/` tier was deleted 2026-09-07. Closing this step closes the
chain.

## Domain basis

S8 gate 1 (translation, rotation, reflection, permutation; charge
conservation; FD force and cell derivative; reference vs production), gate 2
(cutoff continuity, NVE drift over the declared domain), gate 4 (versioned
release with manifest). Drift criterion lifted from
`benchmarks/verify_md_ljcut_nve.py:36-39`: pass when
`|slope · duration| / |E_tot(0)| < 1e-3` with bounded RMS fluctuation.
Periodic impropers with phase ∈ {0, π} and OPLS torsions are achiral, so the
OPLS-AA template energy is reflection-invariant; `improper:harmonic` with
`chi0 ≠ 0` is not (the negative control proves the gate is template-scoped).

## Design

- `tests/test_molpot/test_forcefield/test_potentials.py`: energy invariant
  under `pos → −pos` and a one-axis mirror for the OPLS-AA template; explicit
  non-invariance for `improper:harmonic` with `chi0 = 0.3`.
- `src/molix/md/runner.py`: `EnergyDriftHook(MDHook)` — accumulates
  `(t, E_tot)` per chunk from `MDObservables`, reports `rel_drift =
  |slope · T| / |E0|` and `rms_rel` at `on_run_end`; `MDObservables`
  unchanged; `benchmarks/verify_md_ljcut_nve.py` consumes the hook
  (its `_EnergySampler` is deleted).
- Gate tests (no `regressions/` tier — deleted 2026-09-07): OPLS butane pair in a 20 Å PME box in
  `tests/test_molpot/test_forcefield/test_potentials.py` — T / R / P / reflection ≤ 1e-10,
  `Σq == total_charge`, contribution partition; the 200-step fp64 NVE `rel_drift < 1e-3`
  as a `slow`-marked `EnergyDriftHook` test in `tests/test_molix/test_md/test_runner.py`;
  manifest round trip with sha literals in `tests/test_molpot/test_manifest.py`; FD force +
  strain and cutoff continuity are already pinned by a0-03 / a0-02.
- Close-out: six INDEX rows move to the closed table with the chain tag and
  the manifest sha; `.claude/notes/composable-potentials.md` roadmap row
  flips to closed.

## Files

- `tests/test_molpot/test_forcefield/test_potentials.py`
- `src/molix/md/runner.py`, `src/molix/md/__init__.py` (export), `tests/test_molix/test_md/test_runner.py`
- `benchmarks/verify_md_ljcut_nve.py`
- `.claude/specs/INDEX.md`, `.claude/notes/composable-potentials.md`

## Tasks

- [ ] reflection tests (+ negative control)
- [ ] `EnergyDriftHook` + unit test on a synthetic linear series; benchmark consumes it
- [ ] chain regression with capture header
- [ ] close the chain (INDEX, note)

## Testing

Unit: `test_potentials.py::TestPotentials::test_reflection_invariance`,
`test_harmonic_improper_is_chiral`; `test_runner.py::TestEnergyDriftHook`
(slope on a synthetic linear series exact to 1e-12; `rms_rel` of a constant
series is 0); slow tier: the 200-step NVE drift test.

## Out of scope

Polarization iteration counts (`composable-a1-05` / `b2-03`); strong / weak
scaling; held-out QM accuracy (molexp).
