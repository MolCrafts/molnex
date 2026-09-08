---
title: mm_param_learning scripts and the test-tier sentence (F23, CLAUDE.md)
status: approved
created: 2026-09-08
chain: composable-a0-debt
---

# composable-a0-debt-05-scripts-docs

## Summary
Retire the stale `_probe_nan_term.py`, bring the three `scripts/mm_param_learning` training scripts back to an importable, batch-task-complete state (F12 / F21 / F23), and reconcile CLAUDE.md's "no MD rollouts under tests/" sentence with the two-tier rule.

## Domain basis
F21: the scripts build `OPLSAAForceField` potentials on `DataModule` batches that carry no `NeighborList` task — since a0-03b molpot never enumerates pairs, so they raise on the first nonbonded evaluation. F12: `espaloma_mm_protocol_train.py` never emits `graphs.total_charge`, which the fail-closed `Parameterizer` (a0-02) requires. F23: the peer session's Frame-native collate deleted `TargetSchema` / `FrameSample`; the port depends on that API landing (peer's call — see Tasks).

## Design
- Delete `scripts/mm_param_learning/_probe_nan_term.py` (a debug probe calling private methods with a stale arity).
- The three training scripts (and `scale_audit.py`) were ported to the Frame-native collate by the peer session `molnex-5b` in its own uncommitted series (2026-09-08) — F23 is closed by that series, and the F12 / F21 additions (a `molix.batch.NeighborList(cutoff, symmetry=True)` task in every chain feeding a pair style; a `graphs.total_charge` column where a charge model is built) were requested from the peer because it holds those files; this spec records the outcome in the ledger and does not edit them.
- `CLAUDE.md` § Test layout rules: the "No e2e runs (training loops, MD rollouts, …)" sentence becomes the fast-tier rule, with the slow-tier exception named (bounded rollouts / compile / export checks are `slow`-marked, one gate each) — staged hunk-level through a temporary index because the peer holds other CLAUDE.md hunks.

## Files
- `scripts/mm_param_learning/_probe_nan_term.py` (deleted), `CLAUDE.md`, `.claude/notes/composable-potentials.md`

## Tasks
- [x] delete `_probe_nan_term.py` (+ its orphaned `run_class_i_nan_probe_gh200.sbatch`)
- [x] ledger: F23 closed by the peer's series; F12 / F21 requested from the peer (recorded)
- [ ] CLAUDE.md sentence via temp-index hunk; ledger rows F12 / F21 / F23 updated

## Testing
No unit tests (scripts are molexp-side); verification is import + ruff + a dry construction of the batch task chain on a synthetic Frame where the API allows.

## Out of scope
Running the training workflows (datasets on molexp).
