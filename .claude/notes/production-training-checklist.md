# Production training preflight checklist

This checklist is binding for every production training submission. Complete
it against one versioned molexp calibration experiment/run and attach every
result to that run. An unchecked item, missing evidence, `N/A`, waiver, stale
result, or run mismatch means **NO-GO**. Before sign-off, only explicitly
bounded calibration/validation jobs may run.

## 0. Submission identity

- [x] The run is still classified as calibration/validation; no production
  job has been submitted early.
- [x] Run identity is owned by molexp (`Workspace → Project → Experiment →
  Run`); launchers record a molexp run id, not a git/config hash.
- [x] Every artifact below is written under the run's record dir (or a
  timestamped `runs/<slug>-<utc>/` when launched outside a declared molexp
  run).
- [ ] The intended production configuration is byte-for-byte/config-for-config
  identical to the calibrated configuration except for explicitly enumerated
  production duration and output location.

Launcher: `scripts/mm_param_learning/quasi_prod.py` writes
`runs/quasi-prod-<utc>/`. Direct ClassIFit — not molexp declared-run
replay. Multi-epoch production is still **NO-GO**.

## 1. Dataset and sampling contract

- [x] Train and val pin `dataset:molcrafts/omol25-small@1` (OMol25
  val-neutral, ωB97M-V/def2-TZVPD, kcal/mol after OMol25Small).
  Collection `dataset:molcrafts/class1-universal-2026@1`. v0 val is a
  molecule-level 8:2 self-split of that source (`load_omol25_split`,
  seed 2666). PhAlkEthOH is deferred.
- [x] Molecule-level 8:2 split (`split_records`, seed 2666) before packing.
- [x] All frames of one `molecule_id` stay on one side; overlap raises.
- [x] Loader is OMol25-small only (no undeclared concat).
- [ ] Task sampling audit vs a multi-task mixture — v0 is a single energy
  task, so this item is not applicable until a second task is live.
- [x] Coverage (n_mols, n_atoms min/max, Z set, size buckets 1–8 / 9–16 /
  17–32) is written to `split.json` by `write_split_manifest`.
- [x] Size buckets are the same triple as `align.py` §2.

## 2. Network correctness and numerical parity

- [x] Declared numerical tolerances exist before comparison begins.
- [x] Reference single-molecule and optimized packed-batch paths agree for each
  size bucket on model outputs, energies/forces/parameters, and every active
  per-task loss.
- [x] Parameter-group gradients agree between reference and optimized paths;
  every intended trainable group receives a finite, nonzero gradient.
- [x] One complete optimizer update agrees between the two paths within the
  declared tolerance.
- [x] Permutation, rigid translation/rotation, grouping, charge conservation,
  and other applicable physical invariants pass.
- [x] NaN/Inf checks, outlier reports, gradient norms, clipping incidence, and
  initial prediction/target scales have been inspected and archived.
- [x] Packed batches neither mix graph identities nor leak values across
  molecules/tasks.

Evidence (2026-08-18, GH200 Slurm 1379541 / molq `7117fc40`, molexp run
`1cc36adb`, TF32 off, hidden=64, n_blocks=4, 4 mols/bucket):
small 7–8 / medium 12–16 / large 17–27 all `ok=1`. Packed E/F/loss, leak,
charge, trans/rot/perm, Adam update within declared tols in
`scripts/mm_param_learning/align.py`. TF32-on job `aaff50eb` broke rotation
force and the first Adam step (sign flip at `2·lr`); the alignment gate keeps
TF32 off. Scale audit (Slurm 1383572 / molq `25ed74c4`, molexp `2d16de83`):
OMol25 frames lacked `molecule_id` and collapsed into `ungrouped`, so DFT
totals (~1e6 kcal/mol) entered one energy group and drove `grad_norm` to
1e8. After writing `molecule_id`/`source_record_id` and refusing unlabeled
frames: 16/16 singletons, energy term 0, `grad_norm` 386, pred |F| med 1.2
vs ref 4.9, 16 force outliers >200 kcal/mol/Å, `GradClipHook(1.0)` would
fire every step. Inspected and archived; still not a production GO.

Calibration gates on official 4m/val-neutral coordinates (GH200, molq
`b5b516c9`, run `runs/calibrate-e46dec29b8b6-20260823T150652Z`):
`align_ok=0` — small and medium buckets pass packed-vs-single parity, but
large (18–22 atoms) fails `grad_ok` (gradient residual 1.02 vs tolerance
`1e-5 + 1e-3·scale`); `inspect_ok=0` — dressed energy term 7.16e11 exceeds
`ENERGY_TERM_WARN` (`energy_scale_ok=0`, pred median |E| 352819 vs ref
217501), force scale OK, pre-clip `grad_norm` 3.52e14 so `GradClipHook(1e3)`
fires every step, 6 force outliers. §2 parity does NOT hold on production
coordinates yet.

## 3. Learnability and hyperparameter calibration

- [x] Every active task/head independently overfits a fixed tiny set to its
  predeclared threshold. Overfit gate on official 4m tiny set (8 train / 2
  val, ≤32 atoms, 4000 steps, GH200 n517, molq `68f2064b`): energy MSE
  5.03e12 → 4.91e4, force MSE 6.70e13 → 1.58e9, both < 0.1× step-0,
  `ok=1.0` (`runs/overfit-small-5b68570efd07-20260823T152634Z/overfit.json`).
  Learnability is confirmed; it does NOT clear the §2 energy-scale mismatch
  on official coordinates.
- [x] Single-task force+energy mixture on the 256-molecule mini (Slurm
  1385961) and the 5880-molecule epoch1 (Slurm 1407684) both moved
  loss; clip raised from 1.0 to 1e3 after the scale audit showed
  `grad_norm ~ 400–2e5`.
- [x] Recorded operating point: Adam `lr=1e-4`, `clip_max_norm=1e3`, no
  scheduler, `hidden=64`, `n_blocks=4`, `n_torsion_terms=8` (all powers
  of two, live in `QUASI_PROD` not inside ChemEncoder).
- [x] Three power-of-two capacities: small 32/2/4, medium 64/4/8,
  large 128/8/16 (`PROD_MODELS`). One-day wall-clock campaign
  authorized 2026-08-19.

## 4. Packed batching and target-hardware optimization

- [x] Packed train path is `DataMixture` + `collate_molecules` (one
  forward per packed batch). No per-molecule Python loop in the step.
- [x] Full input/kernel/optimizer profile recorded
  (`runs/profile-188d38b99288-20260823T145111Z/profile.json`): full-step
  median 55.0 ms, profiler self-CUDA 729 679 µs total / 19 298 µs optimizer,
  throughput 2410→4670 samples/s (num_workers 0/2/4).
- [x] Per-bucket batch sizes not split (decision: keep one pack budget).
  OMol25-small is capped at 32 atoms; buckets are contiguous ranges inside
  that cap and the pack is a per-step atom budget, not a per-bucket split.
  Splitting would add loader/collator branches with no HBM or throughput
  benefit at this scale (16384-atom packs use 0.59 GiB).
- [x] HBM sweep GH200 Slurm 1417874 / molq `a03ff86c`, run
  `runs/quasi-prod-d8247ba3046e-20260819T101139Z/pack_sweep.json`:
  4096→0.19 GiB (181 mols), 8192→0.33, 16384→0.59, 32768→1.10,
  65536→2.15, 131072→4.19 GiB (5793 mols). No OOM. 120 GiB card
  still has >90 % free at the largest pack.
- [x] Operating pack `packed_max_atoms=16384` (power of two). Declared
  HBM margin is 16 GiB; the previous sweep at 16384 was 0.59 GiB
  (~500 mols). 131072 (4.19 GiB) still fits but collapses the
  OMol25-small epoch to 1–2 steps. Timed sweep recorded in
  `runs/calibrate-e46dec29b8b6-20260823T150652Z/pack_sweep.json`
  (48 mols / 758 atoms; 4096→131072 = 43.4→40.1 ms/step, no OOM).

## 5. Bounded end-to-end smoke and recovery

- [x] Quasi-prod runner uses the same ClassIFit / DataMixture / Adam /
  `val_omol_mae` path as epoch1. Unique `runs/quasi-prod-*` output.
- [x] Epoch1 (Slurm 1407684) moved train loss 4776 → 3006; val force MAE
  24.08. Single energy task.
- [x] MolRec metrics WAL wired (`MetricsWriter` → `MolRecMetricsHook`):
  metrics.jsonl + status.json; integration-tested.
- [x] `ClassIFit.run(resume_from=, max_steps=)` + `run_resume_smoke`
  (1 step → last.pt → resume → step 2). GH200 evidence from the
  quasi-prod job.
- [x] Preemption / partial-artifact restart: atomic checkpoint write
  (tmp + os.replace, `TorchSaveBackend`) with unit coverage
  (`test_atomic_on_failed_write_keeps_old_file`); resume from `last.pt`
  via `run_resume_smoke`.
- [x] Calibration job `succeeded` (molq `b5b516c9`); its align/inspect
  gates report FAIL on official coordinates (§2 evidence above).

## 6. Final production submission gate

- [ ] No molexp production experiment. Quasi-prod is not production.
- [x] Pins live in `QUASI_PROD`: account `naiss2026-1-53-gpu`, partition
  `gpu`, hardware GH200, coords as §1.
- [x] Output is `runs/quasi-prod-<utc>/` (gitignored),
  not repo-root `checkpoints/`.
- [x] Launcher fail-closed: non-power-of-two sizes, train/val id overlap.
  Direct `python scripts/mm_param_learning/quasi_prod.py`.
- [ ] **decision: NO-GO** for unbounded multi-week production.
  Operator authorized a **1-day / 3-size** campaign on OMol25-small
  8:2. Align uses atol+rtol on dressed totals. Pack 16384. Stop is
  Slurm wall 24h.

```yaml
decision: NO-GO
scope: "prod-day1 three sizes; not unbounded GO"
molexp_experiment: "universal-class-i-v0-calibration-gh200"
molexp_run: "class-i-v0-001"
molhub_collection: "dataset:molcrafts/class1-universal-2026@1"
molexp_calibration_experiment: "universal-class-i-v0-calibration-gh200"
molexp_calibration_run: "class-i-v0-001"
target_hardware: "GH200"
slurm_account: "naiss2026-1-53-gpu"
reviewer: "operator"
reviewed_at: "2026-08-23"
evidence_index: "runs/calibrate-e46dec29b8b6-20260823T150652Z; runs/overfit-small-5b68570efd07-20260823T152634Z; runs/profile-188d38b99288-20260823T145111Z"
```

## Required sign-off record

```yaml
decision: NO-GO
molexp_calibration_experiment: "universal-class-i-v0-calibration-gh200"
molexp_calibration_run: "class-i-v0-001"
molhub_collection: "dataset:molcrafts/class1-universal-2026@1"
target_hardware: "GH200"
slurm_account: "naiss2026-1-53-gpu"
reviewer: "operator"
reviewed_at: "2026-08-23"
evidence_index: "runs/calibrate-e46dec29b8b6-20260823T150652Z; runs/overfit-small-5b68570efd07-20260823T152634Z; runs/profile-188d38b99288-20260823T145111Z"
reason: "calibrate align_ok=0 (large-bucket grad parity) and inspect_ok=0 (energy_scale_ok=0, grad_norm 3.52e14, clip fires every step). Overfit learnability passes; profile passes."
```

The reviewer must change `decision` back to `NO-GO` whenever the code,
environment, data, model, batching, optimizer, precision, or hardware
changes. Recalibration is required before another `GO`.
