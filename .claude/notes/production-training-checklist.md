# Production training preflight checklist

This checklist is binding for every production training submission. Complete
it against one exact configuration fingerprint and attach every result to a
versioned molexp calibration experiment. An unchecked item, missing evidence,
`N/A`, waiver, stale result, or fingerprint mismatch means **NO-GO**. Before
sign-off, only explicitly bounded calibration/validation jobs may run.

## 0. Submission identity

- [x] The run is still classified as calibration/validation; no production
  job has been submitted early.
- [x] One immutable fingerprint covers code revision and dirty-diff hash,
  MolHub collection and source versions, split manifests,
  model dimensions, task heads/losses, optimizer,
  collator, seeds, and target hardware
  (`scripts/mm_param_learning/fingerprint.py`, `QUASI_PROD`).
- [ ] Every artifact below records that same fingerprint.
- [ ] The intended production configuration is byte-for-byte/config-for-config
  identical to the calibrated configuration except for explicitly enumerated
  production duration and output location.

Launcher: `scripts/mm_param_learning/quasi_prod.py` writes
`runs/quasi-prod-<hash>-<utc>/fingerprint.json` and refuses a missing
hash (`require_fingerprint_file`). Direct ClassIFit — not molexp
declared-run replay. Multi-epoch production is still **NO-GO**.

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

## 3. Learnability and hyperparameter calibration

- [ ] Every active task/head independently overfits a fixed tiny set to its
  predeclared threshold. (Not run — 10-conf theater was rejected; energy
  head is still mostly `e0` + singleton per-atom residual.)
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
- [ ] Full input/kernel/optimizer profile not recorded (gpu_bench exists
  but is not the gate).
- [ ] Per-bucket batch sizes not split — OMol25-small is capped at 32
  atoms, one pack budget covers all three buckets.
- [x] HBM sweep GH200 Slurm 1417874 / molq `a03ff86c`, run
  `runs/quasi-prod-d8247ba3046e-20260819T101139Z/pack_sweep.json`:
  4096→0.19 GiB (181 mols), 8192→0.33, 16384→0.59, 32768→1.10,
  65536→2.15, 131072→4.19 GiB (5793 mols). No OOM. 120 GiB card
  still has >90 % free at the largest pack.
- [x] Operating pack `packed_max_atoms=16384` (power of two). Declared
  HBM margin is 16 GiB; the previous sweep at 16384 was 0.59 GiB
  (~500 mols). 131072 (4.19 GiB) still fits but collapses the
  OMol25-small epoch to 1–2 steps. Timed sweep is part of
  `calibrate()` (in flight / pending GH200).

## 5. Bounded end-to-end smoke and recovery

- [x] Quasi-prod runner uses the same ClassIFit / DataMixture / Adam /
  `val_omol_mae` path as epoch1. Unique `runs/quasi-prod-*` output.
- [x] Epoch1 (Slurm 1407684) moved train loss 4776 → 3006; val force MAE
  24.08. Single energy task.
- [ ] molexp metrics WAL not wired for this look-see (on-disk JSON only).
- [x] `ClassIFit.run(resume_from=, max_steps=)` + `run_resume_smoke`
  (1 step → last.pt → resume → step 2). GH200 evidence from the
  quasi-prod job.
- [ ] Preemption / partial-artifact restart not exercised beyond resume
  from `last.pt`.
- [ ] Calibration status `succeeded` waits on the quasi-prod job.

## 6. Final production submission gate

- [ ] No molexp production experiment. Quasi-prod is not production.
- [x] Pins live in `QUASI_PROD`: account `naiss2026-1-53-gpu`, partition
  `gpu`, hardware GH200, coords as §1.
- [x] Output is `runs/quasi-prod-<fingerprint12>-<utc>/` (gitignored),
  not repo-root `checkpoints/`.
- [x] Launcher fail-closed: missing fingerprint file, non-power-of-two
  sizes, train/val id overlap. Direct `python scripts/mm_param_learning/quasi_prod.py`.
- [ ] **decision: NO-GO** for unbounded multi-week production.
  Operator authorized a **1-day / 3-size** campaign on OMol25-small
  8:2. Align uses atol+rtol on dressed totals. Pack 16384. Stop is
  Slurm wall 24h.

```yaml
decision: NO-GO
scope: "prod-day1 three sizes; not unbounded GO"
fingerprint: ""
code_revision: "dirty; prod-day1 launch"
molhub_collection: "dataset:molcrafts/class1-universal-2026@1"
molexp_calibration_experiment: ""
molexp_calibration_run: ""
target_hardware: "GH200"
slurm_account: "naiss2026-1-53-gpu"
reviewer: "operator"
reviewed_at: "2026-08-19"
evidence_index: "runs/calibrate-653d8ec0ebe7-20260819T123919Z"
```

## Required sign-off record

```yaml
decision: GO                 # GO is the only launchable value
fingerprint: "..."
code_revision: "..."
molhub_collection: "<immutable MolHub collection coordinate>"
molexp_calibration_experiment: "..."
molexp_calibration_run: "..."
target_hardware: "GH200"
slurm_account: "<calibrated scheduler account>"
reviewer: "..."
reviewed_at: "..."
evidence_index: "..."
```

The reviewer must change `decision` back to `NO-GO` whenever the code,
environment, data, model, batching, optimizer, precision, or hardware
fingerprint changes. Recalibration is required before another `GO`.
