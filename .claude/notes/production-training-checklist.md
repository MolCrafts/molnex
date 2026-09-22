# Production training preflight checklist

This checklist is binding for every production training submission. An
unchecked item, missing evidence, `N/A`, waiver, or stale result means
**NO-GO**. Before sign-off, only explicitly bounded calibration/validation jobs
may run.

**What the gate is about (operator ruling, 2026-09-16).** The gate is *evidence
that the configuration works*, not bookkeeping about where that evidence is
filed. Two things are therefore explicitly **not** gates:

- **A molexp experiment/run record is optional.** Use it when it helps — it is
  a good place to keep evidence — but an absent or stale molexp record never
  blocks a submission. Evidence under a timestamped `runs/<slug>-<utc>/` is
  sufficient on its own, and §0 below is satisfied by that directory.
- **A clean or committed git tree is not required.** Committing is the
  operator's call and its timing is theirs; "uncommitted work" is never a
  reason to refuse a run. Record what was run (the run dir holds `config.json`)
  and move on.

**Where the launcher and the gates went (operator ruling, 2026-09-17).** Four
molnex scripts this checklist cites no longer exist here:

| Was | Now |
|---|---|
| `scripts/mm_param_learning/prod_day.py` | `lab-new/projects/class-i/train/prod_day.py` — a production day is a molab *workflow* under a project/experiment, not repo code |
| `scripts/mm_param_learning/align.py` | **deleted.** The operator ruled the align gate out; its one surviving fact, the size-bucket triple, was `configs.SIZE_BUCKETS` — itself deleted on 2026-09-20, having had no reader, and being meaningless for a dataset whose molecules are all ~33 atoms. The three tiers are *model* sizes (`configs.PROD_MODELS`), not atom buckets |
| `scripts/mm_param_learning/overfit_gate.py` | **deleted** by the same ruling |
| `scripts/mm_param_learning/profile_class_i.py` | **deleted** — molnex has a profiler (`molix.profiler`); a second one in `scripts/` was duplication |

Items below that name `align.py` / `overfit_gate.py` as the *implementation* of
a gate are therefore satisfied by evidence, not by running those files. The
recorded PASS lines that cite them are historical sign-offs and are left as the
dated record they are. The iron-law items in CLAUDE.md are unchanged: what was
deleted is one particular implementation, not the requirement for evidence.

What stays binding is everything that can make a run silently wrong: data
provenance and units, reference-vs-optimized parity, learnability, batching and
hardware calibration, and a bounded smoke. Those must be re-established
whenever the code, data, batching, precision or hardware changes — that rule is
about *staleness of measurement*, not about version control.

## 0. Submission identity

- [x] The run is still classified as calibration/validation; no production
  job has been submitted early.
- [x] Run identity is owned by the run directory. A molexp id may be recorded
  alongside it and is **optional** (ruling above); launchers never require one,
  and never require a git hash or a clean tree.
- [x] Every artifact below is written under the run's record dir (or a
  timestamped `runs/<slug>-<utc>/` when launched outside a declared molexp
  run).
- [x] The intended production configuration is byte-for-byte/config-for-config
  identical to the calibrated configuration except for explicitly enumerated
  production duration, scheduler account, and output location.
  `production_calibration_drift()` is empty: `PROD_DAY` vs `CALIBRATION`
  differ only in `kind`, `epochs`, `eval_every_epochs`, `wall_hours`,
  `slurm_account`. Science pins (4m/val-neutral, pack 16384, lr, clip,
  energy_scope, seed) match.

Launcher: `lab-new/projects/class-i/train/prod_day.py` writes
`runs/prod-day1-<size>-<utc>/` plus `molexp.json` / `config.json`.
molexp experiment `universal-class-i-v0-prod-day1-gh200` (runs
`f2dd9ed5` small / `90e71a0d` medium / `ef38fb1c` large). Unbounded
multi-epoch production is still **NO-GO**; the GO below is the 1-day
three-size campaign only.

## 1. Dataset and sampling contract

- [ ] **UNMET as of 2026-09-22 — the grouping was never validated, and it was
  wrong.** This item's own word is "grouping". `ProfileIndex` renumbered
  `graphs.group` into `graphs.profile`, and `ClassIPrepare.label` wrote that
  column as the constant `0.0` for every frame on the production path, so every
  conformer of every molecule was one profile. Stages 3 and 4 are
  profile-relative objectives: they subtracted a minimum taken across unrelated
  molecules, making the residual those molecules' absolute-energy spread. Nothing
  raised — the column was present, integral and dense — and the `soft_max` knee
  compressed 1e9-1e13 kcal²/mol² into a logged 1778-2760, which reads as noise.
  Measured on one real torsion-scan batch with stage 2's own checkpoint
  (`projects/class-i/train/energy_scale_probe.py`), before -> after the fix:
  profiles in the batch 1 -> 10; QM label spread per profile **1.83e6 -> 43.94
  kcal/mol**; per-profile squared residual 2.04e12 -> 1.33e5; rms relative-energy
  error 1.43e6 -> 365 kcal/mol. 44 kcal/mol is what a torsion scan should span.
  Fixed in molnex `ea1dd28`: the grouping is the dataset row, `collate_molecules`
  stamps `graphs.row`, and `ProfileIndex` defaults to it.
  **What to take from it for this item:** a grouping column being present and
  well-typed is not evidence that it groups. The check that would have caught it
  in one line is the one the probe now prints — *how many distinct profiles does
  a real batch hold, and what does the label span inside one?*
- [ ] Still open from the same probe, and now well-posed rather than hidden: with
  the grouping fixed, the **proper torsion term varies by 1132 kcal/mol within a
  scan where the QM reference varies by 44** — that is what stage 3 must pull
  down, and it is 25x too strong at the checkpoint stage 3 starts from. The angle
  term is effectively dead (within-scan spread 1.3e-4 kcal/mol) and the bond term
  carries an ~8000 kcal/mol absolute offset (within-scan spread 55), so stage 2's
  partial-Hessian fit is not producing physical bonded parameters — its loss moved
  0.747 -> 0.728 in 3893 steps. LJ is ~1e-6 and contributes nothing, as expected
  for parameters no stage has ever supervised (open item R1).


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
- [x] Size buckets are the same triple as `configs.SIZE_BUCKETS`
  (previously stated as `align.py` §2, which is deleted).

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

Calibration gates on official 4m/val-neutral coordinates (GH200, Slurm
1604451 / molq `0567acff`, run `runs/calibrate-20260823T193404Z`): Huber
energy + inlier-force loss, float64 dressing sum, and zero-energy LJ init
are verified. `inspect_ok=1` — energy term 23.2, force term 10.0 (was
7.16e11 / exploded), pred median |E| 217473 vs ref 217501, force ratio
0.022, pre-clip `grad_norm` 8.94, `clip_would_fire=0`, 6 DFT force
outliers (masked in the loss), 16/16 singleton groups. `align_ok=1`:
small/medium/large all pass packed-vs-single parity (grad residual
6.9e-7 / 5.9e-6 / 5.1e-6). Energy-label shift (+50 kcal/mol on the DFT
target) moves the Huber term on every bucket. §2 holds on production
coordinates.

## 3. Learnability and hyperparameter calibration

- [x] Every active task/head independently overfits a fixed tiny set to its
  predeclared threshold. Overfit gate on official 4m tiny set (8 train / 2
  val, ≤32 atoms, 4000 steps, GH200, molq `f5f4af44`, Slurm 1602274):
  energy Huber 22.4 → 0.16 (median last 200), force Huber drops 13.9 → 9.0
  with pred |F| median 0.32 → 8.4 vs ref 13.9 (scale ≥ 0.5×), clip fire
  7.7 %, `ok=1.0`
  (`runs/overfit-small-20260823T183517Z/overfit.json`).
- [x] Class-I energy-scale root cause fixed as a network-design issue: the
  `LJParameterHead` default `sigmoid(0)=0.5` init (ε≈0.5, σ≈2.5 Å) put a
  `(σ/r)¹²` wall on close nonbonded contacts and exploded the initial
  dressed energy/force scale. The head now zero-inits its final layer,
  biases ε to its floor (zero well depth), and biases σ to a moderate 1.0 Å
  (so `dE/dε ≈ 7` on OMol25's ~0.8 Å contacts instead of the dead-gradient
  a double-floor would give). The energy loss is residualized — MSE of the
  undressed `E_MM` against `E_DFT − e0` in float64 — so it trains on the
  ~O(10) kcal/mol MM scale rather than the float32 ~1e6 kcal/mol dressed
  total, then Huber-ized (`beta=20` kcal/mol) so a later `(σ/r)¹²` spike
  cannot dominate the gradient. Forces are Huber on DFT inliers
  (`|F_ref| ≤ 200` kcal/mol/Å). Verified on the calibrate gate:
  `energy_term` 7.16e11 → 23.2, `grad_norm` 3.52e14 → 8.94,
  `clip_would_fire=0`, `align_ok=1`, pred/ref energy medians match (§2).
- [x] Single-task force+energy mixture on the 256-molecule mini (Slurm
  1385961) and the 5880-molecule epoch1 (Slurm 1407684) both moved
  loss; clip raised from 1.0 to 1e3 after the scale audit showed
  `grad_norm ~ 400–2e5`. Post-Huber step-0 `grad_norm` is 8.94, so
  `clip_max_norm=1e3` does not fire at init.
- [x] Recorded operating point: Adam `lr=1e-4`, `clip_max_norm=1e3`, no
  scheduler, `hidden=64`, `n_blocks=4`, `n_torsion_terms=8` (all powers
  of two, live in `QUASI_PROD` not inside ChemEncoder).
- [x] Three power-of-two capacities: small 32/2/4, medium 64/4/8,
  large 128/8/16 (`PROD_MODELS`). One-day wall-clock campaign
  authorized 2026-08-19.

## 4. Packed batching and target-hardware optimization

- [ ] **UNMET as of 2026-09-22 — the delivered rate is 1/68th of the step rate,
  and every earlier number in this section measured the wrong thing.** Medium's
  production run (job 2787862, stage `mbis`, capacity 131072, 16 workers,
  prefetch 4) delivers **0.67 steps/s**, not the 45 steps/s its own log reports:
  97 batches per epoch in **145 s**, of which the GPU is busy **2.15 s (1.5 %)**.
  Measured three independent ways that agree — checkpoint-hook timestamps in
  `run-2787862.err` (UTC) 144-146 s apart, a live poll of the checkpoint step
  (140/140/160 s per 97 steps), and 8 730 steps in 3 h 40 m.
  `molix.hooks.scalar.StepSpeedHook` cannot see this: it averages a 10-step
  sliding window, and 16 workers x prefetch 4 buffer up to 64 batches, so the
  window that closes and gets logged sits *inside* a prefetch burst. The burst
  rate is the GPU's capability; it is not throughput. **Every ms/step in items 4
  and 5 above is a burst rate** and must be read as "what the accelerator could
  do", never as "what the run does".
- [ ] The cost is per-molecule work in the loader worker, profiled on one real
  batch (4 259 rows, `projects/class-i/train/collate_profile.py`, 27.65 s under
  cProfile; ~19 core-s uninstrumented => **4.5 ms per molecule** against 5 us per
  molecule on the GPU):
  - 18.1 s (65 %) dataset `__getitem__` — molhub `themol.py:582 _read_mbis`
    15.0 s, of which `themol.py:375 _mapped_topology_frame` 7.2 s and h5py
    reads 6.1 s.
  - 12.2 s the collate — `class_i_fit.py:639 execute` 9.3 s
    (`label_sample` 6.3 s, `_with_meta` 4.5 s, `_copy_frame` 3.0 s; `_with_meta`
    was deleted afterwards — `frame.meta["k"] = v` writes through, see
    notes.md),
    `collate_molecules` 2.6 s.
  - Largest single leaf: molrs `schema.column`, **434 418 calls (102 per row)
    for 6.1 s tottime = 22 % of the batch**, reached from `Frame.__setitem__`
    -> `_adopt_schema_dtype` on every column write.
  Sixteen workers at ~80 % CPU each = 12.8 of the 32 allocated cores, so the
  pool is worker-count-bound (host memory caps workers at 16 x prefetch 4 =
  107 GiB of 192), not core-bound.
  **Update 2026-09-22 (molnex `ea1dd28`):** the two per-row Frame rebuilds are
  gone (`_with_meta` rebuilt a frame to set two meta scalars on a false premise —
  `Frame.meta` is a write-through view; `_copy_frame` rebuilt another for an
  isolation no caller wants). Same batch, same profiler: **27.65 s -> 21.16 s**
  overall, the collate/prepare segment **12.2 s -> 3.57 s**, `schema.column`
  calls **434 418 -> 195 914**. The remaining 68 % is molhub's per-row decode
  (`themol.py:582 _read_mbis` 14.4 s, of which `_mapped_topology_frame` 6.87 s is
  an `mp.io.read_smiles` parse per row per epoch).
  Closing the rest is not a shaving job: the prepared per-row form has to be
  computed once and cached, not rebuilt from HDF5 every epoch. The same
  precompute closes the `PartialHessianPairs` re-enumeration already recorded
  under item 5.

- [x] Packed train path is `DataMixture` + `collate_molecules` (one
  forward per packed batch). No per-molecule Python loop in the step.
- [x] Full input/kernel/optimizer profile recorded
  (`runs/profile-20260823T180803Z/profile.json`): full-step median
  69.2 ms, throughput 1857→4702 samples/s (num_workers 0/2/4).
- [x] Per-bucket batch sizes not split (decision: keep one pack budget).
  OMol25-small is capped at 32 atoms; buckets are contiguous ranges inside
  that cap and the pack is a per-step atom budget, not a per-bucket split.
  Splitting would add loader/collator branches with no HBM or throughput
  benefit at this scale (16384-atom packs use 0.59 GiB).
- [x] **4m static shape is capacity-first (2026-09-14/15).** The padder is
  `MoleculePadder.from_counts(extent.counts(), atoms=2**k, cutoff=)` over the
  full-4m extent (scan keyed by the field's list cutoff, edges included) and
  the loader packs under `padder.budgets()` (`batch_counts` /
  `val_batch_counts`); the molecule count per batch floats. The old
  `B × per-molecule max` pad is retired (small@2048 = 17.4 M proper rows,
  OOM, job 2439130). Calibration = `benchmarks/bench_class_i_hbm.py` per
  tier over atoms ∈ {2^16 … 2^19}: `random` view (ms / GiB / fill) and
  `top10` view (largest 10 % by every column; gate = 0 recompiles, 0 padder
  raises). **Done 2026-09-15** (`runs/class-i-hbm/*.json`, GH200 96 GiB): 24
  cells, every one 0 recompiles with 12-20 distinct live graph counts;
  operating point `packed_max_atoms=131072` for all three tiers (2388
  molecules/batch, 12.7 / 14.1 / 20.2 GiB, 143 / 154 / 198 ms, 89-93 % of each
  tier's peak molecule throughput); HBM ceiling 2^19 / 2^19 / 2^18 (large OOMs
  at 2^19 after a 77.5 GiB peak). The json files under `runs/class-i-hbm/` are the
  evidence; linking them to a molexp record is optional (ruling above).
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
- [x] Calibration job `succeeded` (molq `0567acff` / Slurm 1604451); its
  align/inspect gates report PASS on official coordinates (§2).

## 6. Final production submission gate

**Not satisfied (2026-09-20).** The boxes below were ticked against OMol25 and
the model Rondo replaced; three of the four name things that no longer exist.
They are left as the dated record, unticked, so nothing here reads as live.

- [ ] molexp production experiment. The one named here
  (`universal-class-i-v0-prod-day1-gh200`, runs `f2dd9ed5` / `90e71a0d` /
  `ef38fb1c`) is gone; the current record is `lab-new` project `class-i`,
  experiment `prod-day-compile`, runs `day1-{small,medium,large}`. (A molexp
  record is not itself a gate — operator ruling, §"What the gate is about".)
- [x] Pins live in `PROD_DAY`: account `naiss2026-4-715-gpu`, partition `gpu`,
  hardware GH200 — now `dataset:themol/torsion-scan@1`, `shard_role`
  `shard-000`, held out by molecule at `val_frac` 0.05, **not yet calibrated**.
- [x] Output is `runs/prod-day<n>-<size>-<utc>/` under
  `workspaces/lab-new/runs/` (off the 30 GiB /home quota), not repo-root
  `checkpoints/`.
- [x] Launcher fail-closed on non-power-of-two sizes and on a *named* missing
  checkpoint. It no longer refuses a first day outright: a day 1 starts from
  scratch by definition, and refusing that left no way to launch at all.
  `python lab-new/projects/class-i/train/prod_day.py {small,medium,large}`.
- [ ] **decision: NO-GO.** See "Required sign-off record" below for what
  blocks it.

### Superseded: the 2026-08-23 §6 sign-off (does not apply)

```yaml
decision: GO
scope: "prod-day1 three sizes; not unbounded GO"
molexp_experiment: "universal-class-i-v0-prod-day1-gh200"
molexp_run: "f2dd9ed5,90e71a0d,ef38fb1c"
molhub_collection: "dataset:molcrafts/class1-universal-2026@1"
molexp_calibration_experiment: "universal-class-i-v0-calibration-gh200"
molexp_calibration_run: "class-i-v0-001"
target_hardware: "GH200"
slurm_account: "naiss2026-4-715-gpu"
reviewer: "operator"
reviewed_at: "2026-08-23"
evidence_index: "runs/calibrate-20260823T193404Z; runs/overfit-small-20260823T183517Z; runs/profile-20260823T180803Z"
```

## Required sign-off record

**Current decision: NO-GO (2026-09-20).** The previous `GO` is recorded below it
as the dated artefact it is; it does not apply to this configuration. Every one
of the three things the last paragraph of this file names as invalidating a
sign-off has changed since 2026-08-23: the **model** (Rondo did not exist until
2026-09-18, and no checkpoint from the earlier architecture loads into it), the
**data** (`dataset:themol/torsion-scan@1`, not OMol25 — see molnex notes "The
Class-I dataset is THEMol torsion-scan, and a row is 24 molecules"), and the
**batching** (the graph-slot count was being sized 24x short; `MoleculePadder`,
`DataModule` and `TokenBudgetBatchSampler` all changed).

```yaml
decision: NO-GO
scope: "nothing cleared; bounded calibration/validation only"
blocking:
  - "energy target: CLOSED (operator ruling 2026-09-21). THEMol publishes
     absolute energies (sampled record -503415.87 kcal/mol) and the epsilon[Z]
     dressing left ~2.3e5 kcal/mol of per-molecule constant against a
     ~7 kcal/mol torsion barrier, so val_mae_kcal sat at 231651 -> 231645 across
     all three tiers of smoke 2723274 while train_loss moved 81.96 -> 48.34. No
     stage supervises an absolute energy any more: all four declare
     `dressing: False`. The charge and partial-Hessian stages carry no energy term
     at all; stage 2 is ByteFF's `boltzmann_mse` and stage 3 its saturating
     `boltzmann_soft_mse` plus the inlier force Huber, both relative to a
     profile's own minimum with the per-profile offset removed and detached.
     Verified on real rows: a +5e5 / +1e6 kcal/mol per-profile shift leaves the
     energy half bit-identical (stage 2, job 2750175; stage 3, job 2750637). The
     epsilon[Z] table is out of the curriculum, so the metric that selects the
     checkpoint is now the quantity training optimizes."
  - "checklist item 5 is now MET for all four curriculum stages: 35 of 36 cells,
     one grid per stage (`calibration_grid` below). The one loss is
     hessian/medium/524288 to a BrokenPipeError, at a capacity that stage cannot
     use anyway — see the loader finding below. What remains for a GO is items 2
     and 3, not item 5."
  - "the schedule the grid exposed — 2 286 epochs for the large tier and 10 941
     for the small in a 24 h wall over 79 765 train molecules — is no longer a
     wall-clock question alone: the curriculum runs three stages over 61 796 +
     83 951 + 89 161 rows instead of one over 83 951. The stopping rule still
     has to be early stopping on validation rather than `epochs: 1_000_000` +
     `wall_hours: 24`, and the per-stage epoch budget is unset. Operator
     decision."
  - "checklist items 2 and 3: reference-vs-optimized parity and the bounded
     tiny-set overfit test have not been re-established for Rondo on this
     dataset."
target_hardware: "GH200"
slurm_account: "naiss2026-4-715-gpu"
reviewer: "unsigned"
reviewed_at: "2026-09-21"
calibration_grid_per_stage: |
  stage          tier    capacity    ms/step     GiB     mol/s             grid
  -----------------------------------------------------------------------------
  mbis           small    131,072       10.8    2.21   394,906 20260921T105718Z
  mbis           small    262,144       18.5    4.38   463,642 20260921T105718Z
  mbis           small    524,288       36.7    8.72   464,675 20260921T105718Z
  mbis           medium   131,072       21.9    4.45   195,352 20260921T105718Z
  mbis           medium   262,144       40.0    8.82   213,892 20260921T105718Z
  mbis           medium   524,288       74.5   17.57   228,831 20260921T105718Z
  mbis           large    131,072       70.1   12.08    61,066 20260921T105718Z
  mbis           large    262,144      138.1   24.03    61,955 20260921T105718Z
  mbis           large    524,288      267.9   50.43    63,640 20260921T105718Z
  hessian        small    131,072       10.9    2.09   389,670 20260921T113008Z
  hessian        small    262,144      498.0    4.14    17,136 20260921T113008Z
  hessian        small    524,288    29970.7    8.25       569 20260921T113008Z
  hessian        medium   131,072       22.3    4.57   191,016 20260921T113008Z
  hessian        medium   262,144      185.3    9.08    46,055 20260921T113008Z
  hessian        medium   524,288     FAILED                   20260921T113008Z  BrokenPipeError: [Errno 32] Broken pipe
  hessian        large    131,072       66.7   12.16    63,828 20260921T113008Z
  hessian        large    262,144      225.3   24.18    37,873 20260921T113008Z
  hessian        large    524,288    31216.8   48.35       546 20260921T113008Z
  torsion-scan   small    131,072       19.0    3.11   220,167 20260921T105718Z
  torsion-scan   small    262,144       31.3    6.17   268,073 20260921T105718Z
  torsion-scan   small    524,288      180.1   12.33    93,119 20260921T105718Z
  torsion-scan   medium   131,072       30.6    6.24   136,973 20260921T105718Z
  torsion-scan   medium   262,144       53.7   11.64   156,572 20260921T105718Z
  torsion-scan   medium   524,288      107.4   23.24   156,152 20260921T105718Z
  torsion-scan   large    131,072       78.5   14.46    53,348 20260921T105718Z
  torsion-scan   large    262,144      150.5   28.82    55,823 20260921T105718Z
  torsion-scan   large    524,288      300.9   57.62    55,751 20260921T105718Z
  hessian-relax  small    131,072       22.4    3.45   186,956 20260921T123156Z
  hessian-relax  small    262,144       39.8    6.85   206,409 20260921T123156Z
  hessian-relax  small    524,288       73.0   13.70   231,900 20260921T123156Z
  hessian-relax  medium   131,072       32.1    6.31   130,411 20260921T123156Z
  hessian-relax  medium   262,144       60.8   12.61   135,046 20260921T123156Z
  hessian-relax  medium   524,288      117.0   25.15   144,648 20260921T123156Z
  hessian-relax  large    131,072       83.4   15.21    50,152 20260921T123156Z
  hessian-relax  large    262,144      161.8   30.35    50,787 20260921T123156Z
  hessian-relax  large    524,288      320.9   60.59    52,713 20260921T123156Z

  35 of 36 cells calibrated

  Read `ms/step` at 524 288 as contended, not clean. torsion-scan/small/524288
  measured 83.4 ms on one grid and 180.1 ms on another at identical peak memory
  (12.33 GiB) and identical code; the two cheaper capacities of the same tier
  reproduced to 0.1 ms. Peak GiB is reproducible everywhere; step time at the top
  capacity is not, on non-exclusive GPU nodes.

  **Stage 1 is loader-bound above 131 072, by a factor of 700.** Its cells read
  10.9 / 498.0 / 29 970.7 ms for 131 072 / 262 144 / 524 288 — 389 670 down to 569
  molecules/s — on 2 to 8 GiB of a 120 GB card, with genuinely full batches
  (131 061 and 262 116 real atoms, 0.80 graph-slot fill). The GPU is idle: at the
  top capacity one batch is 17 044 rows, each of which `HessianTargets` re-enumerates
  through `PartialHessianPairs`'s Python `permutations` loop, about 44 M interpreter
  operations per batch, and the shard holds only 3.6 such batches so it re-enumerates
  every epoch. Two symptoms confirm it is the loader and not the model:
  hessian/medium/262144 (185.3 ms) is *faster* than hessian/small/262144 (498.0 ms)
  despite five times the parameters, and hessian/medium/524288 died of a broken pipe.
  **So stage 1 runs at 131 072 and nothing larger**, a quarter of what the other three
  stages take, until that pair layout is precomputed into a sidecar the way the
  extent is. That makes the precompute a prerequisite rather than an optimisation.

  The other three stages scale cleanly in both axes — hessian-relax 22.4 / 39.8 /
  73.0 ms with peak memory doubling exactly per step, mbis 10.8 / 18.5 / 36.7,
  torsion-scan 19.0 / 31.3 / (contended) — and 524 288 remains the memory ceiling
  (60.59 GiB for hessian-relax/large, so the next power of two exceeds the card).

  Every stage's own fingerprint differs, which is why one grid could not have served:
  only `hessian` carries a `hessian_pairs` axis, and a row costs 30.8 atoms there and
  in `mbis` against 704.5 in `torsion-scan` and 2 020.6 in `hessian-relax`.

  Six defects were found and fixed between the first grid and the last, every one
  invisible to the bounded smokes because those build a batch by hand and never reach
  the loader, the trainer or `torch.compile`: a unit guard that refused a dataset
  publishing no energy, and a second one on the frame; molpy construction and leaf
  creation inside the compiled region; a delegating `__getattr__` that recursed under
  `pickle` and killed every spawned worker with a message that reads like an OOM; and
  a `WeightedLoss` holding its weights on the host, which eager accepts silently and
  inductor does not. `cpu_cell.py` exists so the host-side half of that class is
  caught in seconds rather than in a GPU queue slot.

superseded_calibration_grid: |
  9/9 cells, job 2727464, snapshot 20260920T132404Z (molrs 026ece44, molpy
  f21ce723, molhub 955efeaf, molnex 8e5a49db — a verified-consistent pair, the
  wheel gate passing on merit rather than by override).

  **SUPERSEDED (2026-09-21), and not merely incomplete.** Two things invalidate
  it. It fingerprints one stage: the caps below carry no `hessian_pairs` axis, so
  they do not describe a stage-1 batch, and a hessian-relax row costs 2 021 atoms
  against torsion-scan's 704, so they do not describe stage 3 either. And the
  stage it *did* fingerprint has since changed objective — torsion-scan now fits
  `BoltzmannProfileLoss` with `dressing: False` instead of the dressed energy
  Huber plus forces, so even its own ms/step and peak GiB are a different step's.
  Four grids are being measured to replace it (jobs 2751981-2751984 on snapshot
  20260921T095451Z); until they land, item 5 is unmet for every stage.

  Row costs, measured from the frozen extents (job 2751930), which is what turns
  an atom capacity into a row count — a row means a different thing per stage and
  is indivisible:

  stage           rows     atoms/row mean   max   graphs/row max   rows @131072
  mbis         430 906              30.9     70                1          4 241
  hessian       61 796              30.8     70                1          4 255
  torsion-scan  83 951             704.5   1632               24            186
  hessian-relax 89 161            2020.6  12180              302             64

  The largest single molecule is 70 atoms across all four, so the fixed
  `RondoSpec.eem_max_atoms=512` covers the dense charge solve everywhere.

  tier    capacity  ms/step  peak GiB  mol/batch     mol/s  atom fill  graphs
  small     131072     20.7      3.52       4229    203814      99.7%       1
  small     262144     36.2      6.99       8274    228598      99.6%       1
  small     524288     68.8     13.94      16678    242553      99.9%       1
  medium    131072     32.5      6.40       4229    129965      99.7%       1
  medium    262144     59.9     12.77       8274    138161      99.6%       1
  medium    524288    114.6     25.49      16678    145544      99.9%       1
  large     131072     83.5     15.35       4229     50621      99.7%       1
  large     262144    158.9     30.61       8274     52072      99.6%       1
  large     524288    308.4     61.19      16678     54080      99.9%       1

  Every cell compiled to a single graph, so the no-graph-breaks iron law holds
  in all nine configurations on this dataset. Atom fill is 99.6-99.9 %
  throughout; the graph-slot count sits above the live count by the padder's
  1.25 graphs_slack, which is what it is for.

  524288 is the top usable capacity: at 61.19 GiB for the large tier the next
  power of two needs ~122 GiB against the GH200's 120 GB. But it is not the
  right production choice — it buys 7 % (large) to 19 % (small) more epochs for
  4x the memory, and equal step counts across tiers serve the scaling-law axis
  better. 131072 stands.

  Step time scales with capacity (1.75-1.91x per doubling) and sub-linearly with
  parameters (5x the parameters costs 1.57x the step; 29x costs 4.03x). The void
  OMol25 grid was flat instead — 85.6 to 91.8 ms across an 8x capacity range —
  which was the loader setting the pace, not the GPU.
evidence_index: >
  workspaces/lab-new/runs/rondo-smoke/smoke-2723274.out (item 6: PASS on all
  four properties for all three tiers — loss movement, validation reported,
  resume past the checkpoint at 8 -> 16, per-tier failure isolation; fullgraph
  compile clean; parameters 41108 / 204188 / 1191212; peak GPU 0.49 / 0.88 /
  1.97 GiB);
  molnex/runs/valence-distrib/summary.json (item 1: the frozen MoleculeExtent
  over all 83951 rows at cutoff 10.0, max 1632 atoms/row);
  workspaces/lab-new/runs/rondo-calibrate/calibration-20260920T133503Z.json
  (item 5: the 9-cell grid above, with the per-cell logs beside it);
  workspaces/lab-new/snapshots/20260920T132404Z/REVISIONS (the four commits
  every job on that snapshot ran, and their wheel's BUILT_FROM)
stage_1_evidence: |
  The partial-Hessian stage's data and loss path is validated, bounded, on real
  rows — not calibrated. Job 2748912 (6 THEMol hessian@1 records, CPU):

  * labels gathered from each record's QM Hessian and the (3N)^2 block dropped;
    224 / 88 / 264 / 294 / 228 / 248 pairs against the cap bound
    2 * (bonds + angles) = 230 / 88 / 264 / 294 / 228 / 248 — exact on five rows,
    six pairs tighter on the one with a ring, which is where a 1-3 pair is also
    a 1-2 pair.
  * the collated label is bit-identical to the raw QM Hessian block on that pair
    (max |d| = 0.000e+00), so collate's atom-pair and slot rebasing are right.
  * padded 1 346 live pairs into a 4 096 cap; the prediction is finite at
    (4096, 9); the loss is 0.699 (dimensionless, trace-relative).
  * `trainable=("bonds","angles","impropers")` holds under backward: bond
    6.84e-05, angle 2.32e-03, improper 2.17e-03, **proper exactly 0**, shared
    encoder 1.69e-03. That is `set_trainable_terms` semantics — the propers are
    still priced so the pair blocks are complete, and cannot move.

  The MM prediction itself is pinned against the full (3N, 3N) Hessian of the
  same `Potentials.energy` to 5.7e-14 on every pair, with bonds, angles, propers
  and impropers all accumulating
  (`tests/test_molpot/test_forcefield/test_potentials.py::TestPartialHessian`).

  Still missing for stage 1: its own calibration cell (item 5) and a bounded
  tiny-set overfit test (item 3).
stage_3_evidence: |
  Stage 3's objective is relative energies plus forces (operator ruling
  2026-09-21) — `BoltzmannProfileLoss(soft_max=PROFILE_SOFT_MAX)` beside the
  inlier force Huber, unit weights, `WeightedLoss`. Validated bounded on real
  `hessian-relax` rows (job 2750637): each relaxation's frames land in one
  profile (58 and 76 frames, so the "~72.5 per row" figure is a mean and not a
  fixed grid), the reference forces are read (|F| ~ 17-19 kcal/mol/A), the energy
  half is invariant to a +5e5 / +1e6 kcal/mol per-profile shift, and with
  `trainable=None` every head moves. Both halves are bounded — the energy by the
  saturation, the force by the Huber's linear regime — which is what makes the
  unit weights defensible.

  **Measured at initialization the two halves are not balanced**: energy 4.118,
  force 0.109, a factor of 38. That is one untrained model on two records and not
  a converged ratio, so it is not yet a reason to reweight — but it is the number
  to watch in the first bounded stage-3 run. If the gap persists the answer is
  `ScaleNormalizedLoss` on each half (dividing by a measured scale, which makes
  both dimensionless), not a hand-tuned weight.

  Still missing for stage 3, as for stages 0-2: its own calibration cell (item 5;
  a relaxation row is ~72.5 frames against torsion-scan's 24) and a bounded
  tiny-set overfit test (item 3).
notes: >
  Step rate in the smoke (0.64 steps/s) is NOT a throughput measurement: the
  smoke runs num_workers=0, so 480 frames per batch go through prepare serially
  on the main process at under 2 GiB peak. Calibration with workers measures it.
```

### Superseded: the 2026-08-23 sign-off (does not apply)

```yaml
decision: GO
scope: "prod-day1 three sizes; not unbounded GO"
molexp_experiment: "universal-class-i-v0-prod-day1-gh200"
molexp_run: "f2dd9ed5,90e71a0d,ef38fb1c"
molexp_calibration_experiment: "universal-class-i-v0-calibration-gh200"
molexp_calibration_run: "class-i-v0-001"
molhub_collection: "dataset:molcrafts/class1-universal-2026@1"
target_hardware: "GH200"
slurm_account: "naiss2026-4-715-gpu"
reviewer: "operator"
reviewed_at: "2026-08-23"
evidence_index: "runs/calibrate-20260823T193404Z; runs/overfit-small-20260823T183517Z; runs/profile-20260823T180803Z"
reason: "calibrate align_ok=1 inspect_ok=1 (Huber residualized energy + inlier force, energy_term 23.2, grad_norm 8.94, clip_would_fire=0). Overfit ok=1. Profile refreshed. PROD_DAY matches CALIBRATION except enumerated duration/account/output."
```

Those three evidence directories were deleted with the rest of the pre-Rondo run
output (molnex `1a76844`), so the sign-off above cannot be re-checked even on its
own terms.

The reviewer must change `decision` back to `NO-GO` whenever the code,
environment, data, model, batching, optimizer, precision, or hardware
changes. Recalibration is required before another `GO`.
