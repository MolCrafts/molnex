# Production training preflight checklist

This checklist is binding for every production training submission. Complete
it against one exact configuration fingerprint and attach every result to a
versioned molexp calibration experiment. An unchecked item, missing evidence,
`N/A`, waiver, stale result, or fingerprint mismatch means **NO-GO**. Before
sign-off, only explicitly bounded calibration/validation jobs may run.

## 0. Submission identity

- [ ] The run is still classified as calibration/validation; no production
  job has been submitted early.
- [ ] One immutable fingerprint covers code revision and dirty-diff hash,
  environment/lockfile, MolHub collection and source versions, split manifests,
  model dimensions, task heads/losses/weights, curriculum stage, optimizer,
  scheduler, precision/compile settings, collator, seeds, and target hardware.
- [ ] Every artifact below records that same fingerprint.
- [ ] The intended production configuration is byte-for-byte/config-for-config
  identical to the calibrated configuration except for explicitly enumerated
  production duration and output location.

## 1. Dataset and sampling contract

- [ ] Every MolHub coordinate resolves to the intended immutable version and
  its provenance, reference method, units, and target schema are recorded.
- [ ] Global molecular identity, cross-source deduplication, and molecule-level
  train/validation/test splitting occur before configuration sampling.
- [ ] All structures from one molecule remain in one split; torsion scans and
  other grouped targets remain intact.
- [ ] No undeclared source or flat concatenated frame pool enters the loader.
- [ ] Task sampling is `task -> molecule/scan -> configuration`; an observed
  sampling audit agrees with the configured mixture within a declared
  statistical tolerance.
- [ ] Rare chemistry and molecular-size coverage are reported for every split.
- [ ] Representative small, medium, and large molecular-size buckets are fixed
  and reused by correctness, performance, and smoke checks.

## 2. Network correctness and numerical parity

- [ ] Declared numerical tolerances exist before comparison begins.
- [ ] Reference single-molecule and optimized packed-batch paths agree for each
  size bucket on model outputs, energies/forces/parameters, and every active
  per-task loss.
- [ ] Parameter-group gradients agree between reference and optimized paths;
  every intended trainable group receives a finite, nonzero gradient.
- [ ] One complete optimizer update agrees between the two paths within the
  declared tolerance.
- [ ] Permutation, rigid translation/rotation, grouping, charge conservation,
  and other applicable physical invariants pass.
- [ ] NaN/Inf checks, outlier reports, gradient norms, clipping incidence, and
  initial prediction/target scales have been inspected and archived.
- [ ] Packed batches neither mix graph identities nor leak values across
  molecules/tasks.

## 3. Learnability and hyperparameter calibration

- [ ] Every active task/head independently overfits a fixed tiny set to its
  predeclared threshold.
- [ ] The configured multi-task mixture overfits a bounded joint tiny set
  without one loss or gradient contribution overwhelming the others.
- [ ] Learning-rate range, optimizer, scheduler, task weights, clipping, and
  initialization have been calibrated with results recorded, not guessed from
  a previous architecture.
- [ ] Small, medium, and large networks use powers of two for every model-size
  hyperparameter and each passes the same correctness/learnability gates.
- [ ] The smallest model is the smallest tested power-of-two configuration
  that still passes the capacity/tiny-overfit and required baseline tests.
- [ ] The largest model is the largest tested power-of-two configuration that
  maximizes useful target-card utilization while retaining the declared HBM
  safety margin and stable packed-batch execution.
- [ ] The medium model is a measured intermediate operating point, not merely
  a name or an untested arithmetic midpoint.

## 4. Packed batching and target-hardware optimization

- [ ] Molecules are packed by a dedicated Collate class; the training step has
  no avoidable Python loop that executes the network once per molecule.
- [ ] A target-hardware profile identifies input wait, CPU collation/copy,
  kernels, synchronization, optimizer, and validation costs.
- [ ] Avoidable serial bottlenecks and dominant regressions have been removed,
  then the optimized path has been re-profiled.
- [ ] Batch size is calibrated separately by molecular-size bucket: small
  systems use larger batches and large systems use smaller batches.
- [ ] Each bucket records peak/steady HBM, reserved HBM safety margin, step
  time, molecules/s, conformations/s, atoms/s, and accelerator utilization.
- [ ] Warm-up/compile cost is separated from steady-state throughput.
- [ ] The selected model/batch pair is the fastest stable measured choice under
  the memory constraint; an OOM boundary and recovery behavior are known.
- [ ] Expected production wall time, checkpoint volume, and storage footprint
  are calculated from measured throughput.

## 5. Bounded end-to-end smoke and recovery

- [ ] A bounded run exercises the exact data mixture, network, optimizer,
  precision, collator, validation, metrics, and checkpoint paths.
- [ ] Training loss moves in the expected direction and frozen validation is
  reported separately for every active physical task.
- [ ] Each configured task is actually sampled and updates its intended head.
- [ ] Losses, gradients, learning rate, throughput, HBM, and data-wait metrics
  are visible in molexp and contain no unexplained discontinuity.
- [ ] Checkpoint save and resume reproduce the expected next-step state and
  metric trajectory within the declared tolerance.
- [ ] Preemption/failure handling, partial-artifact handling, and clean restart
  have been exercised.
- [ ] Calibration status is `succeeded`; failed and superseded calibrations
  cannot satisfy the gate.

## 6. Final production submission gate

- [ ] The molexp production experiment links the succeeded calibration
  experiment, all evidence artifacts, and the exact shared fingerprint.
- [ ] The exact approved MolHub collection coordinate and every role source
  version are pinned in the Molexp experiment; all task weights are recorded.
- [ ] The exact scheduler account/partition is pinned in the calibration and
  production fingerprints and matches the approved experiment configuration.
- [ ] Requested accelerator, precision, memory, wall time, model, size-bucket
  batch policy, and worker counts exactly match calibration.
- [ ] Old/conflicting training jobs are stopped; unrelated running jobs are
  identified explicitly before submission.
- [ ] Output, checkpoint, and log paths are unique, writable, capacity-checked,
  and cannot overwrite a previous run.
- [ ] The launcher validates the fingerprint and calibration status itself and
  exits nonzero on any absent, stale, failed, or mismatched item.
- [ ] A human final review records **GO** below before `sbatch` is called.
- [ ] Immediately after submission, Slurm job ID, resolved config, and launch
  timestamp are recorded in the molexp production run.

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
