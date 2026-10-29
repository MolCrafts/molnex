# Phase 2 — replace the charge and LJ heads under a frozen ByteFF-26 encoder and bonded heads

| Field | Value |
|---|---|
| Opened | 2026-10-28 |
| Chain | rondo-enc (charter `rondo-enc-00-charter.md`, phase 2) |
| Status | done 2026-10-29 — gates 1–6 signed off, D1 / D2 / D1+D2+FT trained on member-01, verdict §8 (D2 replaces the LJ head; D1 misses on torsion under frozen amplitudes) |
| Product | `molzoo.rondo.Rondo` (D1 `PairCoulCut` + EEM, D2 `LJCut` around GAFF2) |
| Depends on | `rondo-enc-01-graph-encoder` (phase 1: parity and the end-to-end gate) |
| Baseline | `molzoo.byteff.ByteFF` — the same encoder and bonded heads with ByteFF-26's own pair heads (`BondChargeCorrection`, `LJCutDirect`) |

## 1. Question

Can Rondo's own physical heads replace ByteFF-26's charge and LJ heads while the
encoder and the bonded heads stay exactly ByteFF-26's? Each head is tested alone and
together, so a difference is attributable to one component (charter: one change per
experiment):

| Run | Charges | σ, ε | Trains |
|---|---|---|---|
| B (baseline, no training) | BCC (ByteFF-26) | direct (ByteFF-26) | — |
| D1 | EEM (`PairCoulCut`), MBIS-supervised | direct (ByteFF-26, frozen) | `atoms.charge` heads |
| D2 | BCC (ByteFF-26, frozen) | GAFF2-anchored window (`LJCut`) | `atoms.sigma`, `atoms.epsilon` heads |
| D1+D2 | EEM | GAFF2 window | both |

D1 and D2 alone need a model that mixes one ByteFF-26 pair head with one Rondo
pair head: `Rondo._new(style)` is the one switch, so each mix is a three-line
subclass in the launcher, not a library option.

## 2. Model composition (caller's, no new API)

    member = ByteFF.load(path, forcefield, dtype=...)       # phase-1 verified
    model  = Rondo(forcefield, dtype=..., **settings)       # D1 / D2 pair heads, fresh
    model.graph.load_state_dict(member.graph.state_dict())
    for style in BONDED: model.heads[style].load_state_dict(member.heads[style].state_dict())
    model.graph.requires_grad_(False); for style in BONDED: model.heads[style].requires_grad_(False)

Frozen means both: no gradient (`requires_grad_(False)`) and no loss names a frozen
table (`JointObjective.trainable`). The optimizer is `Adam(p for p in model.parameters()
if p.requires_grad)`. A check before step 0: the frozen state dict is byte-identical to
the member's after N steps (gate §5.3).

## 3. Objectives (from `scripts/mm_param_learning/configs.py`, restricted)

| Objective | Source | Trainable (phase 2) | Runs |
|---|---|---|---|
| `mbis` | `themol/mbis@1` | `atoms.charge` | D1, D1+D2 |
| `relax` | `themol/hessian-relax@1` | `atoms.sigma`, `atoms.epsilon` (§6.2) | D2, D1+D2 |
| `torsion` | `themol/torsion-scan@1` | `atoms.sigma`, `atoms.epsilon` (§6.2; propers frozen) | D2, D1+D2 |
| `torsion`, `torsion_l1` | `themol/torsion-scan@1` | `propers.k` (proper head output layers only) | the constrained fine-tune of §6, only on a miss |

Every bonded objective (`hessian_k`, `hessian_geom`, `hessian_eq`,
`hessian_eq_improper`, `torsion_l1`) is off: it moves only frozen tables.

## 4. Metrics (every run, and baseline B on the same split)

Held-out by molecule (`themol-molecule-95-5`): charge MAE vs MBIS; relative-energy and
force MAE on hessian-relax; torsion-profile MAE (offset-free, per scan); the
GAFF2 distance of σ / ε (D2 heads only). Reported per element and per functional
group where the count allows.

## 5. Gates (production checklist, `.claude/notes/production-training-checklist.md`)

1. **Data** — the three sources load through MolHub on molpy 0.20 with
   `equi_index` / `propers.linear` / small-ring columns (phase 1, 2998 / 3000), split by
   molecule; units from the manifests (`ClassIPrepare.from_dataset`).
2. **Reference vs packed** — the padded batch path reproduces the per-molecule
   reference on forward, every active loss, gradients and one optimizer step.
3. **Frozen is frozen** — after a bounded run the encoder and bonded state dicts equal
   the loaded member's bit for bit; no frozen table reaches a loss.
4. **Overfit** — each active objective drives a tiny set to near zero.
5. **Calibration** — GH200 batch capacity / step time for the nested batch of the
   active sources (the HBM numbers in `configs.py` predate the port).
6. **Smoke** — 200 steps (one epoch: an eval, `best.pt`, `last.pt`), then a fresh
   process resumes `last.pt` to step 300: loss moves, validation reports, the resume
   continues the counters, the frozen weights stay bit-identical (`phase2/smoke.sbatch`).
7. One timestamped run directory with the evidence: `runs/rondo-enc-phase2-<utc>/`
   (`calibration/`, `smoke/`, `signoff.json`, `runs/<RUN>/`). `phase2/signoff.py` writes
   `signoff.json` from the per-run calibration cells and the two smoke legs, or nothing;
   `heads.py` refuses a production run (no `--smoke` / `--max-steps`, full splits) unless
   the record names the run, its configuration, the capacity and the padded caps it
   composes now (`phase2/production.sbatch`).

## 6. Decisions (2026-10-28) and why

What the reference does with the heads this phase replaces (`byteff2@v1.1.0`,
`example/ByteFF/1_training/{pretrain,train}.yaml`, `model/hybrid_ff.py`; paper §2.5,
Table 1): the charges and σ / ε are **not fitted to QM**. Pre-training regresses them onto
AM1-BCC charges and GAFF-2.2 σ / ε (`ParamMSE`, labels from `gaff2_am1bcc_*.csv`), for
Amber compatibility. The training stage drops that anchor and prices the torsion scans
with the nonbonded energy left differentiable (`ff_require_grad` defaults to True), so the
Boltzmann torsion loss moves q, σ, ε together with the proper `k` from there on. ByteFF-26's
torsion amplitudes were therefore fitted *jointly* with its own intramolecular
nonbonded energy, and an ensemble of five seeds (43–47) is averaged parameter by
parameter.

1. **Per member; develop on member-01.** A D head is trained against one member's frozen
   encoder and bonded heads — there is no single "ensemble encoder" to freeze. Every
   run of the matrix is done on member-01 (seed 43); the winning configuration is
   repeated on member-02 to rule out a seed artefact; the deliverable, if a head passes,
   is five runs averaged as ByteFF-26 averages (each member with its own D heads).
2. **D2 (σ, ε) learns from the torsion scans and the relaxations.** That is where the
   reference's σ / ε got their only QM signal (the torsion stage), and the 1-4 / 1-5
   contacts a torsion scan sweeps are exactly where intramolecular LJ matters; the
   relaxations add off-equilibrium short contacts. The exponential window around GAFF2
   is the anchor the reference's `ParamMSE` was — it keeps σ / ε from absorbing what the
   torsion series should carry.
3. **D1 charges stay MBIS-only.** The point of D1 is a charge with a physical target;
   letting energy and force gradients move it would turn it back into the reference's
   torsion-fitted charge and make the comparison meaningless.

**Expected coupling, and the constrained fine-tune.** Swapping the charges changes the
intramolecular electrostatics the frozen torsion amplitudes were fitted against (MBIS
charges are typically larger in magnitude than AM1-BCC), so D1 alone may move the
torsion profiles even with perfect MBIS charges. That is a measurement, not a failure of
the head. The outline's "constrained joint fine-tuning" is then one step, taken only if
a D run misses the margin in §4: the proper head's output layers train on the torsion
scans with the L1 prior (`torsion`, `torsion_l1`), the encoder and every other bonded
head stay frozen, and the D heads keep their own objectives. Margin for "replaceable":
held-out torsion-profile, relative-energy and force MAE within 10 % of baseline B.

## 7. Work before any run

Done 2026-10-28:

- `configs.py`: tiers carry `Rondo` keywords (`settings`), `byteff26` is the published
  architecture; `OBJECTIVES` gains `torsion_lj`, `relax_lj`, `eval_charge`,
  `eval_torsion`, `eval_relax`; `HEAD_SWAP` holds the run matrix, the member and the
  rate (3e-4). `forcefield.py`: `amber_forcefield()` — the published form, every
  intramolecular pair (50 Å). `class_i_fit.py`: prepared rows `v5` (the new encoder
  columns), the `relax_metrics` score, molpy 0.20 fixes (`Element.by_number`,
  `Quantity.value`, `get_styles`).
- Launcher `lab/projects/rondo-enc/phase2/heads.py`: the four variants (B = `ByteFF`,
  D1 / D2 two-line `_new` overrides, D1+D2 = `Rondo`), member weights into the encoder
  and every shared head, frozen; fingerprints of the frozen and of the trainable heads
  before and after; `phase2.json` per run. Sources read from the HDF5 shards (the old
  LMDB collections predate the new columns), censused, split by molecule.
- `molix.core.steps.joint`: a `"tables"` objective on a source without an energy
  label (MBIS) no longer reads one that is not there (test added, shown to bite).
- CPU dry runs (`phase2/dryrun.sbatch`, 16 train / 8 val molecules per source):
  4581717 — B scored (charge 0.0232, torsion 0.090, relax 1.367), D1+D2 untrained
  (0.082, 0.402, 1.627), both fingerprints unchanged; 4582184 — D1+D2, 30 real steps:
  all three scores fall (0.0817, 0.400, 1.598), the EEM and LJ heads moved, the frozen
  encoder and bonded heads bit-identical.

Gates 2 and 4, 2026-10-29 (CPU, `phase2/gates.sbatch`, 8 molecules per source):

- Data first: molpy's SMILES reader refused RDKit-written THEMol strings (marks next to a
  carbonyl or a terminal `=CH2` read as conflicts) — fixed in molpy (a double bond's
  geometry exists only when both ends carry a mark, OpenSMILES / RDKit); one mbis record
  that is genuinely contradictory is read through the stereo-free
  `mapped_nonisomeric_smiles` in MolHub. Census rerun under molpy 0.20 (counts equal the
  10-11 ones on mbis / torsion).
- **Gate 2 passes for every run** (B, D1, D2, D1+D2, D1+D2+FT): losses ≤ 1.5e-15
  relative, gradients ≤ 2.3e-14 of the largest, one Adam step ≤ 1e-8 of the rate,
  packed + compiled against per-molecule eager in float64. (Per-tensor relative error
  was the wrong metric: the EEM output bias's exact gradient is zero at its zero start —
  charge conservation cancels the sum — so it carries round-off only.)
- **Gate 4** (300 steps at the full rate, scored against the published heads B on the
  same molecules):

  | Run | charge vs MBIS | torsion | relax (E, F) |
  |---|---|---|---|
  | B (published) | 0.0198 | 0.326 | 1.106 |
  | D1 | **0.00039** | — | — |
  | D2 | — | **0.082** | **0.894** |
  | D1+D2 | **0.00037** | 0.463 | 1.124 |
  | D1+D2+FT | **0.00039** | **0.243** | 1.352 |

  D1 and D2 pass. With MBIS charges the frozen bonded heads lose on the relaxations: σ, ε
  alone cannot absorb the electrostatics the torsions were fitted against, and the
  constrained fine-tune (proper output layers, `lr_unfrozen` 2e-5 — 3e-4 diverged from
  step 0) recovers the torsion profiles but not the off-equilibrium energies and forces.
  This is the coupling §6 predicted, measured on a tiny set; it is an operator decision
  whether the next fine-tune also opens the bond / angle geometry (ByteFF's stage 3) or
  the D1 charges are accepted as incompatible with frozen ByteFF-26 bonded heads.

Operator choice 2026-10-29 (option 1): the fine-tune also opens the geometry —
`D1+D2+FT` trains the output layers of the proper amplitudes and of the bond / angle
equilibrium values (`heads.propers.out`, `heads.bonds.out.r0`, `heads.angles.out.theta0`,
at `lr_unfrozen` 2e-5) on the torsion scans (+L1), the relaxations (`relax`, ByteFF's
stage 3) and the QM minima (`hessian_eq`, `hessian_geom`); the force constants and the
shared trunks stay frozen (an energy / force loss that reaches K softens it, A26).
Gate 2 passes (six losses ≤ 2.7e-16, gradients ≤ 4e-15 of the largest; it found and
fixed `MeanTermEnergyLoss` counting an unpadded `bonds` block by a row axis it does
not have). **Gate 4 passes:** charge 0.00036 (B 0.0198), torsion 0.308 (B 0.326),
relax 0.928 (B 1.106); the stationary force at the minima 17.4 → 8.9.

Gate 5 infrastructure: molnex's own GH200 env `.venv-aarch64` (built on a GH200 node by
`.jobs/env-build-aarch64.sbatch`: the x86 env's PyPI pins, molpy compiled for aarch64,
molhub, native ops for sm_90 in `build/csrc-aarch64`); the extents at the published
50 Å (`scan_class_i_extent.py … 50.0`, no matplotlib — not on the allowed surface);
`phase2/calibrate.py` (one process per capacity, recompiles forbidden after the smoke).

Gate 5, first sweep (GH200 4604782, `phase2/calibrate.py`, run D1+D2+FT, four sources,
40 steps per cell after a 3-step compile smoke, recompiles forbidden): every capacity runs.

| atoms / source | step | compute | data wait / step | peak HBM |
|---|---|---|---|---|
| 16 384 | 0.39 s | 0.088 s | 0.23 s | 3.8 GiB |
| 32 768 | 0.62 s | 0.135 s | 0.38 s | 6.8 GiB |
| 65 536 | 1.15 s | 0.241 s | 0.74 s | 13.1 GiB |
| 131 072 | 2.31 s | 0.458 s | 1.56 s | 25.3 GiB |
| 262 144 | 4.80 s | 0.886 s | 3.37 s | 50.4 GiB |

Compute is linear in capacity (~3.4 µs per atom); the delivered step is 4–5× the compute —
the loader, on a cold prepared-row cache (every job starts empty; HDF5 decode per row).
Found and fixed on the way to a clean sweep: the resident validation batches gained
the predicted columns on every pass and recompiled the eval bodies
(`ResidentValidation` now hands out a shallow copy per pass); a label-free `"tables"`
objective was fixed with branches, replaced by `Rondo.tables` (no energy written) and
scorers chosen once at construction (operator rule: no `if` / `try` on the hot path);
MolHub wrote `propers` only for molecules with torsions, so a large batch mixed two
schemas — the block is now always written. Next: a warm v5 prepared-row snapshot, then
the sweep again at steady state, then gate 6.

Still to do: gate 5 (GH200 calibration
of the joint batch at the 50 Å published cutoff — the pair cap, not the atom cap,
binds), gate 6 (smoke on GH200). The GAFF2 census is redone (above).

2026-10-29 afternoon: evidence directory
`workspaces/lab-new/runs/rondo-enc-phase2-20261029T1342Z/`; the steady-state sweep (job
4610664, 16k–262k atoms / source) waits on the warm snapshot 4605616. `heads.py` gained
`--resume` (fails closed without `last.pt`, records `resumed_from_step`) and the sign-off
check; tested on CPU (job 4610690).

**Operator ruling 2026-10-29 (supersedes §6.1's replication and ensemble):** quick
experiments, not a publication — member-01 only, no member-02 repeat, no five-member
average. Production runs: D1, D2, D1+D2+FT (plain D1+D2 was measured at gate 4 and
missed; not rerun). B is scored on the full validation split, not trained. Each run is
capped at 2 h (`HEAD_SWAP["seconds"]`); patience ends most first. Item 5 per run at the
chosen capacity is one GPU job (`phase2/cells.sbatch`), beside the smoke.

Found by the resume test (4610690) and fixed in molix: a baseline `evaluate()` ahead of a
resumed `train()` fired `CheckpointHook`, which wrote the unrestored step-0 state over
`last.pt` (and, its best not yet seeded, over `best.pt`) before `train()` read it back —
the "resume" restarted at step 0 and looked like it worked. `Trainer` now restores at the
first `evaluate` or `train`; the hook seeds its best from the restored checkpoint before
any selection (`test_evaluate_before_train_does_not_clobber_the_resumed_checkpoint`).
`heads.py` records the step the loop starts from and refuses a resume that did not land
there; the dead `register_artifacts` path (`trainer.ctx` never existed) left both hooks.

Gate 5, steady state (GH200 4610664, warm v5 snapshot 4605616 — 72 GiB, 24 min; D1+D2+FT,
60 steps per cell, recompiles forbidden):

| atoms / source | step | compute | data wait / step | peak HBM | atoms / s |
|---|---|---|---|---|---|
| 16 384 | 0.19 s | 0.13 s | 0.03 s | 3.7 GiB | 341 k |
| 32 768 | 0.28 s | 0.14 s | 0.05 s | 6.7 GiB | 469 k |
| 65 536 | 0.51 s | 0.23 s | 0.13 s | 13.0 GiB | 511 k |
| 131 072 | 1.00 s | 0.45 s | 0.29 s | 25.2 GiB | 523 k |
| 262 144 | 1.98 s | 0.86 s | 0.61 s | 50.3 GiB | 531 k |

The warm cache took the data wait from 4–5× compute to a fraction of it. Chosen capacity
**32 768**: 1.4× the throughput of 16 k, within 10 % of the plateau, and ~25 k steps in the
2 h budget. Per-run cells (`cells.sbatch`, 4614016) and the smoke + B score (4614017) at it.

**Validation is a prefix (by design, `JointDataModule.val_dataloader`: zip to the shortest
val loader).** At 32 768 the mbis val set is ~15 batches, so every training eval — and the
`best.pt` choice — scores the same first ~15 batches of each source: all 19 026 mbis
molecules but 819 of 4 054 torsion scans and 280 of 4 304 relaxations. The `eval_relax`
gap (B 4.89 vs untrained 1.67–2.08) is on that prefix. The verdict is read from
`phase2/score.py` instead: each source's own val loader to the end, per molecule (charge
MAE; offset-free profile RMSE, plain and loss-weighted; per-conformer force RMSE), as
median / p90 / p99 / mean in `scores.json`. B on the prefix: charge 0.111 e, torsion
median 0.79 kcal/mol, relax energy median 0.58 kcal/mol, relax force median 5.1 kcal/mol/Å.

## 8. Result (2026-10-29, member-01, held-out by molecule, full val sets)

Production runs (GH200, capacity 32 768, signed off): D1 4614292 (3 200 steps, 15 min),
D2 4614293 (11 200 steps, 33 min), D1+D2+FT 4614294 (6 200 steps, 35 min) — all ended by
patience; frozen encoder and bonded heads bit-identical in every run. Scored by
`phase2/score.py` (4614474 B, 4614966 runs) on every val molecule: 19 026 mbis,
4 054 torsion scans, 4 304 relaxations (271 427 conformers). Median (p90); `last.pt`,
which is within noise of `best.pt` everywhere and better on the charges:

| run | charge MAE, e | torsion RMSE | torsion, weighted | relax E RMSE | relax E, weighted | relax F RMSE |
|---|---|---|---|---|---|---|
| B | 0.111 (0.141) | 0.789 (2.06) | 0.560 (1.28) | 0.589 (1.86) | 0.269 (0.75) | 5.02 (10.4) |
| D1 | **0.018** (0.027) | 0.898 (2.23) | 0.643 (1.46) | 0.571 (1.83) | 0.272 (0.79) | 5.31 (10.7) |
| D2 | 0.111 (0.141) | **0.691** (1.56) | **0.496** (1.05) | **0.449** (1.42) | **0.229** (0.59) | **4.85** (10.0) |
| D1+D2+FT | **0.017** (0.026) | 0.985 (2.26) | 0.643 (1.25) | **0.430** (1.38) | **0.237** (0.57) | **4.46** (9.1) |

Energies kcal/mol (offset-free per profile), forces kcal/mol/Å. Means show the tail:
relax E 11.2 (B) → 1.18 (D2) → 0.85 (D1+D2+FT); relax F 23.1 → 6.4 → 5.6 — the published
direct σ / ε carry a few relaxations with very large errors that the GAFF2-windowed head
does not.

Verdict against §6's margin (within 10 % of B on torsion, relative energy, force):

- **D2 replaces the LJ head: yes.** Better than B on every metric and quantile; the
  relaxation tail collapses.
- **D1 replaces the charge head: no, not under frozen proper amplitudes.** Charges 6× closer
  to MBIS, relaxations unchanged (energy −3 %, force +6 %), but torsion +14 % median
  (+15 % weighted) — the coupling §6 predicted: ByteFF-26's amplitudes were fitted against
  its own BCC electrostatics.
- **D1+D2+FT:** the best relaxations of any run (energy −27 %, force −11 %, tails gone) with
  MBIS charges, but torsion +25 % median / +15 % weighted median (p90 weighted −2 %). The
  constrained fine-tune (proper output layers at 2e-5) did not recover the torsions within
  6 200 steps; patience was on the summed eval loss, which the relaxations dominate, so the
  run stopped on relax, not on torsion.

Next, if the charge swap is pursued: give the torsion its own stopping signal (or weight)
and a larger proper-amplitude rate, or retrain the proper head output fully on the
torsion scans with D1 + D2 fixed. Training-time validation scores a fixed prefix of each
source (above); `score.py` is the measurement of record.
