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

## 2026-10-29 — rondo-enc phase 2 (D1 / D2 / D1+D2+FT, member-01): GO

Spec `rondo-enc-03-heads` §5. Evidence root (one timestamped dir):
`workspaces/lab-new/runs/rondo-enc-phase2-20261029T1342Z/` — `calibration/`, `smoke/`,
`signoff.json`, `runs/<RUN>/`. Scope (operator 2026-10-29): quick comparison, member-01
only, 2 h per run; not a converged or ensembled fit.

1. Data — THEMol mbis / torsion-scan / hessian-relax / hessian HDF5 shard-000 through
   MolHub on molpy 0.20; column parity 2998/3000 vs the ByteFF oracle (phase 1); split by
   molecule (95/5); units from the manifests; censuses exclude bond-stretch and
   GAFF2-untyped rows (logged per source).
2. Reference vs packed — gate 2 passes every run, CPU fp64 and GH200 (4607781).
3. Frozen is frozen — encoder + bonded fingerprints bit-identical in every run so far.
4. Overfit — gate 4 passes (D1, D2, D1+D2+FT with r0/θ0 open).
5. Calibration — warm-cache sweep 4610664 (16k–262k atoms/source); capacity 32 768;
   per-run cells 4614016 (D1 0.20 s, D2 0.23 s, D1+D2+FT 0.28 s per step, ≤ 6.7 GiB).
6. Smoke — 4614017: 200 steps, fresh-process resume at step 200 to 300, eval reports,
   scores fall after warmup, frozen unchanged. Found and fixed a resume that silently
   restarted at step 0 (Trainer / CheckpointHook, spec 03).
7. `signoff.json` pins per run the configuration and per-source padded caps;
   `phase2/heads.py` refuses a production run that does not match it.

## 2026-10-11 — Rondo joint training (ByteFF realignment): what is verified, what is not

The sequential six-stage curriculum this checklist was last signed against no
longer exists. Rondo is `Rondo(ff, spec, wrt=)` on the molrs force-field IR,
trained by `molix.core.steps.JointTrainStep` over three joint stages
(`configs.CURRICULUM`: `s1-hessian-mbis` → `s2-torsion` → `s3-relax`), each
step drawing one batch per source and summing per-objective losses whose
`trainable` sets detach every other table. Plan `rondo-byteff-squishy-iverson`.
Every checkpoint written before 2026-10-11 is invalid.

- [x] Environment: molrs/molpy **0.17 from source** (`molrs-rondo` / `molpy-rondo`
  worktrees at origin/dev f786a891 / f7ddb1e), `.venv-x86_64` rebuilt
  (`build_env.sh --fresh`, `ONLY=` reinstall mode for pure packages); molnex
  fast tier 3427 passed, slow compile tests (Rondo forward / energy /
  partial_hessian; joint bodies) 1 graph / 0 breaks. aarch64 env: job 3676710
  (GH200 n570, 15 min, same molrs/molpy revisions; the 0.15 env kept as
  `.venv-aarch64-015`).
- [x] Data contract: molhub THEMol frames carry `formal_charge`,
  `ring_connectivity`, `min_ring_size`, `gaff2_sigma`, `gaff2_epsilon`,
  `bond_type`, `in_ring` (verified on a real hessian record, 0.5 s decode;
  `AtdTypifier("gaff2")` construction 3.6 s, per-thread cached); impropers
  carry three cyclic rows per centre (`label_sample`).
- [x] Item 2 (parity): style potentials against the molrs oracle
  (`ExplicitTerms`) to 1e-12; forces = −∇E; partial Hessian = full Hessian on
  bonded pairs; mirror / rigid-motion / improper-leg symmetries (`tests/test_molzoo/test_rondo`).
- [x] Bounded real-data smoke (login node, CPU, 8 hessian + 8 MBIS molecules,
  capacity 4096, `small` tier, 2026-10-11 15:28): compiled joint step, 3 steps,
  eval, checkpoint; start losses hessian_k 0.41 / hessian_geom 1539 (uniform
  field at QM minima, before any r0/θ0 learning) / mbis 0.052. Found: an
  atom GAFF2 cannot type (Si in the 4th hessian row) makes the LJ energy NaN by
  design — the GAFF2 census (`Gaff2Census`, `bond_census.py`) must exclude those
  rows; the smoke bypassed it and got a NaN loss on step 0.
- [x] Censuses for all four sources (job 3678026, cpu, 6.5 min; files under
  `workspaces/lab-new/runs/bond-census/*.json`, fingerprinted): bond stretch /
  GAFF2 untyped — hessian 31 / 6 908 of 61 796 (11.2 %), mbis 200 / 50 397 of
  430 906 (11.7 %), torsion-scan 80 / 2 780 of 83 951 (3.3 %), hessian-relax
  2 922 / 0 of 89 161. The GAFF2 exclusions are the ByteFF element filter in
  practice: a direct scan of the first 300 rows finds the hessian shard
  carrying B, P, Si (46/300 rows untyped) while the relax shard's molecules
  are all {H, C, N, O, F, S, Cl, Br, I} (0/300) — its zero is real.
- [x] Item 3 per objective, 8 GAFF2-typeable molecules, train = val, 300 Adam
  steps at lr 3e-3, `small` tier, CPU job 3678072 (`.jobs/overfit/`):
  `hessian_k` partial-Hessian 0.49 → 0.44 with bond K differentiating from the
  uniform 450 (p10 213 / p50 361 / p90 383) — the ~0.4 floor is the Class-I 1-3
  coupling the form cannot express; `hessian_geom` stationary-force 1539 → 18
  (r0 and θ0 move, K held at 450/40 by construction); `mbis` charge MSE
  0.052 → 2.8e-4; `torsion` proper |k| median 2.8 / max 28.5 kcal/mol after
  300 steps (train 3.4, eval 8.8 — the gradient path is live; the amplitude
  ceiling is a stage-2 calibration question); `relax` (job 3678323, capacity
  16384 — a hessian-relax record is a whole relaxation, 5460 atoms as one
  sample): loss 0.56 eval / 0.91 train, gradient path live, **and trained alone
  it collapses K exactly as every earlier chain did** (bond K 450 → p50 8.6,
  angle K 40 → 1.0 in 300 steps): relaxed-frame E/F cannot identify stiffness.
  That is the finding behind the joint design — in `s3-relax` the
  `hessian_k` objective is in every step and the handoff gate's K bounds
  ([0.3×, 3×] of k_ref) would refuse a collapsed checkpoint. Never run
  `relax` as a lone objective.
- [x] Item 6 on the exact launcher path (`term_curriculum.py small --stages
  s1-hessian-mbis --max-steps 10 --capacity 4096 --device cpu`, CPU job
  3679652, 8 min, run `workspaces/lab-new/runs/dryrun-s1-cpu-3679652`): both
  censuses → molecule split (hessian 52 136 / 2 752, mbis 361 483 / 19 026) →
  `stage()` (14 loader workers, 7 per source) → `compile(fullgraph=True)` →
  smoke → baseline eval → 10 steps → `best.pt` → `parameter_report` on real
  nested val batches → `handoff_failures`. Baseline val losses hessian_k 0.49 /
  hessian_geom 2072 / mbis 0.090. The report reads as the untrained field
  should: every bond K 449.8 ± 0.1, angle K 40.02, θ0 by centre hybridisation
  (C4 109.5°, C3 120°, **C2 177.5° against 179.2° measured**), r0 per element
  pair from the covalent radii, proper curvature 0, improper |k| 1e-4, LJ
  log-ratio 0, charge MAE 0.24 e, feature RMS 1.00 on every stream. The gate
  passes on it (its bounds catch collapse and runaway, not "unchanged").
  `--capacity` / `--device` were added to the launcher for this; production
  leaves both at the tier's capacity and `cuda`. Per-record decode measured on
  the login node: hessian 74 ms, mbis 39 ms, torsion-scan 295 ms,
  hessian-relax 168 ms (the 0.5 s above was a first cold call).
- [x] **S1 small, 1500 steps on GH200 (job 3680093, n526, 2026-10-11 18:12-18:37;
  run `workspaces/lab-new/runs/rondo-joint-s1-small-20261011`).** 1.91 steps/s
  at capacity 131072 per source (4 batches per step), peak GPU 8.58 GiB, 13 min
  of training after an 11-min smoke (first-pass decode + compile). Train losses
  at 500 / 1000 / 1500: hessian_k 0.438 / 0.440 / 0.448 (flat), hessian_geom
  27.65 / 24.18 / 23.42, mbis 0.0197 / 0.0106 / 0.0088; grad norm 32 / 110 / 85
  against clip 10 (always clipped). **Found**: every eval "missing" — the
  hessian source sits under two objective names and a persistent-worker
  DataLoader hands both names one iterator, so they split its batches; the val
  set packs into one batch at this capacity, the second name got
  `StopIteration`, the joint val loader yielded nothing, no best.pt, exit 1.
  Fixed in `molix.data.joint._Lockstep` (one iterator per distinct loader,
  shared names get the same batch, each a copy — ByteFF's step) with a unit
  test; `fit()` now fails closed on a scoreless baseline eval. Report built
  from `last.pt` (CPU job 3680621, `.jobs/report-x86.sbatch`,
  `param_report_last1500.json`): **gate passes**. Bond K differentiated by
  chemistry (p50: H1-C4 353, C3-C3 355, H1-C3 376, C4-C4 197, C3-C4 257, C3-N3
  285, C3-O1 724, H1-N3 302; ref 450), r0 chemically right (H-C 1.09, C4-C4
  1.52, C3-C3 1.39, C=O 1.23), angle K 39-72 by centre (ref 40), θ0 by centre:
  **C4 99.7° against 109.6° measured (9.9°, at the gate's 10° limit)**, C3
  126.9 vs 120.1, N3 118 vs 118, S4 102 vs 108, N4 103 vs 109, C2 179.5 vs
  178.9 (the sp centre that every earlier chain failed is right from step 0).
  Charge MAE 0.061 e (untrained 0.24). Improper |k| per row p50 2.4 / p90 9.4.
  LJ log-ratio 0 (no LJ loss in S1, by design). Feature RMS 1.0-1.3. Open:
  hessian_k does not fall in 1500 steps; sp3 θ0 sits 10° under measured while
  the geometry loss is still falling — both are what the rerun (3680577, with
  eval, best.pt and plateau schedule) and a longer S1 must answer.
- [x] **S1 small rerun with the loader fix (job 3680577, n208, 18:34-18:56; run
  `runs/rondo-joint-s1-small-20261011b`)**: eval scores every 200 steps now.
  Eval loss 2050 → 827 → 74.1 → 53.6 → 44.1 → 40.9 → 39.1 → 38.1 → 37.97 at
  step 1500 (hessian_geom 2049 → 37.5, still falling ~1 per 200 steps;
  hessian_k 0.491 → 0.4377 at step 800, then creeping back to 0.449; mbis
  0.0906 → 0.0087); train hessian_geom 23.4 against eval 37.5. 2.50 steps/s,
  peak GPU 18.6 GiB (the shared hessian batch is now cloned for the second
  name), 10 min of training. Report from best.pt (= step 1500): same tables as
  the first run within 1 % (bond K H1-C4 353 / C3-C3 358 / C4-C4 197 / C3-O1
  722, angle K 39-72, charge MAE 0.060 e, C2 θ0 179.5° vs 179.0°).
  **Handoff gate FAILS on one check: θ0 C4 median 99.1° against 109.6°
  measured (10.5° > 10°)**; every other check passes. Not handed on. Open
  questions for the next (longer) S1: whether sp3 θ0 keeps closing or turns
  back once the geometry loss flattens (torsions are zero in S1 by design, so
  the angle term is the only bonded term left to absorb what they would
  carry), and whether hessian_k's rise after step 800 is the two hessian
  objectives trading through the shared encoder.
- [x] **S1 small, unbounded, two clips (user ruling 2026-10-11 ~22:35: "两个都提，
  clip 放到 50")**: `term_curriculum.py --clip NORM` and `CLIP=` in
  `run_term_curriculum.sbatch` added (default stays `configs.CLIP_MAX_NORM`
  10). No `MAX_STEPS`: plateau (factor 0.5, patience 5 evals), early stop
  (patience 50 evals), or the 4 h stage clock ends the stage. Jobs 3685025
  (clip 10, `runs/rondo-joint-s1-small-20261011c-clip10`, n159) and 3685026
  (clip 50, `runs/rondo-joint-s1-small-20261011d-clip50`, n204), both
  `--time=04:45:00` on naiss2026-1-53-gpu, started within a minute. What the
  pair answers: where sp3 θ0 settles once hessian_geom flattens (gate needs
  C4 within 10° of 109.6°), whether hessian_k keeps creeping up, and whether
  letting grad norms 85-135 through (clip 50) changes either.
  **Result (2026-10-12 00:08 / 00:05)**: both early-stopped on patience (50
  evals past best): clip 10 best step 5600 (eval 32.06, hessian_geom 31.6),
  stopped 15600, 1 h 31; clip 50 best step 4800 (32.12), stopped 14800, 1 h 28;
  3.4 steps/s, peak 18.6 GiB. Past the best the eval geometry loss turns
  noisy and rises (40-59 at steps 14400-15000 for clip 10, 45 for clip 50)
  **while plateau had cut lr to 1e-6** — that swing cannot be weight drift and
  is unexplained (candidate: an ill-conditioned part of the geometry
  objective, e.g. the EEM solve on a few hessian-val molecules). hessian_k
  0.437 → 0.46 in both; mbis 0.0087 → 0.0037/0.0043 (charge MAE 0.044/0.046
  e). **Gate fails on more checks than at 1500 steps**: every sp3 centre's
  θ0 median dropped a further ~4° (C4 95.4/95.5°, S4 95.0/100.4°, N4
  96.1/98.3°, P4 93.4/94.3° against measured 109.6/108.2/109.1/107.5), while
  the two runs disagree by 10° on sp2 C3 θ0 (117.0 vs 127.5, measured 120.1)
  and C2 stays 179.5°. Bond K/r0 unchanged within a few % (H1-N3 K 295 →
  464/482 the one mover); angle K unchanged except C2 39.9 → 21/23 (n=216);
  improper |k| p50 2.9 → 6.2 (clip 10) / 2.0 (clip 50). Clip 10 vs 50: no
  difference in any table or in the trajectory (grad norms 80-230 both; one
  1284 spike on clip 50 at step 8000). **Diagnosis**: at a symmetric centre
  (planar trigonal, tetrahedral) a common shift of every θ0 produces zero
  net force on each ligand, so the Force→0 loss on minimum geometries only
  determines θ0 *differences* between inequivalent angles; the common mode is
  a null direction (exact for sp2 — hence 117 vs 127.5 — and near-null for
  sp3, with a consistent weak downward push from whatever residual forces the
  angle term is asked to absorb). hessian_k detaches θ0, so the Hessian's
  (θ−θ0)·∂²θ term that would pin it never reaches it. ByteFF's
  `Partial_Hessian_MAPE` and `Force_MSE` train exactly our sets (`loss.py:319,
  370`), but ByteFF also has `Angle_Energy` / `Bond_Energy` loss types
  (`loss.py:302-314`: the term's energy on the QM-minimum geometry / n_terms,
  trainable θ0 / r0 — the K-weighted regression of θ0 onto the measured
  angle) and a `skip_large_nb` mask on `Force_MSE` (molecules whose
  nonbonded forces exceed a threshold leave the force loss). Proposed next:
  add an `hessian_eq` objective (angle + bond energy at the minimum →
  `angles.theta0`, `bonds.r0`) to S1 and re-check the gate at 1500 steps
  before any longer run. **User ruling 2026-10-12: "加 hessian_eq 和 skip_large_nb 一起上，1500 步看门".**
- [ ] **hessian_eq + skip_large_nb (2026-10-12)**. Code: `Rondo.energy(frame,
  tables=None, styles=None)` sums a named subset of styles (ByteFF
  `calc_terms`; A18); `JointObjective(styles=, nonbonded=)` — `styles` restricts
  the `energy` / `forward` body, `nonbonded` makes the `forward` body write the
  pair styles' own force to `atoms.forces_nonbonded` first (every table
  detached, under `no_grad`, so no graph is kept); `StationaryForceLoss(
  skip_above=, skip_key=)` zeroes every row of a molecule in which any
  non-bonded force component exceeds the threshold (per-graph `index_add` +
  gather, static shapes; divisor counts kept rows only, unlike ByteFF);
  `MeanTermEnergyLoss(namespaces=("bonds","angles"))` = Σ energy of the priced
  styles / Σ live terms (pooled over the two namespaces — ByteFF normalises
  each type in its own loss; one body here). Lab: `OBJECTIVES["hessian_eq"]`
  (hessian source, `output="energy"`, `styles=(bond:harmonic, angle:harmonic)`,
  trainable `bonds.r0`/`angles.theta0`, weight 1), `hessian_geom` gains
  `nonbonded=(pair:lj/cut, pair:coul/cut)`, every stage lists `hessian_eq`;
  `STATIONARY_FORCE_SKIP_ABOVE` = 50 kcal/mol/Å, read off the census (job
  3688938, 1 min 41 on a CPU node, `.jobs/nb_force_census.py` on the clip-10
  best.pt, the whole hessian val set = 2752 molecules, report
  `runs/rondo-joint-s1-small-20261011c-clip10/s1-hessian-mbis/nb_force_census.json`):
  per-molecule max |F component| of the pair styles alone p50 10.8 / p90 20.9
  / p95 26.3 / p99 40.6 / p99.9 72 / max 279; LJ alone max 283, Coulomb alone
  max 34 (the tail is the LJ wall); every style with the S1 parameters p50
  17.7 / p90 33.5. Fraction of molecules above 20 / 30 / 50 / 100: 11.5 % /
  3.5 % / 0.4 % / 0.07 % → 50 removes the wall tail and nothing of the bulk.
  Tests: full fast tier 3443 passed / 15 skipped (login node needs
  `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 -n 6`: at 48 BLAS threads per
  xdist worker it hits the thread limit and dies with INTERNALERROR); slow
  explain green for the two new body shapes (forward+nonbonded, energy+styles)
  and `Rondo.energy(styles=)`. CPU dry run of the exact launcher path with 4
  bodies: job 3688940 passed (7 min 25; baseline eval hessian_k 0.4907 /
  hessian_geom 2047.7 / **hessian_eq 1.123 kcal/mol per term** / mbis 0.0902;
  10 steps; best.pt; report; gate). **S1 small 1500 steps submitted: GPU job
  3689502, `runs/rondo-joint-s1-small-20261012-eq`, `--time=01:00:00`,
  naiss2026-1-53-gpu.** The gate on its best.pt is the question: does sp3 θ0
  stay within 10° of 109.6° with the strain objective in the step.
  **Result (n463, 00:41-01:05, 24 min 41, exit 0): handoff gate PASSES.**
  Eval 2049 → 836 → 74.3 → 49.7 → 42.8 → 39.6 → 37.7 → 36.7 → 37.9 at step
  1500 (hessian_geom 37.1 — same as the baseline's 37.5, so the strain term
  costs the force objective nothing; hessian_k 0.4905 → 0.4352 **falling
  monotonically**, no turn-up — the baseline bottomed at 0.4377 and rose to
  0.449; hessian_eq 1.124 → 0.369 kcal/mol per term; mbis 0.0082). Train loss
  24.2, grad norm 58 / 114 / 262 at steps 500 / 1000 / 1500 (clip 10). 2.24
  steps/s (4 bodies, was 2.50 with 3), peak GPU 22.9 GiB (was 18.6).
  **θ0 medians (eq vs measured)**: C4 107.8 vs 109.6 (p10-p90 105.6-110.8 vs
  106.6-112.9), C3 119.9 vs 120.1, N3 117.5 vs 117.9, N2 112.8 vs 115.3, O2
  109.2 vs 112.3, S4 106.8 vs 108.2, N4 110.4 vs 109.1, P4 110.8 vs 107.5, S2
  96.3 vs 91.9 (n=380), C2 179.5 vs 179.0 — every centre within 5° except S2
  (4.4°); at 1500 steps the baseline had C4 99.1 / C3 125.7 / S4 101.0 / P4
  105.1. Angle K 44-69 (C2 back to 43.7 from the long runs' 21). Bond K / r0
  unchanged from the baseline within a few % (H1-C4 360 / C3-C3 359 / C4-C4
  202 / C3-O1 713; r0 to the thousandth). Charge MAE 0.059 e, improper |k| p50
  4.7, feature RMS 1.0-1.3, LJ untouched. Report:
  `runs/rondo-joint-s1-small-20261012-eq/s1-hessian-mbis/param_report.json`.
  Still open: whether the geometry eval keeps falling past 1500 steps now that
  θ0 is pinned (the baseline's post-5000-step rise and the lr-1e-6 eval swings
  are not yet re-examined), and the skip fraction is not logged (the loss
  cannot write state; a hook reading `atoms.forces_nonbonded` could).
- [ ] **Architecture diagnostics (2026-10-12, CPU job 3691055, 5 min,
  `.jobs/arch_diagnostics.py` on the eq best.pt over the whole hessian val
  set; JSON beside the report)**. (A) The partial-Hessian loss 0.435 is **97 %
  1-3 blocks**: 1-2 rows (176 k) residual 0.0385 (|trace| p50 938), 1-3 rows
  (309 k) residual 0.66 (|trace| p50 26; C-C 1-3 = 1.00, H-C 0.56). The bond
  stiffnesses are fit to 4 %; the 1-3 blocks — angle term only in S1, the QM
  block also carries torsion and through-space stiffness — are the plateau,
  and they are what the angle K is fit against. Loss-form / FF-form, not
  network. (B) Force loss per molecule: p50 26.7, p90 76, p99 190, max 1839;
  top 1 % of molecules carry 9 %, top 10 % carry 33 % — a broad distribution,
  not outliers. **The "train 24 vs eval 37 gap" is the knee**: train reports
  `log_knee(mean, 10)`, 24 ⇔ raw mean 40.6; eval reports the raw mean 37.4. No
  generalisation gap. (C) EEM is well conditioned: η on val atoms p0.1 1.63 /
  p50 3.84 / p99 4.97, none near the 1e-3 floor, max sensitivity 1/η 0.7;
  |q| p50 0.16, p99 0.78, max 1.68. (D) Resolution of the 2-hop encoder:
  bonds 17 k rows → 4.6 k distinct feature vectors, measured-length variance
  explained 0.999 (within-group std 0.006 Å vs 0.195 total); angles 30 k rows
  → 18 k distinct, variance explained 0.991 (within 1.15° vs 7.7° total). The
  receptive field resolves what the S1 data distinguishes. (S) clip-10 run
  best (5600) → last (15600) on val: 31.6 → 40.8; the worst molecules went
  15-38 → 190-260 (alkyl-rich, the sp3 θ0 drift); per-atom charge moved p50
  0.08 e / max 0.29 e between the two checkpoints while mbis val stayed
  0.004 — the charges are underdetermined on hessian-set chemistry by the
  mbis set alone (noted, not acted on). The lr-1e-6 eval swings stay
  unexplained beyond "a strained state is hypersensitive"; watch the eq long
  run (3690962) for a recurrence.
  **Untrained baseline (job 3691174, the 10-step dry-run checkpoint, 3 val
  batches)**: 1-2 residual 0.091 → 0.0385 trained (bond K learned, 2.4×),
  1-3 residual 0.719 → 0.661 (8 %: the angle term's block structure cannot
  follow the QM 1-3 block, so the Hessian objective identifies angle K only
  weakly). Resolution is the same untrained (bonds 0.999, angles 0.988): the
  input channels + 2-hop topology already separate the geometric classes; the
  test says the receptive field suffices, not that the trained features are
  good. Untrained η sits at 10.0 (sigmoid mid-range), trained 1.6-5.0.
  **Architecture review written up**: `.claude/notes/rondo-architecture-review-2026-10-12.md`
  — no defect behind the S1 symptoms; four real issues for S2/S3 (contexts
  blind to their term's bonds; 100-1000× head-gain spread through the shared
  trunk; diagonal EEM non-local / zero start / dead Coulomb gradient at init;
  unbounded K), with fixes and diagnostics. Operator decision pending.
- [ ] **S1 small unbounded run with eq + skip (2026-10-12, GPU job 3690962,
  `runs/rondo-joint-s1-small-20261012-eq-long`, clip 10, patience 50 evals,
  2 h 02 min, 19 600 steps, 3.13 step/s, 22.9 GiB) — gate FAILED on best.pt
  (step 10 000)**: θ0 N3 median 129.3° vs measured 117.9° (per-angle p10 60.8° /
  p90 157.1°), P4 97.1° vs 107.5°; every other centre inside the gate (C4 107.8,
  C3 120.3, N2 112.1, O2 109.2, S4 105.0, N4 107.0, S2 96.0, C2 179.5, S3
  115.8). Trajectory: geometry eval fell 35.0 (step 1 600) → 28.2 (9 600; best
  total 29.4) while hessian_eq **rose** 0.358 → 0.758 — the optimizer traded
  equilibrium strain for force; past 10 000 both train (knee'd geom 20.4 →
  25.9) and eval (29 → 37-49) got worse as the lr halved 1.5e-4 → 5.9e-7, and
  the eval only settled (37.17 → 37.46 over 1 200 steps) at lr ≤ 1.2e-6.
  hessian_k 0.4905 → 0.4317 (flat from step 4 000); mbis 0.0035; charge MAE
  0.039 (1500-step run: 0.059). Bond k moved on the rare pairs between the two
  runs (C-Cl 327 → 189, C-F 376 → 232, H-N 301 → 507, C≡N 934 → 1316); angle k
  at C2 collapsed 43.7 → 14.2 (linear-angle Hessian degeneracy). Patience 50
  evals = 10 000 steps past the best were wasted.
  **Diagnostics (CPU jobs 3693119 / 3693135, `.jobs/skip_theta_diag.py`,
  `.jobs/n3_force_decomp.py`, JSON beside the report)**: (1) the skip mask is
  not the swing — best and last skip the same 10 of 2 752 val molecules, no
  flips, 24 sit in the 40-60 band; the trainer loss 28.23 → 36.31 (+8.07) is
  entirely on the common kept set, so the model genuinely worsened. (2) θ0 by
  centre on best.pt: C3 / C4 / N4 / S4 are shifted > 10° at ≤ 5 % of centres
  (within-centre std 1.5-2.2°). **N3 is the pathology: 3 201 val centres, 67 %
  shifted > 10°, half of those common-mode (the three θ0 together, within-centre
  std 2.6°)**. Planar N3 (Σθ ≥ 355°, 2 164 centres — amide / aniline): θ0 median
  131° vs 120°, 30 % above 140° — the exact force null mode, held by hessian_eq
  alone. Pyramidal N3 (1 037 — amines): bimodal, 46 % below 90° (the worst
  centres have all three θ0 at the 0.5° clamp floor), 26 % above 140°, eq
  strain p50 83 kcal/mol per centre — a common θ0 shift at a pyramidal centre
  is an axial force, so it is cancelling another style's axial force there. P4:
  32 centres, θ0 97° vs 109°, not common-mode (std 7°). The pooled normalisation
  of `MeanTermEnergyLoss` gives the 6 % of angles at N3 and the 0.1 % at P4 a
  proportionally tiny pull, and unlike ByteFF's tanh window θ0 has no
  chemically bounded range (only the 0.5-179.5° clamp). (3) axial-force
  decomposition at N3 (job 3693135, `.jobs/n3_force_decomp.py`): **the
  improper is the culprit**. Every N3 centre carries an improper (ByteFF's
  rule: C/N with three neighbours). At pyramidal N3 on best.pt the improper
  alone pushes the N along the pyramid axis by |F| p50 116 kcal/mol/Å and the
  angle term cancels it (|F| p50 108): on the 477 centres with θ0 ≪ measured
  the medians are improper +139 / angle −138 (total +6); on the 304 with θ0 ≫
  measured, improper −92 / angle +86 (the anti-planar sign the signed cvff
  amplitude allows). Bonds, LJ, Coulomb are all < 6. At planar N3 the
  improper force is < 1 and the θ0 drift there is the pure null mode. Cause:
  `hessian_k` trains `impropers.k` on curvature only, `hessian_geom` cannot
  reach it, so the force objective's only handle on the improper's force at a
  pyramidal minimum was θ0. ByteFF has a third equilibrium objective,
  `ImproperTorsion_Energy` (`loss.py:302-314`, improper energy on the minimum
  trained through the improper k), which I had not ported. **Fixed 2026-10-12**:
  `OBJECTIVES["hessian_eq_improper"]` (hessian source, energy body, styles
  `improper:cvff`, trainable `impropers.k`, weight 1) in all three stages;
  `OBJECTIVE_LOSSES["term_energy_improper"]` =
  `MeanTermEnergyLoss(namespaces=("impropers",))`. **The first CPU dry run of
  the five-objective config (3693163) died on `Dynamo recompile limit exceeded`
  at the baseline eval**: dynamo keys its cache on the function's code object,
  and every `energy_body` closure in `molix.core.steps.joint` shared one, so K
  energy objectives × (train, eval) looked like K recompiles of a single
  function (guards on the closure and on the sub-batch's key count, which
  grows as earlier bodies write their columns); the 4-objective runs sat at ≤ 8
  entries silently and the fifth objective was the ninth. Fixed in
  `joint.py::_own_code`: each body is handed back under a renamed code object
  of its own (`energy_body[mbis]`), one cache per objective per step
  (`test_joint.py::test_each_body_owns_its_code_object`; fast 33 + slow 5
  green). Dry run 3693225 (CPU, 8 min) passes the whole launcher path with the
  five objectives: baseline eval `hessian_eq_improper` 0 at init (zero-init
  amplitude), −3e-6 after 10 steps (roundoff of a non-negative form), gate
  report builds. **Operator 2026-10-12: "提，patience 改 10"** → `PATIENCE`
  50 → 10 evals (2 000 steps; the plateau scheduler needs 5) in `configs.py`,
  and GPU job 3703994 submitted (`runs/rondo-joint-s1-small-20261012-eqimp`,
  SIZE=small, STAGES=s1-hessian-mbis, MAX_STEPS=1500, default clip 10,
  account naiss2026-1-53-gpu, 2 h wall; the earlier 1500-step gate took
  25 min). **Result (11:20): gate fails, worse** — N3 θ0 163.0° vs 117.9°
  (p10 149°), N4 121.5° vs 109.1°, P4 120.7° vs 107.5°; stopped at step 1400
  by the 10× divergence guard (eval 7780, best 54.7 at step 800). The new
  objective itself is the cause: `hessian_eq_improper` went 0 → −0.004 →
  −0.107 → −0.38 → −0.75 → −1.06 → −1.46 → −4.41. The improper strain
  `k (1 − cos 2ψ)` is linear in `k`, and `CvffHead` predicted a **signed** `k`
  from a zero start, so minimising the strain has no floor: `k` runs negative
  wherever ψ ≠ 0, the improper force grows (|k| p90 9.5 → 34.8), the angle θ0
  at every N3 rises to cancel it, and the force eval blows up (94.9 → 1123 →
  7780 from step 1000; hessian_k 0.443 → 0.556). The dry run's −3e-6 after 10
  steps was the first sign of it, misread as roundoff. ByteFF's `k` is
  `4.29 + 3.71·15·tanh(·)` under a configurable `param_range` straight-through
  clamp (`mlparams.py:104-110`); the repository has no training config, so the
  floor it trains with is not documented — but the loss is bounded only with
  one. **Fix (A20, uncommitted)**: `CvffHead` predicts `k_ref · exp(raw)`
  (`heads.improper_k_ref` = 4.291/3 per row, ByteFF's GAFF2 mean per centre
  over three rows), `impropers.k_ref` is a new column;
  `test_classical.py::TestCvffHead` (starts on the reference, stays positive
  for raw = −3); fast tests of heads / rondo / joint 86 green; slow compile
  tests (rondo `test_compile.py` + joint bodies) 10 green. Dry run 3704940
  (CPU, 8.5 min): `hessian_eq_improper` starts at +0.0316 kcal/mol/term (most
  impropers are planar) and falls to 0.03157 after 10 steps, improper |k|
  1.43 at init, report builds, gate passes. GPU gate run resubmitted as
  3705406 (`runs/rondo-joint-s1-small-20261012-eqimp2`, same recipe).
  **Result (3705406, 25 min, n161): gate passes.** N3 θ0 116.5° vs 117.9°
  (p10 109° / p90 123°), N4 110.4°, P4 110.3°, C4 108.1°, C3 121.0°, S4
  107.4°, S2 97.5° (measured 91.9°), C2 179.5°. `hessian_eq_improper` 0.0315 →
  0.0016; improper |k| p10 0.003 / p50 4.26 / p90 7.9 (3689502 without the
  strain objective: 1.7 / 4.7 / 9.5) — pyramidal centres lose their improper,
  planar ones keep it, which is what the objective was for. hessian_k 0.4348
  (3689502: 0.4352), force eval 35.2 (37.1), hessian_eq 0.350 (0.369), mbis
  0.0082, charge MAE 0.058; bond k / r0 tables within ±5 kcal/mol/Å² / 0.004 Å
  of 3689502. 2.14 steps/s, 26.2 GiB. Whether the long-run N3 drift is gone
  needs the unbounded run (patience 10 now). **Operator 2026-10-12:
  "提不限步数的，跑完把 param_report 发我"** → GPU job 3707870
  (`runs/rondo-joint-s1-small-20261012-eqimp-long`, SIZE=small, no
  MAX_STEPS: patience 10 evals / 4 h stage clock, clip 10, account
  naiss2026-1-53-gpu, 4.5 h wall). Still open: pooled normalisation (N3 6 % /
  P4 0.1 % of angles) and no chemical window on θ0.
  **Result (3707870, 4 h 13 min, n442, 40 879 steps at 2.96 step/s): gate
  passes.** Best 22.65 at step 40 600 (eval 36.0 at 1 500); lr cut by the
  plateau scheduler at 26 800 / 30 600 / ~36 000, eval monotone with ±0.05
  wiggles through every cut, hessian_eq 1.12 → 0.22 and never rose — none of
  the 3690962 symptoms. N3 θ0 116.9° vs 117.9° (per-angle p10 108.8 / p90
  123.7 against measured 109.4 / 126.2), P4 105.8 vs 107.5, N4 106.0 vs 109.1,
  S2 96.2 vs 91.9, O2 108.6 vs 112.3; improper |k| p50 3.9 / p90 8.8 with p10
  at zero; charge MAE 0.031; hessian_k 0.4295 (the 1-3 floor). **New finding:
  sp-carbon angle K collapsed to 1.1** (47 at 1 500 steps, 14 in 3690962)
  with every C2 θ0 pinned at the 179.5° ceiling against measured 179.0° (p90
  179.9°): a θ0 below 180° makes the linear geometry the top of a Mexican hat
  whose Cartesian curvature grows as 1/sin θ, and the Hessian fit can only
  match the QM block by K → 0; molpot's angle kernel is C^∞ at linearity, so
  θ0 = 180° exactly is the smooth isotropic bend. Operator 2026-10-12
  (evening): **"先修 bug，最后再长跑！不需要考虑之前的 ckpt！"** → three
  fixes, uncommitted, every earlier checkpoint invalid: (A21) every valence
  context reads the term's own bond features through a per-batch
  `molrep.chem.BondTable` (sort + `searchsorted`, static shapes; the angle /
  proper / improper MLPs widen — ByteFF `OutputWithBond`; the Kekulé probe of
  the architecture review); (A22) `ExpAround(lo=, hi=)` straight-through
  windows on K / ε [1/8, 8], improper k [1/64, 8], r0 / σ [0.7, 1.4]; (A23)
  θ0 ceiling 179.5° → 180° and the linear reference 177.5° → 180°. Not done on
  purpose: θ0 "chemical window" (ByteFF's tanh on θ0 spans ±149°, i.e. none),
  per-term normalisation of hessian_eq (ByteFF's two means with weight 1 vs
  our one pooled mean: a constant factor), gain rebalancing (ByteFF's own
  gains have the same r0 : θ0 : torsion ratio; measured instead by
  `.jobs/head_grad_probe.py`, job 3723477: at init hessian_geom r0 1.1e-2
  vs θ0 1.5e-3 per unit raw, bond K ≈ angle K ≈ 2e-5 ≫ improper 1.7e-6; after
  30 steps the trunk share is hessian_geom ~1e-4 to 3e-4 against ~1e-6 for
  hessian_k — the geometry objective owns the trunk ~100×; no S1 cost, the
  S2/S3 lever is a per-head backward scale). Tests: fast 364 (chem, heads,
  rondo, potentials) + joint 33; slow compile (Rondo forward / energy /
  styles / partial_hessian / eval forward + 5 joint bodies) 10 green, 1
  graph / 0 breaks with the BondTable inside the traced body. CPU dry runs
  3723026 (A21 + A22) and 3723579 (A21 + A22 + A23) on the exact launcher
  path **both pass** (8-9 min each; 3723579: C2 θ0 180.0° at init, angle K
  40.01, improper |k| 1.43, baseline hessian_k 0.4832, gate passes on the
  untrained field). GPU: gate run 3723815 (1500 steps,
  `runs/rondo-joint-s1-small-20261012-fix3-gate`, 2 h wall) and the long run
  3723816 (`…-fix3-long`, no MAX_STEPS, 4.5 h wall) submitted together with
  `--dependency=afterok:3723815` — the launcher exits with
  `term_curriculum.py`'s status, so a failed gate cancels the long run.
  **Gate run 3723815 (29 min, n137): passes**, every S1 metric at or better
  than 3705406: eval 34.55 (36.00), hessian_k 0.4339 (0.4348), geom 33.8
  (35.2), hessian_eq 0.340 (0.350), mbis 0.0079, charge MAE 0.060. C2 θ0 p50
  180.0° (p10 177.8°), C2 angle K 42.3 at 1500 steps (47 before — the long
  run is the A23 test); N3 116.3° vs 117.9°, N4 107.9, P4 106.3, S2 100.2
  (91.9), O2 109.0 (112.3); improper |k| p10 0.022 = the 1/64 floor, p50 4.12,
  p90 7.98. 2.10 step/s, 28.4 GiB (+2.2 GiB for the wider context MLPs).
  Long run 3723816 released by the dependency.
  **2026-10-12 21:00, user "提，两条都提"**: (a) S1 large gate run 3732774
  (`SIZE=large STAGES=s1-hessian-mbis MAX_STEPS=1500`,
  `runs/rondo-joint-s1-large-20261012-gate`, 4.5 h wall) — doubles as the
  large-tier memory probe under the joint step at the tier's 131072
  capacity (old single-objective path measured 20.2 GiB large vs 12.7 small
  at this capacity; small joint is 28.4 GiB, so ~45 GiB expected; fallback
  `--capacity 65536` if it OOMs). (b) S2 small gate run 3732775
  (`STAGES=s2-torsion MAX_STEPS=1500`, `INIT_FROM` = the long run's
  `best.pt`, `runs/rondo-joint-s2-small-20261012-gate`) with
  `--dependency=afterok:3723816`, so it starts only if the long run passes
  its handoff gate. (c) s2-torsion has never run under the joint path (no
  `s2-torsion` run directory exists), so a CPU dry run of the exact S2
  launcher path, `.jobs/dryrun-s2-x86.sbatch` (job 3732773, 10 steps,
  capacity 4096, init from the 3723815 gate `best.pt`), runs now; a bug it
  finds can be fixed before (b) starts, since the chained job reads the code
  at start.
  **Dry run 3732773 passes** (8 min 50 s, n233, exit 0): torsion-scan
  census 80 bond-stretch + 2780 GAFF2-untyped of 83951 scans excluded, split
  77115 / 4054 scans; six objectives (`torsion` → `propers.k`); smoke 3
  steps, baseline eval torsion 4.965 (propers at zero), step 10 torsion
  4.961 with the five S1 metrics unchanged from the 3723815 checkpoint;
  report builds, handoff passes. The S2 launcher path is exercised end to
  end; the chained gate run (b) stands.
  **Large gate 3732774 OOMs at 131072** (n538, 15 min): censuses, loader
  plan, three smoke train steps all pass; the baseline eval dies in the
  fourth of five bodies (`hessian_eq_improper`) allocating the proper
  context input `(368896, 896)` with 90.5 GiB already allocated by PyTorch.
  Cause, found and fixed: `Trainer._run_eval_phase` kept the previous
  batch's `outputs` (loss with grad, predictions = the sub-batches holding
  the whole autograd graph) in its loop variable through the next batch's
  forward, so the eval peak was ~2× the train peak for any step that
  evaluates with grad on (`del outputs` after the scalar collection;
  `tests/test_molix/test_core/test_trainer_eval.py::
  test_eval_phase_releases_a_batch_before_running_the_next`, weakref on the
  returned loss, fails without the line; core tier 430 green). The launcher
  now passes `CAPACITY` through to `--capacity`; resubmitted as 3735480 at
  65536 (`runs/rondo-joint-s1-large-20261012-gate-c65536`, started 19:52 on
  n128, within a minute of the fix landing — whether it imported the fixed
  trainer is not provable from the log). Large's production capacity
  (65536 or 131072) is decided from a measured peak under the fixed eval
  loop, not from the broken run.
  **3735480 (65536) OOMs too, at the same 90.5 GiB, in the first training
  step** after the baseline eval passed, and the 131072 memory probe 3736461
  (eval-loop fix in) in the baseline eval — the ceiling is a leak, not a
  footprint. Root cause: the eval bodies write their predictions
  (`forces_pred`, `partial_hessian`, every table column) onto the sub-batch
  **attached** to the whole eval forward graph, and the validation batches
  are resident (`ResidentValidation`), so every val batch's eval graph stayed
  alive through the next val batch and through training. Fixed in
  `DefaultEvalStep.on_eval_batch`: with grad on, the loss and every attached
  column of the batch / predictions are replaced by their detached selves
  once the loss is computed (`molix.core.steps.eval.release`; tests in
  `test_eval.py::TestEvalReleasesTheGraph`, the multitask and step-protocol
  tests that asserted "keeps the graph" inverted / removed).
  **Performance restructure (user "搞一下性能优化", 2026-10-12 evening):**
  the S1 step ran `model.tables` (encoder + every head) **six** times per
  step — once per objective on its own host-cloned copy of the one hessian
  batch (hessian_k, hessian_geom, hessian_eq, hessian_eq_improper), a fifth
  under no_grad for the nonbonded force, and once on the mbis batch, where
  `output="energy"` then priced LJ + Coulomb + every bonded term on 4.4 M
  pair rows for a loss that reads `atoms.charge` only. Now
  (`molix.core.steps.joint`): `JointObjective(source=)` — objectives on one
  source share the sub-batch and **one** `tables` prediction, one traced
  region per source returning the per-objective losses, each objective
  keeping its own trainable views; `Derivative.forward(frame,
  **energy_kwargs)` hands the tables to the energy; `output="tables"` for
  the charge objective (no energy); the nonbonded force uses the shared
  tables detached. Encoder passes per S1 step: 6 → 2; host copies per
  step: 5 → 2; the mbis pair potentials gone. Lab: `objectives()` passes
  `source=`, `JointDataModule({source: module})`, `OBJECTIVES["mbis"]
  ["output"] = "tables"`, `param_report(tables_from="hessian",
  charges_from="mbis")`. Tests: molix fast 1428, whole fast tier 3469, slow
  joint explain 6 (a four-objective source region is 1 graph / 0 breaks),
  Rondo compile 5. `.jobs/head_grad_probe.py` predates the per-source bodies
  (note at its top; its D2 numbers stand). CPU dry run 3738119 on the exact
  S1 launcher path, then the GPU A/B against 3723815 (2.03 step/s, 28.4 GiB).
  **Dry run 3738119 passes** (12.5 min, n221, exit 0): baseline eval on the
  untrained field bit-for-bit the pre-restructure dry run's (hessian_k
  0.4832, geom 2046.6, hessian_eq 1.123, improper 0.0316, mbis 0.0902), 10
  steps, report, handoff. GPU A/B submitted (small, S1, 1500 steps,
  `runs/rondo-joint-s1-small-20261012-perf-gate`); user: "先确保可以正确训练小
  模型" — the gate must pass and the 1500-step metrics must match 3723815
  before the large tier is retried.
  **Long run 3723816 (3 h 10 min, n65): trains cleanly, fails the handoff
  gate.** Patience stop at 26 400 steps, best eval 24.09 at 24 400 (five lr
  cuts: 11600 / 14800 / 20800 / 24000 / 25800), eval monotone 28.3 (4000) →
  24.1; hessian_k 0.4329, hessian_eq 0.218, mbis 0.00233, charge MAE 0.034,
  2.60 steps/s, 28.4 GiB. A23 confirmed: C2 θ0 180.0° (p10 179.8°), angle K
  25.2 (p10 16.7 / p90 27.5) — down from 42 at 1500 steps but no collapse
  (3707870: 1.1), and 21–43 kcal/mol/rad² is the physical range of a linear
  bend. Other centres: N3 116.9 vs 117.9, N2 112.3 vs 115.3, O2 109.5 vs
  112.3, S2 95.6 vs 91.9. **Failure**: feature RMS improper 5.92 (window
  0.3–3; the other streams 1.0–1.2); improper |k| p10 0.022 = the 1/64 floor,
  p50 = p90 = 3.13. Mechanism: the straight-through window (A22) holds the
  value at the floor but passes the whole gradient, so rows the strain
  objective legitimately wants at zero (pyramidal centres) pull on the
  improper stream's LayerNorm gain forever — unclamped (3707870) that push
  is self-limiting (k → 6e-16, stream 1.29). Fix **A24**: the windows
  saturate (ByteFF's tanh, `w·tanh(x/w)` per side, slope one at the
  reference) — `ExpAround` / `LinearAround` in `molpot.heads.classical`;
  heads tests rewritten (247 green), Rondo compile 5 green. The chained S2
  gate 3732775 was cancelled by Slurm (dependency never satisfied). Next
  long run = perf restructure + A24, after the A/B.
  **A/B 3741657 passes** (small, S1, 1500 steps, n548): every metric
  within noise of 3723815 — eval 34.78 (34.55), hessian_k 0.4343 (0.4339),
  geom 34.0 (33.8), hessian_eq 0.339 (0.340), eq_improper 0.0020 (0.0019),
  mbis 0.0078 (0.0079), train 23.1 (23.0); handoff passes. **2.75 steps/s
  (2.03, +35 %) and peak GPU memory 7.79 GiB (28.4, ×3.6 less).** Small
  trains correctly on the restructured step. Submitted next (user "提，两条
  都提" + "最后再长跑"): A24 small gate → A24 small long run (afterok) → S2
  small gate (afterok, init from the long run's best.pt), and the large gate
  at the tier's 131072 capacity (perf + A24; it is the large memory
  calibration).
  **A24 small gate 3742959 passes** (25 min, n65): eval 34.23, hessian_k
  0.4336, geom 33.4, hessian_eq 0.340, eq_improper 0.0022, mbis 0.0080,
  charge MAE 0.061; C2 θ0 180.0 (p10 180.0) / K 45.7; improper |k| p10
  0.030 / p50 4.58 / p90 8.42, improper feature RMS 1.14; 2.71 steps/s,
  7.80 GiB. The A24 long run 3742960 is released (its improper feature RMS
  at 20 k+ steps is the A24 evidence).
  **Large gate 3742962 at 131072 passes** (26.5 min, n502; the first
  large-tier run that fits): **28.75 GiB peak, 2.17 steps/s** (small 7.8
  GiB / 2.71); eval 19.94 at 1500 steps (small: 34.2 at 1500, 24.1 after
  26 400), hessian_k 0.4305, geom 19.3, hessian_eq 0.227, mbis 0.0025,
  charge MAE 0.035; C2 180.0 / K 28.1, N3 117.4 vs 117.9, P4 101.7 vs
  107.5, S2 94.5 vs 91.9; feature RMS 1.00–1.05. This is the item-5
  calibration for large (capacity 131072, HBM 28.75 of 95 GiB, 2.17
  steps/s, resident val fits). Submitted the S1 large long run
  (`runs/rondo-joint-s1-large-20261012-a24-long`, no MAX_STEPS, 4.5 h
  wall) per the gate → long → report pattern.
  **S1 large long run 3743201 passes** (2 h 09, n95, 17 800 steps, patience
  stop, best 15 800): eval **10.85** (small A23 long 24.09), hessian_k 0.428,
  geom 10.34, hessian_eq 0.161, eq_improper 0.0016, mbis 0.00062, charge MAE
  0.017 e; lr cut twice (1.5e-4 at ~13 600, 7.5e-5 at ~17 200). θ0: C2 180.0
  (p10 178.7) vs 179.0, C4 107.6 vs 109.6, N2 111.6 vs 115.3, O2 108.0 vs
  112.3, the rest within 2°; angle K C2 16.9 (p10 7.1); bond K H1-C4 358 /
  C3-C3 351 / C4-C4 201 / C3-O1 728; improper |k| 0.022 (floor) / 3.90 /
  9.70, **improper feature RMS 1.11 at 17 800 steps** (A23 small: 5.92);
  2.65 steps/s, 28.75 GiB. `hessian_k` is the one loss neither tier moves
  (0.428 large / 0.433 small from 2 000 steps on) — open, see spec appendix.
  Submitted the **S2 large gate** (1500 steps, cap 131072, init from this
  run's `best.pt`, `runs/rondo-joint-s2-large-20261012-a24-gate`) chained
  `afterok` the S2 small gate 3742961 — the user's "两条都过了，再从 S1
  large 的 best.pt 走 S2 large → S3 large"; Slurm cancels it if the small
  S2 gate fails.
  **A24 small long run 3742960 passes — A24 confirmed** (3 h 01, n139,
  37 800 steps, patience stop, best 35 800): eval **22.09** (A23 long run
  3723816: 24.09 at 26 400 and a failed gate), **improper feature RMS 1.33
  at 37 800 steps** (A23: 5.92), improper |k| 0.022 (floor) / 3.96 / 8.70,
  C2 θ0 180.0 with p10 = p90 = 180.0, C2 K 24.6; hessian_k 0.430, geom
  21.5, hessian_eq 0.211, eq_improper 0.0017, mbis 0.0018, charge MAE
  0.030 e; lr cut four times (20 400 / 24 600 / 33 400 / 37 200); 3.82
  steps/s, 7.80 GiB. Its tables match the large run's to a few percent.
  The chained S2 small gate 3742961 started on n139 (2026-10-13 ~00:00,
  first S2 GPU run); the S2 large gate 3747335 waits on it.
  **S2 small gate 3742961 passes** (33 min, n139; the first S2 GPU run,
  init from 3742960's best.pt): eval 27.80 → 26.21, torsion 5.72 → 4.39
  (still falling), proper Σn²|k_n| 2.66 / 6.90 / 11.25, proper stream RMS
  1.14; S1 objectives hold (hessian_k 0.432, geom 21.2, mbis 0.0018, charge
  MAE 0.030, C2 180.0 / K 22.0; N3 θ0 116.7 → 113.2). **Watch**: improper
  |k| p50 3.96 → 0.42 (p90 8.70 → 5.19) in 1500 steps while eq_improper
  halved — fresh Adam at 3e-4 (S1 ended at 1.9e-5) lets the improper-strain
  objective beat the partial Hessian; read the S2 long run before judging.
  1.41 steps/s, 9.58 GiB. The S2 large gate 3747335 started on n470
  (2026-10-13 ~00:00); submitted the **S2 large long run 3748659**
  (`runs/rondo-joint-s2-large-20261012-a24-long`, init from the S1 large
  best.pt, no step cap) chained `afterok` the gate — gate → long → report.
  **S2 large gate 3747335 passes** (36 min, n470, init from 3743201's
  best.pt): torsion 5.27 → 3.57, best eval 14.83 at 1000 but 16.09 at 1500
  (hessian_k 0.428 → 0.450, geom 10.26 → 11.85, hessian_eq 0.165 → 0.224:
  the S1 objectives regress under the fresh Adam at 3e-4); charge MAE
  0.018; N3 θ0 117.0 → 113.1 (same 113 as the small S2 gate — systematic
  once torsion joins), C2 180.0 / K 17.3; improper |k| 0.022 / 5.28 /
  10.07 (up, where small went down tenfold); proper Σn²|k_n| 2.94 / 8.77 /
  13.84; **1.28 steps/s, 39.69 GiB**. The S2 large long run 3748659
  started on n88 (2026-10-13 01:34). If the S1 tables do not re-settle as
  the plateau scheduler cuts lr, the fix is the S2 start: a lower initial
  rate (1e-4) or carrying the S1 Adam state.
  **S3 preflight (2026-10-13 01:40–02:00)**: S3 (`s3-relax`) had never been
  dry-run, so `.jobs/dryrun-s3-x86.sbatch` (S2 template, `--stages
  s3-relax`) was added. 3748999 at the dry-run capacity 4096 failed in
  `MoleculePadder.from_counts`: a hessian-relax **row is a whole
  relaxation trajectory** (frozen extent 16d9296f: median 60 frames / 1 891
  atoms per row, p90 100 / 3 234, max 302 frames / 12 180 atoms, pairs p50
  58 k, max 508 k at 10 Å) — same shape class as a torsion-scan row, so the
  CPU dry run needs `CAPACITY=16384` (GPU 131072 holds ~60 rows a batch).
  3750954 at 16384 passed the padder and failed closed on the init
  checkpoint (large S2 gate best.pt handed to the small dry-run model);
  3750992 re-run from the S2 small gate's best.pt. Checked three relax rows
  through `THEMolHessianRelax().load("shard-000", cache=False)` (the
  lab's own path; there is no mrec LMDB for hessian/mbis/torsion — only the
  relax shard has one): every atom carries `gaff2_sigma/epsilon`,
  `formal_charge`, `ring_connectivity`, `min_ring_size`; bonds carry
  `bond_type`/`in_ring`; meta has `energy` and `step_index` per frame. So
  the 0-of-89 161 GAFF2-untyped census is a population difference, not a
  missing column (the census raises on a missing column).
  **S3 dry run 3750992 passes** (10 min, n224, CPU, capacity 16384, small,
  init from the S2 small gate): the full S3 path — seven objectives incl.
  `relax` (`relax_profile_force` → every bonded table + `atoms.sigma/
  epsilon`), four padders, compile, smoke, baseline eval (relax 3.81,
  torsion 4.39, S1 objectives at their S2 values), 10 steps, best.pt,
  param_report, handoff passes. 0.23 steps/s on 16 CPU cores. Chained the
  **S3 large gate 3751063** (1500 steps, cap 131072, init from the S2 large
  long run's best.pt) `afterok` 3748659 and the **S3 large long run
  3751064** `afterok` the gate — the user's S2 large → S3 large chain. The
  S2 long run's verdict lands ~40 min before the S3 long run can start;
  if the stage-start lr must change, `scontrol hold 3751064` while it is
  still pending.
  **S2 large long run 3748659 DIVERGED** (46 min, n88; patience stop at
  2800, best 14.32 at 800): eval 14.3 → 24.6 at 2800 (hessian_k 0.428 →
  0.696, geom 10.3 → 19.5, hessian_eq 0.165 → 0.353; torsion 5.27 → 4.08),
  train losses flat (12.3–12.9; train geom 8.2 → 9.6), pre-clip grad norm
  420–600 vs 125–280 in S1 — under clip 10 the torsion gradient owns the
  step. The handoff "passes" only because it reads the step-800 best.pt.
  Diagnosis: the torsion profile loss grows proper amplitudes (Σn²|k|
  p50 9.0 / p90 14.6 and rising) that no S1 objective can push back on
  (hessian_k trains stiffnesses, geom/eq train r0/θ0, propers.k detached
  in all three), and those amplitudes put forces and curvature on the
  hessian-set minima; the kneed train loss hides the outliers the plain
  eval counts. ByteFF stage 2 bounds exactly this with `mask_by_circstd`
  (train torsion k only where the dihedral rotates across the scan),
  `max_config`/`set_grad_max` (per-table gradient cap) and an L1/L2 aux
  loss on ProperTorsion_k — none implemented here. **Held 3751063 (S3
  gate) and 3751064 (S3 long)** (`scontrol hold`; release with `scontrol
  release`). CPU report on the step-2800 last.pt (3751903; 3751893 died
  on the stale `tables_from="hessian_k"` in `.jobs/report_from_ckpt.py`,
  fixed to the source key) **confirms it**: proper Σn²|k| 3.8/9.0/14.6 →
  17.8/34.2/56.0, improper |k| p50 4.72 → 0.022 (all on the floor), θ0
  N3 112.8 → 105.3 (gate fails), O2 → 103.5, P4 → 103.0; bond K/r0
  unchanged, feature RMS 1.0–1.17.
  **A25 landed (2026-10-13 03:00, not committed)**: the paper (§2.4, Table
  1) names an L1 norm on the proper amplitudes and stage rates 1e-4 /
  2e-5 — `molix.core.losses.NormPrior` (+ `TestNormPrior`, 6 tests; loss +
  joint suites 154 green), lab `OBJECTIVES["torsion_l1"]` (`"tables"`
  body on the torsion-scan batch, `propers.k`, weight `TORSION_L1_WEIGHT` =
  1.0) in S2 and S3, `LR_TORSION` = 1e-4, `LR_RELAX` = 2e-5. The circular-std
  mask and per-table gradient cap (code-only in ByteFF) are not adopted
  yet. **S2 CPU dry run on A25 passes** (3752027, 10.5 min, n226, small,
  cap 4096, init from the S1 small long run): seven objectives incl.
  `torsion_l1` (0 at step 0 — the propers start at zero — 1.3e-5 after 10
  steps), smoke, baseline eval, best.pt, param_report, handoff passes.
  **2026-10-13 09:40 the user authorised autonomous iteration** ("自动迭代，
  直到能达到生产机训练为止"): gate → judge → long → next stage, fixing and
  resubmitting on failure, param_report to the user after every run.
  Dropped a duplicate `TORSION_L1_WEIGHT` definition in `configs.py` (the
  original stayed when the constant moved above `OBJECTIVES`; same value).
  Submitted the **S2 large gate on A25, 3765199** (1500 steps, cap 131072,
  lr 1e-4 from the stage, init from 3743201's best.pt, name
  `rondo-joint-s2-large-20261013-a25-gate`) and the **S2 large long run
  3765202** (`--hold`, `afterok` 3765199; released only if the gate passes
  the A25 criteria: Σn²|k| p50 near 9, improper |k| p50 > 1, torsion eval
  still falling, S1 objectives' eval not rising). The held S3 jobs
  3751063/3751064 (init from the diverged S2 best.pt) stay held; they need
  `scancel` from the user and a re-chain from the A25 S2 best.pt.
  **S2 large gate on A25 passes its criteria (3765199, 38 min, n129,
  12:10)**: eval 16.12 → 14.16 monotone (A24 gate: 14.83 best, 16.09 at
  1500), torsion 5.27 → 3.63 (same fit as A24's 3.57) at Σn²|k| p10/p50/p90
  **0.87 / 2.31 / 7.85** (A24 gate 8.77 p50; diverged run 34.2), improper
  |k| p50 **2.71** (floor 0.022 only at p10), S1 objectives flat/falling
  (hessian_k 0.4276, geom 9.81, hessian_eq 0.170, improper strain
  0.00135, mbis 0.00062), θ0 N3 111.7 (−6.2° vs measured; watch), N2 −4.0,
  O2 −4.2, C2 180.0 / K 18.6, charge MAE 0.017, feature RMS 1.00–1.11,
  pre-clip grad norm 34 (3e-4 run: 420–600), 1.28 steps/s, 39.69 GiB.
  Released the S2 long run 3765202; queued the **S3 large gate 3768159**
  (`--hold`, afterok 3765202, 1500 steps, cap 131072, init from the S2
  long's best.pt) and the **S3 large long run 3768163** (`--hold`, afterok
  3768159); each is released only after its predecessor passes judgement.
  **S2 large long run on A25 passes (3765202, 60 min, n38, 13:21)**: best
  14.13 at 2600, patience stop at 4600 after one lr cut (5e-5 at 3800);
  eval plateau 14.1–14.4 from 1400 on (A24 at 3e-4 went 14.3 → 24.6), one
  transient spike at 3800 (16.04, L1 term 0.16 → 0.66) reversed by 4000.
  torsion 3.44; S1 objectives hold (hessian_k 0.428, geom 9.94,
  hessian_eq 0.180, improper strain 0.0013, mbis 0.00061). Report on best:
  Σn²|k| 0.93/2.44/10.2, improper |k| p50 2.36 (declining 3.90 → 2.71 →
  2.36), θ0 N3 111.3 (−6.7°, drifting; window ends 107.9), N2 −4.0, O2
  −4.1, C2 180.0/K 19.8, charge MAE 0.017, feature RMS 1.00–1.11, 1.96
  steps/s, 39.69 GiB. Released the **S3 large gate 3768159** on this
  best.pt (first GPU S3 on A25; judge: relax loss falls, S1/S2 evals hold,
  LJ log ratio median ≤ 0.3, impropers do not collapse, memory at 131072
  with four padders); the S3 long run 3768163 stays held until then.
  **S3 large gate on A25 fails judgement (3768159, 75 min, n36, 14:47) — K
  collapse**: handoff passes mechanically (K medians still inside
  [0.3×, 3×] of 450) but hessian_k rises monotonically 0.428 → 0.469 and the
  bond K fall 30–70 % in 1500 steps (H1-C4 358 → 249, C3-C3 349 → 154,
  C3-O1 731 → 230), angle K −20…−35 %, improper |k| p50 2.36 → 1.88;
  hessian_geom 9.94 → 4.28 and hessian_eq 0.180 → 0.109 only because every
  force at the QM minimum scales with K. relax 2.88 → 2.25, torsion 3.31,
  LJ log ratio ε −0.18 median, pre-clip grad norm 100–180 (S2: 34),
  0.46 steps/s, **57.36 GiB** (first S3 memory number at 131072). ByteFF's
  stage-3 energy/force loss trains every bonded table but under a per-table
  gradient cap with no public values; fix A26 (configs.py, uncommitted):
  `OBJECTIVES["relax"]["trainable"]` = bonds.r0, angles.theta0, propers.k,
  atoms.sigma, atoms.epsilon — the curvatures stay the partial Hessian's.
  Resubmitted from the S2 long's best.pt: **S3 gate 3776940** (A26, 1500
  steps) and **S3 long 3776941** (`--hold`, afterok 3776940). Judge the gate
  on: hessian_k stays at 0.428, bond K within ±10 % of the S2 long's table,
  relax still falling (A25 gate reached 2.25 with K free), N3 θ0 not worse,
  LJ ε log ratio median ≤ 0.3. 3768163 stays held (stale) for the
  operator's scancel, with 3751063/3751064.
  **Throughput question raised by the operator (10-13 afternoon)**: S3 steps
  at 2.24 s against S2's 0.57 s (wall per step incl. evals; evals are ~12 s
  per 200 steps, no recompiles in .err) and 57 vs 40 GiB. Two driver-side
  suspects: `clear_runtime_cache()` wipes the prepared-row cache at every
  stage start (every job cold; relax rows are whole trajectories) and
  `loader_plan`'s 16 workers are divided among the sources (S3: 4 per
  source). The log table had no data-wait column — added
  `compute_step_per_second` and `data_wait_s` (curriculum.py Log keys).
  **A26 gate 3776940 held** until the step time is explained; bounded
  profiler `profile_joint.py` + `profile_joint.sbatch` (train dir) submitted
  as **3782146** (1-proc cold/warm per source with collate stage split,
  per-source production loaders, lockstep, GPU per-source bodies + whole
  step + peak memory + kernel table, slowest source at 16 workers).
  Stale: 3781360 failed (inherited SIZE check), 3782141 held (truncated
  sbatch) — both for the operator's scancel.
  **Profiler 3782146 (n463, 27 min) — the step is data-bound, the cache is
  per job**: the whole S3 step on a resident nested batch is **0.380 s** (per
  source fwd/bwd: hessian 0.044/0.057, mbis 0.032/0.029, torsion 0.035/0.042,
  relax 0.037/0.055 s; peak 54.8 GiB, mm/addmm 60 % of CUDA time), against
  2.24 s/step delivered. One process, cold → warm batch: hessian 169/51/36 →
  1.0 s, mbis 160/100/75 → 0.5–0.8 s, torsion 28/20/18 → 0.1 s, relax
  15.5/14/11.6 → 0.05–0.1 s; 4 workers: the first cold batch 10–44 s, warm
  ~0. Rows per 131072-atom batch: hessian/mbis ~4 300, torsion ~170 scans,
  relax ~59 trajectories (median 71 frames) → one pass over relax is ~1 390
  steps, torsion ~450, mbis ~83, hessian 14: the S3 gate never warmed, and
  S2's 0.57 s/step was the 450-step cold phase averaged over 4 600 steps.
  Root: molhub's runtime cache root is `/scratch/local/molcache/<jobid>` (per
  job), `clear_runtime_cache()` was moot. **Fix (lab, uncommitted)**:
  `warm_prepared_cache.py` + `.sbatch` (CPU partition, x86 env built from the
  same molrs/molpy revisions) drains every source's train+val loader once and
  snapshots each `<digest>.prpm` to
  `workspaces/lab-new/prepared-cache/` with a manifest; `term_curriculum.py`
  now calls `restore_prepared_cache()` once per job (copytree onto the
  runtime root; fails closed without a snapshot, `PREPARED_SNAPSHOT=none`
  runs cold) and `report_prepared()` prints warm/cold per source after
  `stage()`. Warm-up job **3785149** submitted (24 workers, 256 GB, 3 h). Not
  changed: `loader_plan._MAX_WORKERS=16` (memory would allow 30; warm
  batches at 4 workers/source ≈ 0.25 s < 0.38 s GPU, so not the bound once
  warm). Expected S3 step ≈ 0.4–0.45 s (5× faster), S2 similar.
  **Warm-up 3785149 (n282, 21 min) filled the 214 GB scratch**: the stored
  row carried the 10 Å cutoff graph and relation for every conformer (~36 B
  a pair): mbis 22 GB, torsion 99 GB (1.2 MB a scan), relax 88 GB at 31 % of
  its rows (3.3 MB a trajectory; ~285 GB complete), hessian 4.5 GB. The
  batch-level rebuild is vectorised (`NeighborList._execute_batched` 0.17–
  0.2 s + `PairRelation` 0.14 s on a 131072-atom synthetic batch, one
  thread), so the graph is not worth storing. **Changed (molnex src,
  uncommitted)**: the collate skips exactly the tasks the rows were captured
  with (`PreparedRows.tasks` → `_CollateFn(captured=)`, keyed on `task_id`;
  `DataModule._make_collate_fn(dataset)`), the `per_row` class flag on
  `NeighborList` / `PairRelation` is gone; tests updated + 2 added (108
  pass). **Lab**: `class_i_fit._with_prepared` stores rows with no per-row
  task, field `class-i.prepared.v4.<kind>` (no cutoff in the id);
  `loader_plan._MAX_WORKERS` 16 → 24 (155 GiB under the memory model). The
  v3 snapshot (214 GB on nobackup) was deleted. Submitted: cold CPU dry run
  of the S3 path on v4 (**3786080**, `.jobs/dryrun-s3-x86.sbatch`, cap 4096,
  10 steps, `PREPARED_SNAPSHOT=none`) and the v4 warm-up (**3786081**).
  Then: check sizes (expect ~3× smaller, ≲130 GB), release the A26 gate
  3776940 (its driver restores the snapshot and logs warm/cold, data_wait).
  **Both pass (18:10)**: cold CPU dry run 3786166 (small, cap 16384 — 4096
  cannot hold a 12 180-atom relax trajectory row; init S1 small fix3-long
  best.pt, the S2 large one is the wrong tier) runs stage → smoke →
  baseline → 10 steps → eval → param_report on prepared v4 with the cutoff
  graph rebuilt per batch; it stops at the handoff on an untrained improper
  stream (rms 5.89), which is the dry run's documented outcome. v4 warm-up
  3786081 (n290, 11 min): **74.3 GiB** — hessian 2.2, mbis 5.9, torsion
  16.8, relax 49.5 (all 86 239 rows) — against 214+ GB for v3; snapshot +
  manifest under `workspaces/lab-new/prepared-cache/`. **A26 S3 gate 3776940
  released** on it (first run with the restored cache, 6 workers/source,
  data_wait/compute columns in the log).
  **Architecture / engineering cleanup (operator: 「把代码清理出来，确保架构
  符合要求、工程上干净」, 10-13 evening; three independent review passes
  over molix, molpot+molrep, molzoo+scripts+lab; all uncommitted).** Gates
  before and after: test mirror, ruff, ruff-format (this chain's files), ty
  0 errors; fast tier 3480 → 3481 passed, slow compile 11 passed (Rondo
  forward/energy/partial_hessian and every joint body still 1 graph /
  0 breaks). molix: prediction keys live in `molix.schema`
  (`ENERGY_PRED_KEY`, `FORCES_PRED_KEY`, `NONBONDED_FORCES_KEY`,
  `HESSIAN_KEY`); `molix.frame.beside_labels` is the one place a model's
  write is moved beside its label (Rondo.forward and both joint bodies use
  it; the joint `"forward"` op lands the force at `forces_pred` through
  `Derivative(outputs=)`); `JointObjective(trainable=())` is refused;
  `JointTrainStep.source_losses` is the public per-source primitive
  (`forward_loss` loops over it); the collate skips exactly the tasks the
  prepared rows were captured with (`PreparedRows.tasks` → `_CollateFn
  (captured=)`), `per_row` flags gone; `PreparedRows.store`,
  `PreparedMmap.rows_present()`, `DataModule.release()`; `collate.rebase_kind`
  / `OFFSET_TARGETS` public (no private cross-module imports);
  `collator.Prepare` protocol; subset views read through public
  `indices`/`source`/`dataset`; one package-wide import-boundary test
  (`tests/test_molix/test_init.py`); legacy loss tests moved to their mirror
  paths; history narration in docstrings replaced by design statements.
  molpot/molrep: `molpot.Potentials` / `molpot.potentials.Potentials` is the
  style collection (`StylePotentials` gone), the top-level bonded names are
  the molrs-style classes (the tabulated path's `bonds.harmonic` etc. by
  module path only), the lazy `__getattr__` container shims are direct
  imports, `BondedPotential.values` reads `bonds.bond_types` (the collate's
  spelling — was `type`, never hit by a collated batch), `Potentials.energies`
  / the new `ImproperPeriodic` / `_style_params` helper / `fourier_basis`
  export gone, `CvffHead.sign` buffer owns the cvff form, heads declare
  `references`; tests: registry duplicate-claim, pair `shifts`/`pair_diff`
  geometry, bond type column. molzoo: Rondo has no `OUTPUT_COLUMNS` /
  `_index` / `table_names` (the bonded potentials own their index,
  `table_columns()` is what the heads produce), hybridization once per
  `tables()` with the linear-centre proper mask applied there, one energy
  loop, `forcefield: molpy ForceField`. scripts: `cyclic_impropers` /
  `IMPROPER_ROWS_PER_CENTRE` live in `molix.data.valence_columns` (the
  three-rows contract is molix's), `ClassIPrepare.bond` refuses a frame
  without bonds (no position-based inference), the prepared-row walkers use
  public `source`/`indices`/`path` (+ `HessianTargets.source`,
  `prepared_kind`), dead code out (`profile_force_*`, `TASK_CODE_ENERGY`,
  `label_sample(group=)`, `get_max_*`, `class_i_data_module(slack=,
  *_counts=)`, the census-key shim, `CALIBRATION`/`TORSION_ATOMS_PER_ROW`/
  unread `PROD_DAY` keys, `ForceFieldDeclaration.save/load`). lab: one
  curriculum loop `curriculum.run(size, stages=, split=, after_stage=)`
  shared by `curriculum.py` (LMDB split) and `term_curriculum.py` (censused
  HDF5 split + `handoff_gate`), `stage_datamodule()` split out of `stage()`
  (the warm-up builds no model), argparse CLIs, census files renamed to
  `<shard>.bond-stretch.json` (shim gone), `profile_joint.py` import fixed,
  the 16 pre-joint scripts and their 11 sbatch files moved to
  `train/attic/` (none of them imported). Spec (both copies): §2.1
  total_charge integer-valued, §2.2 prediction keys, A14/A22 note A24,
  R2 notes A26, §7.2 test name, scheduler sentence out of the appendix.
  **Not done, on purpose**: tabulated-path classes keep their names in their
  modules (unit 03); `PeriodicHead.periodicity` stays a persistent buffer
  (the S2 checkpoint carries it); `publish_terms` still re-evaluates a
  combined loss's terms for reporting (no-op on the joint step);
  `param_report` composes the encoder twice; `fit` finds the checkpoint dir
  by hook walk; `ForceFieldDeclaration` belongs upstream as
  `ForceField.to_dict`; the other lab tools' lint. **Validation of the
  refactored path**: cold CPU dry run of S3 on the new drivers 3788296
  (small, cap 16384, 10 steps): **bit-identical to the pre-refactor dry run
  3786166** — baseline eval 47.5274, step-10 eval 47.3202, every per-objective
  value equal, same report, same documented stop at the improper-stream
  handoff; 11 min, 3 workers/source on the 16-CPU node. **A26 gate 3776940
  released on the cleaned tree** (restores the v4 snapshot).
  **A26 gate 3776940 passes (n419, 43 min, 20:32)**: 0.66 s/step wall incl.
  evals (3768159: 2.24), delivered 1.67 steps/s vs accelerator 3.2 — data
  wait 428 s of 993 s, so the loader still takes ~45 % of the step at 6
  workers/source (next lever: `_MAX_WORKERS` 24 → 30 fits the memory model);
  55.7 GiB; all four sources warm (74 GB restored in 32 s). Judgement:
  hessian_k 0.4277 flat, bond/angle K unchanged against S2, improper |k|
  2.36 → 2.52, hessian_geom 9.94 → 8.96, hessian_eq 0.180 → 0.166, relax
  2.88 → 2.76 (slow; the A25 descent to 2.25 was K softening), torsion 3.35,
  Σn²|k| p50 2.89 (p90 8.0; watch the L1 term 0.12–0.31), N3 θ0 112.5 (up
  from 111.3), LJ ε log ratio median −0.18. **S3 long run 3776941 released**
  (no step cap; patience 10 evals / 4 h clock). Judge it on: relax still
  falling, hessian_k flat, Σn²|k| not growing past ~4, N3 θ0 not falling,
  ε median ≤ 0.3; then the S3 long's best.pt is the first production
  candidate (GAFF2 baseline still owed).
- [x] Item 5 calibration (large tier, nested batches, 131072): measured by
  3742962 above — 28.75 GiB, 2.17 steps/s, three sources resident. Small:
  7.80 GiB, 2.71 steps/s (3742959). Medium: not run (not needed yet).
- [ ] Item 5 calibration on GH200: nested batches (three sources resident at
  once), capacity / HBM / step rate / scratch for the prepared caches — not run.
- [ ] S1 small bounded run (1500 steps) → handoff gate (`param_report.py`:
  k within [0.3×, 3×] of k_ref, θ0 within 10° of measured per centre incl. C2,
  r0 ∈ [0.8, 2.6], feature RMS, σ/ε log-ratio ≤ 0.3) → S1 large → S2 → S3.
- [ ] Open: the lab's other probes (`calibrate.py`, `overfit_gate.py`,
  `prod_day.py`, `smoke.py`, `stage*_smoke.py`, `theta0_probe.py`,
  `hessian_breakdown.py`) still name the deleted API; `molhub.dataset.molrec`
  does not import under molpy 0.17 (`mrec.read_trajectory` gone) — the HDF5
  path the driver uses is unaffected.

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

- [ ] **UNMET, and 2026-09-22 is the bill for it.** Three defects lived in stage 2
  for its whole four-hour run, and a bounded tiny-set overfit of the
  partial-Hessian head would have shown every one of them in minutes:
  1. `ClassIPrepare.label` re-derived the valence families and assigned whole new
     blocks, discarding the `hpair_<i>_<j>` slot columns `HessianTargets` had
     written on them. Bonds kept theirs (`ClassIPrepare.bond` short-circuits on an
     existing bond block); angles, propers and impropers lost theirs, and
     `Potentials.partial_hessian` **silently skipped** a namespace with no slot
     column. So the stage supervised the bond term alone. Its angle term was still
     at 0.004 kcal/mol afterwards, and the loss cannot go below the 64 % of pairs
     that are 1-3 or further.
  2. Every bonded force constant initialised at `softplus(0) = 0.693` against QM
     labels asking for ~932 (1-2 pairs) and ~22 (1-3) kcal/mol/Å² — three orders
     of magnitude, traversed through a loss whose gradient is scale-free.
  3. The profile grouping of item 1.
  None of the three raised, and the logged loss moved plausibly (0.747 → 0.728)
  the whole time. **A loss that moves is not evidence that a head is learning.**
  The check that closes this item per head is the cheap one: overfit ~8 molecules
  to near-zero and assert it gets there. A head that cannot overfit eight
  molecules has no gradient, and that is exactly what three of four bonded
  families had.
- [ ] **The gate exists now** (`projects/class-i/train/overfit_gate.py`), and it
  reports three independent things per stage rather than one word: whether every
  parameter group the objective should reach carries a gradient after one backward,
  whether the loss blows up, and whether it falls by that objective's own factor.
  One factor cannot serve four objectives — a plain MSE has a zero floor, while the
  partial-Hessian residual is a normalised L1 a Class-I harmonic model cannot drive
  to zero (it converges to ~0.40 and stays, which a fall-by-10x rule called a
  failure).
- [x] **No head is severed.** Every `severed` list matches its stage's own
  `trainable` and objective, and `hessian` now shows `angle=0.058` where the lost
  slot columns had left it at exactly nothing. `atom_head` (Lennard-Jones) is dead
  in the first three stages, which is the open R1 rather than a defect: no stage
  before `profile_forces` supervises a nonbonded parameter.
- [ ] **Stage 4 diverges in the gate and the mechanism is not identified.** Eight
  candidates are dead by measurement, each recorded so nobody re-runs them:
  1. The Boltzmann weight depending on the prediction — a **real defect**, fixed
     (molnex `ee12ae1`): the gradient was 58 % off the loss's own finite difference.
     It gives stage 4 a 3x better minimum and cures torsion-scan (3/3 bounded), but
     not this stage.
  2. Force-term dominance — no: gradient norms 221 (profile) against 132 (force).
  3. A prediction-dependent inlier mask — no: `outlier_norm` uses the **label**.
  4. The two terms trading off — no: both rise together, two orders of magnitude.
  5. Charge / LJ runaway — not the sole cause: freezing them moves the blow-up from
     step ~500 to ~1400.
  6. Missing gradient clipping — no: `clip=10` ends at 1082, `clip=100` at 1934.
     (Worth keeping anyway: the tier's 1e3 never fires against a measured 884.)
  7. The force term at all — no: the **profile-only** objective on the same data
     also diverges, 2.73 at step 400 to 873 by 1400.
  8. Weight concentration on relaxation profiles — no, and the opposite: 56.7 of 58
     and 73.5 of 76 frames carry weight (torsion-scan: 17.8-21.5 of 24), and the
     label spread is 12-16 kcal/mol, not the hundreds a relaxation was assumed to
     span.
  **What the data says instead:** on relaxation frames at initialisation the model
  is already within **2.55 kcal/mol rms** of the QM relative energies, with every
  term sane (bonds 451 kcal/mol absolute but only 27.9 of within-profile spread,
  LJ ~1e-6). It improves to step ~400 and then walks away. So this is an optimizer
  leaving a good solution, not a bad objective or bad data.
  The step that leaves is Adam on each head's final `Linear.weight`: the steady
  update is `lr` per coordinate, the weight is dotted with the hidden vector, and
  the physical parameter (`r0`, a torsion amplitude, `sigma`) therefore moves by
  `lr * ||h||_1` per step. Global grad clipping scales `m` and `v` together, so
  it does not change that step. `hessian-relax` now sets `scale_head_output`:
  those weights take `lr / in_features`; biases and the encoder keep `lr`. The
  gate has not been re-run on the grouped optimizer.
- [x] **Protection, since the good state exists and would not be saved.** The
  instability is intra-epoch — best at step ~400 against a ~1300-step production
  epoch — so an epoch-boundary eval takes its first reading after the blow-up and
  writes `best.pt` from it. `configs.STAGE_EVAL_STEPS` gives the two profile stages
  a 200-step eval cadence. Not a cure, and not evidence about production either:
  the gate diverges on **two rows**, and a production batch averages thousands of
  frames, which usually stabilises. Whether it diverges there is unmeasured.
- [x] **The four objectives' gradients differ by ~3000x, and one lr served all
  four** (`charge` 0.199, `partial_hessian` 0.071, `profile_forces` **616** at
  initialisation). Swept: `torsion-scan` falls 3.2x at 1e-4 and 26.9x at 3e-3;
  `hessian-relax` is stable at 3e-4 and, at 1e-3, never improves on its step-0 loss
  and climbs to 4689. `PROD_DAY`'s 1e-4 was the safe end, so stage 3 was starved by
  configuration — the shape the production run showed before it died. Fixed as
  `configs.STAGE_LR` (molnex `45832dc`); on it, every stage is stable and nothing
  diverges.
  Fixed in molnex `7ea99c5`; measured effect at initialisation, same batch and
  probe (`projects/class-i/train/hessian_pairing_probe.py`): loss 0.7464 → 0.7189
  (heads) → **0.5671** (slots restored), which is below the 0.728 that four hours
  of the stage reached; the 1-3 predicted block trace went from exactly 0.00 to a
  median of +4.14 against a label median of −22.


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
