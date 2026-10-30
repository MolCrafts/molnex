# Phase 4 — MACE replaces ByteFF-26's encoder under its frozen heads

| Field | Value |
|---|---|
| Opened | 2026-10-29 |
| Chain | rondo-enc (charter `rondo-enc-00-charter.md`); encoder spec `rondo-enc-02-mace.md` |
| Status | done 2026-10-30: M-DP passes the gate (within 10 % of B); EGT fine-tuned (D2) still better |
| Scope | quick experiment (operator 2026-10-29): member-01 only, no seed repeat, no ensemble |

## 1. Question

Can a geometry-reading encoder (MACE on one reference conformer per molecule) be trained to
feed ByteFF-26 member-01's **frozen** heads — bonded, bond-charge-correction charges,
direct σ / ε — and price THEMol as well as the published covalent-graph encoder does?
"No" is a valid result: it would say a conformer-based backbone is not an economical
replacement for the topological one at this interface.

## 2. Composition (lab-side, `lab/projects/rondo-enc/phase4/`)

`ByteFF(ff, dtype=float32, encoder=MaceConformerEncoder(spec))` on phase 2's field
(`amber_forcefield`), every head loaded from member-01 and frozen; the encoder trains from
scratch. The teacher — member-01's covalent-graph encoder, frozen — rides in a lab
subclass that writes both feature sets onto the batch whenever the model computes
features, so the distillation loss reads columns like every other loss and the encoder
runs once per call. Same sources, splits, packing, capacity, eval objectives and per-molecule
scorer as phase 2.

| run | trains the encoder on |
|---|---|
| M-D | feature distillation only: relative MSE of atom and bond features to the teacher's, on the mbis, torsion-scan and hessian-relax molecules |
| M-DP | starts from M-D's `best.pt`; M-D's objectives plus physical supervision through the frozen heads: `torsion_profile` (q, σ, ε, proper k) and `relax_metrics` (r0, θ0, proper k, σ, ε — ByteFF's relax tables; curvatures and charges stay the teacher's via distillation) |

M-DP cannot start from scratch: the frozen heads turn an untrained encoder's features into
wild parameters (gate 4, 2026-10-29, first attempt: relax loss 5.6e20, gradient norm inf at
step 0). The launcher refuses M-DP without `--init-from`.

B (the published model) is phase 2's scored baseline, same split, not re-run.

## 3. Metrics

1. Phase 2's per-molecule held-out scores (`score.py`): charge MAE vs MBIS, offset-free
   torsion RMSE (plain and Boltzmann-weighted), relax energy RMSE, relax force RMSE.
2. **Parameter agreement with B** on the held-out molecules: per table (bond k, r0, angle
   k, θ0, proper k, charge, σ, ε) the median absolute difference.
3. **Cross-conformer spread** on the held-out torsion scans: each conformer read out on its
   own geometry (no gather); per profile, the std of the predicted proper k and charges.
   Says how much the reference choice matters.
4. **Cost**: encoder forward time per batch at the run capacity, EGT vs MACE, on GH200.

## 4. Gate

MACE replaces EGT if the better M run's held-out medians are within 10 % of B's on
torsion, relax energy and relax force, and charge-vs-MBIS within 10 % (B: 0.111 e,
0.789, 0.589 kcal/mol, 5.02 kcal/mol/Å). Otherwise the result is a measured no, with
where the gap is (which tables disagree, metric 2).

## 5. Production checklist

1. **Data** — phase 2's sources, splits and prepared rows, unchanged (evidence carried).
2. **Reference vs packed** — the padded batch reproduces per-molecule features, the
   distillation loss and its encoder gradient (GPU check, `gate_packed.py`).
3. **Frozen is frozen** — heads and teacher fingerprinted before and after; bit-identical.
4. **Overfit** — `gate_overfit.py`, 4 molecules per source, GH200. M-D from scratch: every
   distillation loss ≤ 0.5× its start (passed: 0.12–0.16 in 300 steps). M-DP from M-D: the
   physics objectives it adds ≤ 0.5× start; the distillation objectives it inherits are
   anchors the fit trades against and must stay ≤ 1.5× start. At 1e-3 M-DP oscillated and
   ended above its start on every loss (job 4625482), so M-DP trains at 1e-4; at 1e-4 over
   300 steps torsion 0.14×, relax 0.59× still falling, distillation 1.14–1.16×
   (job 4625781), but over 800 steps the gradient norm climbed after ~400 and the fit
   ran away (charges 21 e, job 4625890) — a torsion profile fitted through
   intramolecular Coulomb. Fixes: charges leave `torsion_swap` (they follow the teacher
   through distillation only), the gate clips as production does, the rate is ByteFF's
   refinement 2e-5. Then 1500 steps are monotone and stable: torsion 0.115×, relax
   0.515× and flattening at 5.0 against B's 0.51 on the same molecules, distillation
   ≤ 1.19× (job 4626199). The relax floor is the init's: it was the 300-step smoke
   encoder (features 77 % off the teacher), and relax trains r0 / θ0 / proper k / σ / ε
   with every curvature detached, so wrong k are beyond its reach. The decisive run
   starts from M-D production's `best.pt`. From there, with distillation at weight 1,
   the physics gradients walked the encoder off the teacher's features after ~200 steps
   (job 4626569); M-DP's anchors (`anchor_*`) are weighted 30 (probes at 10 / 30, jobs
   4626667 / 4626668). Final bar for a run that starts from another (phase 2's gate-4
   bar for a fine-tune near B), judged at the best step (lowest weighted total — what
   `best.pt` keeps), scored metrics read from that state: anchors ≤ 1.5× start, and each
   added objective at least as good as B on the metric it trains. **Passed** (job
   4626854): best step 99, anchors 0.93–1.00×, torsion 0.26× (0.286 vs B 0.551), relax
   0.48× (0.388 vs B 0.520), charge 0.028 vs B 0.025 (not trained by physics).

Gates 2 (GH200, fp64; M-D 1.7e-14, M-DP 2.7e-9 of the largest gradient), 5 (cells: M-D
0.32 s/step, M-DP 0.39 s/step, 39 GiB, at 16 384), 6 (smoke 200 → resume → 300, frozen
bit-identical, encoder moved; M-D and M-DP each under `smoke/<RUN>/`) and 7 passed;
`signoff.json` signs M-D and M-DP (M-DP starts from M-D production's `best.pt`).
5. **Calibration** — GH200 step time / HBM at 16k / 32k / 65k atoms, steady state; the
   THEMol neighbour count within `r_max` (for `avg_num_neighbors`).
6. **Smoke** — 200 steps, resume to 300, frozen unchanged, encoder moved.
7. **Evidence** — `runs/rondo-enc-phase4-<utc>/`; `signoff.py` writes `signoff.json`, the
   launcher fails closed without it.

## 6. Result

Evidence `runs/rondo-enc-phase4-20261029T2017Z/` (calibration/, smoke/M-D, smoke/M-DP, signoff.json,
runs/M-D, runs/M-DP, scores.json, compare.json). Capacity 16 384 atoms per source (M-DP needs 39 GiB
there and does not fit 32 768; M-D 0.32 s/step at 16 384, 0.48 s at 32 768).

**M-D** (job 4625477): 11 600 steps, stopped by patience, heads and teacher bit-identical.
Held-out by molecule, full val sets, median / mean:

| run | charge (e) | torsion | torsion (w) | relax E | relax F |
|---|---|---|---|---|---|
| B | 0.111 / 0.11 | 0.789 / 1.03 | 0.560 / 0.70 | 0.589 / 11.2 | 5.02 / 23.1 |
| M-D | 0.111 / 0.11 | 0.851 / 1.12 | 0.622 / 0.77 | 0.737 / 10.9 | 6.19 / 23.7 |
| M-D vs B | 0 % | +8 % | +11 % | +25 % | +23 % |

Charge and torsion within the 10 % margin; the relaxations miss it by ~25 %. Parameters
against B (median |Δ| / median |B|): bond k 3.0 / 379, r0 0.0011 / 1.34 Å (p90 0.008),
angle k 0.77 / 50, θ0 0.27 / 112°, proper k 0.035 / 0.13 (the loosest), improper k
0.36 / 3.9, charge 0.010 / 0.11 e, σ 0.005 / 3.1 Å, ε 0.0008 / 0.094. Cross-conformer
spread on the torsion scans (each conformer as its own reference): proper k 0.0055,
charge 0.0016 e — 4 % and 1.4 % of B's typical magnitude; the reference choice barely
matters. Encoder forward per 16 k-atom batch: MACE 13–20 ms, EGT 10–16 ms (32 k: 25–29
vs 12.5 ms).

**M-DP** (job 4627223): from M-D's `best.pt`, 2e-5, anchors weighted 30; 11 800 steps
(1 h 32 m), stopped by patience, heads and teacher bit-identical, distillation 0.078 →
0.080 on the training prefix (the anchors held). Scored by jobs 4628794 / 4628795.

| run | charge (e) | torsion | torsion (w) | relax E | relax E (w) | relax F |
|---|---|---|---|---|---|---|
| B | 0.111 / 0.11 | 0.789 / 1.03 | 0.560 / 0.70 | 0.589 / 11.2 | 0.269 / 1.25 | 5.02 / 23.1 |
| M-D | 0.111 / 0.11 | 0.851 / 1.12 | 0.622 / 0.77 | 0.737 / 10.8 | 0.311 / 1.28 | 6.19 / 23.7 |
| **M-DP** | 0.114 / 0.11 | 0.785 / 0.99 | 0.551 / 0.66 | 0.517 / 3.52 | 0.247 / 0.59 | 5.11 / 10.5 |
| M-DP vs B | +2.4 % | −0.5 % | −1.5 % | −12 % | −8 % | +1.8 % |
| phase 2 D2 (EGT, same physics fine-tune) | 0.111 / 0.11 | 0.707 / 0.87 | 0.506 / 0.60 | 0.440 / 1.23 | 0.229 / 0.36 | 4.86 / 6.45 |
| M-DP vs D2 | +2.4 % | +11 % | +9 % | +18 % | +8 % | +5 % |

Parameters against B, M-DP (median |Δ|): bond k 3.3, r0 0.0012 Å, angle k 0.83, θ0 0.30°,
proper k 0.047, improper k 0.36, charge 0.011 e, σ 0.009 Å, ε 0.002 — the trained tables
(proper k, σ, ε) moved away from B, the rest stayed at M-D's distance. Cross-conformer
spread unchanged (proper k 0.0054, charge 0.0016 e).

**Verdict: yes, at the gate's bar.** M-DP is within 10 % of B on every gated metric
(charge +2.4 %, torsion −0.5 %, relax E −12 %, relax F +1.8 %) and well under B's mean
tails (relax E mean 3.5 vs 11.2, relax F 10.5 vs 23.1). Distillation alone (M-D) is not
enough: it reproduces charges and torsions but not the relaxations (+25 %). The fairer
comparison — both encoders fine-tuned on the same physics — still favours EGT: phase 2's
D2 beats M-DP by 5–18 % on every physical metric, so MACE is an adequate substitute
under frozen heads, not a better one, and it costs ~1.3–2× the encoder forward time.

Caveats: one member, one seed; M-DP's gate-4 criterion was changed during this phase
(fine-tune bar = match or beat B on each trained metric, anchors ≤ 1.5×, judged at the
best step — §5.4), and the anchor weight 30 was chosen from two probes, not tuned.
