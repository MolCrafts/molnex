# The bond head's C–I stiffness, fine-tuned on the DFT Hessian

| Field | Value |
|---|---|
| Opened | 2026-10-30 |
| Chain | rondo-enc (charter `rondo-enc-00-charter.md`); found by `rondo-enc-08-bond-floor` §2 |
| Status | done 2026-10-30: gate not met by any run. L1-anchored fine-tune (BKA2/BKA3) moves only C–I and leaves every score unchanged, but C–I stops at 0.68 × K_dft (bulk 1.11); freer runs reach C–I only by moving every bond |
| Scope | operator 2026-10-30 ("微调键头修 C–I"); quick experiment, member-01, one seed |

## 1. Defect

Against the DFT Hessian of THEMol's held-out hessian molecules (`K_dft = −u·H_ij·u / 2`,
spec 08 §2) the published bond head is 2× too soft on C–I: over 36 C–I bonds `K / K_dft`
median 0.49 (DFT `K` median 119), against 1.11 over all 88 063 bonds. Every bond B puts
below `K` 80 is a C–I.

## 2. Run `BK`

`HEAD_SWAP["runs"]["BK"]` (`scripts/mm_param_learning/configs.py`), phase 2's machinery
(`lab/projects/rondo-enc/phase2/heads.py`):

* model B (ByteFF-26 member-01, published heads and encoder, all frozen) at bond floor 80;
* **only `heads.bonds.out.k` trains** — the bond head's K output MLP (8 tensors); its
  trunk, r0 output and every other head stay published, so nothing but bond `K` can move;
* objective `hessian_k` (partial Hessian, ByteFF's stage-1 curvature loss; with only the
  K output unfrozen its angle / improper gradients reach no parameter);
* rate `lr_unfrozen` 2e-5 (ByteFF's refinement rate);
* eval `hessian_k` + the phase-2 scores (charge, torsion, relax), so `best.pt` is the
  Hessian fit and the other scores watch it.

Evidence `runs/rondo-enc-bondft-<utc>/` (its own `signoff.json`; phase 2's stays as signed).

## 3. Gate

BK fixes C–I if, on the held-out hessian molecules (`phase6/bondk.py`):

1. C–I `K / K_dft` median within 10 % of the all-bond median (≈ 1.11);
2. the all-bond median moves by ≤ 2 %;

and on the phase-2 held-out scores (B's set) no metric is worse than B by more than 1 %,
and the 48-molecule LAMMPS MD (spec 07) breaks nothing.

## 4. Production checklist

All passed; `runs/rondo-enc-bondft-20261030T1054Z/signoff.json` signs BK at 32 768.

* 2 — packed vs reference, 8 hessian molecules: `hessian_k` rel 1.3e-16, CPU fp64 (job
  4639578) and GH200 compiled (4639579).
* 4 — overfit, 8 molecules, 300 steps: `hessian_k` 0.4360 → 0.4321 (0.991×), beats B on the
  same molecules, gradients finite. The bond K output is a small share of the partial
  Hessian (angle and improper blocks are frozen), so the loss cannot fall far.
* 5 — cell at 32 768 (4639693): 0.22 s/step, 3.2 GiB, caps measured (hessian source:
  207 360 Hessian pairs).
* 6 — smoke (4639694): 200 steps then resume to 300, frozen bit-identical, only
  `heads.bonds.out.k` moved; eval `hessian_k` 0.4449 → 0.4409, torsion 0.375 → 0.364,
  relax 1.51 → 1.26 on the training prefix.
* 7 — `signoff.json`.

Found, not fixed (pre-existing, also in phase 2's calibration logs 4602706 / 4603263 /
4603480 / 4604105): at interpreter exit a DataLoader worker aborts ("terminate called
without an active exception", `_MultiProcessingDataLoaderIter.__del__`). It fires after
every result and checkpoint is written and the job carries on; a native thread pool
inside a forked worker is the likely cause. Owner: molix data loading.

## 5. Result

Production `runs/rondo-enc-bondft-20261030T1054Z/runs/BK` (job 4640082, 3 200 steps, 22 min,
patience; frozen bit-identical, only `heads.bonds.out.k` moved). Eval `hessian_k`
0.4449 → 0.4403. Checks: `phase6/bondk.py` (4640354), phase-2 scores (4640355), LAMMPS
MD + parity (4640356, `phase6` evidence `md-BK/`, `parity-BK/`).

`K / K_dft` (median) on the held-out hessian molecules, B → BK:

| bonds | n | B | BK | BK K / B K |
|---|---|---|---|---|
| all | 88 063 | 1.106 | 1.002 | — |
| C–I | 34 | **0.486** | **0.934** | 1.86 |
| C–C | 29 872 | 1.213 | 0.987 | 0.82 |
| C–N | 10 722 | 1.268 | 1.008 | 0.80 |
| C–H | 36 813 | 1.053 | 1.006 | 0.95 |
| C–S | 1 092 | 1.349 | 0.931 | 0.72 |
| C–Cl | 499 | 1.386 | 1.031 | 0.75 |

Held-out per-molecule medians (B's set), B → BK: charge 0.1110 → 0.1110, torsion 0.7889 →
0.7714 (−2.2 %), torsion (w) −1.2 %, **relax E 0.5894 → 0.6765 (+14.8 %)**, relax E (w)
+1.0 %, **relax F 5.021 → 4.529 (−9.8 %)**. MD: 0 / 48 broken, worst bond 0.74 r0 (row
2747; B at floor 80: 0.79), geometry RMSD median 0.140 → 0.149 (BK better on 16 / 48).
Export parity unchanged.

**Verdict: gate not met.** Criterion 1 holds (C–I 0.934 against the all-bond 1.002, within
7 %), but criterion 2 fails — the all-bond median moved −9.4 % — and relax E is 15 %
worse. With the angle and improper curvatures frozen, the partial-Hessian loss is met by
pulling every bond K onto its one-dimensional DFT projection: the published K sit 10–35 %
above it (likely because ByteFF fitted them jointly with the angle K, which share those
Hessian blocks — not verified), and
the K output alone moves them all. C–I is fixed as a side effect of a global retune that
trades relax energy for relax force.

## 6. Anchored fine-tunes (BKA1–3)

`hessian_k` plus an L1 anchor of `bonds.k` to a frozen copy of the published bond head
(`BondKAnchorLoss`: mean `|K / K_published − 1|`, lab `phase2/heads.py` `AnchoredBonds`),
weights 0.03 / 0.1 / 0.3 (runs BKA1 / BKA3 / BKA2, `HEAD_SWAP`). An L1 anchor rather than a
squared one: a bond moves only where the Hessian's pull beats the weight, so a few badly
wrong bonds can move while the near-right bulk stays put; a squared anchor shrinks every
move by one factor. Gates 2 / 4 / 5 / 6 / 7 passed for all three (signed with BK in
`signoff.json`); productions 4642187 (BKA1), 4642188 (BKA2), 4643667 (BKA3).

| run (weight) | C–I K/K_dft | all bonds | bonds moved > 5 % (C–I) | relax E vs B | relax F vs B | MD broken / 48 |
|---|---|---|---|---|---|---|
| B | 0.486 | 1.106 | — | — | — | 0 |
| BK (none) | 0.934 | 1.002 | ~all | +14.8 % | −9.8 % | 0 |
| BKA1 (0.03) | 0.814 | 1.047 | 46 644 (34) | +8.7 % | −7.0 % | 0 |
| BKA3 (0.1) | 0.679 | 1.105 | 31 (29) | 0.0 % | 0.0 % | 0 |
| BKA2 (0.3) | 0.679 | 1.106 | 38 (30) | 0.0 % | 0.0 % | 0 |

The anchor does what it was built for — at 0.1 and 0.3 only the C–I bonds move and nothing
else changes — but C–I stops at 0.68 × K_dft at both weights, short of criterion 1 (within
10 % of the bulk 1.11). Below 0.1 the whole bond population moves with it. So no anchored
run meets the gate: the bond head's K output alone cannot raise C–I to the bulk ratio
without retuning every bond, at least on this objective. **Kept**: BKA2/BKA3 as a safe
partial correction (C–I 0.49 → 0.68, no other change); not adopted as a default.

## 7. Next

Not opened. Candidates: weight the partial-Hessian loss per element pair (C–I is 34 of
88 063 bonds, so its share of `hessian_k` is negligible), or fit C–I on the iodine
molecules alone with the bulk anchored.
