# The bond head's K floor

| Field | Value |
|---|---|
| Opened | 2026-10-30 |
| Chain | rondo-enc (charter `rondo-enc-00-charter.md`); found by `rondo-enc-07-validation` |
| Status | done 2026-10-30: floor 80 by default; D2's collapse gone, held-out scores unchanged |
| Scope | operator 2026-10-30 ("修键 k 下限"); no training |

## 1. Defect

Phase 6's LAMMPS MD collapsed one held-out molecule under D2 (val relax row 2747, the
zwitterion N⁻–C(=O)–N⁺H₂–C): its C–N⁺ bond, 1.664 Å in DFT, went to 0.70 Å (0.43 r0).
The bond head put that bond exactly on its window's floor — `BondHarmonic.window_constants`
`k` clamps at 80 in the published ``½k`` units, LAMMPS `K` 40 — beside charges
+0.69 / −0.80 on its ends; intramolecular 1-4 Coulomb across the bond then compresses it.
B reached 0.65 r0 on the same molecule. M-DP read the bond as `K` 162, r0 1.665.

## 2. Diagnosis — the floor against the DFT Hessian

`lab/projects/rondo-enc/phase6/bondk.py` (jobs 4636215 B, 4636283 M-DP; evidence
`runs/rondo-enc-phase6-20261030T0907Z/bondk/`): on every held-out molecule of THEMol's
hessian source (2 752 molecules, 88 063 bonds) each bond's DFT stiffness along its axis,
`K_dft = −u·H_ij·u / 2`, against the predicted `K`.

* Bulk agreement: `K / K_dft` median 1.11 (B and M-DP; the projection leaves out the angle
  terms' share along the axis, so > 1 is expected).
* DFT's softest bonds: 0.05 % below `K` 80, 0.1 % below 89, 1 % below 145 (C–S, C–C, C–Br,
  N–O … — every element pair).
* B's softest predictions are one family: all 30 bonds B puts below `K` 80 are neutral C–I,
  and over every C–I bond B's `K` is 0.49 × `K_dft` (median; `K_dft` median 119). The
  published head is systematically 2× too soft on C–I. M-DP: no bond below 80, C–I at
  0.69 ×.
* Exactly on the floor: none of the 88 063; the zwitterion of §1 is the one case seen.

## 3. Fix

The floor becomes a head setting: `HeadsSpec.bond_k_floor` (LAMMPS `K`, kcal mol⁻¹ Å⁻²),
`BondHarmonic(..., k_floor=)` replacing the published clamp (`2·k_floor` in the head's
units). Rondo's default is **80** — DFT's 0.05th percentile: the clamp only binds where the
head is below almost every real bond. `ByteFF.load` keeps the published 40
(`PUBLISHED_BOND_K_FLOOR`) so a released member is reproduced as released. The phase 2 / 4
lab builds pin 40 (`heads.BOND_K_FLOOR`), the value their signed-off runs used.

Not fixed here: the C–I bias beyond the floor (B's C–I `K` between 80 and the DFT ~119
stays as published) — that wants the bond head fine-tuned on `hessian_k`; and D2's own LJ
ε on charged nitrogens (§4).

## 4. Result

Evidence `runs/rondo-enc-phase6-20261030T0907Z/` (`md-floor80/`, `parity-floor80/`,
`scores.json` entries `*:floor80`), jobs 4636356 (MD + parity), 4636357 (B, D2 scores),
4636358 (M-DP scores). The lab sets the floor with `BOND_K_FLOOR=80`.

Held-out per-molecule medians (full val sets), floor 40 → 80:

| model | charge | torsion | torsion (w) | relax E | relax F |
|---|---|---|---|---|---|
| B | 0.1110 → 0.1110 | 0.7889 → 0.7889 | 0.5596 → 0.5596 | 0.5894 → 0.5894 | 5.0214 → 5.0239 |
| D2 | 0.1110 → 0.1110 | 0.7069 → 0.7069 | 0.5056 → 0.5056 | 0.4398 → 0.4398 | 4.8568 → 4.8599 |
| M-DP | 0.1136 → 0.1136 | 0.7849 → 0.7849 | 0.5513 → 0.5513 | 0.5173 → 0.5173 | 5.1114 → 5.1122 |

Only relax force moves (+0.02–0.06 %); every mean moves < 0.1 %.

LAMMPS MD, the same 48 held-out molecules as spec 07:

| model | broken, 40 → 80 | worst bond / r0 on row 2747, 40 → 80 | every other molecule |
|---|---|---|---|
| B | 0 → 0 | 0.65 → 0.79 | ≥ 0.886, unchanged |
| D2 | **1 → 0** | 0.43 → 0.72 | ≥ 0.880, unchanged |
| M-DP | 0 → 0 | 0.835 → 0.835 | ≥ 0.871, unchanged |

Export parity at floor 80 is unchanged (≤ 2.4e-8 per term, ≤ 4.8e-7 kcal/mol/Å).

**Verdict.** The floor at the DFT 0.05th percentile removes the collapse at no measurable
cost. It does not make the zwitterion right: D2 still compresses that bond to 0.72 r0 (its
LJ ε on the N⁺ is 1.88 kcal/mol, B's 0.48) and B to 0.79 r0, where M-DP's 0.835 comes
from reading the bond correctly (`K` 162, r0 1.665). Open: the bond head's C–I bias
(fine-tune on `hessian_k`), an ε bound for D2's LJ head on charged atoms.

Lab defect found and fixed on the way: `score.py` (phases 2, 4) and `compare.py` read
`scores.json` once and rewrote it after scoring, so two jobs on one evidence directory
dropped each other's entries (M-DP:floor80 was lost and restored from its log). They now
merge under a file lock (`score.merge`).
