# Phase 6 — Rondo validated and MD-ready (lean)

| Field | Value |
|---|---|
| Opened | 2026-10-30 |
| Chain | rondo-enc (charter `rondo-enc-00-charter.md`) |
| Status | done 2026-10-30: export at parity for all three; B and M-DP MD-ready on 48 held-out molecules, D2 collapses one bond (1 / 48) |
| Scope | lean (operator 2026-10-30, "开 Phase 6"): the models phases 2 and 4 already trained, member-01, no new training; one seed |

## 1. Question

Are the trained Rondo models usable as a static force field in a production MD engine, and
do they hold up off the training distribution? Three candidates, one per encoder story:

| model | what it is | checkpoint |
|---|---|---|
| B | ByteFF-26 member-01, published | none (official weights) |
| D2 | B with the GAFF2-window LJ head (phase 2's winner) | `runs/rondo-enc-phase2-20261029T1342Z/runs/D2/checkpoints/best.pt` |
| M-DP | MACE encoder under B's frozen heads (phase 4) | `runs/rondo-enc-phase4-20261029T2017Z/runs/M-DP/checkpoints/best.pt` |

## 2. Units

1. **Static export** — the parameters of one molecule, computed once, written as a LAMMPS
   deck by molpy's own writer (operator ruling 2026-10-23: molpy owns backend
   serialization; molnex's `molix.ff_export` is not used). Every term row is its own type,
   so nothing is averaged; the conversion is generic over styles because each
   `Term.potential_params` already speaks molrs's parameter names. Prototyped lab-side,
   promoted to a Rondo primitive with a unit test once parity holds:
   `Rondo.parameterize(frame, *, graph=0) -> (molpy.Frame, molpy ForceField)` — atoms
   (positions, charge, mass, one type each), one block per bonded namespace (`propers` as
   molpy's `dihedrals`), the field's own pair styles / cutoffs / special bonds; a
   `(rows, m)` column becomes molrs's `name1 … namem` (molpy's `dihedral periodic` takes
   `k1…k4`, `periodicity1…4`, `phase1…4`, written as LAMMPS `fourier`). molpy numbers
   LAMMPS types in label order (`bond0, bond1, bond10, …`), so a reader maps a type
   number through the data file's `Type Labels`, never by definition order.
2. **Export parity** — molnex energy and forces vs LAMMPS `run 0` on the exported deck,
   per term, on held-out molecules of every model. Gate: relative 1e-6 on each term's
   energy, 1e-5 kcal/mol/Å on forces, after any engine constant (Coulomb `qqr2e`) is
   named and accounted for, never absorbed into the tolerance.
3. **MD in LAMMPS** — held-out molecules (relax set, stratified by size), each model:
   minimise, Langevin 300 K equilibration, then NVE. Reported: blow-ups (non-finite
   energy, a bond beyond 2× its r0), NVE relative drift, and the RMSD of the minimised
   geometry to the DFT-relaxed one.
4. **Scaffold-novel subset** — *dropped* (2026-10-30). Novelty is only definable against
   our own 95/5 molecule split, but every candidate stands on ByteFF-26's encoder or heads,
   trained by ByteDance on data whose overlap with THEMol's val molecules is unknown; a
   "novel vs seen" split would measure our split, not generalisation. THEMol carries no
   scaffold-disjoint or test split (molhub `themol.py`; only a 95/5 by-molecule split
   exists) and no independent set is on disk. An honest out-of-distribution test needs an
   independent dataset — out of this lean scope, recorded as the open gap.
5. **Cost** — parameterisation time per molecule (encoder + heads, once per topology),
   EGT vs MACE.

## 3. Gate

Rondo is MD-ready if every model's export is at parity (unit 2) and no held-out molecule
blows up in unit 3. The scaffold-novel scores and the drift are reported, not gated.

## 4. Production checklist

No training runs in this phase, so the checklist's training gates do not apply; scoring and
MD jobs are bounded evaluation jobs.

## 5. Result

Evidence `runs/rondo-enc-phase6-20261030T0907Z/` (`parity/<model>/parity.json`,
`md/md.json`, every deck and log), job 4635973 (CPU, 8 min; the first pass, 4635763,
judged bonds one-sided and missed the collapse below). LAMMPS 30 Mar 2026 (x86_64). Lab
`rondo-enc/phase6/` (`export.py`, `md.py`, `validate.sbatch`).

**Export parity** (unit 2) — 16 held-out relax molecules per model, the DFT-relaxed
geometry, molnex float64 vs LAMMPS `run 0`. Largest relative error per term over the 16:

| model | bond | angle | dihedral | improper | LJ | Coulomb | max \|ΔF\| kcal/mol/Å |
|---|---|---|---|---|---|---|---|
| B | 3.8e-11 | 1.7e-11 | 2.9e-12 | 1.9e-8 | 1.2e-11 | 1.2e-8 | 4.8e-7 |
| D2 | 3.8e-11 | 1.7e-11 | 2.9e-12 | 1.9e-8 | 2.3e-11 | 1.2e-8 | 4.8e-7 |
| M-DP | 3.3e-11 | 1.5e-11 | 1.5e-12 | 2.4e-8 | 1.5e-10 | 1.2e-8 | 4.0e-7 |

Every term within the 1e-6 gate. The Coulomb residual is the engine constant, named: the
field's `AMBER_COULOMB` 332.0637141 against LAMMPS `real`'s `qqr2e` 332.06371 (1.2e-8).
The improper's 2e-8 is relative to energies of ~7e-7 kcal/mol (absolute 2e-14).

**MD** (unit 3) — 48 held-out molecules (13–50 atoms) per model: minimise, 20 ps
Langevin 300 K, 50 ps NVE at 0.5 fs, no constraints. Statistics over the runs that did
not break.

| model | broken | bond / r0 range | RMSD to DFT, median / mean / p90 / max (Å) | \|NVE drift\| median / max (kcal/mol/ns/atom) | σ(E)/⟨KE⟩ | parameterise (s, CPU) |
|---|---|---|---|---|---|---|
| B | 0 / 48 | 0.65 – 1.11 | 0.140 / 0.222 / 0.545 / 1.60 | 0.0039 / 0.015 | 0.10 % | 0.18 |
| D2 | **1 / 48** | 0.43 – 1.11 | 0.162 / 0.230 / 0.529 / 1.13 | 0.0035 / 0.022 | 0.10 % | 0.18 |
| M-DP | 0 / 48 | 0.84 – 1.10 | 0.172 / 0.259 / 0.802 / 1.07 | 0.0045 / 0.022 | 0.10 % | 1.10 |

*The broken molecule* (val row 2747, C4H8N2O, a zwitterion N⁻–C(=O)–N⁺H₂–C whose C–N⁺
bond DFT relaxes to 1.664 Å). B's bond head puts that bond on its window floor
(LAMMPS K 40, molrs k 80 — `BondHarmonic.window_constants`) at r0 1.607, beside charges
+0.69 / −0.80 on its ends. In MD the bond compresses to 0.65 r0 under B (unphysical, not
past the 0.5 threshold) and to 0.43 r0 (0.70 Å) under D2, whose LJ head gives the N⁺
ε = 1.88 kcal/mol (B 0.48). M-DP's encoder reads the same bond as K 162, r0 1.665 — the
DFT length — and the molecule is quiet (0.84 r0 at worst). Every other run of every model
keeps all bonds within 0.8–1.11 r0.

Energy is conserved to ~0.1 % of the kinetic energy everywhere. Minimised geometries are
close to DFT for all three; D2 and M-DP beat B on 24 and 22 of the 48 molecules, so the
median gaps (0.14 vs 0.16–0.17 Å) are within molecule-to-molecule scatter — no model is
better at geometry, despite D2's and M-DP's better relaxation energies (specs 03, 05).
Of each model's three worst RMSDs, most have bond lengths within 0.03 Å of DFT (the
difference is conformational); the exceptions are bonded: val row 1373, an N–O bond DFT
puts at 1.564 Å, predicted r0 1.379 (B) / 1.366 (M-DP), and row 2747 above.
Parameterisation is once per topology: 0.18 s (EGT) vs 1.1 s (MACE) per molecule on 8 CPU
threads, the MACE time including all 45 relaxation frames of the row (the encoder reads
the profile's first).

**Verdict.** Export is exact for every encoder (`Rondo.parameterize` → molpy → LAMMPS).
The MD gate is met by B and M-DP and **missed by D2** on one charged molecule — and B
only escapes by threshold: the root is B's bond head saturating at its k floor on an
unusual C–N⁺, which D2's larger LJ ε turns into a collapse. A conformer-reading encoder
(M-DP) recovers that bond. Open: the bond head's k floor on charged bonds (a constraint
or a wider window; D2 is not production-ready until then), an independent (non-THEMol)
or scaffold-disjoint test set (unit 4, §2), condensed-phase MD.
