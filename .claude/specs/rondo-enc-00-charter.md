# rondo-enc: one force field, interchangeable encoders

| Field | Value |
|---|---|
| Opened | 2026-10-28 |
| Status | charter; 01, 03, 05 and 07 done, 02 implemented, 04 / 06 not opened |
| Source | operator's "ByteFF-like Modular Encoder — Development Specification v0.1" (2026-10-28), bound here to molnex types |
| Product | `molzoo.rondo.Rondo` (`src/molzoo/specs/rondo.md` stays the math contract and is rewritten with 01's code) |

## Hypothesis

Different molecular encoders can be trained independently and calibrated to one feature
interface, so a frozen set of force-field heads is reused across them without losing physical
accuracy. Failure is a valid outcome: it says a geometry-dependent backbone is not an economical
replacement for topology-based static parameterization.

## What the source draft becomes in this tree

| Draft | Here |
|---|---|
| `CovalentGraphEncoder`, reference head | ByteFF-26's EGT and `PreMMBonded`, ported with official weights (01) |
| `MACEEncoder` | `MaceConformerEncoder` in `molrep/encoders/mace_conformer/` over `molrep.encoders.mace` blocks (02) |
| `CanonicalAdapter` | owned by each encoder: identity for ByteFF-26, `nn.Linear` for MACE; no adapter class |
| `EncoderFeatures` | `molrep.chem.features.EncoderFeatures(atom, bond)`, replacing `ChemEmbeddings` |
| `encoder.encode(batch)` | `Rondo.features(frame, *, encoder=)`; evaluates no head |
| standalone trainer | the existing `Trainer` / `JointTrainStep` (accumulation, autocast, checkpoints already there); no new trainer |
| temporary pretraining heads | caller-side modules in the pretraining step, never on `Rondo` |

Binding rules that override the draft: no `src/rondo/`, no YAML tree, no factory constructors,
no training-stage arguments on `Rondo`, no position-reading encoder per MD step (the MACE encoder
runs on a reference conformer at parameterization time only), dtype from the caller.

## Phases

Operator outline 2026-10-28 (supersedes the earlier five-phase table). Near term: phases 1
and 2; no large-scale encoder pretraining first, and ByteFF is never retrained from scratch.
Three interfaces stay independently replaceable — encoder → (canonical adapter) → physical
heads → (parameters) → energy model — so each experiment changes one component.

| # | Spec | Content | Gate | State |
|---|---|---|---|---|
| 1 | `rondo-enc-01-graph-encoder` | modularize ByteFF-26: EGT encoder + weights, `EncoderFeatures` (adapter = identity), bonded / charge (BCC) / LJ heads as separate terms, encoder ∥ heads ∥ potentials; the previous message-passing encoder kept as a second encoder | numerical agreement with ByteFF-26, all five members; MolHub columns equal the oracle's | done; end-to-end gate passed 2026-10-28 (499 / 500); open R1 (resonance step) |
| 2 | `rondo-enc-03-heads` | D1 (EEM, MBIS) and D2 (GAFF2 LJ, exp windows) with EGT + bonded heads frozen; D1, D2, both; energy / force / charge / torsion-profile comparison; constrained joint fine-tune if needed | full production checklist | done 2026-10-29 (member-01, quick comparison): D2 (GAFF2-window LJ) beats B on every metric; D1 (EEM/MBIS) charges 6× closer to MBIS but torsion +14 % under frozen amplitudes; D1+D2+FT best relaxations, torsion not recovered (spec 03 §8) |
| 3 | `rondo-enc-04-pretrain` (not opened) | one training interface for EGT and MACE; Uni-Mol-style masked atom/bond, geometry denoising, physical supervision; curated THEMol subsets; random vs ByteFF vs re-pretrained init | transfer beats random init | not started; parallel to 2, not its prerequisite |
| 4 | `rondo-enc-02-mace`, `rondo-enc-05-swap` | MACE encoder + MACE → canonical adapter under the frozen reference heads; feature distillation + physical supervision; cross-conformer consistency; EGT vs MACE cost | MACE replaces EGT without retraining the heads, or a clear no | done 2026-10-30 (spec 05): yes at the 10 % bar — M-DP (distillation + torsion/relax, from M-D) is within 2.4 % of B on charge, torsion, relax F and 12 % better on relax E; distillation alone (M-D) misses relax by 25 %; EGT fine-tuned on the same physics (phase 2 D2) still beats M-DP by 5–18 %; MACE forward 1.3–2× EGT |
| 5 | `rondo-enc-06-residual` (not opened) | new bonded forms; optional `ΔE_ML(G, R)`; static MM vs hybrid ML/MM; OMol25 intermolecular data if needed; reconcile with `rondo-geom-01` first | NVE, throughput, held-out accuracy | not started |
| 6 | `rondo-enc-07-validation` | encoder × head × physics grid; THEMol + independent sets; MD energy conservation / stability; GPU time, memory, initialization; one active encoder at inference; static parameters computed once per unique topology and cached | benchmark + an MD-ready Rondo | done 2026-10-30 (lean, spec 07): `Rondo.parameterize` → molpy → LAMMPS at parity (≤ 2.4e-8) for B, D2, M-DP; 48 held-out molecules each: B and M-DP stable, D2 collapses one zwitterion's C–N⁺ (B's bond k at its window floor, D2's large LJ ε), M-DP recovers that bond; NVE drift ~0.004 kcal/mol/ns/atom, geometry RMSD 0.14–0.17 Å; open: bond-k floor on charged bonds, independent set, condensed phase |

Datasets: THEMol for 1–4 and 6 (H C N O F P S Cl Br I, B3LYP-D3(BJ)/DZVP, CC BY-NC 4.0 — a
commercial use of a THEMol-trained model needs a licence check first). OMol25 only in 5, under
its own protocol; THEMol and OMol25 labels are never mixed as one level of theory.
Splits are by molecule (`THEMolShardDataset.molecule_index()`), plus a scaffold-disjoint set.

## Stale evidence

`rondo-train-01-static-c1`'s calibration (job 4282710) was taken on the Espaloma encoder and
does not carry over.
