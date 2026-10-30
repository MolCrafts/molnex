# Rondo's covalent-graph encoder and heads, ported from ByteFF-26; Rondo-owned interfaces

| Field | Value |
|---|---|
| Opened | 2026-10-28 |
| Chain | rondo-enc (charter `rondo-enc-00-charter.md`) |
| Status | Units 1–3 landed (committed 2026-10-29 with phase 2); full reproduction passed through the in-tree `ByteFF.load`; encoders moved to `molrep.encoders`; data-side unit landed (molpy `assign_symmetry_classes` / `perceive_linear_dihedrals` / `assign_small_rings`, MolHub columns on molpy 0.20, one-row impropers); column parity 2998 / 3000 THEMol molecules, the two misses are ByteFF's resonance step (§8); **end-to-end gate passed 2026-10-28** (MolHub frames → padded Class-I data path → `ByteFF.load`, 499 / 500 molecules, §8) — phase 1 complete |
| Product | `molzoo.rondo.Rondo` |
| Reference | `ByteDance-Seed/byteff2@663451d65c72d6c1d556494ab8c0897bbdc0383a` (tag `v1.1.0`), read-only at `work/lab-new/references/byteff2` |
| Weights | HF `ByteDance-Seed/byteff2`, revision `v1.1.0`, `ByteFF-26/models/member-0{1..5}/optimal.pt` (Apache-2.0), at `molcrafts/.build-alloc/byteff-weights/`; sha256 below |
| Paper | Zheng *et al.*, "Data-driven parametrization of molecular mechanics force fields for expansive chemical space coverage", *Chem. Sci.* 2025, arXiv [2408.12817](https://arxiv.org/abs/2408.12817) |

## 0a. Package layout (operator, 2026-10-28, latest — supersedes §0 where they differ)

- molzoo subpackages are networks (`mace`, `allegro`, `pinet`, `rondo`,
  `byteff`); encoders and their blocks are `molrep/encoders/<arch>/`
  (`covalent_graph`, `message_passing`, `mace`, `allegro`, `pinet`); terms and
  heads are molpot.
- `molzoo/byteff/` is the published network: `ByteFF(Rondo)` with
  `BondChargeCorrection` + `LJCutDirect`, and its official loader **in tree**
  (`ByteFF.load(path, forcefield, *, dtype)` over `molzoo/byteff/checkpoint.py`
  `ByteFFKeys`). The out-of-tree `convert.py` / `keys.py` are superseded.
- The Espaloma-style encoder stays as the second encoder,
  `MessagePassingEncoder`, returning `EncoderFeatures`.
- `CheckpointRemap(conversions={type: fn})` replaces `ConvertedWeight`
  (`molzoo.mace.checkpoint.MACE_CONVERSIONS`).

## 0. Naming and weights (operator, 2026-10-28, supersedes the conflicting parts of §1)

- Nothing under `src/` is named after the source model. Modules are named for
  what they do (`CovalentGraphEncoder`, `EdgeAttentionLayer`, `ChemicalEmbedding`,
  `TanhWindow`, `SymmetricHead`, `LJCutDirect`, `BondChargeCorrection`,
  `RondoBCC`); docstrings keep the paper reference.
- `src/` has no reader of the upstream checkpoint format. The weights are
  converted **out of tree** (`work/lab/projects/rondo-enc/parity/convert.py`,
  `keys.py`) into an ordinary `state_dict` of our modules and loaded with
  `load_state_dict`.
- The upstream combined nonbonded block ("LJEs") has no counterpart: its σ/ε
  MLPs load into the LJ term (`LJCutDirect`, `pair:lj/cut`) and its bond charge
  corrections into the Coulomb term (`BondChargeCorrection`, `pair:coul/cut`),
  two independent terms. A registered `coul/long` style is molpot work, not
  this spec (open item).
- The five-member parameter mean is a caller composition over
  `Rondo.predict` / `Rondo.price`, done out of tree.

## 1. Rulings this spec implements (operator, 2026-10-28)

1. ByteFF replaces Espaloma. `molrep.chem.ChemEncoder` and its stack are deleted; earlier Rondo
   checkpoints are not kept.
2. Reuse ByteFF-26 where it works, but its internals never constrain Rondo:
   - the encoder is ByteFF-26's EGT with its pretrained weights; its adapter is the identity;
   - the bonded heads are ByteFF-26's (`PreMMBonded`) with their pretrained weights, including
     the symmetric assembly of bond / angle / proper / improper features;
   - charges stay D1 (EEM, MBIS-supervised) and LJ stays D2 (GAFF2-anchored, exp windows);
   - energy evaluation stays modular: a head or a potential changes without touching the encoder.
3. Order: reproduce ByteFF-26 exactly → replace charge and LJ heads with EGT and bonded heads
   frozen → decide on joint fine-tuning.
4. Rondo owns both interfaces: encoder → heads (`EncoderFeatures`) and heads → potentials
   (`Term.predict` columns, molrs IR conventions).

"Train from scratch" applies to the old Rondo checkpoints, the D1/D2 heads and the MACE encoder,
not to the ByteFF-26 encoder and bonded heads.

## 2. The interfaces

```
frame ─▶ encoder ─▶ EncoderFeatures(atom (N, 256), bond (N_bonds, 256))
                     ├─ bonded Terms: ByteFF symmetric assembly → (k, r0), (k, θ0), k_1..4, k_imp
                     ├─ LJCut (D2), PairCoulCut/EEM (D1)              read .atom
                     └─ RondoBCC: LJCutDirect (σ, ε) + BondChargeCorrection (charges) — two terms
      ─▶ Potentials(frame, params)   (molpot, molrs IR: LAMMPS K without ½, degrees)
```

`EncoderFeatures` (molrep, so molpot heads can import it) carries exactly two tensors, in the
batch atom order and `bonds.bond_index` order. Bond features are undirected (ByteFF averages the
two directions after the last layer). Equivalence averaging is part of the contract: the atom
and bond features of topologically equivalent atoms/bonds are equal.

## 3. Paper ↔ code map (ByteFF2 v1.1.0 → molnex)

Config of every ByteFF-26 member (`fftrainer_config_in_use.yaml`, identical across members):
`feature_layer: atom_embedding_dim 32, node_mlp (64, 256, 2), edge_mlp (64, 256, 2), gelu`;
`gnn_layer: EGT, gnn_dims (256, 256, 4), heads 16, at_channels 16, ffn_dims (256, 2), jk cat`;
`ff_block: MMBonded (pre_mlp (256,256,3), post_mlp (256,256,1), out_mlp (256,4), tanh 20)`,
`LJEs (pre_mlp (512,512,4), out_mlp (256,3), tanh 20)`; elements H C N O F P S Cl Br I.

| ByteFF2 | Location | molnex |
|---|---|---|
| Atom input: `[Z−1, degree, formal_charge, ring_connectivity, min_ring_size]`, five embeddings **concatenated** (32+8+8+8+8), `node_mlp` (PyG `MLP`, `plain_last=False`: GELU after the last layer too), then equivalence mean | `graph_block.py:FeatureLayer` | `molrep/encoders/covalent_graph/layers.py` `ChemicalEmbedding`; element table 86 rows (Z−1) |
| Bond input: `[in_ring, order ∈ {single, double, triple, aromatic}]`, embeddings concatenated, `edge_mlp`, equivalence mean | same | same module; `bonds.bond_type` 1/2/3/4 → index 0/1/2/3 |
| `edge_in_lin`, 4 × `EGTConv` (pre-LN attention over directed covalent edges: logit `q_i·k_j/√16 + lin_edge(LN e)`, softmax per target, gated by `σ(lin_gate(LN e))`, residual + pre-LN FFN; edge stream `e += lin_proj_edge(logits)`, then pre-LN FFN residual), GELU after every layer (jk set), JK concat (4×256) → `lin` → 256, `edge_out_lin`, mean of the two directions | `gnn.py:EGTConv`, `BasicGNN.forward`, `graph_block.py:GNNLayer` | `molrep/encoders/covalent_graph/layers.py` `EdgeAttentionLayer`; `molrep/encoders/covalent_graph/encoder.py` `CovalentGraphEncoder` |
| Directed edge order: bond *b* → edges `2b` (i→j), `2b+1` (j→i); message flow source → target (PyG default) | `data.py:265-275, 296` | built in the encoder from `bonds.bond_index`, no host sync |
| Symmetric assembly: for each term, copies of `[x_i, e_ij, x_j, …]` (bond, angle, proper: forward + reversed; improper: three legs `[x_c, e_c,leg, x_leg]`), `pre_mlp[term]` on each copy, **summed**, `post_mlp[term]` (built with `pre_mlp_dims`), `out_mlp[param]` | `mm_bonded.py:_symmetric_pooling` | `molpot.classical` bonded terms (rewritten) |
| Output transform: `20·tanh(d)·std + mean`, straight-through clamp to `[min, max]`; (std, mean, min, max): bond_k (200, 700, 80, 4000), bond_r0 (0.19, 1.3, 0.5, 5.0), angle_k (60, 130, 40, 1000), angle_d0 (10, 120, 20, 180), proper_k (1.2, 0.4, −20, 20) ×4, improper_k (3.7, 4.3, 0, 20) | `mm_bonded.py:PreMMBonded` | same, in the term heads |
| Hypervalent patch: P/S with degree ≥ 5 → bonds touching them k=1000, r0=0; angles centred on them k=278.44, θ0=0; propers/impropers containing them k=0 | `mm_bonded.py:patch_bonded_terms`, `data.py:301-323` | in the term heads, from `atoms.Z` and the degree (computed from `bonds.bond_index`) |
| Linear-proper mask: SMARTS `[*:1]~[#6X2;!r5;!r6:2]~[*:3]~[*:4]`, `[!$([#7H1]):1]~[#7X2+:2]~[!$([#7H1]):3]~[*:4]` → `proper_k = 0` | `mol_utils.py:match_linear_proper` | input column `propers.linear` (MolHub) |
| Energies: bond `½k(r−r0)²`, angle `½k(θ−θ0)²` (θ0 in degrees), proper `Σ_{n=1..4} k_n(1+cos(nφ − φ_n))`, `φ_n = 0` odd / `π` even, improper `k(1+cos(2φ − π))` on **one** row per C/N centre with three neighbours, legs sorted ascending | `ff_kernels.py`, `data.py:get_impropers` | IR conversion at the head output: `K_bond = k/2`, `K_angle = k/2` (LAMMPS no-½), torsion and improper map to `dihedral:periodic` / an improper style with the same phase convention |
| Equivalence classes: RDKit `CanonicalRankAtoms(breakTies=False)` on the graph with every bond single, charges 0; bond class = sorted pair of atom ranks | `bytemol/core/rkutil/symmetry.py`, `mol_utils.py:find_equivalent_index` | input column `atoms.equi_index` (MolHub, producer in §5); a bond's class is derived in the model from its endpoints' atom classes (`Equivalence.bonds`) |
| Canonical resonance structure before featurization | `bytemol/core/rkutil/resonance.py:get_canonical_resoner` | MolHub writes bond orders and formal charges of that structure; parity compares inputs first |
| LJEs: BCC charge `q = equi-mean(formal) + Σ ±20·tanh(out(pre(fwd) − pre(rev)))`, `σ, ε = 20·tanh(mlp) + 20`; arithmetic σ / geometric ε, 1-4 scales 0.8333 / 0.5, no cutoff | `mm_nonbonded.py` | split into two terms of `RondoBCC`: `LJCutDirect` (σ, ε) and `BondChargeCorrection` (charges); `Rondo` keeps D1/D2 |
| Five members combined: the **mean of each final parameter** (after transform, clamp, patch, mask) over the members; `std / max(mean, 1)` is reported as uncertainty. Energy is priced on the mean parameters, not averaged | `hybrid_ff.py:EnsembleModel` | an ensemble averages parameter tables before `Potentials`; member-level parity first, then the mean |

Not ported: torch_geometric, bytemol, RDKit (molnex surface rule). Featurization that needs
cheminformatics (equivalence, resonance, linear SMARTS, improper rows) is a MolHub column.

## 4. Data contract changes (Rondo inputs)

Added: `atoms.equi_index` (int64, the index of the first atom of the atom's
topological class; `collate_molecules` rebases it by the atom offset, the padder
makes every ghost its own class) and `propers.linear` (bool, ByteFF's SMARTS
mask). Bond classes are not a column: `Equivalence.bonds` derives them from the
atom classes with one sort, which reproduces ByteFF's partition exactly
(parity §8). Changed: `impropers` is **one row per C/N centre with three
neighbours**, centre first, legs ascending (ByteFF2), replacing the three
cyclic rows (A15). Dropped as an encoder channel: `graphs.total_charge`
(ByteFF-26 does not read it; EEM still does). Degree is computed in the encoder
from `bonds.bond_index`.

## 5. Producers of the encoder's input columns

RDKit is not on the molhub/molnex surface. Every column that reads the molecule's
form is computed in molpy (Rust core, Python binding) and written by MolHub
(`molhub/dataset/themol_format.py`, row cache `themol.rows.v3`):

| column | molpy | rule |
|---|---|---|
| `atoms.equi_index` | `perceive.assign_symmetry_classes` | colour refinement on the element graph, bond orders and charges erased; seeds (element, degree, declared H, **ring bonds at the atom**). Not the SSSR ring count: on a bridged system the SSSR keeps some of a set of equivalent rings and splits equivalent atoms (bicyclo[2.2.2]octane; 8 / 500 THEMol molecules before the fix) |
| `propers.linear` | `perceive.perceive_linear_dihedrals` | ByteFF's two SMARTS on molpy's matcher (`smiles` feature) |
| `bonds.in_ring`, `atoms.ring_connectivity`, `atoms.min_ring_size` | `perceive.assign_small_rings(mol, 8)` | every simple cycle of at most 8 atoms (`bytemol` `MAX_RING_SIZE`), not the SSSR: a macrocycle bond on no small ring is not a ring bond (4 / 3000 porphyrin-like molecules before the fix) |
| `bonds.bond_type` | SMILES reader | a `/` or `\` between two aromatic atoms is aromatic (RDKit, Daylight) and keeps its direction (10 / 500 before the fix) |

Gate: on the parity set, every column equals the oracle's for every molecule; a
mismatch is listed by SMILES in §8, not tolerated silently.

## 6. Units of work

| Unit | Content | Done when |
|---|---|---|
| 1 | `CovalentGraphEncoder` + `layers.py` | key table covers every `graph_block.*` key; features equal the oracle |
| 2 | bonded terms rewritten (assembly, transform, patch, mask, IR conversion) | parameters and bonded energy equal the oracle |
| 3 | `Rondo` wiring; `predict` / `price` primitives; `RondoBCC`; out-of-tree conversion (`convert.py`) to plain `state_dict`s | all 260 upstream tensors land in our modules, none missing |
| 4 | equivalence columns (molrs → molpy → molhub), `propers.linear`, ByteFF improper rows | column gate in §5 — **done** 2026-10-28 except ByteFF's resonance step (§8) |
| 5 | delete the Espaloma stack and `CoordinateEncoder` | no consumer left, tests mirrored |

## 7. Gates

- **Parity** (out of tree, `lab/projects/rondo-enc/parity/`, oracle env
  `lab-new/references/byteff2-oracle-env`): each member, ~100 THEMol molecules plus charged,
  linear, ring, hypervalent P/S cases. Equal inputs, atom/bond features, every bonded parameter
  after conversion, BCC charges and σ/ε, bonded and total MM energy. fp64 relative ≤ 1e-10.
- **Unit tests** (fast tier, mirrored): EGT permutation equivariance; bond reversal and
  equivalent atoms give equal features; angle i↔k, proper reversal and improper leg permutation
  give equal parameters; the IR conversions; the key table is complete.
- **Compile** (slow): `Rondo.energy` and `forward` explain to 0 graph breaks on a padded batch.

## 8. Record

> **Layout since 2026-10-28** (operator ruling: encoders and their blocks live in molrep): the
> encoder is `molrep/encoders/covalent_graph/` (`layers.py`, `encoder.py`, `spec.py`),
> `RondoBCC` is `molzoo.byteff.ByteFF` (`potential.py`, key table `checkpoint.py:ByteFFKeys`),
> `EncoderFeatures` is `molrep.encoders.features`. The record below names the layout as it
> landed on 2026-10-28; the names above are where the code is now.

Module layout as landed: `molzoo/rondo/layers.py` (`ChemicalEmbedding`, `EdgeAttentionLayer`,
`Equivalence`), `encoder.py` (`CovalentGraphEncoder`), `checkpoint.py` (`the out-of-tree key table (`lab/projects/rondo-enc/parity/keys.py`)`),
`Rondo.predict` / `Rondo.price` / `RondoBCC`; `molpot/classical/heads.py`
(`SymmetricHead`, `TanhWindow`, D2 `LJHead`), `terms.py` (bonded terms on
ByteFF heads, `Hypervalent`); `molrep.chem.features.EncoderFeatures`,
`molrep.embedding.mlp.DenseMLP`. Every model piece is an `nn.Module`
(operator rule 2026-10-28).

Parity 2026-10-28 (`lab/projects/rondo-enc/parity/`, oracle dump `oracle-v1.pt`,
27 molecules incl. PF6⁻, SF6, azide, nitro, allene, caffeine; all five members,
fp64): encoder atom features ≤ 2.1e-15 and bond features ≤ 1.8e-15 relative,
degree exact; bonded parameters ≤ 2.6e-14; bond and improper energies ≤ 4.3e-14;
angle ≤ 2.5e-9 (azide, 3.5e-3 kcal/mol); proper ≤ 4.2e-8 — the oracle builds
its 180° phase as `int64 tensor × π` in float32 (π off by 8.7e-8 rad,
`ff_kernels.py:386`), molnex uses exact 180°. Full reproduction (`parity_full.py`, oracle `oracle-v2.pt`, converted
`state_dict`s in `converted/`; `RondoBCC` on the force field of `convert.py`):
every parameter of every member — bonded, BCC charges, σ, ε — ≤ 2.7e-14
relative; the five-member mean ≤ 8.6e-15; total energy (bonded + 1-4 + all
pairs) ≤ 2.7e-7 relative per member and 4.9e-8 for the ensemble. The largest
residual is formamide (members 2, 4: totals −1.13 / 0.47 kcal/mol, 1.1e-7 /
1.3e-7 kcal/mol absolute), the same float32-π proper phase. `PairRelation`'s
1-2/1-3/1-4 classes reproduce the reference's exclusion and 1-4 lists on every
ring system in the set (cyclopentane, dioxane, pyridine, caffeine). All 260
upstream tensors land in `RondoBCC`'s modules, none missing.


Weights sha256: member-01 `04db22bc…387d5b`, member-02 `707e2951…afb71a`, member-03
`a588f64b…46311d`, member-04 `021e8a9d…c5c1e6`, member-05 `1d2f8aea…61dfce`; the five configs
are byte-identical (`be02c06e…caa757`). Full digests in `.build-alloc/byteff-weights/SHA256SUMS`.

Column parity 2026-10-28 (`lab/projects/rondo-enc/parity/themol_columns.py` in the molhub
env → `themol_compare.py` in the oracle env against ByteFF2's own `MonoData`; THEMol
`hessian@1` shard-000, first 3000 records): element, ring connectivity, smallest ring,
`equi_index` partition, bond set, `in_ring`, linear-proper set and proper count equal on
every molecule. Formal charge and bond order differ on 2 / 3000, both ByteFF's
`get_canonical_resoner` (RDKit `ResonanceMolSupplier`, fewest charges, then the
lexicographically smallest RDKit SMILES), which rewrites the input before featurizing:
`[B:1]1=[C-:2]…` → `[B-]#[C]`, and a quinoxaline di-N-oxide whose `[n+][O-]` it writes
`n=O`. Not ported — it is defined by RDKit's enumerator and canonical SMILES order;
molhub keeps the charges and orders the SMILES states. Deviation **R1**, open: the
encoder sees different inputs on such molecules, so their parameters can differ from
ByteFF-26's.

End-to-end gate 2026-10-28 (`lab/projects/rondo-enc/parity/e2e_parity.py`, job
`.jobs/e2e-parity.sbatch`; THEMol `hessian@1` shard-000 rows 0–499): MolHub frames through
the training data path — `ClassIPrepare` per sample, `MoleculeCollator`, neighbour list at
50 Å, `PairRelation`, `TotalCharge`, `MoleculePadder` (ghost atoms, dead valence rows),
graph schema — in batches of eight, priced by each `ByteFF.load` member and the
five-member mean, against `oracle_dump.py` on the same mapped SMILES and THEMol
coordinates. Every parameter keyed by its atoms: bonded ≤ 2.2e-13, charges ≤ 5.0e-11,
σ ≤ 6.7e-15, ε ≤ 6.7e-13 relative (members; the mean ≤ 3.5e-12). Energies: every term but
the torsion ≤ 1.2e-10 kcal/mol absolute; the torsion ≤ 2.5e-6 kcal/mol, the oracle's
float32-π phase (above). One molecule is R1 (the boron anion) and is excluded. Found
and fixed on the way: the encoder read a padded batch's dead bond rows (all joining two
ghosts) as degree and bond order, out of the embeddings' range — it now counts live
bonds only (`bonds.mask`); the improper term carried every trivalent centre that molpy
emits, where the reference's `get_impropers` takes C and N only — `ImproperCvff` now
zeroes the others (22 / 500 molecules: three-coordinate B, S⁺, sulfoxide S);
`label_sample`'s valence graph is built from Z and bonds (molpy 0.20 refuses the
frame's int64 columns).
