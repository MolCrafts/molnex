# Rondo Specification

This page is the implementation contract for `molzoo.rondo`. It is not a
tutorial; use the MolZoo user guide for theory narrative and worked examples.

| Field | Value |
|-------|-------|
| Module | `molzoo.rondo` |
| Entry point | `Rondo(forcefield, *, dtype, encoder=None, **settings)`. One force field (`forcefield.canonical()`), priced by one `Potentials`; `Rondo.graph` is the encoder — `molrep.encoders.covalent_graph.CovalentGraphEncoder` (ByteFF-26's edge-augmented graph transformer, no coordinates) built from `settings`, or the encoder passed as `encoder=` — returning `EncoderFeatures(atom, bond)`; one `molpot.classical` term per priced style under `Rondo.heads`. `predict` / `price` are the primitives `energy` composes; `tables` writes the predicted columns without an energy; `forward` differentiates `atoms.pos` and writes `atoms.forces_pred`; `parameterize` writes one molecule's static field as molpy data. The published pair heads are `molzoo.byteff.ByteFF`. |
| Paper | Encoder and bonded heads: Zheng *et al.*, "Data-driven parametrization of molecular mechanics force fields for expansive chemical space coverage", *Chem. Sci.* (2025) — ByteFF, ported with the ByteFF-26 weights. Lineage: Wang *et al.*, "End-to-end differentiable construction of molecular mechanics force fields", *Chem. Sci.* **13**, 12016 (2022) — Espaloma. |
| arXiv | [2408.12817](https://arxiv.org/abs/2408.12817) (ByteFF); lineage [2010.01196](https://arxiv.org/abs/2010.01196) (Espaloma) |
| DOI | [10.1039/D2SC02739A](https://doi.org/10.1039/D2SC02739A) (Espaloma) |
| Reference implementation | Encoder and bonded heads: `ByteDance-Seed/byteff2@663451d65c72d6c1d556494ab8c0897bbdc0383a` (tag `v1.1.0`); weights HF `ByteDance-Seed/byteff2` revision `v1.1.0`, `ByteFF-26/models/member-0{1..5}/optimal.pt`. Training recipe line references (§3.7, §6): `bytedance/byteff@90ef54a7c38e58f162adc23f3e8f457aaf922d46`. Lineage: `choderalab/espaloma@413eb55074037540e1414eb365da7304bd5d72be` |
| Spec status | stable |

## 1. Scope

Rondo is MolNex's **learnable Class-I force field**: an encoder reads a
molecule and one head per style emits every *open* parameter of the Class-I
force field it is built on, and the same module evaluates that field through
the molpot potentials of the field's styles.

It is a *recipe*, in the MolZoo sense — the named composition of primitives
that already live in `molrep` and `molpot`. The force field **is** the
Hamiltonian: a molpy `ForceField` declares the styles, cutoff, mixing rule,
Coulomb constant and 1-4 scales once (operator ruling 2026-10-06);
`Rondo` keeps `forcefield.canonical()` and `Potentials.from_forcefield`
returns that one `Potentials`. Rondo holds it, adds the encoder and one
parameter head per style it prices, and owns no energy of its own.

Rondo owns two interfaces and nothing between them (`.claude/specs/rondo-enc-01-graph-encoder.md`
§1 ruling 4, §2): encoder → heads is `EncoderFeatures(atom (N, D), bond
(N_bonds, D'))`, nothing else; heads → potentials is `Term.predict(features,
frame, potential)`, columns in molrs conventions (LAMMPS `K` without ½,
degrees). No encoder builds angle, torsion or improper features; the bonded
heads assemble them from the atom and bond rows.

### 1.1 What Rondo owns

| Concern | Rondo surface |
|---------|---------------|
| The parameters a style leaves open, as functions of the features | one class per LAMMPS style in `molpot.classical.TERMS` (`BondHarmonic`, `AngleHarmonic`, `DihedralPeriodic`, `ImproperCvff`, `LJCut`, `PairCoulCut`), the same names as `molpot.potentials`, registered on `Rondo.heads` at their `head_key` (`bonds`, `angles`, `propers`, `impropers`, `lj`, `charge`) |
| The encoder | `Rondo.graph`: `CovalentGraphEncoder(spec.encoder, dtype=)` when `encoder=None`, else the injected module (e.g. `molrep.encoders.mace_conformer.MaceConformerEncoder`), registered as `graph` either way so the heads load by the same keys. The heads are sized from `graph.width` / `graph.bond_width`. `Rondo.features(frame)` is the encoder call; no head runs |
| Which terms the model carries | one `Potentials`, `Potentials.from_forcefield` of `forcefield.canonical()`. A later style is a force field that declares it; a subclass overrides `load_state_dict` to accept the parent's checkpoint |
| What a training step calls | `Rondo.predict` once per source, then per objective `Rondo.energy_at(frame, tables, styles=)` (every carried style, or a named subset), `Rondo.write_tables` (the columns, no energy) or `Rondo.partial_hessian(frame, tables)`; the step detaches the columns a loss does not name (`JointTrainStep.keep`) |
| Energy and derivatives as one model forward | `Rondo.forward` = `molix.autograd.Derivative(Rondo.energy, wrt=("atoms.pos",), outputs=(FORCES_PRED_KEY,))`, writing `atoms.forces_pred` |
| One molecule's static field | `Rondo.parameterize(frame, *, graph=0) -> (molpy.Frame, molpy.ForceField)` (§4) |
| The configuration contract | `RondoSpec(encoder=CovalentGraphEncoderSpec, heads=HeadsSpec)`, built from the constructor keywords (§4) |

### 1.2 What Rondo does not own

| Concern | Owner |
|---------|-------|
| The functional form, its settings and units | the molpy `ForceField` (molrs IR); `ForceField.canonical()` rewrites a typed `dihedral opls` as `dihedral periodic` |
| The torch implementation of each style | `molpot.potentials` (`bond:harmonic`, `angle:harmonic`, `dihedral:periodic`, `improper:cvff`, `pair:lj/cut`, `pair:coul/cut` — the styles `TERMS` has heads for) |
| Topological equivalence and per-batch bond lookup | `molrep.chem.equivalence.Equivalence`, `molrep.chem.bond_table.BondTable` |
| The published pair heads (bond charge corrections, direct σ / ε) and the ByteFF-26 loader | `molzoo.byteff.ByteFF` (`BondChargeCorrection`, `LJCutDirect`, `ByteFF.load`) |
| Which loss trains which parameters | the training step (`molix.core.steps.JointObjective.trainable`) |
| Pair relations (1-2 / 1-3 / 1-4) and the cutoff pair list | `molix.data.PairRelation`, `molix.neighbor.NeighborList` |
| Input columns: formal charge, ring features, bond order, equivalence classes, linear-proper mask, improper rows, GAFF2 reference σ/ε | molpy perception (`assign_symmetry_classes`, `perceive_linear_dihedrals`, `assign_small_rings`), written by MolHub (THEMol decoding; `rondo-enc-01` §5) |
| Serialising a static field for an engine | molpy (`molpy.io.write_lammps_data`, `molpy.io.lammps.LammpsForcefieldWriter`) |
| The per-term partial Hessian | `molpot.forcefield.partial_hessian.TermHessian` |
| The derivative op | `molix.autograd.Derivative` |
| Losses, padding, collation, batch tasks | `molix` |
| Datasets, splits, units, provenance | MolHub |
| Training configuration, scheduling, calibration evidence | MolExp / MolAb |

### 1.3 Lineage and the deliberate departures

Rondo sits in the Espaloma → ByteFF family: a GNN over the covalent graph
predicts MM parameters, and the MM energy is differentiated for forces. After
the 2026-10-11 realignment its *training* follows ByteFF — joint
multi-dataset training with per-loss parameter sets, equilibrium forces for
`r0` / `θ0`. Since 2026-10-28 (`rondo-enc-01`) its encoder and bonded heads
**are** ByteFF-26's — the edge-augmented graph transformer and `PreMMBonded`,
ported with the published weights and reproduced to 2.7e-14 (§7.1). Two choices
depart from ByteFF-26; `molzoo.byteff.ByteFF` is the same model without them:

| # | Rondo | ByteFF-26 (`molzoo.byteff.ByteFF`) | Consequence |
|---|-------|-------------------------------------|-------------|
| D1 | Partial charges from **electronegativity equalisation** (`PairCoulCut`: `molpot.charge.ChiEtaHead` → `ElectronegativityEqualization`), supervised by MBIS charges | bond charge corrections on the equivalence-averaged formal charges (`BondChargeCorrection`) | Charges respond to the solved electrostatic problem; one dense per-molecule solve; reads `graphs.total_charge` |
| D2 | Lennard-Jones σ / ε **learned around the atom's GAFF2 type** (`LJCut` → `LJHead`: `σ = σ_GAFF2·exp(0.1·raw)`, `ε = ε_GAFF2·exp(raw)`, windowed, zero start) | σ, ε predicted directly, `t·tanh(mlp) + t` (`LJCutDirect`) | Starts exactly on GAFF2; gas-phase intramolecular data gives little signal, so a stage that names `atoms.sigma` / `atoms.epsilon` carries a prior toward the reference (R1) |

The bond head's lower `K` clamp is also Rondo's: 80 kcal mol⁻¹ Å⁻² by default
against the published 40 (`HeadsSpec.bond_k_floor`, A27).

### 1.4 Status of this document

`stable`. The 2026-10-28 encoder / head revision (`rondo-enc-01`) is folded
into §1–§6. §5 carries no `unknown` row. §7.1 is claimed for the encoder and
the bonded heads (ByteFF-26 parity) and not for Rondo's own pair heads or any
trained model.

## 2. Public Contract

### 2.1 Required Inputs

| Direction | TensorDict path | Shape | Dtype | Contract |
|-----------|------------------|-------|-------|----------|
| In | `atoms.Z` | `(N,)` | int64 | Atomic numbers; the element embedding reads `Z − 1` (86 rows). Ghost rows carry the padder's `pad_atom_type` |
| In | `atoms.pos` | `(N, 3)` | the model's `dtype` | Positions, Å. Differentiated against. Never read by `CovalentGraphEncoder`; `MaceConformerEncoder` reads each profile's first conformer, detached |
| In | `atoms.batch` | `(N,)` | int64 | Graph membership |
| In | `atoms.formal_charge` | `(N,)` | int64 | Encoder channel, clamped to ±2. Also the starting charge of `BondChargeCorrection` (ByteFF) |
| In | `atoms.ring_connectivity` | `(N,)` | int64 | Encoder channel: ring bonds at the atom (embedding rows 0..6) |
| In | `atoms.min_ring_size` | `(N,)` | int64 | Encoder channel: smallest ring of at most 8 atoms through the atom, `0` none (embedding rows 0..8) |
| In | `atoms.equi_index` | `(N,)` | int64 | First atom of the atom's topological class; `collate_molecules` rebases it, the padder makes every ghost its own class. Atom and bond features are averaged over these classes |
| In | `graphs` | `(B,)` namespace | — | Required: the graph count is `frame["graphs"].batch_size[0]` |
| In | `graphs.total_charge` | `(B,)` | the model's `dtype`, integer-valued | The EEM constraint $Q$ (D1). Required iff `pair coul/cut` is declared; not an encoder channel |
| In | `atoms.gaff2_sigma`, `atoms.gaff2_epsilon` | `(N,)` | the model's `dtype` | Reference σ (Å) / ε (kcal/mol) of the atom's GAFF2 type (D2). Required iff `pair lj/cut` is declared (`KeyError` naming the column); a `NaN` reference makes a `NaN` parameter rather than pricing silently |
| In | `bonds.bond_index` | `(2, N_bonds)` | int64 | Covalent topology. Bond features and the bond table are in this order |
| In | `bonds.bond_type` | `(N_bonds,)` | int64 | 1 single, 2 double, 3 triple, 4 aromatic; `0` on a dead row |
| In | `bonds.in_ring` | `(N_bonds,)` | int64 | Encoder channel, 0 / 1 |
| In | `angles.atomi` / `atomj` / `atomk` | `(N_angles,)` | int64 | Valence angles; `atomj` is the vertex |
| In | `propers.atomi` … `atoml` | `(N_propers,)` | int64 | Proper torsions |
| In | `propers.linear` | `(N_propers,)` | bool | ByteFF's linear-centre SMARTS (molpy `perceive_linear_dihedrals`); a missing column raises `KeyError` by name |
| In | `impropers.atomi` … `atoml` | `(N_impropers,)` | int64 | **One row per three-connected centre**, centre first, legs ascending (ByteFF's `get_impropers`). Only a C or N centre carries the term |
| In | `impropers.centre`, `leg1` / `leg2` / `leg3` | `(N_impropers,)` | int64 | Optional: the chemical centre and legs when the stored order is another engine's (AMBER puts the centre third). Features come from these, the dihedral from the stored order |
| In | `pairs.pair_index` | `(E, 2)` | int64 | Cutoff pair list at the force field's pair cutoff; read when a pair style is declared, and by `MaceConformerEncoder` (both directions) |
| In | `pairs.shifts` or `pairs.pair_diff` | `(E, 3)` | the model's `dtype` | Lattice remainder of each pair for the pair styles. `MaceConformerEncoder` reads `pair_diff` detached as its edge vectors |
| In | `pairs.relation` | `(E,)` | int8 | `0` none, `1`/`2`/`3` = 1-2/1-3/1-4, from `molix.data.PairRelation` |
| In | `graphs.profile` | `(B,)` | int64 | Only with `MaceConformerEncoder`: batch-local profile id (`molix.data.ProfileIndex`); the first graph of a profile is its reference conformer |
| In | `hessian_pairs`, `<ns>.hpair_<i>_<j>` | — | — | Only for `partial_hessian` |
| In | `atoms.mask` / `graphs.mask` / `<ns>.mask` | bool | — | Ghost-row, dead-slot and dead-term masks under static shapes; `bonds.mask` also restricts the degree and the hypervalent count to live bonds |

A batch that lacks a column the encoder or a priced style reads raises
`KeyError` on that key.

### 2.2 Outputs

`Rondo.forward` returns the batch. `energy` writes `graphs.energy` and every
column the priced styles' heads predict, onto each namespace the batch holds;
`forward` adds the force at `atoms.forces_pred` (`FORCES_PRED_KEY` in
`molix.schema`, the op's output key for `atoms.pos`) — a training batch holds
the DFT force at `atoms.forces` (A6). The energy lands where a labelled batch
holds its DFT energy; where it must sit beside that label, the training step
wraps the call in `molix.frame.beside_labels` (`graphs.energy →
graphs.energy_pred`, `ENERGY_PRED_KEY`), as `molix.core.steps.JointTrainStep`
does. The model moves nothing. `tables` writes the same parameter columns and
leaves `graphs.energy` untouched.

| Column | Shape | Written by | Contract |
|--------|-------|------------|----------|
| `graphs.energy` | `(B,)` | `energy` | Class-I energy, kcal mol⁻¹; a training step copies it to `graphs.energy_pred` beside the label (`beside_labels`) |
| `atoms.forces_pred` | `(N, 3)` | `forward` | `-dE/dpos`, kcal mol⁻¹ Å⁻¹ |
| `bonds.k`, `bonds.r0` | `(N_bonds,)` | `energy`, `tables` | LAMMPS `K` (kcal/mol/Å², half the head's `k`), `r0` (Å) |
| `angles.k`, `angles.theta0` | `(N_angles,)` | `energy`, `tables` | `K` (kcal/mol/rad², half the head's `k`), `θ0` **degrees** |
| `propers.k` | `(N_propers, 4)` | `energy`, `tables` | Signed amplitudes of $n = 1..4$, kcal/mol |
| `impropers.k` | `(N_impropers,)` | `energy`, `tables` | cvff amplitude, kcal/mol, in $[0, 20]$ |
| `atoms.sigma`, `atoms.epsilon`, `atoms.sigma_ref`, `atoms.epsilon_ref` | `(N,)` | `energy`, `tables` (`pair lj/cut`) | Å, kcal/mol, and the GAFF2 references they were predicted around (the `gaff2_*` inputs). `ByteFF` writes `sigma` / `epsilon` only |
| `atoms.charge` | `(N,)` | `energy`, `tables` (`pair coul/cut`) | Solved partial charges, $e$. The MBIS label stays `atoms.atomic_charge` |
| `hessian_pairs.partial_hessian` | `(P, 9)` | the caller, from `partial_hessian` | MM 3×3 blocks, kcal mol⁻¹ Å⁻² |

No bonded reference column (`*_ref`) is written: the bonded heads have no
per-row reference (§3.4).

## 3. Forward Contract

### 3.1 Notation

| Symbol | Meaning | Code anchor |
|--------|---------|-------------|
| $N$, $B$, $E$ | Atom rows, graph slots, pair rows (capacities, ghosts included) | `atoms.Z`, `graphs`, `pairs.pair_index` |
| $x_a$, $e_b$ | Atom / bond feature row handed to the heads | `EncoderFeatures.atom`, `.bond` |
| $h_a$, $\epsilon_{s\to t}$ | Encoder node state and directed edge state | `CovalentGraphEncoder` |
| $\mathrm{Eq}[\cdot]$ | Mean over a topological equivalence class | `Equivalence` |
| $t$; $(s, \mu, lo, hi)$ | Output saturation; a parameter's window std, mean and clamp | `HeadsSpec.tanh_output`; `window_constants`, `TanhWindow` |
| $\chi_i, \eta_i, q_i$ | Electronegativity, hardness, partial charge | `ChiEtaHead`, `atoms.charge` |

Energies are kcal mol⁻¹, lengths Å, **angles degrees** (molrs convention),
charges $e$ throughout; there is no output unit conversion.

### 3.2 Chemical perception

`CovalentGraphEncoder` (ByteFF-26's `graph_block.py:Graph2DBlock`) reads
**topology only** — no coordinate enters any equation below, so the predicted
parameters are geometry-independent and the deployed force is exactly
$-\partial E/\partial \mathbf{r}$.

Input features (`ChemicalEmbedding`): five atom embeddings and two bond
embeddings, concatenated, through an MLP that ends in GELU, then averaged over
equivalence classes:

$$
x^{(0)}_a = \mathrm{Eq}_a\!\left[\mathrm{MLP}_\text{node}\big(E_Z(Z_a - 1) \Vert E_d(d_a) \Vert E_q(\operatorname{clamp}(q^\text{f}_a, \pm 2)) \Vert E_\rho(\rho_a) \Vert E_m(m_a)\big)\right],
\qquad
e^{(0)}_b = \mathrm{Eq}_b\!\left[\mathrm{MLP}_\text{edge}\big(E_\text{ring}(\iota_b) \Vert E_o(o_b - 1)\big)\right]
$$

with $d_a$ the atom's degree over live bonds (counted from `bonds.bond_index`
and `bonds.mask`), $\rho_a$ ring connectivity, $m_a$ smallest ring, $\iota_b$
ring membership, $o_b$ bond order. A bond's class is the unordered pair of its
endpoints' atom classes (`Equivalence.bonds`). A dead bond row reads order
index 0 and counts toward no degree.

Each bond $b$ is two directed edges, $2b$ ($i \to j$) and $2b{+}1$ ($j \to
i$), $\epsilon^{(0)} = W_\text{in}\, e^{(0)}$. Then $L$ = `n_layers`
pre-LN edge-augmented attention layers (`EdgeAttentionLayer`, ByteFF's
`EGTConv`), with $q, k, v$ from $\mathrm{LN}(h)$ and $g = \mathrm{LN}(\epsilon)$:

$$
\ell_{s\to t} = \frac{q_t \cdot k_s}{\sqrt{d_h}} + W_b\, g_{st}, \qquad
\alpha_{s\to t} = \operatorname*{softmax}_{s \to t}\, \ell_{s\to t}, \qquad
h_t \leftarrow h_t + W_o \sum_{s\to t} \sigma(W_g\, g_{st})\,\alpha_{s\to t}\, v_s
$$

$$
h \leftarrow h + \mathrm{FFN}(\mathrm{LN}\,h), \qquad
\epsilon \leftarrow \epsilon + W_p\,\ell, \qquad
\epsilon \leftarrow \epsilon + \mathrm{FFN}_e(\mathrm{LN}\,\epsilon), \qquad
h \leftarrow \mathrm{GELU}(h)
$$

per head ($d_h$ = `head_dim`, `heads` heads; the softmax runs over the edges
entering one atom, an explicit segment softmax with no torch_geometric). The
outputs: $x = W_\text{jk}\,[h^{(1)} \Vert \dots \Vert h^{(L)}]$ (width
`width`) and $e_b = \tfrac12\,(W_\text{out}\epsilon_{2b} + W_\text{out}\epsilon_{2b+1})$
(width `edge_mlp[1]`, `bond_width`). Atom and bond rows of equivalent atoms
and bonds are equal (`EncoderFeatures` contract).

**Injected encoder.** `MaceConformerEncoder` (`rondo-enc-02`) returns the same
interface from geometry: every MACE layer's `0e` block on each conformer
(`pairs.pair_diff`, detached), gathered from each profile's first conformer
(`graphs.profile`); atom rows `MLP[s_i ‖ E(formal charge)]`, bond rows
`½(MLP[s_i ‖ s_j ‖ φ(r)] + MLP[s_j ‖ s_i ‖ φ(r)]) + E(order) + E(in_ring)`
with `φ` a Gaussian basis of the reference bond length; both averaged over the
equivalence classes. No gradient reaches `atoms.pos` through it, so the energy
is a Class-I energy at fixed parameters.

### 3.3 Valence contexts — permutation symmetry is structural

Every bonded head (`SymmetricHead`, ByteFF's `PreMMBonded._symmetric_pooling`)
reads the term's own atom and bond rows, in every order that names the same
term:

| Term | Copies (each a concatenation) |
|------|-------------------------------|
| bond $ij$ | $[x_i \Vert e_{ij} \Vert x_j]$, $[x_j \Vert e_{ij} \Vert x_i]$ |
| angle $ijk$ | $[x_i \Vert e_{ij} \Vert x_j \Vert e_{jk} \Vert x_k]$ and its reverse |
| proper $ijkl$ | $[x_i \Vert e_{ij} \Vert x_j \Vert e_{jk} \Vert x_k \Vert e_{kl} \Vert x_l]$ and its reverse |
| improper, centre $c$ | $[x_c \Vert e_{c\ell} \Vert x_\ell]$ for each of the three legs $\ell$ |

$$
y = \mathrm{post}\Big(\sum_{\text{copies}} \mathrm{pre}(\text{copy})\Big), \qquad
\mathrm{raw}_p = \mathrm{out}_p(y)
$$

`pre` and `post` are `DenseMLP`s (`bonded_hidden` → `bonded_width`,
`bonded_layers` linears, GELU between), one `out_p` per parameter
(`bonded_out_hidden`, `bonded_out_layers`). The sum makes every parameter
invariant under bond reversal, angle and proper reversal and any permutation
of the improper's legs. $e_{ab}$ is the row of the bond $a$–$b$ found in a
per-batch `BondTable` over `bonds.bond_index` (one sort of the pair keys,
`searchsorted` per term; static shapes, no loader column); a pair that is not
a bond reads row 0, and a padded term's energy is masked.

### 3.4 Parameter heads

Each bonded parameter is ByteFF-26's output transform (`TanhWindow`,
`mm_bonded.py:PreMMBonded` / `CustomClamp`):

$$
v = \operatorname{clamp}\!\big(t \tanh(\mathrm{raw})\, s + \mu,\ lo,\ hi\big),
$$

forward the clamped value exactly, gradient of the unclamped one
(straight-through). $t$ = `tanh_output` = 20, so the `tanh` reaches past both
bounds and the clamp binds. The constants are ByteFF-26's (GAFF2 statistics),
in the head's units ($\tfrac12 k$, degrees); the term converts once to molrs:

| Parameter | $(s, \mu, lo, hi)$, head units | Column | Conversion |
|-----------|-------------------------------|--------|------------|
| bond $k$ | $(200, 700, 2\cdot$`bond_k_floor`$, 4000)$ | `bonds.k` | $K = k/2$: $[$`bond_k_floor`$, 2000]$, 350 at raw 0 |
| bond $r_0$ | $(0.19, 1.3, 0.5, 5.0)$ Å | `bonds.r0` | — |
| angle $k$ | $(60, 130, 40, 1000)$ | `angles.k` | $K = k/2$: $[20, 500]$ |
| angle $\theta_0$ | $(10, 120, 20, 180)$ degrees | `angles.theta0` | — (180° inclusive) |
| proper $k_n$, $n = 1..4$ | $(1.2, 0.4, -20, 20)$ each | `propers.k` `(M, 4)` | phases $\gamma_n = 0°$ (odd $n$), $180°$ (even $n$) |
| improper $k$ | $(3.7, 4.3, 0, 20)$ | `impropers.k` | ByteFF's $k[1 + \cos(2\psi - \pi)]$ is cvff with $d = -1$, $n = 2$: $k[1 - \cos 2\psi]$, not halved |

Then, per row:

- **Hypervalent patch** (`Hypervalent`, ByteFF `patch_bonded_terms`): an atom
  that is P or S with five or more live bonds. A bond touching it takes head
  $k = 1000$ ($K = 500$), $r_0 = 0$; an angle centred on it head $k = 278.44$
  ($K = 139.22$), $\theta_0 = 0$; a proper or improper containing it $k = 0$.
  Applied in the head's units, before $K = k/2$.
- **Linear centre**: a proper with `propers.linear` set has $k_n = 0$ (A16).
- **Improper centre**: only a carbon or nitrogen centre carries $k$; molpy
  emits a row for every three-connected atom, and the others (B, S⁺,
  sulfoxide S) are zero.
- **Bond floor**: `BondHarmonic(..., k_floor=)` replaces the published
  lower clamp (head 80, $K$ 40) with $2\cdot$`bond_k_floor`; Rondo's default
  `bond_k_floor` is 80 (A27).

The bonded heads have no zero start (`DenseMLP` default init); the published
weights are ByteFF-26's.

Rondo's pair heads (one Linear–SiLU–Linear MLP per parameter, width `hidden`,
**last layer zeroed**, so the model starts on the reference, A12):

| Parameter | Map from the pre-activation | Reference | Head |
|-----------|-----------------------------|-----------|------|
| LJ $\sigma$ | $\sigma_\text{GAFF2}\, e^{0.1\,\mathrm{raw}}$ on $[0.7, 1.4]\,\sigma_\text{GAFF2}$ | `atoms.gaff2_sigma` | `LJCut` (`LJHead`) |
| LJ $\varepsilon$ | $\varepsilon_\text{GAFF2}\, e^{\mathrm{raw}}$ on $[\tfrac18, 8]\,\varepsilon_\text{GAFF2}$ | `atoms.gaff2_epsilon` | `LJCut` (`LJHead`) |
| $\chi$, $\eta$ | $5\tanh$, $10^{-3} + (20 - 10^{-3})\,\mathrm{sigmoid}$ | — | `PairCoulCut` (`ChiEtaHead`) |

The LJ windows are reached by **saturation** (`ExpAround(lo=, hi=)`, A24):
on each side of the reference the exponent is bent as $w\tanh(x/w)$ with $w$
the distance to that bound, so the slope at the reference is exactly one and
the gradient fades as a bound is approached.

Torsions carry a **signed** $k_n$ at fixed phases — ByteFF's and GAFF2's form.
Every term is even in the dihedral, so a mirror image has the same energy; and
$\partial E/\partial k_n = 1 + \cos(n\phi - \gamma_n)$ does not vanish at
$k_n = 0$, where an $|a|$ parameterization would (A13).

### 3.5 Charge equilibration

Per molecule the KKT system
$\begin{bmatrix} \mathbf{A} & \mathbf{1} \\ \mathbf{1}^{\mathsf T} & 0 \end{bmatrix}
\begin{bmatrix} \mathbf{q} \\ \lambda \end{bmatrix} =
\begin{bmatrix} -\boldsymbol{\chi} \\ Q \end{bmatrix}$
with a `DiagonalHardnessKernel`, $Q$ from `graphs.total_charge`, solved on a
fixed $(B, n_\text{max})$ layout, $n_\text{max}$ = `HeadsSpec.eem_max_atoms`,
`eem_jitter` added on the diagonal. A molecule over capacity fails closed
(eager: raises; compiled: `NaN`).

### 3.6 Class-I energy

`Rondo.energy` maps the tables onto parameters (`Term.potential_params`) and
calls that one `Potentials`:

| Style | Potential | Parameters from the tables |
|-------|-----------|----------------------------|
| `bond harmonic` | $K (r - r_0)^2$ | `bonds.k`, `bonds.r0` |
| `angle harmonic` | $K (\theta - \theta_0)^2$, $\theta_0$ in degrees | `angles.k`, `angles.theta0` |
| `dihedral periodic` | $\sum_{n=1}^{4} k_n [1 + \cos(n\phi - \gamma_n)]$ | `propers.k`; periodicities and phases are the head's buffers |
| `improper cvff` | $k [1 + d\cos(n\psi)]$, $d = -1$, $n = 2$, on the stored row (centre first) | `impropers.k`; `sign`, `periodicity` are the head's buffers |
| `pair lj/cut` | $w\, 4\varepsilon_{ij}[(\sigma_{ij}/r)^{12} - (\sigma_{ij}/r)^6]$, the field's mixing and cutoff | `atoms.epsilon`, `atoms.sigma` |
| `pair coul/cut` | $w\, C\, q_i q_j / (\epsilon_r r)$, the field's `coulomb` and `dielectric` | `atoms.charge` |

$w$ is the field's `special_bonds` weight of `pairs.relation`; each pair counts
once (`source < target`); dead rows are multiplied by their `mask`. A typed
`dihedral opls` reaches Rondo as `dihedral periodic` (molpy `canonical()`);
an untyped one is refused by name.

### 3.7 Derivatives, the three primitives and the compile seam

`forward = Derivative(energy, wrt=("atoms.pos",), outputs=(FORCES_PRED_KEY,))`,
built in `__init__`, writes `atoms.forces_pred`. It is not a constructor
argument. The training step marks its leaves before the traced body.

The joint training step (`molix.core.steps.joint`) draws one batch per source
and runs one compiled region per source (`traced_bodies`). Each objective of
the source shares one `predict(frame)` — the encoder and the heads run once
per source per step, whatever the number of objectives (the 2026-10-12 memory
fix; lost in e966239, restored 2026-10-30) — then detaches the columns its loss
does not name (`JointTrainStep.keep(tables, trainable)`) and runs one primitive,
chosen when the step is built: the derivative op over `energy_at(frame, tables,
styles=)`, `energy_at` itself, `write_tables` (a loss that supervises a
predicted column itself, the MBIS charges: no energy is written) or
`partial_hessian(frame, tables)`. The model knows nothing about objectives.
Tables are predicted **inside** the traced region (A5).

`energy(frame, styles=("bond:harmonic", "angle:harmonic"))` asks that
`Potentials` for the named styles only — ByteFF's `calc_terms` (A18). Three
objectives use it: the bond and angle energy on a QM minimum, per term, trained
through `r0` / `θ0` (ByteFF `Bond_Energy` / `Angle_Energy`,
`loss.py:302-314`; molix `MeanTermEnergyLoss`); the improper energy on the
same minimum trained through the improper `k` (ByteFF
`ImproperTorsion_Energy`, same lines, A19); and the pair styles' force alone,
written aside by the step (`JointObjective(nonbonded=)`) so the
stationary-force loss can leave out a clashing molecule (ByteFF
`skip_large_nb`, `loss.py:373-380`). The styles are a fixed tuple of names, so
the sum is straight-line code and traces as one graph.

`tests/test_molzoo/test_rondo/test_compile.py` (`slow`) asserts
`graph_count == 1` and `graph_break_count == 0` for the forward, the energy,
the energy of named styles, the partial Hessian and the grad-off eval forward.

## 4. Configuration Contract

`Rondo(forcefield, *, dtype, encoder=None, **settings)`. The keyword settings
are the fields of `CovalentGraphEncoderSpec` and `HeadsSpec`; the internal
`molzoo.rondo.spec.validate` checks them into `RondoSpec(encoder=, heads=)`
(frozen, `extra="forbid"`), and an unknown setting raises `TypeError` by name.
A caller does not build the spec object. The functional form is the force
field. The defaults are ByteFF-26's sizes (the reference
`fftrainer_config_in_use.yaml`), except Rondo's pair heads and the bond floor.

`CovalentGraphEncoderSpec` (`molrep.encoders.covalent_graph.spec`):

| Field | Meaning | Default |
|-------|---------|---------|
| `atom_embedding_dim` | Width of the element embedding | `32` |
| `channel_dim` | Width of the degree / charge / ring / bond embeddings | `8` |
| `node_mlp` | `(hidden, out, layers)` of the atom input MLP; `out` must equal `width` (else `ValueError`) | `(64, 256, 2)` |
| `edge_mlp` | `(hidden, out, layers)` of the bond input MLP; `out` is the bond feature width | `(64, 256, 2)` |
| `width` | Node and edge width of the transformer; the atom feature width | `256` |
| `n_layers` | Transformer layers $L$; the receptive field in covalent hops | `4` |
| `heads` | Attention heads per layer | `16` |
| `head_dim` | Width of one head | `16` |
| `ffn` | `(hidden, layers)` of the feed-forward MLPs | `(256, 2)` |

`HeadsSpec` (`molzoo.rondo.spec`):

| Field | Meaning | Default |
|-------|---------|---------|
| `bonded_hidden`, `bonded_width`, `bonded_layers` | Hidden width, output width and linears of each bonded head's `pre` and `post` | `256`, `256`, `3` |
| `bonded_out_hidden`, `bonded_out_layers` | Hidden width and linears of each bonded parameter MLP | `256`, `4` |
| `tanh_output` | $t$ of the bonded window (§3.4) | `20.0` |
| `bond_k_floor` | Lowest bond `K` the bond head puts out, LAMMPS kcal mol⁻¹ Å⁻² | `80.0` |
| `hidden` | Hidden width of Rondo's pair heads (D1, D2) | `64` |
| `lj_sigma_gain`, `lj_epsilon_gain` | §3.4 | `0.1`, `1.0` |
| `eem_max_atoms`, `eem_jitter` | §3.5 | `512`, `1e-4` |
| `lj_direct_hidden`, `lj_direct_layers` | `ByteFF` only: the σ and ε MLPs of `LJCutDirect` | `256`, `3` |
| `bcc_hidden`, `bcc_width`, `bcc_layers`, `bcc_out_hidden`, `bcc_out_layers` | `ByteFF` only: `BondChargeCorrection`'s `pre` and `out` | `512`, `512`, `4`, `256`, `3` |
| `pair_tanh_output` | `ByteFF` only: $t$ of σ, ε and the charge transfer | `20.0` |

**Bond floor.** `bond_k_floor` replaces the published window's lower clamp
(head `k` 80, LAMMPS `K` 40). Rondo's 80 is the 0.05th percentile of the DFT
bond stiffness on THEMol's held-out Hessians (`rondo-enc-08-bond-floor`);
`ByteFF.load` builds with `PUBLISHED_BOND_K_FLOOR` = 40 unless the caller
passes `bond_k_floor`, so a released member is reproduced as released.

**Injected encoder.** `encoder=` takes an encoder built by the caller at
`dtype` — `CovalentGraphEncoder` or `MaceConformerEncoder` (`rondo-enc-02`,
`MaceConformerEncoderSpec`: ByteFF-26's widths 256 / 256 by default). It is
registered as `graph`, as the built one is, and the heads are sized from its
`width` / `bond_width`. A `CovalentGraphEncoderSpec` field passed beside it
raises `TypeError` (it would configure nothing); `HeadsSpec` fields are
accepted. `spec.encoder` then holds the covalent-graph defaults and does not
describe the injected encoder.

**Loading.** `load_state_dict` is strict. `load_state_dict(state_dict,
prefixes=("graph.",))` (or `"heads."`, …) copies only those prefixes into the
existing tensors — a missing or unexpected key under a named prefix raises
`KeyError` — so an encoder and the heads load independently. A subclass on a
richer force field overrides `load_state_dict` so a checkpoint of the parent
loads (`strict=False`); the new head stays at its initial value.
`energy(styles=)` asks the same `Potentials` for a subset of its styles,
named as classes in `molpot.classical` or as the force field's style keys.

**Static field.** `Rondo.parameterize(frame, *, graph=0) -> (molpy.Frame,
molpy.ForceField)` predicts once (`no_grad`) and writes molecule `graph` as
molpy data: every atom and every term row becomes its own type at its
predicted parameters — nothing is averaged — on the styles, style parameters
and `special_bonds` of `forcefield`; masses from the element, charges on the
`atoms` block; one block per bonded namespace (`propers` as molpy's
`dihedrals`); a `(rows, m)` column (the periodic series) becomes molrs's
`name1 … namem`. Padding rows are not written. It runs eagerly with host
syncs, once per topology — not a hot path. molpy serialises the pair
(`write_lammps_data`, `LammpsForcefieldWriter`); export parity with LAMMPS is
recorded in `rondo-enc-07-validation`.

**ByteFF.** `molzoo.byteff.ByteFF(Rondo)` overrides the pair styles' heads
(`LJCutDirect`, `BondChargeCorrection`); everything else is Rondo's.
`ByteFF.load(path, forcefield, *, dtype, **settings)` reads one member's
`optimal.pt` through `CheckpointRemap` over `ByteFFKeys` and raises
`ValueError` if a checkpoint tensor has no home or a model tensor is not
provided. The five-member mean is the caller's: average `predict` over the
members, then `price` once.

The optimizer is `Adam(model.parameters())`. Which of the carried tables a
loss moves stays on `JointObjective(trainable=)`.

## 5. Reference Crosswalk

| Concept | Espaloma | ByteFF (byteff2 v1.1.0) | MolNex anchor | Status |
|---------|----------|--------|---------------|--------|
| Backbone | SAGEConv stack | Edge-augmented graph transformer (`EGTConv` ×4, JK concat), equivalence-class averaging | `CovalentGraphEncoder`, `EdgeAttentionLayer` | matched (ByteFF; parity §7.1) |
| Input features | element | element, degree, formal charge, ring connectivity, smallest ring; bond order, in-ring | `ChemicalEmbedding` | matched (ByteFF) |
| Equivalence classes | — | RDKit `CanonicalRankAtoms(breakTies=False)`; bond class = pair of atom classes | `atoms.equi_index` (molpy `assign_symmetry_classes`), `Equivalence.bonds` | matched (ByteFF; column parity in `rondo-enc-01` §8) |
| Canonical resonance structure | — | `get_canonical_resoner` rewrites bond orders and charges before featurizing | not ported; MolHub keeps the input's charges and orders | departed (R3) |
| Reads coordinates? | no | no | `CovalentGraphEncoder` never; `MaceConformerEncoder` one reference conformer, detached | matched / adapted (A28) |
| Valence symmetry | Janossy pooling | symmetric pooling over the term's orderings | `SymmetricHead` | matched (ByteFF) |
| Term bond features | — | the term's own bonds interleaved with its atoms | `_Bonded.copies` + `BondTable` (A21) | matched (ByteFF) |
| Bond / angle form | harmonic | harmonic ($\tfrac12 k$) | `bond:harmonic`, `angle:harmonic`, $K = k/2$ | matched |
| Proper torsion | Fourier | signed $k_n$, $n = 1..4$, phases 0/π by parity | `DihedralPeriodic` → `dihedral:periodic` | matched (ByteFF; the reference's float32 π, §7.1) |
| Improper | periodic | $k[1+\cos(2\psi-\pi)]$, one row per C/N centre, legs ascending | `ImproperCvff` → `improper:cvff`, $d = -1$, $n = 2$ | matched (ByteFF) |
| Linear-centre propers | — | zeroed by SMARTS | `propers.linear` (molpy `perceive_linear_dihedrals`) | matched |
| Hypervalent P / S | — | `patch_bonded_terms` | `Hypervalent` | matched |
| Parameter bounds | — | `20·tanh·std + mean`, straight-through clamp | `TanhWindow` | matched, except the bond `K` floor (A27) |
| Charges | charge-equilibration read-out | bond charge corrections | EEM KKT solve, MBIS supervision (`PairCoulCut`); ByteFF's in `molzoo.byteff.ByteFF` | departed (D1) |
| Lennard-Jones | OpenFF template | σ, ε predicted directly | learned around GAFF2 (`LJCut`); ByteFF's (`LJCutDirect`) in `molzoo.byteff.ByteFF` | adapted (D2) |
| Ensemble | — | mean of each final parameter over five members, energy on the mean | caller: mean of `predict`, one `price` | matched |
| Training | QM E/F | joint per-step sum over datasets, per-loss trainable sets, 3 stages | `molix.core.steps.JointTrainStep` + `JointObjective.trainable` | matched |
| Stage-1 geometry target | — | MM force → 0 at the QM minimum | `molix.core.losses.StationaryForceLoss` | matched |
| Force computation | autograd | autograd | `molix.autograd.Derivative` | matched |

## 6. MolNex Adaptations

| ID | Adaptation | Reason | Risk | Validation |
|----|------------|--------|------|------------|
| A5 | Tables are predicted inside the traced region | an external `grad_fn` breaks the force's `autograd.grad` under compile | low | `test_compile.py` |
| A6 | The force lands at `atoms.forces_pred`, the op's output key; the energy stays at `graphs.energy` and the training step moves it beside its label (`molix.frame.beside_labels`) | a training batch holds the DFT labels at `atoms.forces` / `graphs.energy`; the model moves nothing | high if regressed | `test_model.py::TestRondo::test_forces_are_the_negative_gradient_of_the_energy`, `tests/test_molix/test_frame.py::TestBesideLabels` |
| A7 | Atom and bond features averaged over topological equivalence classes (`atoms.equi_index`, `Equivalence`), by every encoder | ByteFF's `equi_features`: equivalent atoms get equal parameters by construction, also from a conformer that breaks the symmetry | high if an encoder drops it (§10.4) | `test_covalent_graph/test_encoder.py::test_resonance_equivalent_atoms_get_equal_features`, `test_mace_conformer/test_encoder.py::test_equivalent_atoms_and_bonds_get_equal_rows`, `test_chem/test_equivalence.py` |
| A9 | Per-objective weights, no per-table gradient clamp | global clip + per-loss `trainable` cover it | low | revisit if one table dominates |
| A12 | Rondo's pair heads start on their reference: the last layer of each LJ and χ/η MLP is zeroed | one lr serves every parameter; the model starts on GAFF2 and on $q = Q/N$ | low | `test_classical/test_heads.py::TestHeadProtocol`, `TestLJHead::test_starts_exactly_on_the_reference_well` |
| A13 | Torsions as signed $k_n$ at fixed phases (not $\lvert a\rvert$) | gradient at $k_n = 0$; ByteFF / GAFF2 form | low | `test_classical/test_terms.py::TestDihedralPeriodic::test_the_series_is_n_1_to_4_at_phases_0_and_180`, `test_model.py::test_a_mirror_image_has_the_same_energy` |
| A15 | One improper row per C/N centre, centre first, legs ascending; the head is symmetric in the legs, the dihedral is taken on the stored row; an engine's own order is named in `impropers.centre` / `leg1..3` | ByteFF's `get_impropers` (replaces the three cyclic rows of 2026-10-11) | low | `test_terms.py::TestImproperCvff` |
| A16 | Linear-centre propers zeroed from `propers.linear` | undefined dihedral; ByteFF's SMARTS, now a molpy-perceived column (replaces the bond-order rule) | low | `test_model.py::test_propers_through_a_linear_centre_are_silent`, `test_terms.py::TestDihedralPeriodic::test_a_batch_without_the_linear_column_is_refused_by_name` |
| A17 | LJ around GAFF2 with a prior, not frozen | operator ruling 2026-10-11: trainable, GAFF2 start | medium | `test_model.py::test_lennard_jones_starts_on_gaff2` |
| A18 | `energy(styles=)` prices a named subset of styles | ByteFF `calc_terms`: the equilibrium-strain objective and `skip_large_nb` need the bonded, resp. pair, energy alone. The force at a minimum cannot set `θ0`'s common mode at a symmetric centre — sp3 `θ0` walked 109° → 95° on it (GH200 3680577, 3685025/3685026) | low | `test_model.py::test_energy_of_named_styles_sums_only_those`, `test_compile.py::test_the_energy_of_named_styles_is_one_graph` |
| A19 | The improper `k` is also trained by the improper strain on the QM minimum (training-side: `OBJECTIVES["hessian_eq_improper"]`, no model code) | ByteFF `ImproperTorsion_Energy`. With the Hessian objective alone the improper's planarising force at a pyramidal N (\|F\| p50 116 kcal/mol/Å) had no parameter the force objective could reach, and every `θ0` at the centre walked together to cancel it (GH200 3690962: N3 θ0 129° vs 118°, 46 % of amine centres below 90°) | low | `.jobs/n3_force_decomp.py` on 3690962 best.pt; gate run 3705406 passes (with A20) |
| A20 | The improper `k` is non-negative by construction — the window's lower clamp is 0 (§3.4) | the strain $k(1-\cos 2\psi)$ is linear in $k$: with a signed $k$ the A19 objective is unbounded below (GH200 3703994: 0 → −4.4 kcal/mol/term in 1400 steps). A negative `k` would also mean a minimum at ψ = 90° | low | `test_heads.py::TestTanhWindow::test_the_clamp_holds_the_value_and_passes_the_gradient` |
| A21 | Every bonded head reads the features of the term's own bonds, interleaved with its atoms (§3.3); the bond rows come from a per-batch `BondTable` (sort + `searchsorted`, static shapes) rather than a loader column | ByteFF's assembly. Read from atom rows alone, a torsion about a single and one about a double bond of a Kekulé ring get identical inputs (probe 2026-10-12) | low | `test_chem/test_bond_table.py`; the reversal and leg-order tests of `test_terms.py` |
| A24 | Rondo's LJ windows saturate ($w\tanh(x/w)$ per side, slope one at the reference) instead of a straight-through clamp | a straight-through clamp holds the value but passes the whole gradient, so a row the loss wants at a bound pulls on its features forever (GH200 3723816: improper stream RMS 5.9 at 26 400 steps, with the old reference·exp improper window) | low | `test_heads.py::TestExpAround::test_the_window_is_approached_by_saturation_and_the_gradient_fades` |
| A25 | The torsion stage carries ByteFF's auxiliary loss — the L1 norm of the proper amplitudes (`molix.core.losses.NormPrior` on `propers.k`, a `"tables"` objective on the torsion-scan batch, weight 1.0) — and the stages run at ByteFF's rates (torsion 1e-4, off-equilibrium 2e-5; were 3e-4 and 1e-4) | Paper §2.4 / Table 1: "L1-norm regularization was applied to the force constants of proper torsions to restrain redundant degrees of freedom", `TorsionLossType.l1_norm`. GH200 3748659 (S2 large, no prior, 3e-4): a scan measures one dihedral and every other proper of the molecule turns a little as the constrained conformers relax; six free amplitudes per row fit those wiggles with constants nothing in the stage prices back (the hessian objectives detach `propers.k`) — Σn²\|k_n\| median 9 → 34 in 2 000 steps, the impropers driven to their floor, N3 θ0 off by 13°, eval 14.3 → 24.6 while the kneed train loss stayed flat | The weight is the paper's unstated knob: too high and the profiles under-fit (torsion loss stalls above its no-prior value), too low and the amplitudes grow again. The circular-std mask and the per-table gradient cap ByteFF's code also carries (`mask_by_circstd`, `max_config`) are not adopted until a run shows the L1 prior and the rate are not enough | §7.4: an S2 gate on A25 must hold Σn²\|k_n\| near its step-800 scale (p50 ≈ 9) and the improper \|k\| median above 1 while the torsion loss keeps falling; the S2 long run must not raise the S1 objectives' eval |
| A26 | The off-equilibrium objective (`relax`) trains the equilibrium geometry (`bonds.r0`, `angles.theta0`), the proper amplitudes and σ/ε only; the curvatures `bonds.k`, `angles.k`, `impropers.k` stay the partial Hessian's (were: every bonded table + σ/ε) | ByteFF's `energyforce_loss` names every bonded table trainable (loss.py `set_trainable_terms([Bond_k, Angle_k, ProperTorsion_k, ImproperTorsion_k, Bond_length, Angle_theta])`) under a per-table gradient cap (`max_config`) whose values the repository does not carry. GH200 3768159 (S3 large, 2e-5, 1 500 steps): the relax forces take the cheap direction — a softer K — and the Hessian objective loses the tug-of-war: bond K −30 … −70 % (stiff bonds most, C=O 731 → 230), angle K −20 … −35 %, hessian_k 0.428 → 0.469 monotone while hessian_geom / hessian_eq fall only because every force at the minimum scales with K | A harmonic bond's K is the curvature at the minimum; forces along a relaxation sit on the anharmonic side and a harmonic fit to them is always softer than the Hessian K, so the two objectives disagree by construction and the one with the larger gradient wins. Rondo resolves it by ownership (each table moved by the data that defines it), ByteFF by a cap whose value is not public. Revisit (restore K to the relax set under a per-table cap) if the A26 relax loss stalls far above its A25 value | §7.4: the A26 S3 gate must hold hessian_k at its S2 value (0.428) and the bond K table within ±10 % of the S2 long run's, with the relax loss still falling; an S3 run that later shows the relax loss stuck above ~2.5 reopens the question |
| A27 | The bond head's lower `K` clamp is `HeadsSpec.bond_k_floor`, 80 by default (published: 40); `ByteFF.load` keeps 40 | a zwitterion's C–N⁺ collapsed in LAMMPS MD with its `K` on the published floor (`rondo-enc-07`); 80 is DFT's 0.05th percentile on THEMol's held-out Hessians (`rondo-enc-08`) | low: held-out scores move < 0.1 % (`rondo-enc-08` §4) | `test_terms.py::TestBondHarmonic::test_k_never_falls_below_the_floor` |
| A28 | An encoder may be injected (`encoder=`); `MaceConformerEncoder` reads one reference conformer per profile, detached, and feeds ByteFF-26's heads unchanged | the rondo-enc hypothesis: encoders interchangeable behind one feature interface (`rondo-enc-00`, `-02`, `-05`) | low (positions detached: no ∂parameter/∂R term) | `test_model.py::test_an_injected_encoder_feeds_every_head_and_trains_on_the_energy`, `test_a_covalent_setting_beside_an_injected_encoder_is_refused`; `test_mace_conformer/test_encoder.py::test_no_gradient_reaches_the_positions` |

Superseded on 2026-10-28 by `rondo-enc-01` (the Espaloma-style encoder and the
reference·exp heads are deleted; their checkpoints are not kept, ruling 1):
A11 (pre-norm covalent hops, every channel layer-normalized — the
transformer's own pre-LN layers replace it), A14 (`θ0` reference from
hybridization — ByteFF's window, mean 120°), A22 (multiplicative windows
around per-row references — `TanhWindow` on the bonded parameters; A24 keeps
the saturating window for LJ), A23 (linear-centre `θ0` reference of 180° —
the window's ceiling is 180° inclusive, no per-row reference). The
covalent-radius and VSEPR references and `Hybridization` are no longer read.

### 6.1 Open methodological risks

| ID | Risk | Detail |
|----|------|--------|
| R1 | Non-bonded parameters carry little signal in gas-phase data | σ/ε start on GAFF2 and move only in a stage that names them with a prior on `log(σ/σ_ref)`, `log(ε/ε_ref)`; condensed-phase data would close this |
| R2 | One objective driving every parameter leaves families degenerate | **Addressed** by per-loss `trainable` and the joint schedule: Hessian → `k`, equilibrium forces → `r0`/`θ0`, MBIS → charges, torsion scans → propers, off-equilibrium E/F → `r0`/`θ0`, propers and σ/ε, never the curvatures (A26) |
| R3 | ByteFF's canonical-resonance step is not ported | on a molecule it rewrites, the encoder sees other bond orders and charges than ByteFF-26 does (2 / 3000 THEMol molecules; `rondo-enc-01` §8 R1) |
| R4 | The bonded windows are straight-through clamps (ByteFF's) | a row held at a bound keeps passing the whole gradient — the failure A24 removed from the old windows (3723816). The improper floor 0 is where A19 sends a pyramidal centre; watch the improper feature RMS when the bonded heads train |

## 7. Validation Contract

### 7.1 Research Reproduction

**Claimed for the encoder and the bonded heads: ByteFF-26 parity**, recorded in
`.claude/specs/rondo-enc-01-graph-encoder.md` §8 (oracle: `byteff2@v1.1.0`,
all five members, fp64):

| Check | Result |
|-------|--------|
| Parity 2026-10-28, 27 molecules incl. PF6⁻, SF6, azide, nitro, allene, caffeine | encoder atom features ≤ 2.1e-15 and bond features ≤ 1.8e-15 relative, degree exact; bonded parameters ≤ 2.6e-14; bond and improper energies ≤ 4.3e-14; angle ≤ 2.5e-9 (azide, 3.5e-3 kcal/mol); proper ≤ 4.2e-8 — the oracle builds its 180° phase with a float32 π (off by 8.7e-8 rad, `ff_kernels.py:386`), molnex uses exact 180° |
| Full reproduction with the published pair heads (now `ByteFF.load`) | every parameter of every member — bonded, BCC charges, σ, ε — ≤ 2.7e-14 relative; the five-member mean ≤ 8.6e-15; total energy ≤ 2.7e-7 relative per member and 4.9e-8 for the ensemble (largest residual formamide, the same float32-π phase); all 260 upstream tensors land, none missing |
| End-to-end gate, THEMol `hessian@1` shard-000 rows 0–499 through the padded Class-I data path | bonded ≤ 2.2e-13, charges ≤ 5.0e-11, σ ≤ 6.7e-15, ε ≤ 6.7e-13 relative (members; the mean ≤ 3.5e-12); every energy term but the torsion ≤ 1.2e-10 kcal/mol, the torsion ≤ 2.5e-6 kcal/mol (float32 π); 499 / 500 molecules, the boron anion excluded (R3) |

The parity runs predate `bond_k_floor` and used the published window (floor
40, which `ByteFF.load` keeps). Not
claimed: Rondo's own pair heads (D1, D2), the MACE conformer encoder, and every
trained model (§7.4 and the rondo-enc phase specs).

### 7.2 Symmetry and Shape Tests

| Claim | Test path |
|-------|-----------|
| Bonded parameters invariant under bond, angle and proper reversal and improper leg order | `tests/test_molpot/test_classical/test_terms.py` (`TestBondHarmonic`, `TestAngleHarmonic`, `TestDihedralPeriodic`, `TestImproperCvff`), `test_heads.py::TestSymmetricHead` |
| `K = k/2`, θ0 in degrees, $n = 1..4$ at 0°/180°, cvff $k(1 - \cos 2\psi)$, the bond floor, the hypervalent patch, C/N improper centres | `test_terms.py` |
| `TanhWindow` holds the value and passes the gradient | `test_heads.py::TestTanhWindow` |
| Encoder reads no coordinate | `tests/test_molrep/test_encoders/test_covalent_graph/test_encoder.py::test_positions_are_never_read` |
| Encoder features follow an atom permutation; equivalent atoms equal; a bond's reversal changes nothing; dead bond rows leave real rows unchanged | `test_covalent_graph/test_encoder.py`, `test_layers.py` |
| $\sum_i q_i = Q$ | `test_model.py::test_charges_meet_the_declared_total_charge` |
| $\mathbf{F} = -\partial E / \partial \mathbf{r}$ | `test_model.py::test_forces_are_the_negative_gradient_of_the_energy` |
| Mirror image has the same energy | `test_model.py::test_a_mirror_image_has_the_same_energy` |
| Rigid motion invariance | `test_model.py::test_energy_is_invariant_under_a_rigid_motion` |
| Linear-centre propers priced at zero | `test_model.py::test_propers_through_a_linear_centre_are_silent` |
| Partial Hessian equals the full Hessian on the bonded pairs | `test_model.py::test_partial_hessian_is_the_full_hessian_on_the_bonded_pairs` |
| `forward` writes the force at `atoms.forces_pred` and the energy at `graphs.energy`; `-dE/dpos` against `energy` | `test_model.py::TestRondo::test_forces_are_the_negative_gradient_of_the_energy` |
| `tables` writes `energy`'s columns and no energy | `test_model.py::test_tables_writes_the_columns_energy_writes_and_no_energy` |
| Unsupported style refused by name; untyped `dihedral opls` refused | `test_model.py::test_an_unsupported_style_is_refused_by_name`, `test_untyped_opls_requires_ir_conversion` |
| `energy(styles=)` sums the named styles only; an unknown name is refused | `test_model.py::test_energy_of_named_styles_sums_only_those`, `test_model.py::test_a_style_selection_outside_the_field_is_refused_by_name` |
| Injected encoder feeds every head; a covalent setting beside it is refused | `test_model.py::test_an_injected_encoder_feeds_every_head_and_trains_on_the_energy`, `test_a_covalent_setting_beside_an_injected_encoder_is_refused` |
| `parameterize` types every row at its predicted parameters; molpy writes one coefficient per row | `test_model.py::test_a_static_field_types_every_row_at_its_predicted_parameters`, `test_molpy_writes_the_static_field_one_coefficient_per_row` |
| `ByteFF.load` fills every tensor | `tests/test_molzoo/test_byteff/test_potential.py::TestByteFF::test_load_fills_every_tensor_of_the_model` |
| forward / energy / energy(styles=) / partial_hessian / eval forward compile at 1 graph, 0 breaks | `test_compile.py` (`slow`) |

### 7.3 Engineering Benchmark

Training runs live in the molab workspace (`lab-new/projects/class-i/train/`),
not in this repo. Measure on a compute node.

### 7.4 Run Log

Every row below ran on the encoder and heads superseded on 2026-10-28 (§6);
`rondo-enc-01` §1 ruling 1 keeps none of their checkpoints. The runs on the
current encoder and heads are recorded in their own specs
(`.claude/specs/rondo-enc-03-heads.md`, `-05-swap`, `-07-validation`,
`-08-bond-floor`, `-09-bond-k-ci`).

| run_id | date | commit | dirty | dataset | config | steps | train | val | note |
|--------|------|--------|-------|---------|--------|-------|-------|-----|------|
| 1–3 | 2026-09-17/18 | uncommitted | 1 | themol torsion scan 0 | hidden=64 n_blocks=3 | 400 | 0.027 | 0.091 | pre-realignment, old API |
| 3680577 | 2026-10-11 | 66cc421+ | 1 | S1 joint: themol hessian + mbis | small (width 32, 2 hops, hidden 32), cap 131072, lr 3e-4, clip 10 | 1500 | 23.8 | 37.97 | first S1 after the loader fix; gate FAILS on C4 θ0 99.1° (measured 109.6°) |
| 3685025 / 3685026 | 2026-10-11 | 66cc421+ | 1 | S1 joint | as above, unbounded, clip 10 / 50 | 15600 / 14800 (best 5600 / 4800) | 24.6 / 25.1 | 32.06 / 32.12 | patience stop; gate fails on 4 / 3 sp3 centres (C4 95.4°); sp2 θ0 117 vs 127.5 across the two — θ0 common mode is a null direction of the force loss; clip no effect |
| 3689502 | 2026-10-12 | 66cc421+ | 1 | S1 joint + `hessian_eq` (bonded strain, ByteFF Angle/Bond_Energy) + `skip_large_nb` 50 | small, cap 131072, lr 3e-4, clip 10 | 1500 | 24.2 | 37.88 | **gate passes**: C4 θ0 107.8°, C3 119.9°, S4 106.8°, N4 110.4°, P4 110.8°, C2 179.5°; hessian_k falls monotonically to 0.435; hessian_eq 1.12 → 0.37 kcal/mol/term; 22.9 GiB, 2.24 steps/s |
| 3690962 | 2026-10-12 | 66cc421+ | 1 | S1 joint + `hessian_eq` + `skip_large_nb` 50, unbounded (patience 50 evals) | small, cap 131072, lr 3e-4 → plateau ÷2, clip 10 | 19600 (best 10000) | 23.9 | 37.46 (best 29.4) | **gate fails**: N3 θ0 129.3° vs 117.9° (per-angle p10 61° / p90 157°), P4 97.1° vs 107.5°; every other centre inside the gate. hessian_eq rose 0.36 → 0.76 while the force eval fell to 28.2; past step 10000 train and eval both worsen (lr ≤ 1.2e-6 only settles). Cause (`.jobs/n3_force_decomp.py`): at pyramidal N3 the improper alone pushes the N along the pyramid axis by |F| p50 116 kcal/mol/Å and the angle term cancels it (θ0 ≪ measured: improper +139 / angle −138); `hessian_k` trains the improper k on curvature only and the force objective cannot reach it, so θ0 was its only handle. ByteFF's `ImproperTorsion_Energy` was missing → `hessian_eq_improper` added (A19). 3.13 steps/s, 22.9 GiB |
| 3703994 | 2026-10-12 | 66cc421+ | 1 | S1 joint + `hessian_eq` + `hessian_eq_improper` (A19) + `skip_large_nb` 50, patience 10 evals | small, cap 131072, lr 3e-4, clip 10 | 1400 (stopped: eval 10× its best) | — | 7780 (best 54.7 at 800) | **gate fails, worse**: N3 θ0 163.0° vs 117.9° (p10 149°), N4 121.5° vs 109.1°, P4 120.7° vs 107.5°. `hessian_eq_improper` fell 0 → −0.004 → −0.38 → −1.06 → −4.41: the signed improper `k` made the strain objective unbounded below, the improper force grew (\|k\| p90 9.5 → 34.8) and the N3 `θ0` rose to cancel it; the force eval blew up from step 1000 (94.9 → 1123 → 7780) and hessian_k 0.443 → 0.556. Fix A20 (`k` positive by construction). 2.33 steps/s, 26.2 GiB |
| 3705406 | 2026-10-12 | 66cc421+ | 1 | S1 joint + `hessian_eq` + `hessian_eq_improper` with A20 (positive improper `k`) + `skip_large_nb` 50, patience 10 evals | small, cap 131072, lr 3e-4, clip 10 | 1500 | 23.4 | 36.00 | **gate passes**: N3 θ0 116.5° vs 117.9° (p10 109° / p90 123°), N4 110.4°, P4 110.3°, C4 108.1°, C3 121.0°, S4 107.4°, C2 179.5°; `hessian_eq_improper` 0.0315 → 0.0016 kcal/mol/term, improper \|k\| p10 0.003 / p50 4.26 / p90 7.9 (vs 1.7 / 4.7 / 9.5 without the strain objective: pyramidal centres lose their improper, planar ones keep it); hessian_k 0.4348 (3689502: 0.4352), force eval 35.2 (37.1), hessian_eq 0.350 (0.369), charge MAE 0.058; bond tables within ±5 of 3689502. 2.14 steps/s, 26.2 GiB |
| 3707870 | 2026-10-12 | 66cc421+ | 1 | S1 joint, five objectives with A19 + A20, `skip_large_nb` 50, unbounded (patience 10 evals, 4 h stage clock) | small, cap 131072, lr 3e-4 → plateau ÷2 at 26800 / 30600 / ~36000, clip 10 | 40879 (clock; best 40600) | 18.4 | 22.65 (best) | **gate passes**; the 3690962 failure modes do not recur: hessian_eq 1.12 → 0.22 monotone (never rose), eval monotone with ±0.05 wiggles through three lr cuts, best kept moving to the end. N3 θ0 116.9° vs 117.9° (p10 108.8 / p90 123.7 vs measured 109.4 / 126.2), P4 105.8 vs 107.5, N4 106.0 vs 109.1, S2 96.2 vs 91.9, O2 108.6 vs 112.3, C4 107.8 vs 109.6, C3 120.9 vs 120.1; improper \|k\| p10 6e-16 / p50 3.9 / p90 8.8; charge MAE 0.031; hessian_k 0.4295 (the 1-3 floor), hessian_eq_improper 0.0013, mbis 0.00199. **New finding**: sp-carbon angle `K` collapsed to 1.1 (p10 0.8 / p90 2.0; 47 at 1500 steps, 14 in 3690962) with every C2 `θ0` pinned at the 179.5° ceiling (measured 179.0, p90 179.9): a θ0 below 180° is a Mexican hat at a linear centre and the Hessian fit can only kill it by `K → 0` → A23. 2.96 steps/s |
| 3723815 | 2026-10-12 | 66cc421+ | 1 | S1 joint, five objectives, **A21 + A22 + A23** (term bond features, parameter windows, θ0 = 180°), `skip_large_nb` 50, patience 10 evals | small, cap 131072, lr 3e-4, clip 10 | 1500 | 23.0 | 34.55 | **gate passes**, every S1 metric at or better than 3705406 (same recipe without the three fixes): hessian_k 0.4339 (0.4348), hessian_geom eval 33.8 (35.2), hessian_eq 0.340 (0.350), mbis 0.0079 (0.0082), charge MAE 0.060 (0.058). C2 θ0 p50 180.0° (p10 177.8°) with angle K 42.3 (3705406: 179.5° / 47 — the long run decides A23); N3 θ0 116.3° vs 117.9°, N4 107.9, P4 106.3, C4 108.2, C3 121.3, S2 100.2 (91.9), O2 109.0 (112.3); improper \|k\| p10 0.022 (the new 1/64 floor) / p50 4.12 / p90 7.98. 2.10 steps/s, 28.4 GiB (wider context MLPs: +2.2 GiB, −2 %) |
| 3723816 | 2026-10-12 | 66cc421+ | 1 | S1 joint, five objectives, A21 + A22 + A23, `skip_large_nb` 50, unbounded (patience 10 evals, 4 h stage clock) | small, cap 131072, lr 3e-4 → plateau ÷2 at 11600 / 14800 / 20800 / 24000 / 25800, clip 10 | 26400 (patience; best 24400) | 19.4 | 24.09 (best) | **handoff gate fails**: feature RMS improper 5.92 (window 0.3–3; atom 1.06, bond 1.06, angle 1.20); improper \|k\| p10 0.022 (floor) / p50 = p90 = 3.13 — the straight-through window let rows held at the floor pull on the stream forever → A24. Everything else healthy and A23 confirmed: eval monotone 28.3 (4000) → 24.1, hessian_k 0.4329, hessian_eq 0.218, mbis 0.00233, charge MAE 0.034; **C2 θ0 180.0° (p10 179.8°) with angle K 25.2 (p10 16.7 / p90 27.5)** — down from 42 at 1500 steps but no collapse (3707870: 1.1), and a linear bend of 21–43 kcal/mol/rad² is the physical range; N3 116.9° vs 117.9°, N2 112.3 vs 115.3, O2 109.5 vs 112.3, S2 95.6 vs 91.9, C4 107.9 vs 109.6, C3 121.0 vs 120.1. 2.60 steps/s, 28.4 GiB |
| 3741657 | 2026-10-12 | 66cc421+ | 1 | S1 joint, A21 + A22 + A23, **training step restructured** (one `tables` per source shared by its objectives, `output="tables"` for mbis, eval releases its graphs) — the A/B of 3723815 on the same model | small, cap 131072, lr 3e-4, clip 10 | 1500 | 23.1 | 34.78 | **gate passes**, metrics within noise of 3723815: hessian_k 0.4343 (0.4339), hessian_geom 34.0 (33.8), hessian_eq 0.339 (0.340), hessian_eq_improper 0.0020 (0.0019), mbis 0.0078 (0.0079). **2.75 steps/s (2.03) and peak 7.79 GiB (28.4)**: ×3.6 less memory, +35 % throughput; the remaining step is loader-bound |
| 3742959 | 2026-10-12 | 66cc421+ | 1 | S1 joint, A21–A23 + **A24** (saturating windows) on the restructured step; the gate before the A24 long run | small, cap 131072, lr 3e-4, clip 10 | 1500 | 23.0 | 34.23 | **gate passes**: hessian_k 0.4336, geom 33.4, hessian_eq 0.340, eq_improper 0.0022, mbis 0.0080, charge MAE 0.061 — all within noise of 3741657 / 3723815. C2 θ0 180.0° with p10 180.0° (the reference on its bound is held exactly) and angle K 45.7; improper \|k\| p10 0.030 (off the 0.022 floor: saturation, not a wall) / p50 4.58 / p90 8.42; improper feature RMS 1.14. 2.71 steps/s, 7.80 GiB |
| 3742962 | 2026-10-12 | 66cc421+ | 1 | S1 joint, A21–A24, restructured step — **first large-tier run that fits** (3732774 / 3735480 / 3736461 OOMed at 90.5 GiB before the step fixes) | **large** (width 128, 8 hops, hidden 128), cap 131072, lr 3e-4, clip 10 | 1500 | 17.0 | 19.94 | **gate passes**, and at 1500 steps already below where small sits after 26 400 (24.1): hessian_k 0.4305, geom 19.3, hessian_eq 0.227, eq_improper 0.0019, mbis 0.0025, charge MAE 0.035. C2 θ0 180.0 / K 28.1; N3 117.4 vs 117.9, C4 107.5, C3 120.5, S2 94.5 (91.9), P4 101.7 (107.5); improper \|k\| 0.024 / 3.87 / 9.00; feature RMS 1.00–1.05. **2.17 steps/s, 28.75 GiB** at the tier capacity (the item-5 calibration for large) |
| 3743201 | 2026-10-12 | 66cc421+ | 1 | S1 joint, A21–A24, restructured step — **large long run** (no step cap, init from scratch) | **large** (width 128, 8 hops, hidden 128), cap 131072, lr 3e-4, clip 10, plateau ×0.5 / patience 5 evals, early stop 10 evals | 17 800 (patience stop; best 15 800) | 8.20 | **10.85** | **gate passes** (2 h 09, n95): eval 2048 → 11.09 at 12 000 → best 10.85 at 15 800, lr cut twice (3e-4 → 1.5e-4 at ~13 600 → 7.5e-5 at ~17 200), stopped by patience at 17 800. hessian_k **0.428** (the floor every run sits on from 2 000 steps, small or large), geom 10.34, hessian_eq 0.161, eq_improper 0.0016, mbis 0.00062, charge MAE **0.017 e** (small A23 long: 24.09 / 0.433 / 23.5 / 0.218 / 0.0012 / 0.0023). θ0 per centre vs measured: C4 107.6 (109.6), C3 120.3 (120.1), N3 117.0 (117.9), N2 111.6 (115.3), O2 108.0 (112.3), S4 110.7 (108.2), N4 106.8 (109.1), S2 91.3 (91.9), **C2 180.0 (179.0), p10 178.7**, P4 108.6 (107.5); angle K C4 56, C3 56, O2 66, S4 78, C2 16.9 (p10 7.1); bond K H1-C4 358, C3-C3 351, C4-C4 201, C3-N3 314, C3-O1 728, H1-N3 497, C2-N1 1332; improper \|k\| 0.022 (floor) / 3.90 / 9.70, improper feature RMS **1.11** at 17 800 steps (A23 small long: 5.92 at 26 400 — A24 holds); feature RMS 1.00–1.06. **2.65 steps/s, 28.75 GiB**. Next: S2 large gate from its `best.pt`, chained after the S2 small gate 3742961 |
| 3742960 | 2026-10-12 | 66cc421+ | 1 | S1 joint, A21–A24, restructured step — **small long run on A24** (the A23 long run 3723816 re-run with saturating windows; no step cap) | small (32/2/32), cap 131072, lr 3e-4, clip 10, plateau ×0.5 / patience 5 evals, early stop 10 evals | 37 800 (patience stop; best 35 800) | 18.30 | **22.09** | **gate passes** (3 h 01, n139) — **the A24 verdict**: improper feature RMS **1.33 after 37 800 steps** (A23: 5.92 at 26 400, the only failure), improper \|k\| 0.022 (floor) / 3.96 / 8.70, C2 θ0 **180.0 with p10 = p90 = 180.0** / K 24.6 (p10 13.8). Better than A23 at every step and at the end (22.09 vs 24.09; A23 stopped at 26 400): hessian_k 0.430, geom 21.5, hessian_eq 0.211, eq_improper 0.0017, mbis 0.0018, charge MAE 0.030 e; lr cut four times (20 400 / 24 600 / 33 400 / 37 200). θ0 vs measured: C4 107.8 (109.6), C3 120.9 (120.1), N3 116.7 (117.9), N2 111.5 (115.3), O2 109.1 (112.3), S4 108.2 (108.2), N4 105.9 (109.1), S2 95.0 (91.9), P4 105.5 (107.5); bond K H1-C4 359, C3-C3 350, C4-C4 202, C3-O1 727, H1-N3 495 — the same tables as the large run to within a few percent. **3.82 steps/s, 7.80 GiB** (the long run amortises the compile the 1500-step A/B paid) |
| 3742961 | 2026-10-12 | 66cc421+ | 1 | **S2 joint (first S2 run on a GPU)**: S1's five objectives + `torsion` (`BoltzmannProfileLoss`, soft-max 100, → `propers.k`), init from the S1 small long run's `best.pt` (3742960) | small (32/2/32), cap 131072 per source (hessian / mbis / torsion-scan 83 951 rows, 2 782 excluded), lr 3e-4 (fresh Adam, 500-step warmup), clip 10 | 1500 | 21.63 | 26.21 | **gate passes** (33 min, n139). torsion 5.72 (step 0, propers at zero) → 4.39 and still falling; proper n²-curvature Σn²\|k_n\| per row 2.66 / 6.90 / 11.25 (p10/p50/p90), proper stream RMS 1.14. The S1 objectives hold: hessian_k 0.432, geom 21.2, hessian_eq 0.223, mbis 0.0018, charge MAE 0.030; θ0 within 1° of S1 except N3 113.2 (116.7 → measured 117.9); C2 180.0 (p10 = p90) / K 22.0. **Watch**: eq_improper 0.0018 → 0.0010 while improper \|k\| p50 fell 3.96 → 0.42 (p90 8.70 → 5.19, p10 at the floor) in 1500 steps — the fresh Adam at 16× the lr S1 converged at lets the improper-strain objective (ByteFF `ImproperTorsion_Energy`) win over the partial Hessian; the S2 long run decides whether it recovers as the plateau scheduler cuts lr. **1.41 steps/s, 9.58 GiB** (torsion batch: 24 conformers per scan row) |
| 3747335 | 2026-10-12 | 66cc421+ | 1 | **S2 joint, large**, init from the S1 large long run's `best.pt` (3743201) | **large** (128/8/128), cap 131072 per source, lr 3e-4 (fresh Adam, 500-step warmup), clip 10 | 1500 (best 1000) | 12.75 | 14.83 (best) / 16.09 (step 1500) | **gate passes** (36 min, n470). torsion 5.27 → 3.57 (small S2 gate: 4.39); proper Σn²\|k_n\| 2.94 / 8.77 / 13.84; charge MAE 0.018; θ0 within 1° of S1 except **N3 117.0 → 113.1** (measured 117.9; the small S2 gate landed at 113.2 from 116.7 — the same shift in both tiers once torsion joins) and O2 108.0 → 107.4; C2 180.0 / K 17.3; improper \|k\| 0.022 / **5.28** / 10.07 (S1 large end 3.90 — up here, down tenfold in the small S2 gate: the fresh-optimizer kick lands differently per tier, not a systematic S2 pull). **The S1 objectives regress at step 1500 under the kick**: hessian_k 0.428 → 0.450, geom 10.26 → 11.85, hessian_eq 0.165 → 0.224, so eval 14.83 at 1000 → 16.09 at 1500; the plateau scheduler has not cut yet. The S2 large long run 3748659 reads whether the S1 tables re-settle as lr decays. **1.28 steps/s, 39.69 GiB** (S1 large: 2.65 / 28.75) |
| 3748659 | 2026-10-13 | 66cc421+ | 1 | **S2 joint, large, long run** (no step cap), init from the S1 large long run's `best.pt` | large, cap 131072, lr 3e-4 (fresh Adam, 500-step warmup), clip 10, plateau ×0.5 / 5 evals, early stop 10 evals | 2800 (patience stop; **best 800**) | 12.87 (step 2500) | 14.32 (best) / 24.60 (step 2800) | **Diverged — the stage is a failure although the handoff passes on the step-800 `best.pt`.** Eval after the warmup reached 3e-4: 14.3 (800) → 15.2 → 15.2 → 16.2 → 14.7 → 15.3 → 15.4 (2000) → 16.7 (2200, lr cut to 1.5e-4) → 19.6 → 15.6 → **24.6** (2800): hessian_k 0.428 → **0.696**, geom 10.26 → **19.46**, hessian_eq 0.165 → 0.353, while torsion fell 5.27 → 3.0–4.1 and the *train* losses stayed flat (train total 12.3–12.9, train geom 8.2 → 9.6, train hessian_k 0.43–0.50). The train/eval gap on the hessian objectives (geom 8.2 vs 10.2 at 500, 9.6 vs 14.8 at 2500; S1 large had 12.9 vs 13.4) is the kneed train loss hiding **outlier structures whose stationary forces grow** — the S1 geometry objectives train only `r0`/`θ0` and the Hessian objective only the stiffnesses, so neither can push back on the proper amplitudes the torsion profile loss grows (Σn²\|k_n\| p50 9.0 / p90 14.6 at step 800 and rising), and those amplitudes exert forces and curvature at the QM minima of the hessian set. Pre-clip grad norm 420–600 (S1: 125–280; clip 10 normalises every step, so the torsion gradient owns the direction). Diagnosis against ByteFF `loss.py`: ByteFF's stage 2 has the same trainable-term split but three mechanisms Rondo's S2 lacks — `mask_by_circstd` (the profile loss trains only the dihedrals that actually rotate across the scan's conformers), `max_config`/`set_grad_max` (a per-table gradient cap), and the `l1_norm`/`l2_norm` aux loss on `ProperTorsion_k`. **1.56 steps/s, 39.69 GiB.** The chained S3 gate 3751063 / long 3751064 are held (`scontrol hold`); the CPU report on the step-2800 `last.pt` (3751903, `param_report_last2800.json`) confirms it: proper Σn²\|k_n\| p10/p50/p90 **3.8 / 9.0 / 14.6 → 17.8 / 34.2 / 56.0** (×4 in 2 000 steps), improper \|k\| p50 **4.72 → 0.022** (every centre on the floor — the torsions now carry the out-of-plane curvature and the improper-strain objective kills k), θ0 N3 112.8 → **105.3** (measured 118; the stage's own gate fails on it), O2 108.5 → 103.5, P4 107.2 → 103.0, angle K C3 55 → 65, N3 57 → 67; bond K / r0 unchanged, charge MAE 0.018 → 0.022, feature RMS still 1.0–1.17 (no stream drift — this is not an A24-class failure) |
| 3765199 | 2026-10-13 | 66cc421+ | 1 | **S2 joint, large, A25 gate**, init from the S1 large long run's `best.pt` (3743201) | large, cap 131072, **lr 1e-4** (fresh Adam, 500-step warmup), clip 10, **+ L1 prior on `propers.k`, weight 1.0** (A25) | 1500 (best 1500) | 11.18 | 14.16 (step 1500 = best) | **Passes the A25 criteria** (38 min, n129). Eval 16.12 → 15.47 → 14.72 → 14.67 → 14.63 → 14.58 → 14.73 → 14.31 → **14.16** (the A24 gate at 3e-4: best 14.83 at 1000, 16.09 at 1500). torsion 5.27 → 3.63 (A24 gate 3.57 — the same fit) at proper Σn²\|k_n\| **0.87 / 2.31 / 7.85** (A24 gate 2.94 / 8.77 / 13.84; the diverged run 17.8 / 34.2 / 56.0): the same profile fit at a quarter of the amplitude — the prior removes redundancy rather than under-fitting; the L1 term (mean \|k\|) 0 → 0.12. Improper \|k\| 0.022 / **2.71** / 7.36 (S1 end 3.90; the diverged run put every centre on the 0.022 floor). **The S1 objectives hold**: hessian_k 0.4281 → 0.4276, geom 10.26 → 9.81, hessian_eq 0.165 → 0.170, improper strain 0.00164 → 0.00135, mbis 0.00064 → 0.00062. θ0 within 2° of S1 except **N3 111.7** (measured 117.9, −6.2°; the A24 gate went to 113.1 — the torsion stage pulls N3 in both recipes, the long run reads whether it keeps going; the handoff window ends at 107.9), N2 111.3 (−4.0), O2 108.1 (−4.2); C2 180.0 / K 18.6. Bond K / r0 unchanged, charge MAE 0.017, feature RMS 1.00–1.11. Pre-clip grad norm **34** (the 3e-4 run: 420–600). **1.28 steps/s, 39.69 GiB.** The S2 long run 3765202 is released; the S3 gate 3768159 / long 3768163 are queued held behind it |
| 3765202 | 2026-10-13 | 66cc421+ | 1 | **S2 joint, large, A25 long run** (no step cap), init from the S1 large long run's `best.pt` (3743201) | large, cap 131072, lr 1e-4 (fresh Adam, 500-step warmup), clip 10, L1 prior on `propers.k` weight 1.0, plateau ×0.5 / 5 evals, early stop 10 evals | 4600 (patience stop; **best 2600**) | 11.66 (step 4600) | **14.13** (best) / 14.75 (step 4600) | **Passes — the stage converges instead of diverging** (60 min, n38). Eval sits on a 14.1–14.4 plateau from step 1400 (A24 at 3e-4: 14.3 → 24.6 over the same span), one transient spike at 3800 (16.04: the L1 term jumped 0.16 → 0.66 and hessian_k/geom with it; back to 14.15 at 4000 after the one lr cut to 5e-5) — the prior pulls the amplitudes back but the dynamics touch the edge. torsion 5.27 → **3.44** (best; the diverged run reached 3.0–4.1 only by wrecking S1). S1 objectives at best: hessian_k 0.428 (flat), geom 9.94 (start 10.26), hessian_eq 0.180 (start 0.165, +9 %), improper strain 0.0013 (start 0.0016), mbis 0.00061. param_report on best: proper Σn²\|k_n\| **0.93 / 2.44 / 10.2** (gate 0.87 / 2.31 / 7.85 — p90 creeping), improper \|k\| 0.022 / **2.36** / 7.0 (S1 end 3.90 → gate 2.71 → 2.36: a slow decline, p10 on the floor), θ0 N3 **111.3** (−6.7°; S1 end 117.0, gate 111.7 — still drifting, window ends at 107.9), N2 −4.0, O2 −4.1, C2 180.0 / K 19.8, bond K / r0 unchanged, charge MAE 0.017, feature RMS 1.00–1.11, LJ untouched (S2 does not train it). **1.96 steps/s, 39.69 GiB.** Gain over the 1500-step gate is small (14.16 → 14.13; torsion 3.63 → 3.44): S2 at 1e-4 converges within ~2 500 steps. The S3 gate 3768159 is released on this `best.pt` |
| 3768159 | 2026-10-13 | 66cc421+ | 1 | **S3 joint, large, A25 gate** (first GPU S3 on the joint recipe), init from the S2 large long run's `best.pt` (3765202) | large, cap 131072, **lr 2e-5** (fresh Adam, 500-step warmup), clip 10, 1500 steps, relax trainable = every bonded table + σ/ε, LJ prior weight 1.0 | 1500 | 10.06 (step 1500) | **10.60** (step 1500; 10.54 at 1400) | **Handoff passes mechanically, judgement fails — K collapse** (75 min, n36, **0.46 steps/s, 57.36 GiB**). relax 2.88 → 2.25 (flattening from 1200), torsion 3.45 → 3.31, mbis flat, LJ log ratio σ −0.02 / ε **−0.18** median (p10 −0.32), feature RMS 1.00–1.12; but **hessian_k rises monotonically 0.428 → 0.469** (+9.5 %) and the bond K on best fall 30–70 % in 1 500 steps: H1-C4 358 → 249, C3-C3 349 → 154, C4-C4 202 → 128, C3-N2 384 → 152, **C3-O1 731 → 230**; angle K 56 → 44 (C4), 59 → 39 (N3); improper \|k\| p50 2.36 → 1.88. hessian_geom 9.94 → 4.28 and hessian_eq 0.180 → 0.109 "improve" because a softer K scales every force at the QM minimum down — the same cheap direction S1 closed for the geometry objectives by detaching K there. Pre-clip grad norm 100–180 (S2: 34) with clip 10: the relax gradient sets the direction. θ0 N3 111.1 (−6.9°; still drifting), C2 180.0 / K 15.3. The S3 long run 3768163 is **not** released; A26 resubmits the gate with the curvatures out of the relax set (3776940 gate, 3776941 long held) |
| 3776940 | 2026-10-13 | 66cc421+ | 1 | **S3 joint, large, A26 gate** (relax trains `r0`/`θ0`, `propers.k`, σ/ε; curvatures stay the Hessian's), init from the S2 large long run's `best.pt` (3765202); first run on the restored prepared-row snapshot (v4, no stored cutoff graph) and the cleaned tree | large, cap 131072, lr 2e-5 (fresh Adam, 500-step warmup), clip 10, 1500 steps, 6 workers/source, LJ prior weight 1.0 | 1500 | 13.58 (step 1500) | **15.83** (step 1500; 17.01 at 0) | **Passes** (43 min, n419; smoke/compile 19 min, training **0.66 s/step** wall incl. evals — 2.24 s on 3768159 — delivered 1.67 steps/s, accelerator 3.2 steps/s, cumulative data wait 428 s of 993 s, **55.7 GiB**; every source warm from the 74 GB snapshot restored in 32 s). hessian_k **0.4277** (flat; A25 gate 0.469 ↑), bond K unchanged against the S2 table (H1-C4 358, C3-C3 352, C4-C4 202, C3-O1 734), angle K unchanged (C4 56.6, N3 58.3), improper \|k\| p50 2.36 → **2.52**; hessian_geom 9.94 → **8.96** and hessian_eq 0.180 → 0.166 — a real equilibrium gain, r0/θ0 moving on the relax forces with K held; relax 2.88 → **2.76** (slow, noisy; with K free the A25 gate reached 2.25 by softening K) ; torsion 3.45 → 3.35, Σn²\|k_n\| p50 2.44 → 2.89 (p90 8.0, L1 term 0.12–0.31: watch); θ0 N3 **112.5** (S2 111.3, the drift turned back toward 117.9), C4 −2.0, N2 −2.4, O2 −3.1, C2 180.0 / K 20.2; LJ log ratio σ −0.02 / ε −0.18 median (p10 −0.32); charge MAE 0.017; feature RMS 1.00–1.11. S3 long run 3776941 released on it |

## 8. System Boundary

| Concern | Owner | Contract |
|---------|-------|----------|
| Functional form | molpy `ForceField` | styles, settings, special bonds — declared once; Rondo prices `canonical()` |
| Style kernels | `molpot.potentials` | molrs conventions, molpy oracle to 1e-10 |
| Covalent-graph encoder | `molrep.encoders.covalent_graph` | topology and integer channels only; `EncoderFeatures(atom, bond)` |
| Injected encoders | `molrep.encoders.mace_conformer` | the same `EncoderFeatures`, positions detached |
| Equivalence classes, bond lookup | `molrep.chem` (`Equivalence`, `BondTable`) | static shapes, no host sync |
| Predicted parameters | `Rondo.energy`, `Rondo.tables` | written on the batch beside the rows (`bonds.k`, `propers.k`, `atoms.charge`, …) |
| Per-loss parameter sets | `molix.core.steps.JointObjective` | `JointTrainStep.keep(tables, trainable)` on the source's one `predict`, then `energy_at(..., styles=)`, `write_tables`, `partial_hessian(frame, tables)` |
| Derivatives | `molix.autograd.Derivative` | the single force op |
| Static export | `Rondo.parameterize` → molpy | one type per row; molpy writes the engine files |
| Input columns | molpy perception, written by MolHub | `formal_charge`, ring features, `equi_index`, `bond_type`, `in_ring`, `propers.linear`, one improper row per centre, `gaff2_sigma/epsilon` |

## 9. Version Pinning

| Item | Value |
|------|-------|
| Reference commits | `ByteDance-Seed/byteff2@663451d…` (`v1.1.0`, encoder and bonded heads), `bytedance/byteff@90ef54a…` (training-recipe line references), `choderalab/espaloma@413eb55…` (lineage) |
| ByteFF-26 weights | HF `ByteDance-Seed/byteff2` revision `v1.1.0`, `ByteFF-26/models/member-0{1..5}/optimal.pt` (Apache-2.0); sha256 in `.claude/specs/rondo-enc-01-graph-encoder.md` §8 |
| molpy | `molcrafts-molpy>=0.18` (`pyproject.toml`); the encoder's MolHub columns are written on molpy 0.20 (`rondo-enc-01`) |
| GAFF2 reference | AmberTools25 `dat/leap/parm/gaff2.dat` (GAFF 2.2.30), MOD4 RE section; σ = 2R*/2^{1/6} |
| Dependencies | `molrep.encoders.covalent_graph`, `molrep.encoders.mace_conformer`, `molrep.encoders.features`, `molrep.chem`, `molpot.classical`, `molpot.charge`, `molpot.potentials`, `molpot.forcefield.partial_hessian`, `molix.autograd` |
| Public docs mirror | `docs/molzoo/specs/rondo.md` |

## 10. Drift Policy

### 10.1 Triggers

Re-run `/molzoo-spec rondo` or the `molzoo-auditor` when a head changes its
emitted keys, the backbone changes (§10.4), the supported style set changes,
the charge model changes, a §7.4 row regresses >10 %, or the compile seam
moves.

### 10.2 Breaking changes

The ten-section structure is immutable.

### 10.3 Code-before-spec

This spec is `stable`; code may be written against it.

### 10.4 Backbone-change gate

An encoder that does not average its atom and bond rows over
`atoms.equi_index` voids A7 and may not be built or injected without a test
asserting equal rows for chemically equivalent atoms.

## Appendix A. Maintenance Log

- 2026-10-30: Body rewritten to the code; the 2026-10-28 revision section is
  folded in and deleted, status `stable` in the header and §1.4. §1–§4:
  `CovalentGraphEncoder` (ByteFF-26's transformer) in place of the
  Espaloma-style `ChemEncoder`; `encoder=` injection (`MaceConformerEncoder`,
  covalent settings beside it a `TypeError`); `tables`, `parameterize`,
  `load_state_dict(prefixes=)`; inputs `atoms.equi_index`, `propers.linear`,
  one improper row per C/N centre, `graphs.total_charge` only for EEM;
  outputs without the bonded `*_ref` columns (only `atoms.sigma_ref` /
  `epsilon_ref` remain); the bonded heads as `TanhWindow` with ByteFF-26's
  constants, `K = k/2`, the hypervalent patch, cvff `d = −1, n = 2`, propers
  `n = 1..4` at 0°/180°; `HeadsSpec` / `CovalentGraphEncoderSpec` fields and
  defaults, `bond_k_floor` 80 (`ByteFF.load` 40). §3.7 describes the joint
  step as `molix.core.steps.joint` runs it. §5 / §6 rows rewritten or marked
  superseded (A11, A14, A22, A23); A27 (bond floor), A28 (injected encoder),
  R3 (resonance step), R4 (straight-through bonded windows) added. §7.1
  claimed: ByteFF-26 parity from `rondo-enc-01` §8. §7.2, §8, §9, §10.4
  updated to the current tests, owners and pins. No code changed.
- 2026-10-27: `Rondo(forcefield, *, dtype=)`. The force field is one
  `Potentials` (`Potentials.from_forcefield`); Rondo holds that object and adds
  one parameter head per style it prices. `terms=` and `wrt=` are not
  constructor arguments. `forward` differentiates `atoms.pos` and writes
  `atoms.forces_pred`. A later style is a force field that declares it, loaded
  with `load_state_dict(strict=False)`.
- 2026-10-27: `load_state_dict` writes only a present prefix (`graph.`, `coordinate.`, `heads.`, `potentials.`); a missing prefix and an encoder the model was not built with are not written, and any other first component is unexpected.
- 2026-10-12: `energy(styles=)` (A18) after the S1 runs 3680577 / 3685025 /
  3685026 showed sp3 `θ0` walking 109° → 99° → 95° under the stationary-force
  loss alone: a common `θ0` shift at a symmetric centre moves no atom, so the
  force loss leaves it free. S1 gains `hessian_eq` (ByteFF `Bond_Energy` /
  `Angle_Energy`: bonded strain on the minimum, per term, through `r0` / `θ0`)
  and the stationary-force loss gains ByteFF's `skip_large_nb` on the pair
  styles' own force. No checkpoint invalidated (the model's parameters are
  unchanged).
- 2026-10-12 (later): the unbounded run on that configuration (3690962) failed
  the gate on N3 / P4 `θ0`. The per-style axial-force decomposition at N3
  centres showed the angle term cancelling the improper's force at pyramidal
  nitrogens — the improper `k` was trained on curvature only and the force
  objective could not reach it. ByteFF's third equilibrium objective,
  `ImproperTorsion_Energy`, had not been ported; S1–S3 gain
  `hessian_eq_improper` (A19). Training-side only; no model change.
- 2026-10-12 (later still): the gate run on A19 (3703994) diverged — the
  improper strain is linear in a signed `k`, so the new objective had no
  floor. `CvffHead` now predicts $k_\text{ref}\,e^{\mathrm{raw}}$ from
  ByteFF's GAFF2 mean (A20, `heads.improper_k_ref`); `impropers.k_ref` is a
  new output column. Checkpoints from before carry a signed improper table
  and are not comparable (none are reused).
- 2026-10-12 (night, operator: "搞一下性能优化！"): the first long run on A21–A23
  (3723816) trained cleanly for 26 400 steps and failed only the handoff
  gate's feature-RMS window — the improper stream drifting under a
  straight-through clamp — so the windows now saturate (A24). The large tier
  could not hold one step at the tier capacity (3732774 / 3735480 / 3736461,
  90.5 GiB at two capacities): two leaks and one waste in the *training
  step*, not the model — the eval loop kept the previous batch's outputs
  through the next forward, the eval bodies left their graphs on the resident
  validation batches, and the five objectives each re-ran the encoder on
  their own copy of the one hessian batch. The step now predicts `tables`
  once per source and shares them (§3.7), the charge objective prices no
  energy, and evaluation releases its graphs. Every checkpoint stays valid
  (the model did not change).
  The large tier then trained to its gate in one sitting (3743201: best 10.85
  at 15 800 steps, 2.65 steps/s, 28.75 GiB), with the improper stream at RMS
  1.11 after 17 800 steps, and the small long run (3742960) closed the A24
  question: 37 800 steps, improper stream RMS 1.33, C2 θ0 held at 180.0 from
  p10 to p90, and a better end point than A23 (22.09 vs 24.09).
  S2 then ran on a GPU for the first time (3742961, small, torsion-scan
  joining the five S1 objectives from the S1 `best.pt`): the torsion loss
  moves from the zero-proper baseline and the S1 tables hold, but the
  improper amplitude median dropped tenfold in 1 500 steps under the fresh
  optimizer, and the large S2 gate (3747335) showed the other face of the
  same kick: torsion 5.27 → 3.57, improper amplitude up, N3 θ0 down 4° in
  both tiers, the S1 objectives regressing between 1 000 and 1 500 steps
  at the full rate — and the S2 long run (3748659, large) read it: eval
  diverged after the warmup (best 14.3 at 800 → 24.6 at 2 800, hessian_k
  0.43 → 0.70) while the train losses stayed flat. The torsion profile loss
  grows proper amplitudes nothing else constrains at the QM minima; ByteFF
  bounds that with a circular-std mask on the scanned dihedrals, a per-table
  gradient cap and an L1/L2 aux loss on the torsion amplitudes, none of
  which Rondo's S2 had. The paper names the L1 norm and a stage rate of
  1e-4 (2e-5 for the fine-tune), so those two are now in (A25); the
  code-only mask and cap wait for evidence they are needed. The S3 chain
  (3751063 / 3751064) stays held until an S2 run on A25 passes. The one loss no tier moves is `hessian_k`, 0.428 on
  large and 0.433 on small from 2 000 steps on: a label-noise or form floor
  of the partial-Hessian objective, not capacity (open). **2026-10-13 noon: the S2 large gate on A25 (3765199) passes its criteria** — Σn²\|k_n\| p50 2.3 (not 9 and not 34), improper \|k\| p50 2.7, torsion 5.27 → 3.63 (the no-prior fit at a quarter of the amplitude), the S1 objectives flat or falling, pre-clip grad norm 34 against 420–600. The user had authorised autonomous iteration (gate → judge → long run → next stage); the S2 long run 3765202 runs and the S3 chain 3768159 / 3768163 waits held on it. Watch N3 θ0 (111.7, −6.2° from measured) in the long run. **13:21 the S2 long run (3765202) passes**: best 14.13 at 2600, patience stop at 4600, the S1 objectives hold on a plateau, torsion 3.44, Σn²\|k_n\| p50 2.4, improper \|k\| p50 2.36, N3 θ0 111.3 (−6.7°, drifting slowly), one transient amplitude spike at 3800 that the prior reversed. S3 gate 3768159 released on its `best.pt`; the S3 long run 3768163 waits held. **14:47 the S3 gate (3768159) fails judgement — K collapse**: the handoff passes mechanically but hessian_k rises 0.428 → 0.469 and the bond K fall 30–70 % in 1 500 steps (C=O 731 → 230) while hessian_geom halves for the wrong reason (every force at the minimum scales with K). ByteFF trains K on its stage-3 energy/force loss too, but under a per-table gradient cap the repo gives no values for; Rondo takes the curvatures out of the relax set instead (A26). Resubmitted: S3 gate 3776940 (A26), S3 long 3776941 held behind it. First S3 numbers on GH200: 0.46 steps/s, 57.4 GiB at 131072 with four padders. **20:32 the A26 gate (3776940) passes on the cleaned tree and the restored prepared-row snapshot**: hessian_k flat at 0.428, bond and angle K unchanged, improper \|k\| up to 2.52, hessian_geom 9.94 → 8.96, relax 2.88 → 2.76 (slow — the fast relax descent of A25 was K softening), N3 θ0 back up to 112.5; 0.66 s/step wall (2.24 before: the loader was the bound, every job started on an empty node-local cache and the stored cutoff graph made the cache 3× too big to snapshot), 55.7 GiB. S3 long run 3776941 released.
- 2026-10-12 (evening, operator: "先修 bug，最后再长跑，不需要考虑之前的 ckpt"):
  the two structural findings of the architecture review
  (`.claude/notes/rondo-architecture-review-2026-10-12.md`) are in: every
  valence context reads the term's own bond features through a per-batch
  `BondTable` (A21; the angle / proper / improper MLPs grow their input width,
  so **every checkpoint before this is invalid**), and every multiplicative
  parameter is bounded around its reference (A22). The unbounded run 3707870
  (gate passes, N3 θ0 116.9° vs 117.9°) showed the sp-carbon angle `K`
  collapsing to 1.1 with every C2 `θ0` pinned at the 179.5° ceiling: the
  ceiling and the 177.5° linear reference become 180° (A23). Not done, on purpose: a
  θ0 "chemical window" — ByteFF's `tanh` on θ0 spans ±149°, i.e. none; per-term
  normalisation of `hessian_eq` — ByteFF's `Bond_Energy` / `Angle_Energy` are
  two means with weight 1 against our one pooled mean, a constant factor;
  head-gain rebalancing — ByteFF's gains have the same r0 : θ0 : torsion ratio
  as ours (measured in the D2 probe, `.jobs/head_grad_probe.py`).
- 2026-10-11: Realignment with ByteFF (plan `rondo-byteff-squishy-iverson`).
  `Rondo(ff, spec, wrt=)` on the molrs force-field IR through
  `Potentials.from_forcefield`; `outputs` / `trainable` / `energy_terms` /
  `param_groups` and the per-objective machinery removed from the model — the
  training step owns them. Heads moved to `molpot.classical` with per-row
  references (covalent radii, hybridization angle, GAFF2 σ/ε); torsions as
  signed $k_n$ at fixed phases; three improper rows per centre; linear-centre
  mask; ByteFF input channels on the encoder. **Every earlier checkpoint is
  invalid**, deliberately.
- 2026-10-06: Drift repair against the 2026-09-29 restructure.
- 2026-09-17: Created; filled from `scripts/mm_param_learning/`.
