# Rondo Specification

This page is the implementation contract for `molzoo.rondo`. It is not a
tutorial; use the MolZoo user guide for theory narrative and worked examples.

| Field | Value |
|-------|-------|
| Module | `molzoo.rondo` |
| Entry point | `Rondo(forcefield, spec, wrt=, dtype=)` (config `RondoSpec`) |
| Paper | Not a reproduction. Lineage: Wang *et al.*, "End-to-end differentiable construction of molecular mechanics force fields", *Chem. Sci.* **13**, 12016 (2022) — Espaloma. Structure aligned with: Zheng *et al.*, "Data-driven parametrization of molecular mechanics force fields for expansive chemical space coverage", *Chem. Sci.* (2025) — ByteFF. |
| arXiv | Lineage [2010.01196](https://arxiv.org/abs/2010.01196) (Espaloma); structure [2408.12817](https://arxiv.org/abs/2408.12817) (ByteFF) |
| DOI | [10.1039/D2SC02739A](https://doi.org/10.1039/D2SC02739A) (Espaloma) |
| Reference implementation | `choderalab/espaloma@413eb55074037540e1414eb365da7304bd5d72be`, `bytedance/byteff@90ef54a7c38e58f162adc23f3e8f457aaf922d46` |
| Spec status | stable |

## 1. Scope

Rondo is MolNex's **learnable Class-I force field**: a graph neural network
that reads a molecule's covalent topology and emits every *open* parameter of
the Class-I force field it is built on, and evaluates that field in the same
module through the molpot potentials of the field's styles.

It is a *recipe*, in the MolZoo sense — the named composition of primitives
that already live in `molrep` and `molpot`. The force field **is** the
Hamiltonian: a molpy `ForceField` declares the styles, cutoff, mixing rule,
Coulomb constant and 1-4 scales once (operator ruling 2026-10-06), and
`Potentials.from_forcefield(ff)` turns it into torch potentials. Rondo adds
the encoder and one head per parameter table, nothing else, and owns no kernel.

### 1.1 What Rondo owns

| Concern | Rondo surface |
|---------|---------------|
| The parameters a style leaves open, as functions of the graph | `Rondo.heads` — `BondHead`, `AngleHead`, `PeriodicHead`, `CvffHead`, `LJHead` (`molpot.heads.classical`) and the charge model (`molpot.charge.AtomicChargeModel`) |
| The chemical-perception backbone and its width | `Rondo.encoder` → `molrep.chem.ChemEncoder` |
| Hybridization read off the bond list (reference angles, linear centres) | `Rondo.hybridization` → `molrep.chem.Hybridization` |
| The three primitives a training step composes | `Rondo.tables`, `Rondo.energy` (every style, or `styles=` a named subset), `Rondo.partial_hessian` |
| Energy and derivatives as one model forward | `Rondo.forward` = `molix.autograd.Derivative(Rondo.energy, wrt)` |
| The configuration contract | `RondoSpec(encoder=EncoderSpec, heads=HeadsSpec)` |

### 1.2 What Rondo does not own

| Concern | Owner |
|---------|-------|
| The functional form, its settings and units | the molpy `ForceField` (molrs IR) |
| The torch implementation of each style | `molpot.potentials` (`bond:harmonic`, `angle:harmonic`, `dihedral:periodic`, `dihedral:opls`, `improper:cvff`, `pair:lj/cut`, `pair:coul/cut`) |
| Which loss trains which parameters | the training step (`molix.core.steps.JointObjective.trainable`) |
| Pair relations (1-2 / 1-3 / 1-4) and the cutoff pair list | `molix.data.PairRelation`, `molix.data.NeighborList` |
| Input columns: formal charge, ring features, bond order, GAFF2 reference σ/ε | MolHub (THEMol decoding) |
| The per-term partial Hessian | `molpot.forcefield.partial_hessian.TermHessian` |
| The derivative op | `molix.autograd.Derivative` |
| Losses, padding, collation, batch tasks | `molix` |
| Datasets, splits, units, provenance | MolHub |
| Training configuration, scheduling, calibration evidence | MolExp / MolAb |

### 1.3 Lineage and the deliberate departures

Rondo sits in the Espaloma → ByteFF family: a GNN over the covalent graph
predicts MM parameters, and the MM energy is differentiated for forces. After
the 2026-10-11 realignment its *structure* follows ByteFF — joint
multi-dataset training with per-loss parameter sets, equilibrium forces for
`r0` / `θ0`, ByteFF's input features, signed-`k` fixed-phase torsions, a
GAFF2 non-bonded start. Two choices still depart from both predecessors:

| # | Rondo | Espaloma / ByteFF | Consequence |
|---|-------|-------------------|-------------|
| D1 | Partial charges from **electronegativity equalisation** (`molpot.charge`: `ChiEtaHead` → `ElectronegativityEqualization`), supervised by MBIS charges | Espaloma: charge-equilibration read-out. ByteFF: AM1-BCC-style bond-charge corrections regressed to AM1-BCC | Charges respond to the solved electrostatic problem; one dense per-molecule solve |
| D2 | Lennard-Jones σ / ε **learned around the atom's GAFF2 type** (`LJHead`: `σ = σ_GAFF2·exp(0.1·raw)`, `ε = ε_GAFF2·exp(raw)`, zero start) | ByteFF regresses predicted σ/ε to GAFF2 labels or uses them unchanged | Starts exactly on GAFF2; gas-phase intramolecular data gives little signal, so a stage that names `atoms.sigma` / `atoms.epsilon` carries a prior toward the reference (R1) |

### 1.4 Status of this document

`stable`. §5 carries no `unknown` row. §7.1 stays `not claimed` by
construction.

## 2. Public Contract

### 2.1 Required Inputs

| Direction | TensorDict path | Shape | Dtype | Contract |
|-----------|------------------|-------|-------|----------|
| In | `atoms.Z` | `(N,)` | int64 | Atomic numbers; ghost rows carry `0` |
| In | `atoms.pos` | `(N, 3)` | the model's `dtype` | Positions, Å. Differentiated against; never read by the encoder |
| In | `atoms.batch` | `(N,)` | int64 | Graph membership |
| In | `atoms.formal_charge` | `(N,)` | int64 | Encoder channel (clamped to ±2) |
| In | `atoms.ring_connectivity` | `(N,)` | int64 | Encoder channel: ring bonds on the atom (clamped to 4) |
| In | `atoms.min_ring_size` | `(N,)` | int64 | Encoder channel: smallest ring through the atom, `0` none (rings above 8 read as 8) |
| In | `graphs.total_charge` | `(B,)` | the model's `dtype`, integer-valued | Encoder channel (read as an integer class, clamped to ±5) and the charge model's constraint; no neutral fallback |
| In | `atoms.gaff2_sigma`, `atoms.gaff2_epsilon` | `(N,)` | the model's `dtype` | Reference σ (Å) / ε (kcal/mol) of the atom's GAFF2 type; `NaN` for an untyped atom, which makes the batch energy `NaN` rather than pricing silently. Required iff `pair lj/cut` is declared |
| In | `bonds.atomi` / `bonds.atomj` or `bonds.bond_index` | `(N_bonds,)` / `(2, N_bonds)` | int64 | Covalent topology |
| In | `bonds.bond_type` | `(N_bonds,)` | int64 | Bond order code (1 single, 2 double, 3 triple, 4 aromatic): encoder channel and hybridization input |
| In | `bonds.in_ring` | `(N_bonds,)` | int64 | Encoder channel |
| In | `angles.atomi` / `atomj` / `atomk` | `(N_angles,)` | int64 | Valence angles; `atomj` is the vertex |
| In | `propers.atomi` … `atoml` | `(N_propers,)` | int64 | Proper torsions |
| In | `impropers.atomi` … `atoml` | `(N_impropers,)` | int64 | Impropers, **centre first**; **three rows per centre**, the cyclic orders `[c,i,j,k]`, `[c,j,k,i]`, `[c,k,i,j]` (ByteFF's `k/3` split, `classical_forcefield.py:261-284`) |
| In | `pairs.pair_index` | `(E, 2)` | int64 | Cutoff pair list at the force field's pair cutoff; read when a pair style is declared |
| In | `pairs.pair_diff` or `pairs.shifts` | `(E, 3)` | the model's `dtype` | Lattice remainder of each pair (detached) |
| In | `pairs.relation` | `(E,)` | int8 | `0` none, `1`/`2`/`3` = 1-2/1-3/1-4, from `molix.data.PairRelation` |
| In | `hessian_pairs`, `<ns>.hpair_<i>_<j>` | — | — | Only for `partial_hessian` |
| In | `graphs.num_atoms` | `(B,)` | int64 | Graph sizes |
| In | `atoms.mask` / `graphs.mask` / `<ns>.mask` | bool | — | Ghost-row, dead-slot and dead-term masks under static shapes |

Every encoder channel column is required: a batch that lacks one raises
`KeyError` naming it.

### 2.2 Outputs

`Rondo.forward` returns the batch. `energy` writes `graphs.energy` and every
predicted table as columns, references included, so losses and priors read
them from the batch; `forward` adds the force at `atoms.forces_pred`
(`FORCES_PRED_KEY` in `molix.schema`, the op's output key for `atoms.pos`) —
a training batch holds the DFT force at `atoms.forces` (A6). The energy lands
where a labelled batch holds its DFT energy; where it must sit beside that
label, the training step wraps the call in `molix.frame.beside_labels`
(`graphs.energy → graphs.energy_pred`, `ENERGY_PRED_KEY`), as
`molix.core.steps.JointTrainStep` does. The model moves nothing.

| Column | Shape | Written by | Contract |
|--------|-------|------------|----------|
| `graphs.energy` | `(B,)` | `energy` | Class-I energy, kcal mol⁻¹; a training step copies it to `graphs.energy_pred` beside the label (`beside_labels`) |
| `atoms.forces_pred` | `(N, 3)` | `forward` (when `"atoms.pos" in wrt`) | `-dE/dpos`, kcal mol⁻¹ Å⁻¹ |
| `bonds.k`, `bonds.r0`, `bonds.k_ref`, `bonds.r0_ref` | `(N_bonds,)` | `energy` | LAMMPS `K` (kcal/mol/Å²), `r0` (Å) and their references |
| `angles.k`, `angles.theta0`, `angles.k_ref`, `angles.theta0_ref` | `(N_angles,)` | `energy` | `K` (kcal/mol/rad²), `θ0` **degrees** and references |
| `propers.k` | `(N_propers, T)` | `energy` | Signed amplitudes per periodicity, kcal/mol |
| `impropers.k`, `impropers.k_ref` | `(N_impropers,)` | `energy` | Amplitude (positive) and reference, kcal/mol |
| `atoms.sigma`, `atoms.epsilon`, `atoms.sigma_ref`, `atoms.epsilon_ref` | `(N,)` | `energy` | Å, kcal/mol and the GAFF2 references |
| `atoms.charge` | `(N,)` | `energy` | Solved partial charges, $e$. The MBIS label stays `atoms.atomic_charge` |
| `hessian_pairs.partial_hessian` | `(P, 9)` | the caller, from `partial_hessian` | MM 3×3 blocks, kcal mol⁻¹ Å⁻² |

## 3. Forward Contract

### 3.1 Notation

| Symbol | Meaning | Code anchor |
|--------|---------|-------------|
| $N$, $B$, $E$ | Atom rows, graph slots, pair rows (capacities, ghosts included) | `atoms.Z`, `graphs.num_atoms`, `pairs.pair_index` |
| $h^{(t)}_a$, $h^{(t)}_{ij}$ | Atom / bond features after $t$ hops | `ChemEncoder` |
| $h_{ijk}$, $h_{ijkl}$, $h^{\text{imp}}_{cijk}$ | Angle / proper / improper features | `molrep.chem.context.*` |
| $c_a$ | Summed channel embeddings of atom $a$'s extra columns | `DiscreteChannels` |
| $d_a$, $\pi_a$ | Degree and π count of atom $a$ from the bond orders | `Hybridization` |
| $\chi_i, \eta_i, q_i$ | Electronegativity, hardness, partial charge | `ChiEtaHead`, `atoms.charge` |
| $T$ | Fourier terms per proper torsion | `HeadsSpec.proper_periodicity` |

Energies are kcal mol⁻¹, lengths Å, **angles degrees** (molrs convention),
charges $e$ throughout; there is no output unit conversion.

### 3.2 Chemical perception

The encoder reads **topology only** — no coordinate enters any equation
below, so the predicted parameters are geometry-independent and the deployed
force is exactly $-\partial E/\partial \mathbf{r}$.

Atom seed with the channel sum, bond seed with its channel sum, then $L$
pre-norm covalent hops:

$$
h^{(0)}_a = \mathrm{Embed}(Z_a) + c_a, \qquad
c_a = \sum_{\text{channels } f} \mathrm{Embed}_f\!\left(\operatorname{clamp}(x_{f,a} + o_f)\right)
$$

$$
h^{(0)}_{ij} = \tfrac{1}{2}\left[\mathrm{MLP}(h_i \Vert h_j) + \mathrm{MLP}(h_j \Vert h_i)\right] + c_{ij}, \qquad
h^{(t+1)}_{ij} = \mathrm{BondCtx}\!\left(\mathrm{LN}(h^{(t)}_i), \mathrm{LN}(h^{(t)}_j)\right) + c_{ij} + \mathrm{Lin}(\mathrm{LN}(h^{(t)}_{ij}))
$$

$$
h^{(t+1)}_a = h^{(t)}_a + \mathrm{MLP}\!\left(\sum_{b \,\ni\, a} h^{(t+1)}_{ab}\right)
$$

The channels are ByteFF's atom and bond features (§2.1): formal charge, total
charge (per graph, broadcast through `atoms.batch`), ring connectivity,
smallest ring; bond order and ring membership (`EncoderSpec.atom_channels`,
`bond_channels`). Every feature channel the encoder returns is layer-normalized
before a head reads it; padded rows are zeroed after the norm (A11).

### 3.3 Valence contexts — permutation symmetry is structural

$$
h_{ijk} = \tfrac{1}{2}\left[\mathrm{MLP}(h_i \Vert h_{ij} \Vert h_j \Vert h_{jk} \Vert h_k) + \mathrm{MLP}(h_k \Vert h_{jk} \Vert h_j \Vert h_{ij} \Vert h_i)\right],\quad
h_{ijkl} = \tfrac{1}{2}\left[\mathrm{MLP}(h_i \Vert h_{ij} \Vert h_j \Vert h_{jk} \Vert h_k \Vert h_{kl} \Vert h_l) + \mathrm{MLP}(h_l \Vert h_{kl} \Vert h_k \Vert h_{jk} \Vert h_j \Vert h_{ij} \Vert h_i)\right],\quad
h^{\text{imp}}_{cijk} = \tfrac{1}{3}\sum_{l \in \{i,j,k\}} \mathrm{MLP}\!\left(h_c \Vert h_{cl} \Vert h_l\right)
$$

where $h_{ab}$ is the post-hop feature of the bond $a$–$b$ **of the term
itself**, looked up by endpoint pair in a per-batch `BondTable` (one sort of
the pair keys, `searchsorted` per term; static shapes, no loader column) —
ByteFF's `OutputWithBond` (`layers.py:486-510`, A21). A term whose pair is not
a bond (a padded row) gathers a zero bond feature. A batch that carries a
valence namespace without `bonds` is refused by name.

Graph-automorphism equivalence is not enforced separately: message passing
over the covalent graph is automorphism-equivariant by construction (A7).

### 3.4 Parameter heads

Every head is a two-layer MLP per parameter whose **last layer starts at
zero**, so `raw = 0` is a per-row reference the batch supplies and the model
begins as a chemically sensible uniform field (A12, A14):

| Parameter | Map from the pre-activation | Reference | Head |
|-----------|-----------------------------|-----------|------|
| bond $K$ | $K_\text{ref}\, e^{\mathrm{raw}}$ on $[\tfrac18, 8]\,K_\text{ref}$ | 450 kcal/mol/Å² (`HeadsSpec.bond_k_ref`) | `BondHead` |
| bond $r_0$ | $r_\text{ref}\, e^{0.1\,\mathrm{raw}}$ on $[0.7, 1.4]\,r_\text{ref}$ | $r^\text{cov}_i + r^\text{cov}_j$ (`CovalentRadiusSum`, molpy `Element.covalent`) | `BondHead` |
| angle $K$ | $K_\text{ref}\, e^{\mathrm{raw}}$ on $[\tfrac18, 8]\,K_\text{ref}$ | 40 kcal/mol/rad² | `AngleHead` |
| angle $\theta_0$ | $\theta_\text{ref} + 20°\cdot\mathrm{raw}$ saturating toward $[0.5°, 180°]$ (A24), **180° inclusive** (A23) | the centre's VSEPR angle: 180° linear ($d=2,\pi>1.5$), 120° trigonal ($\pi>0.5, d\le3$), 109.47° otherwise (`Hybridization.angle_reference`) | `AngleHead` |
| proper $k_n$ | $\mathrm{raw}_n / n$, phase $\gamma_n = 0$ (odd $n$), $180°$ (even $n$) fixed | $0$ | `PeriodicHead` |
| improper $k$ | $k_\text{ref}\, e^{\mathrm{raw}}$ on $[\tfrac{1}{64}, 8]\,k_\text{ref}$, `sign = −1` fixed | $4.29 / 3$ per row (ByteFF's GAFF2 mean per centre, `definitions.py:113`) | `CvffHead` |
| LJ $\sigma$ | $\sigma_\text{GAFF2}\, e^{0.1\,\mathrm{raw}}$ on $[0.7, 1.4]\,\sigma_\text{GAFF2}$ | `atoms.gaff2_sigma` | `LJHead` |
| LJ $\varepsilon$ | $\varepsilon_\text{GAFF2}\, e^{\mathrm{raw}}$ on $[\tfrac18, 8]\,\varepsilon_\text{GAFF2}$ | `atoms.gaff2_epsilon` | `LJHead` |
| $\chi$, $\eta$ | $5\tanh$, sigmoid on $[10^{-3}, 20]$, zero start | — | `ChiEtaHead` |

Every window is reached by **saturation** (`ExpAround(lo=, hi=)`,
`LinearAround`, A22 + A24): on each side of the reference the pre-activation
is bent as $w\tanh(x/w)$ with $w$ the distance to that bound, so the slope at
the reference is exactly one and the gradient fades as a bound is approached
— ByteFF's `tanh_output` (`mlparams.py:104-110`). It was a straight-through
clamp until 2026-10-12 (A24): that holds the value but keeps passing the full
gradient, and a row the loss legitimately wants *at* a bound then pulls on its
features forever (3723816: improper stream RMS 5.9 at 26 400 steps). The
improper's window reaches lower because the strain objective (A19) sends a
pyramidal centre's amplitude to zero; the floor is where it rests.

Torsions carry a **signed** $k_n$ at fixed phases — ByteFF's and GAFF2's form
($K_n = 2k_n$ for the OPLS form, matched exactly when `dihedral opls` is
declared with $n = 1..4$). Every term is even in the dihedral, so a mirror
image has the same energy; and $\partial E/\partial k_n = 1 + \cos(n\phi -
\gamma_n)$ does not vanish at the zero start, where an $|a|$ parameterization
would (A13).

Propers through a **linear centre** (an sp atom at position $j$ or $k$) are
multiplied by zero before pricing: their dihedral is undefined
(`Rondo._proper_mask`, ByteFF `mlparams.py:151-153`, A16).

### 3.5 Charge equilibration

Per molecule the KKT system
$\begin{bmatrix} \mathbf{A} & \mathbf{1} \\ \mathbf{1}^{\mathsf T} & 0 \end{bmatrix}
\begin{bmatrix} \mathbf{q} \\ \lambda \end{bmatrix} =
\begin{bmatrix} -\boldsymbol{\chi} \\ Q \end{bmatrix}$
with a `DiagonalHardnessKernel`, $Q$ from `graphs.total_charge`, solved on a
fixed $(B, n_\text{max})$ layout, $n_\text{max}$ = `HeadsSpec.eem_max_atoms`.
A molecule over capacity fails closed (eager: raises; compiled: `NaN`).

### 3.6 Class-I energy

`Rondo.energy` maps the tables onto the parameters each style's potential
reads and sums `Potentials.from_forcefield(ff)`:

| Style | Potential | Parameters from the tables |
|-------|-----------|----------------------------|
| `bond harmonic` | $K (r - r_0)^2$ | `bonds.k`, `bonds.r0` |
| `angle harmonic` | $K (\theta - \theta_0)^2$, $\theta_0$ in degrees | `angles.k`, `angles.theta0` |
| `dihedral periodic` | $\sum_n k_n [1 + \cos(n\phi - \gamma_n)]$ | `propers.k` (masked at linear centres), periodicities and phases of the head |
| `dihedral opls` | $\tfrac12 K_1(1+\cos\phi) + \tfrac12 K_2(1-\cos2\phi) + \tfrac12 K_3(1+\cos3\phi) + \tfrac12 K_4(1-\cos4\phi)$ | $K_n = 2 k_n$ |
| `improper cvff` | $k [1 - \cos(n\psi)]$ on each of the three cyclic rows of a centre | `impropers.k` |
| `pair lj/cut` | $w\, 4\varepsilon_{ij}[(\sigma_{ij}/r)^{12} - (\sigma_{ij}/r)^6]$, the field's mixing and cutoff | `atoms.epsilon`, `atoms.sigma` |
| `pair coul/cut` | $w\, C\, q_i q_j / (\epsilon_r r)$, the field's `coulomb` and `dielectric` | `atoms.charge` |

$w$ is the field's `special_bonds` weight of `pairs.relation`; each pair counts
once (`source < target`); dead rows are multiplied by their `mask`. The three
improper rows per centre make the improper energy symmetric under any
permutation of the outer legs (the even series maps each cyclic dihedral to
the negative of another, A15).

### 3.7 Derivatives, the three primitives and the compile seam

`forward = Derivative(energy, wrt)`, built in `__init__` like every molzoo
potential; the training step marks its leaves before the traced body. A joint
training step composes the primitives itself: it predicts `tables` **once per
source batch**, and every objective on that source (`JointObjective.source`;
the four Hessian objectives share the hessian batch, ByteFF's "every loss of a
dataset on the one batch drawn from it") detaches the tables its loss does
not name (`JointObjective.trainable`, views of the one prediction) and prices
`energy(frame, tables)` under its own `Derivative` (the tables travel as the
op's keyword argument), calls `partial_hessian(frame, tables)`, or — for a
loss that supervises a predicted table itself, the MBIS charges — writes the
kept tables onto the batch and prices no energy (`output="tables"`). The
model knows nothing about objectives. Tables must be predicted **inside** the
region the caller traces (A5); one traced region per source (2026-10-12: the
per-objective regions cost the S1 step five encoder passes on one hessian
batch, and the large tier could not hold a step at the tier capacity). The
evaluation step detaches the loss and every column it wrote once the loss is
computed — the validation batches are resident, and a graph left on them
stayed alive through training (GH200 3732774 / 3735480, 90 GiB).

`energy(frame, tables, styles=("bond:harmonic", "angle:harmonic"))` sums the
named styles only — ByteFF's `calc_terms` (A18). Three objectives use it: the
bond and angle energy on a QM minimum, per term, trained through `r0` / `θ0`
(ByteFF `Bond_Energy` / `Angle_Energy`, `loss.py:302-314`; molix
`MeanTermEnergyLoss`); the improper energy on the same minimum trained through
the improper `k` (ByteFF `ImproperTorsion_Energy`, same lines, A19 — the cvff
minimum is planarity, so at a pyramidal centre the stiffness the Hessian wants
is also a force the minimum does not have, and this is what turns it off); and
the pair styles' force alone, written aside by the step
(`JointObjective(nonbonded=)`) so the stationary-force loss can leave out a
clashing molecule (ByteFF `skip_large_nb`, `loss.py:373-380`). The styles are a
fixed tuple of names, so the sum is straight-line code and traces as one graph.

`tests/test_molzoo/test_rondo/test_compile.py` asserts `graph_count == 1` and
`graph_break_count == 0` for the forward, the energy and the partial Hessian.

## 4. Configuration Contract

`RondoSpec(encoder=EncoderSpec(...), heads=HeadsSpec(...))`; the functional
form is the force field.

| Field | Meaning | Default |
|-------|---------|---------|
| `encoder.width` | Feature width of every channel and hop MLP | `64` |
| `encoder.n_blocks` | Covalent hops $L$ | `4` |
| `encoder.local_attention` | Softmax over incident bonds | `False` |
| `encoder.num_elements` | Atomic-number vocabulary | `119` |
| `encoder.atom_channels` | `formal_charge` (5, offset 2), `total_charge` (graph, 11, offset 5), `ring_connectivity` (5), `min_ring_size` (9) | ByteFF's |
| `encoder.bond_channels` | `bond_type` (5), `in_ring` (2) | ByteFF's |
| `heads.hidden` | Head MLP hidden width | `64` |
| `heads.bond_k_ref`, `heads.bond_r0_gain` | §3.4 | `450`, `0.1` |
| `heads.angle_k_ref`, `heads.angle_theta0_gain_deg` | §3.4 | `40`, `20` |
| `heads.proper_periodicity` | $n$ of the proper series | `(1,2,3,4,5,6)` (`(1,2,3,4)` for `dihedral opls`) |
| `heads.improper_periodicity` | $n$ of the improper | `2` |
| `heads.improper_k_ref` | §3.4, per row | `4.291 / 3` |
| `heads.lj_sigma_gain`, `heads.lj_epsilon_gain` | §3.4 | `0.1`, `1.0` |
| `heads.eem_max_atoms`, `heads.eem_jitter` | §3.5 | `512`, `1e-4` |

`Rondo(ff, spec, wrt=("atoms.pos",), dtype=torch.float64)`: `wrt=()` makes `forward` the energy
alone. The optimizer is `Adam(model.parameters())`: a head with no loss has a
zero gradient and Adam leaves it where it starts.

## 5. Reference Crosswalk

| Concept | Espaloma | ByteFF | MolNex anchor | Status |
|---------|----------|--------|---------------|--------|
| Backbone | SAGEConv stack | Edge-augmented Graph Transformer, equivalence-class averaging | `molrep.chem.blocks.ChemGraphBlock` | adapted |
| Input features | element | element, total charge, formal charge, ring connectivity, smallest ring; bond order, in-ring | `EncoderSpec.atom_channels` / `bond_channels` | matched (ByteFF) |
| Reads coordinates? | no | no | never | matched |
| Valence symmetry | Janossy pooling | equivalence averaging | `molrep.chem.context.*` | matched |
| Term bond features | — | the term's own bonds interleaved with its atoms (`OutputWithBond`) | `molrep.chem.BondTable` lookup + `context.*` (A21) | matched (ByteFF) |
| Bond / angle form | harmonic | harmonic (½K) | `bond:harmonic`, `angle:harmonic` (LAMMPS $K$) | matched |
| Proper torsion | Fourier | signed $k_n$, phases 0/π by parity | `PeriodicHead` → `dihedral:periodic` | matched (ByteFF) |
| Improper | periodic | $k[1+\cos(2\psi-\pi)]$, $k/3$ over 3 permutations | `CvffHead` → `improper:cvff`, three rows per centre | matched (ByteFF) |
| Linear-centre propers | — | zeroed by SMARTS | `Rondo._proper_mask` from bond orders | matched |
| Parameter reference | — | GAFF2 mean/std affine | per-row reference + zero start (§3.4) | adapted |
| Parameter bounds | — | `tanh_output` + `param_range` straight-through clamp | log-space window around the reference, reached by `tanh` saturation (`ExpAround(lo=, hi=)`, A22 + A24) | matched (ByteFF's `tanh`; no straight-through stage — ours binds, theirs does not) |
| Charges | charge-equilibration read-out | AM1-BCC regression | EEM KKT solve, MBIS supervision | departed (D1) |
| Lennard-Jones | OpenFF template | GAFF2 labels | learned around GAFF2 (`LJHead`) | adapted (D2) |
| Training | QM E/F | joint per-step sum over datasets, per-loss trainable sets, 3 stages | `molix.core.steps.JointTrainStep` + `JointObjective.trainable` | matched |
| Stage-1 geometry target | — | MM force → 0 at the QM minimum | `molix.core.losses.StationaryForceLoss` | matched |
| Force computation | autograd | autograd | `molix.autograd.Derivative` | matched |

## 6. MolNex Adaptations

| ID | Adaptation | Reason | Risk | Validation |
|----|------------|--------|------|------------|
| A5 | Tables are predicted inside the traced region | an external `grad_fn` breaks the force's `autograd.grad` under compile | low | `test_compile.py` |
| A6 | The force lands at `atoms.forces_pred`, the op's output key; the energy stays at `graphs.energy` and the training step moves it beside its label (`molix.frame.beside_labels`) | a training batch holds the DFT labels at `atoms.forces` / `graphs.energy`; the model moves nothing | high if regressed | `test_model.py::TestRondo::test_forces_are_the_negative_gradient_of_the_energy`, `tests/test_molix/test_frame.py::TestBesideLabels` |
| A7 | No automorphism orbit averaging | message passing is automorphism-equivariant | high if the backbone changes (§10.4) | gate |
| A11 | Pre-norm hops, every channel layer-normalized | feature RMS grew to 1e3–1e4 without it (job 3213256) | low | `test_encoder.py` |
| A12 | Heads predict deviations from a reference, zero start | one lr serves every parameter | low | `test_classical.py` |
| A13 | Torsions as signed $k_n$ at fixed phases (not $|a|$) | gradient at the zero start; GAFF2/OPLS form | low | `test_classical.py::test_phases_alternate…`, `test_model.py::test_a_mirror_image_has_the_same_energy` |
| A14 | $\theta_0$ reference from hybridization, linear map, straight-through clamp (the clamp superseded by A24's saturating window) | sigmoid around 109.47° held sp carbon at 146° vs 177.5° (jobs 3321932–3322130) | low | `test_classical.py::TestAngleHead` |
| A15 | Three cyclic improper rows per centre | energy must be leg-symmetric like the features | low | `test_model.py::test_improper_energy_is_symmetric_in_the_outer_legs` |
| A16 | Linear-centre propers zeroed from bond orders | undefined dihedral | low | `test_model.py::test_propers_through_a_linear_centre_are_silent` |
| A17 | LJ around GAFF2 with a prior, not frozen | operator ruling 2026-10-11: trainable, GAFF2 start | medium | `test_model.py::test_lennard_jones_starts_on_gaff2` |
| A18 | `energy(styles=)` prices a named subset of styles | ByteFF `calc_terms`: the equilibrium-strain objective and `skip_large_nb` need the bonded, resp. pair, energy alone. The force at a minimum cannot set `θ0`'s common mode at a symmetric centre — sp3 `θ0` walked 109° → 95° on it (GH200 3680577, 3685025/3685026) | low | `test_model.py::test_energy_of_named_styles_sums_only_those`, `test_compile.py::test_the_energy_of_named_styles_is_one_graph` |
| A19 | The improper `k` is also trained by the improper strain on the QM minimum (training-side: `OBJECTIVES["hessian_eq_improper"]`, no model code) | ByteFF `ImproperTorsion_Energy`. With the Hessian objective alone the improper's planarising force at a pyramidal N (|F| p50 116 kcal/mol/Å) had no parameter the force objective could reach, and every `θ0` at the centre walked together to cancel it (GH200 3690962: N3 θ0 129° vs 118°, 46 % of amine centres below 90°) | low | `.jobs/n3_force_decomp.py` on 3690962 best.pt; gate run 3705406 passes (with A20) |
| A20 | The improper `k` is positive by construction, $k_\text{ref}\,e^{\mathrm{raw}}$ (was a signed amplitude from a zero start) | the strain $k(1-\cos 2\psi)$ is linear in $k$: with a signed $k$ the A19 objective is unbounded below and ran 0 → −4.4 kcal/mol/term in 1400 steps while every N3 `θ0` went to 163° to cancel the force (GH200 3703994). ByteFF's `k` is $4.29 + 3.71 \cdot 15 \tanh(\cdot)$ inside a configurable `param_range` clamp (`mlparams.py:104-110`); no training config is in the repository, so the floor it trains with is not known. A negative `k` would also mean a minimum at ψ = 90° | low | `test_classical.py::TestCvffHead::test_stays_positive_for_a_negative_raw`; gate run 3705406 (improper \|k\| p10 0.003 / p50 4.26 / p90 7.9: amines go to zero, planar centres keep their amplitude) |
| A21 | Every valence context reads the features of the term's own bonds, interleaved with its atoms (§3.3); the bond rows come from a per-batch `BondTable` (sort + `searchsorted`, static shapes) rather than a loader column | ByteFF `OutputWithBond` (`layers.py:486-510`; the slots are data, `mol_data.py:119-122`). The contexts had read atom features only, so a torsion head had to decode its central bond order from a two-hop atom summary; on a vertex-transitive graph it cannot: in a Kekulé ring the torsion about a single bond and the one about a double bond got bit-identical features while the bond features differed by 2.6 (probe 2026-10-12). The central bond order is the main determinant of a torsion barrier, so the gap would have shown in S2. Lookup on the device instead of a loader column: no collate / padder / prepared-cache change, no new batch contract, same cost as one sort of the bond list | low | `test_encoder.py::TestChemEncoderTermBonds::test_a_torsion_knows_its_central_bond_order` (cyclobutadiene: identical atom features, distinct torsion features); `test_context.py` reversal / leg-swap tests with the bonds swapped along; Rondo and joint-body compile tests still 1 graph / 0 breaks |
| A22 | Every multiplicative parameter is held to a window around its reference with a straight-through gradient (as first landed; A24 made the window saturate): $K$, $\varepsilon$ on $[\tfrac18, 8]$, improper $k$ on $[\tfrac1{64}, 8]$, $r_0$, $\sigma$ on $[0.7, 1.4]$ (§3.4) | ByteFF bounds every parameter (`tanh_output`, `param_range`); `ExpAround` was open above, so one outlier Hessian row could carry a $K$ to 1e4–1e5 — `PartialHessianLoss(blowup=)` drops the row but the head was free to go there. The improper floor is lower because A19 legitimately sends a pyramidal centre's amplitude to zero (3705406: p10 0.003) | low | `test_classical.py::TestExpAround::test_the_bounds_hold_the_value_but_pass_the_gradient`, `TestBounds` |
| A23 | The angle `θ0` ceiling is 180° inclusive and the linear-centre reference is 180° (were 179.5° / 177.5°) | $K(\theta-\theta_0)^2$ with $\theta_0 = 180°$ is $K\,\lvert\delta\rvert^2/r^2$, a smooth isotropic bend, and molpot's angle kernel is C^∞ at linearity (`geom.GEOM_EPS`); with any $\theta_0 < 180°$ the linear geometry is the top of a Mexican hat whose Cartesian curvature $2K(\theta-\theta_0)\,\partial^2\theta/\partial x^2$ grows as $1/\sin\theta$, so the partial-Hessian fit can only match the QM block by $K \to 0$. GH200 3707870: sp-carbon angle $K$ 47 (1 500 steps) → 14 (3690962, 19 600) → 1.1 (40 879) with every C2 `θ0` pinned at the 179.5° ceiling against measured 179.0° (p90 179.9°). ByteFF's `θ0` map reaches 180° (`113.9 + 148.5·tanh`) and its kernel is the same `atan2` form (`ff_kernels.py:87`) | low | `test_hybridization.py::test_the_linear_reference_is_exactly_180_degrees`, `test_classical.py::TestAngleHead::test_theta0_reaches_180_degrees_and_no_further`, `test_model.py::test_an_sp_centre_starts_at_a_linear_angle`; the C2 angle $K$ of the next long run is the evidence |
| A24 | Windows saturate (ByteFF's `tanh`: $w\tanh(x/w)$ per side, slope one at the reference) instead of a straight-through clamp (A22 as first landed) | GH200 3723816 (S1 small, 26 400 steps, best eval 24.09): ≥ 10 % of improper rows at the $1/64$ floor — where A19 legitimately sends a pyramidal centre — and the improper stream's LayerNorm gain drifted to an RMS of 5.9 (atom / bond / angle 1.06 / 1.06 / 1.20); the handoff gate refused the checkpoint. A straight-through clamp holds the value but passes the whole gradient, so a push that is self-limiting unclamped ($k\,\partial/\partial\mathrm{raw} \to 0$ as $k \to 0$; 3707870 reached $k$ = 6e-16 with the stream at 1.29) becomes perpetual at a floor. Cost: a unit of `raw` is the gain only at the reference (r0 +9.7 % instead of +10 % at one unit; θ0 19.3–19.9° of 20° on the 60–177° references) and a row deep in saturation returns slowly | low | `test_classical.py::TestExpAround::test_the_window_is_approached_by_saturation_and_the_gradient_fades`, `TestLinearAround::test_a_reference_on_the_bound_is_held_there`; the improper feature RMS of the next long run is the evidence |
| A25 | The torsion stage carries ByteFF's auxiliary loss — the L1 norm of the proper amplitudes (`molix.core.losses.NormPrior` on `propers.k`, a `"tables"` objective on the torsion-scan batch, weight 1.0) — and the stages run at ByteFF's rates (torsion 1e-4, off-equilibrium 2e-5; were 3e-4 and 1e-4) | Paper §2.4 / Table 1: "L1-norm regularization was applied to the force constants of proper torsions to restrain redundant degrees of freedom", `TorsionLossType.l1_norm`. GH200 3748659 (S2 large, no prior, 3e-4): a scan measures one dihedral and every other proper of the molecule turns a little as the constrained conformers relax; six free amplitudes per row fit those wiggles with constants nothing in the stage prices back (the hessian objectives detach `propers.k`) — Σn²\|k_n\| median 9 → 34 in 2 000 steps, the impropers driven to their floor, N3 θ0 off by 13°, eval 14.3 → 24.6 while the kneed train loss stayed flat | The weight is the paper's unstated knob: too high and the profiles under-fit (torsion loss stalls above its no-prior value), too low and the amplitudes grow again. The circular-std mask and the per-table gradient cap ByteFF's code also carries (`mask_by_circstd`, `max_config`) are not adopted until a run shows the L1 prior and the rate are not enough | §7.4: an S2 gate on A25 must hold Σn²\|k_n\| near its step-800 scale (p50 ≈ 9) and the improper \|k\| median above 1 while the torsion loss keeps falling; the S2 long run must not raise the S1 objectives' eval |
| A26 | The off-equilibrium objective (`relax`) trains the equilibrium geometry (`bonds.r0`, `angles.theta0`), the proper amplitudes and σ/ε only; the curvatures `bonds.k`, `angles.k`, `impropers.k` stay the partial Hessian's (were: every bonded table + σ/ε) | ByteFF's `energyforce_loss` names every bonded table trainable (loss.py `set_trainable_terms([Bond_k, Angle_k, ProperTorsion_k, ImproperTorsion_k, Bond_length, Angle_theta])`) under a per-table gradient cap (`max_config`) whose values the repository does not carry. GH200 3768159 (S3 large, 2e-5, 1 500 steps): the relax forces take the cheap direction — a softer K — and the Hessian objective loses the tug-of-war: bond K −30 … −70 % (stiff bonds most, C=O 731 → 230), angle K −20 … −35 %, hessian_k 0.428 → 0.469 monotone while hessian_geom / hessian_eq fall only because every force at the minimum scales with K | A harmonic bond's K is the curvature at the minimum; forces along a relaxation sit on the anharmonic side and a harmonic fit to them is always softer than the Hessian K, so the two objectives disagree by construction and the one with the larger gradient wins. Rondo resolves it by ownership (each table moved by the data that defines it), ByteFF by a cap whose value is not public. Revisit (restore K to the relax set under a per-table cap) if the A26 relax loss stalls far above its A25 value | §7.4: the A26 S3 gate must hold hessian_k at its S2 value (0.428) and the bond K table within ±10 % of the S2 long run's, with the relax loss still falling; an S3 run that later shows the relax loss stuck above ~2.5 reopens the question |
| A9 | Per-objective weights, no per-table gradient clamp | global clip + per-loss `trainable` cover it | low | revisit if one table dominates |

### 6.1 Open methodological risks

| ID | Risk | Detail |
|----|------|--------|
| R1 | Non-bonded parameters carry little signal in gas-phase data | σ/ε start on GAFF2 and move only in a stage that names them with a prior on `log(σ/σ_ref)`, `log(ε/ε_ref)`; condensed-phase data would close this |
| R2 | One objective driving every parameter leaves families degenerate | **Addressed** by per-loss `trainable` and the joint schedule: Hessian → `k`, equilibrium forces → `r0`/`θ0`, MBIS → charges, torsion scans → propers, off-equilibrium E/F → `r0`/`θ0`, propers and σ/ε, never the curvatures (A26) |

## 7. Validation Contract

### 7.1 Research Reproduction

Not claimed: the charge model, data and encoder differ from both predecessors.

### 7.2 Symmetry and Shape Tests

| Claim | Test path |
|-------|-----------|
| Bond / angle / proper / improper features invariant under their index symmetries | `tests/test_molrep/test_chem/test_context.py` |
| Encoder reads no coordinate | `test_encoder.py::test_same_topology_different_pos_embeddings_identical` |
| A declared channel changes the features; a missing one is refused | `test_encoder.py::TestChemEncoderChannels` |
| $\sum_i q_i = Q$ | `test_model.py::test_charges_meet_the_declared_total_charge` |
| $\mathbf{F} = -\partial E / \partial \mathbf{r}$ | `test_model.py::test_forces_are_the_negative_gradient_of_the_energy` |
| Mirror image has the same energy | `test_model.py::test_a_mirror_image_has_the_same_energy` |
| Rigid motion invariance | `test_model.py::test_energy_is_invariant_under_a_rigid_motion` |
| Improper energy symmetric in the outer legs | `test_model.py::test_improper_energy_is_symmetric_in_the_outer_legs` |
| Linear-centre propers priced at zero | `test_model.py::test_propers_through_a_linear_centre_are_silent` |
| Partial Hessian equals the full Hessian on the bonded pairs | `test_model.py::test_partial_hessian_is_the_full_hessian_on_the_bonded_pairs` |
| `forward` writes the force at `atoms.forces_pred` and the energy at `graphs.energy`; `-dE/dpos` against `energy` | `test_model.py::TestRondo::test_forces_are_the_negative_gradient_of_the_energy` |
| Unsupported style refused by name | `test_model.py::test_an_unsupported_style_is_refused_by_name` |
| `energy(styles=)` sums the named styles only; an unknown name is refused | `test_model.py::test_energy_of_named_styles_sums_only_those`, `test_model.py::test_a_style_selection_outside_the_field_is_refused_by_name` |
| forward / energy / energy(styles=) / partial_hessian compile at 1 graph, 0 breaks | `test_compile.py` (`slow`) |

### 7.3 Engineering Benchmark

Training runs live in the molab workspace (`lab-new/projects/class-i/train/`),
not in this repo. Measure on a compute node.

### 7.4 Run Log

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
| Functional form | molpy `ForceField` | styles, settings, special bonds — declared once |
| Style kernels | `molpot.potentials` | molrs conventions, molpy oracle to 1e-10 |
| Covalent-graph perception | `molrep.chem` | topology and integer channels only |
| Parameter tables | `Rondo.tables` | namespace → parameter → tensor, with references |
| Per-loss parameter sets | `molix.core.steps.JointObjective` | detach between `tables` and `energy` |
| Derivatives | `molix.autograd.Derivative` | the single force op |
| Input columns | MolHub | `formal_charge`, ring features, `bond_type`, `in_ring`, `gaff2_sigma/epsilon`, three improper rows |

## 9. Version Pinning

| Item | Value |
|------|-------|
| Reference commits | `choderalab/espaloma@413eb55…`, `bytedance/byteff@90ef54a…` |
| molrs / molpy | 0.17 (`ff.def_style`, `special_bonds`, `to_section`) |
| GAFF2 reference | AmberTools25 `dat/leap/parm/gaff2.dat` (GAFF 2.2.30), MOD4 RE section; σ = 2R*/2^{1/6} |
| Dependencies | `molrep.chem`, `molpot.heads.classical`, `molpot.charge`, `molpot.potentials`, `molpot.forcefield.partial_hessian`, `molix.autograd` |
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

A backbone with structural or positional encodings voids A7 and may not land
without a test asserting equal parameters for chemically equivalent atoms.

## Appendix A. Maintenance Log

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
  training step owns them. Heads moved to `molpot.heads.classical` with per-row
  references (covalent radii, hybridization angle, GAFF2 σ/ε); torsions as
  signed $k_n$ at fixed phases; three improper rows per centre; linear-centre
  mask; ByteFF input channels on the encoder. **Every earlier checkpoint is
  invalid**, deliberately.
- 2026-10-06: Drift repair against the 2026-09-29 restructure.
- 2026-09-17: Created; filled from `scripts/mm_param_learning/`.
