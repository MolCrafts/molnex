# Rondo Specification

This page is the implementation contract for `molzoo.rondo`. It is not a
tutorial; use the MolZoo user guide for theory narrative and worked examples.

| Field | Value |
|-------|-------|
| Module | `molzoo.rondo` |
| Entry point | `Rondo` (config `RondoSpec`) |
| Paper | Not a reproduction. Lineage: Wang *et al.*, "End-to-end differentiable construction of molecular mechanics force fields", *Chem. Sci.* **13**, 12016 (2022) — Espaloma. Contrast: Zheng *et al.*, "Data-driven parametrization of molecular mechanics force fields for expansive chemical space coverage", *Chem. Sci.* (2025) — ByteFF. |
| arXiv | Lineage [2010.01196](https://arxiv.org/abs/2010.01196) (Espaloma); contrast [2408.12817](https://arxiv.org/abs/2408.12817) (ByteFF) |
| DOI | [10.1039/D2SC02739A](https://doi.org/10.1039/D2SC02739A) (Espaloma) |
| Reference implementation | `choderalab/espaloma@413eb55074037540e1414eb365da7304bd5d72be`, `bytedance/byteff@90ef54a7c38e58f162adc23f3e8f457aaf922d46` |
| Spec status | stable |

## 1. Scope

Rondo is MolNex's **learnable Class-I force field**: a graph neural network
that reads a molecule's covalent topology and emits every parameter of a
general Class-I functional form, which `molpot` then evaluates as an ordinary
classical potential.

It is a *recipe*, in the MolZoo sense — the named composition of primitives
that already live in `molrep` and `molpot`. It owns the choice of encoder
width and depth, which parameter heads are mounted, and which styles the
template force field declares. It owns no kernel of its own.

### 1.1 What Rondo owns

| Concern | Rondo surface |
|---------|---------------|
| The classical functional form being fitted | `RondoForceField(ClassIForceField)` — harmonic bond and angle, general periodic proper and improper, Lennard-Jones 12-6, Coulomb; 1-4 scaling and the open-Coulomb radius are declared here |
| The chemical-perception backbone and its width | `Rondo` → `molrep.chem.ChemEncoder` |
| Which parameter heads are mounted, and their capacity | `Rondo` → `molpot.Parameterizer` with the six `molpot` heads |
| Energy and forces as one prediction mapping | `RondoPotential` — the host-side bind / traced-derivative split |
| The configuration contract | `RondoSpec` |

### 1.2 What Rondo does not own

| Concern | Owner |
|---------|-------|
| Class-I energy kernels, exclusions, pair styles | `molpot.forcefield` |
| The parameter heads themselves | `molpot.composition.mm_heads`, `molpot.composition.heads`, `molpot.charge` |
| Graph blocks, valence context pooling, embeddings | `molrep.chem` |
| The derivative op (`E → F`, `E → virial`) | `molix.autograd.Derivative` |
| Losses, padding, collation, batch tasks | `molix` |
| Datasets, splits, units, provenance | MolHub |
| Training configuration, scheduling, calibration evidence | MolExp / MolAb |

### 1.3 Lineage and the three deliberate departures

Rondo sits in the Espaloma → ByteFF family: a GNN over the covalent graph
predicts MM parameters, and the MM energy is differentiated for forces. The
current implementation descends directly from the Espaloma architecture table
(SAGEConv message passing → Janossy-pooled valence features → parameter
heads). Three choices depart from **both** predecessors and are the reason
Rondo is a distinct model rather than a port:

| # | Rondo | Espaloma / ByteFF | Consequence |
|---|-------|-------------------|-------------|
| D1 | Partial charges from **electronegativity equalisation** (`molpot.charge`: `ChiEtaHead` → `ElectronegativityEqualization`) | Espaloma: charge-equilibration read-out on graph features. ByteFF: AM1-BCC-style additive bond-charge corrections | Rondo's charges respond to the solved electrostatic problem rather than being a fixed function of the graph alone; costs one dense per-molecule solve |
| D2 | Lennard-Jones `sigma` / `epsilon` are **learned** (`LJParameterHead`) | ByteFF **inherits GAFF-2.2 non-bonded parameters unchanged** for Amber compatibility | Full parameterisation with no vendor anchor — and no restoring force on the non-bonded parameters from intramolecular data alone (see risk R1, §6) |
| D3 | Proper torsions are trained in **Cartesian** amplitudes `(a_n, b_n)`, mapped to `(k, gamma)` only at export | Both train `(k, phi0)` directly | `(k, gamma)` has a coordinate singularity at `k = 0`, which an honest fit passes through; `(a, b)` is smooth there |

### 1.4 Status of this document

`stable`. Every section describes the contract as implemented, and §5 carries
no `unknown` row. §7.1 stays `not claimed` **by construction** — Rondo is not
a reproduction, so no published metric is a valid target for it (§7.1) — which
is a filled section, not a missing one.

## 2. Public Contract

### 2.1 Required Inputs

| Direction | TensorDict path | Shape | Dtype | Contract |
|-----------|------------------|-------|-------|----------|
| In | `atoms.Z` | `(N,)` | int64 | Atomic numbers; ghost rows carry `0` |
| In | `atoms.pos` | `(N, 3)` | `config.ftype` | Positions, Å. Differentiated against; never read by the encoder |
| In | `atoms.batch` | `(N,)` | int64 | Graph membership |
| In | `bonds.atomi` / `bonds.atomj` | `(N_bonds,)` | int64 | Covalent topology (column form) |
| In | `angles.atomi` / `atomj` / `atomk` | `(N_angles,)` | int64 | Valence angles; `atomj` is the vertex |
| In | `propers.atomi` … `atoml` | `(N_propers,)` | int64 | Proper torsions |
| In | `impropers.atomi` … `atoml` | `(N_impropers,)` | int64 | Impropers; **centre is `atomi`** |
| In | `pairs.pair_index` | `(E, 2)` | int64 | Cutoff pair list at the template's own radius |
| In | `graphs.total_charge` | `(B,)` | `config.ftype` | Required by the charge model; no neutral fallback |
| In | `graphs.num_atoms` | `(B,)` | int64 | Graph sizes |
| In | `atoms.mask` / `graphs.mask` | `(N,)` / `(B,)` | bool | Ghost-row and dead-slot masks under static shapes |

### 2.2 Outputs

| Direction | TensorDict path | Shape | Written when | Contract |
|-----------|------------------|-------|--------------|----------|
| Out | `atoms.chem_features` | `(N, D_a)` | always | Continuous chemical perception, written by the encoder |
| Out | `bonds` / `angles` / `propers` / `impropers` `.chem_features` | `(N_x, D_x)` | always | Per-interaction features |

`RondoPotential` returns its predictions as a mapping rather than writing them
onto the batch — the batch's `graphs.energy` and `atoms.forces` are label
columns and are explicitly restored (§3.7, adaptation A6):

| Key of the returned mapping | Shape | Present when | Contract |
|-----------------------------|-------|--------------|----------|
| `energy` | `(B,)` | always | Class-I energy plus dressing, in `energy_unit` |
| `energy_mm` | `(B,)` | always | Undressed Class-I energy — what a residualised loss fits |
| `forces` | `(N, 3)` | `"atoms.pos" in wrt` | `-dE/dpos` |
| `virial` | `(B, 3, 3)` | `"graphs.strain" in wrt` | Inference path only |
| `e0` | `(B,)` | `dressing is not None` | Frozen per-element reference sum, float64 |

**Charges are a parameter, not a batch column.** The solved partial charges
(`e`) are returned as the `charge` column of the `atoms` *parameter table* —
`Rondo.tables(batch)["atoms"]["charge"]` on the hot path, and a filled
`ForceField` from `Rondo(batch)` on the export path. They are not written back
onto `batch["atoms"]`. `Rondo.forward` returns the filled field; the encoder's
`*.chem_features` are the only thing it writes onto the batch.

## 3. Forward Contract

### 3.1 Notation

| Symbol | Meaning | Code anchor |
|--------|---------|-------------|
| $N$ | Atom rows in the packed batch, ghosts included | `atoms.Z` |
| $B$ | Graph slots, dead slots included | `graphs.num_atoms` |
| $E$ | Cutoff pair rows (capacity, not live count) | `pairs.pair_index` |
| $h^{(t)}_a$ | Atom features after $t$ covalent hops, $(N, D_a)$ | `ChemEncoder`, `h_atom` |
| $h^{(t)}_{ij}$ | Undirected bond features, $(N_\text{bonds}, D_b)$ | `ChemEncoder`, `h_bond` |
| $h_{ijk}$, $h_{ijkl}$, $h^{\text{imp}}_{ijkl}$ | Angle / proper / improper features | `AngleContext`, `ProperContext`, `ImproperContext` |
| $\chi_i, \eta_i$ | Electronegativity and hardness of atom $i$ | `ChiEtaHead` |
| $q_i$ | Partial charge, $e$ | `atoms.charge` |
| $T$ | Number of Fourier terms per proper torsion | `RondoSpec.n_torsion_terms` |
| $\phi$ | Torsion dihedral angle, rad | `molpot` torsion kernel |

Energies are kcal mol⁻¹, lengths Å, angles rad, charges $e$ (LAMMPS
`units real`) throughout the container; `RondoSpec.energy_unit` rescales only
the value that leaves `RondoPotential`.

### 3.2 Chemical perception

The encoder reads **topology only** — no coordinate enters any of the
following equations. This is what makes the predicted parameters
geometry-independent, and therefore makes the deployed force exactly
$-\partial E/\partial \mathbf{r}$ rather than a partial derivative that drops
a parameter-response term.

Atom seed, then the bond seed, then $L$ covalent hops:

$$
h^{(0)}_a = \mathrm{Embed}(Z_a), \qquad
h^{(0)}_{ij} = \mathrm{BondEmbed}\!\left(h^{(0)}_i, h^{(0)}_j, \tau_{ij}\right)
$$

$$
h^{(t+1)}_{ij} = \mathrm{BondCtx}\!\left(h^{(t)}_i, h^{(t)}_j, \tau_{ij}, h^{(t)}_{ij}\right), \qquad
h^{(t+1)}_a = h^{(t)}_a + \mathrm{MLP}\!\left(\sum_{b \,\ni\, a} h^{(t+1)}_{ab}\right)
$$

with $\tau_{ij}$ the optional bond type and the sum over bonds incident on
$a$ (each undirected bond scatters onto **both** endpoints). Under
`local_attention=True` the plain sum becomes a per-atom softmax over incident
bonds, sharing one undirected score per bond — the normalisation is over an
atom's own covalent neighbourhood, never over all $E$.

| Quantity | Shape | Code anchor |
|----------|-------|-------------|
| $h^{(0)}_a$ | $(N, D_a)$ | `molrep.chem.embed.AtomChemEmbedding` |
| $h^{(t)}_{ij}$ | $(N_\text{bonds}, D_b)$ | `molrep.chem.context.BondContext` |
| one hop | — | `molrep.chem.blocks.ChemGraphBlock` |

### 3.3 Valence contexts — permutation symmetry is structural

Each valence feature is mean-pooled over the symmetry orbit of its index
tuple, so the emitted parameter cannot depend on which way the interaction
was enumerated. This is the Janossy-pooling construction of Espaloma §5 and
it is a **property of the architecture**, not a training outcome:

$$
h_{ijk} = \tfrac{1}{2}\left[\mathrm{MLP}(h_i \Vert h_j \Vert h_k) + \mathrm{MLP}(h_k \Vert h_j \Vert h_i)\right]
$$

$$
h_{ijkl} = \tfrac{1}{2}\left[\mathrm{MLP}(h_i \Vert h_j \Vert h_k \Vert h_l) + \mathrm{MLP}(h_l \Vert h_k \Vert h_j \Vert h_i)\right]
$$

$$
h^{\text{imp}}_{ijkl} = \mathrm{MLP}\!\left(h_i \;\Vert\; h_j + h_k + h_l\right)
$$

The improper form is symmetric under **any** permutation of the three outer
legs with the centre $i$ fixed — the batch-schema convention is centre-first
(`impropers.atomi` is the centre).

| Claim | Consequence |
|-------|-------------|
| Angle features invariant under $(i,j,k) \to (k,j,i)$ | One angle, one parameter, regardless of enumeration order |
| Proper features invariant under $(i,j,k,l) \to (l,k,j,i)$ | Same, for torsions |
| Improper features invariant under $S_3$ on outer legs | Same, for impropers |
| Bond features invariant under $(i,j) \to (j,i)$ | Same, for bonds |

Graph-automorphism equivalence (chemically equivalent atoms receiving equal
embeddings) is **not** enforced separately: message passing over the covalent
graph is equivariant to graph automorphism by construction, so equivalent
atoms already receive equal features. A backbone that adds structural or
positional encodings breaks that guarantee and would require the explicit
orbit-averaging that ByteFF performs — see adaptation **A7**.

### 3.4 Parameter heads

Every head is a two-layer MLP from its interaction's feature to that
interaction's parameter row. Heads never see a coordinate.

| Interaction | Head | Emits | Units |
|-------------|------|-------|-------|
| bond | `BondParamHead` | $k$, $r_0$ | kcal mol⁻¹ Å⁻², Å |
| angle | `AngleParamHead` | $k$, $\theta_0$ | kcal mol⁻¹ rad⁻², rad |
| proper | `ProperTorsionParamHead` | $a_n$, $b_n$ for $n = 1 \dots T$ | kcal mol⁻¹ |
| improper | `ImproperParamHead` | periodic amplitudes | kcal mol⁻¹ |
| atom (LJ) | `LJParameterHead` | $\sigma$, $\varepsilon$ | Å, kcal mol⁻¹ |
| atom (charge) | `ChiEtaHead` | $\chi$, $\eta$ | kcal mol⁻¹ $e^{-1}$, kcal mol⁻¹ $e^{-2}$ |

The proper head emits **Cartesian** amplitudes. The canonical polar table is
recovered only at IR / `to_potentials` / export time:

$$
k_n = \sqrt{a_n^2 + b_n^2}, \qquad \gamma_n = \operatorname{atan2}(b_n, a_n)
$$

A barrier that passes through zero is ordinary during a fit, and at $k_n = 0$
the phase $\gamma_n$ is undefined — $(k, \gamma)$ has a coordinate
singularity there and no gradient, while $(a, b)$ is smooth through the
origin. The polar pair is the storage form; the Cartesian pair is the
trainable one. There is no OPLS output form: OPLS-AA is this series at
$n = 1,2,3,4$ with $\gamma = 0, \pi, 0, \pi$, a manifold the exporters detect.

### 3.5 Charge equilibration

Charges are not read off the graph. Per molecule, Rondo solves the
electronegativity-equalisation KKT system

$$
\begin{bmatrix} \mathbf{A} & \mathbf{1} \\ \mathbf{1}^{\mathsf T} & 0 \end{bmatrix}
\begin{bmatrix} \mathbf{q} \\ \lambda \end{bmatrix}
=
\begin{bmatrix} -\boldsymbol{\chi} \\ Q \end{bmatrix}
$$

with $\mathbf{A}$ assembled by a `HardnessKernel` (`DiagonalHardnessKernel`
by default; `ShieldedCoulombKernel` is the Ohno-screened alternative), $Q$
the molecule's total charge from `graphs.total_charge`, and $\lambda$ the
Lagrange multiplier enforcing $\sum_i q_i = Q$. Charge conservation is
therefore a **constraint of the solve**, not a post-processing correction.

The solve is batched onto a fixed $(B, n_\text{max})$ dense layout with
$n_\text{max} =$ `RondoSpec.eem_max_atoms`. Sizing it from the batch instead
(`counts.max().item()`) is a host sync and a data-dependent shape, which is a
graph break on a hot path. A molecule over capacity fails closed twice:
eagerly it raises naming the capacity; inside a compiled region every charge
of the batch becomes `NaN` — never a truncated, finite, wrong charge.

### 3.6 Class-I energy

`molpot` assembles the filled field into a `Potentials` container and
evaluates

$$
E = \underbrace{\sum_{\text{bonds}} \tfrac{1}{2} k_{ij}\,(r_{ij} - r^0_{ij})^2
  + \sum_{\text{angles}} \tfrac{1}{2} k_{ijk}\,(\theta_{ijk} - \theta^0_{ijk})^2}_{\text{harmonic valence}}
$$

$$
{}+ \underbrace{\sum_{\text{propers}} \sum_{n=1}^{T} \left[a_n \cos (n\phi) + b_n \sin (n\phi)\right]
  + \sum_{\text{impropers}} k_\psi \left[1 + \cos(n\psi - \gamma)\right]}_{\text{torsional}}
$$

$$
{}+ \underbrace{\sum_{i<j} s^{\text{LJ}}_{ij}\, 4\varepsilon_{ij}\left[\left(\frac{\sigma_{ij}}{r_{ij}}\right)^{12} - \left(\frac{\sigma_{ij}}{r_{ij}}\right)^{6}\right]
  + \sum_{i<j} s^{q}_{ij}\, \frac{q_i q_j}{4\pi\varepsilon_0 r_{ij}}}_{\text{non-bonded, over the cutoff pair list}}
$$

where $s^{\text{LJ}}$ and $s^{q}$ are the 1-2 / 1-3 / 1-4 exclusion scales
declared on `RondoForceField`, and $\sigma_{ij}$, $\varepsilon_{ij}$ follow
the declared mixing rule. The pair sum runs over the list built by
`RondoForceField.neighbor_list()` — **the field owns the radius it prices**,
so a list built at a different radius than the model charges for is
impossible by construction rather than by convention.

### 3.7 Derivatives and the compile seam

`RondoPotential` splits one step into a host-side half and a traced half:

| Half | Content | Traceable |
|------|---------|-----------|
| host | Fill the field, assemble `Potentials`, bind this batch's topology (exclusion table, edge-orientation check, net-charge report) | no |
| traced | `molix.autograd.Derivative` over the container's energy | yes |

The container is assembled **once per run** and only `bind_topology` runs per
batch. Running both halves in one compiled region costs 22 graph breaks on a
real batch; none of them is physics. Split, the same op is 1 graph / 0 breaks
in inference and one break in training at the autograd leaf — measured
2026-09-18 on GH200 (job 2616579), `Compiler.explain(potential.energy, batch)`
reporting `graph_count=1, graph_break_count=0`.

**What a global config read costs here.** The split alone is not sufficient.
Until 2026-09-18 the same measurement reported **13 graphs / 12 breaks**, all
from one line: `molrep.chem.encoder._empty` built its `(0, D)` stand-in for an
absent namespace with `dtype=config.ftype`. `molcfg.Config` resolves every
attribute through a custom `__getattribute__`, which dynamo cannot trace
(`TypeError: expected 1 argument, got 2`), so each read broke the graph and
dynamo then *skipped the enclosing function* rather than breaking gracefully.
The fix is that the dtype was never the global's to state: an empty tensor
standing in for features derived from the atom embedding takes that
embedding's dtype, and `like.new_zeros((0, D))` already carries it. A butane
batch has no impropers, so the path was live on the smallest real input.
The rule this instance is a case of is CLAUDE.md's "pydantic / config
attribute reads" under forbidden hot-path break sources.

The parameter heads run **inside** `energy()`, i.e. inside whatever region
the caller compiles. Predicting the tables outside and handing them in does
not work: a table carrying a `grad_fn` from outside makes
`torch.autograd.grad` untraceable.

$$
\mathbf{F} = -\frac{\partial E}{\partial \mathbf{r}}, \qquad
\mathbf{W} = \frac{\partial E}{\partial \boldsymbol{\epsilon}}
$$

Forces are the training path (`wrt=("atoms.pos",)`). The virial adds
`"graphs.strain"` and is an **inference path only**: one strain leaf is one
lattice, so it requires a single-graph batch carrying `graphs.cell` and a
placed list with `pairs.shifts`, which a packed training batch is not.

## 4. Configuration Contract

| `RondoSpec` field | Meaning | Default | Constraint / note |
|-------------------|---------|---------|-------------------|
| `hidden` | Shared feature width for atom / bond / angle / proper / improper and the head MLPs | `64` | Power of two (calibration policy) |
| `n_blocks` | Covalent hops $L$ | `4` | $\ge 0$; receptive field is $L$ bonds |
| `n_torsion_terms` | Fourier terms $T$ per proper | `8` | $\ge 1$; real capacity, not an ignored argument |
| `local_attention` | Per-atom softmax over incident bonds instead of a sum | `False` | |
| `num_elements` | Atomic-number vocabulary | inherited | |
| `num_bond_types` | Bond-type table size; `0` disables type conditioning | `0` | |
| `eem_max_atoms` | Fixed per-molecule capacity of the dense EEM solve | `512` | Must cover the largest molecule in the train set; `None` re-enables the host sync and is not a compiled-path option |
| `cutoff` | Open-Coulomb / LJ pair radius, Å | `10.0` | Declared on the field; `neighbor_list()` refuses to invent one |
| `scale_14_charge`, `scale_14_lj` | 1-4 exclusion scales | `0.5`, `0.5` | 1-2 and 1-3 are `0.0`; part of the parameterisation being fitted, so the caller states them |
| `mixing` | LJ combination rule | `"geometric"` | |

`RondoPotential` takes the remaining knobs, because they describe what leaves
the potential rather than what the parameterizer predicts — one fact, one
home:

| `RondoPotential` argument | Meaning | Default | Constraint / note |
|---------------------------|---------|---------|-------------------|
| `dressing` | Frozen per-element reference energies added after the unit conversion | `None` | A kcal mol⁻¹ table; pairing it with a non-kcal `energy_unit` raises |
| `energy_unit` | Unit of the reported energy | kcal mol⁻¹ | Only rescales the output; the container is always kcal mol⁻¹ |
| `wrt` | What the derivative op differentiates against | `("atoms.pos",)` | Adding `"graphs.strain"` is inference-only (§3.7) |
| `compiler` | Applied to the traced half alone | `None` | Eager otherwise |

## 5. Reference Crosswalk

Rondo is not a port, so this table maps **concepts**, not lines. `departed`
marks a deliberate difference, argued in §1.3 and §6.

| Concept | Espaloma | ByteFF | MolNex anchor | Status |
|---------|----------|--------|---------------|--------|
| Backbone | SAGEConv stack over the covalent graph | Edge-augmented Graph Transformer (EGT) | `molrep.chem.blocks.ChemGraphBlock` | adapted |
| Reads coordinates? | no | no | `ChemEncoder` — coordinates never read | matched |
| Valence symmetry | Janossy pooling over the index orbit | Averaging over chemically equivalent atoms and bonds | `molrep.chem.context.*` | matched (Espaloma); see A7 |
| Bond / angle form | harmonic | harmonic | `bond:harmonic`, `angle:harmonic` | matched |
| Proper torsion form | Fourier series | Fourier series, $(k, \phi_0)$ | `dihedral:periodic`, trained as $(a_n, b_n)$ | departed (D3) |
| Improper form | periodic | $k_\psi[1 + \cos(2\psi - \pi)]$ | `improper:periodic`, amplitude and multiplicity free | adapted |
| Stage-1 target | — | 3×3 mixed blocks of the QM Hessian on the atom pairs a bonded term couples, relative to $\lvert\mathrm{tr}\rvert$ | `PartialHessianPairs` + `Potentials.partial_hessian` + `PartialHessianLoss` | matched |
| Stage-2 target | relative conformational energies, molecule-centered | each conformer against its own profile's **minimum**, Boltzmann-weighted, optimal per-profile offset | `BoltzmannProfileLoss` over `ProfileIndex` | matched (ByteFF) |
| Stage-3 target | QM energies and forces | forces (`Force_MSE`, driven to zero on QM minima) | the same profile-relative energy in its saturating form (`boltzmann_soft_mse`) **plus** the inlier force Huber against QM forces | adapted (A10) |
| Charge supervision | — | none — AM1-BCC is fixed input | MBIS partial charges (`themol/mbis@1`) via `AtomicChargeLoss`, fitted before the bonded stages | departed (D1 consequence) |
| Partial charges | charge-equilibration read-out on graph features | AM1-BCC-style additive bond-charge corrections | `molpot.charge`: `ChiEtaHead` → `ElectronegativityEqualization` KKT | departed (D1) |
| Total-charge conservation | read-out construction | output-module construction | Lagrange multiplier in the KKT solve | adapted |
| Lennard-Jones | from the OpenFF template | **inherited unchanged from GAFF-2.2** | `LJParameterHead`, learned | departed (D2) |
| Training signal | QM energies and forces | staged: partial-Hessian → torsion profiles → off-equilibrium $E$/$F$ | the same three stages, `scripts.mm_param_learning.configs.CURRICULUM` | matched (see R2) |
| Non-bonded training data | condensed-phase and QM | explicitly out of scope | gas-phase single molecules (OMol25) | departed (see R1) |
| Force computation | autograd on the MM energy | autograd on the MM energy | `molix.autograd.Derivative` | matched |

## 6. MolNex Adaptations

| ID | Adaptation | Reason | Risk | Validation |
|----|------------|--------|------|------------|
| A1 | Proper torsions trained as Cartesian $(a_n, b_n)$, converted to $(k, \gamma)$ at export | $(k, \gamma)$ is singular at $k = 0$, which a fit passes through | low | `tests/test_molpot/test_composition/test_mm_heads.py` |
| A2 | No vendor dialect in the trained form; OPLS is a detected manifold at export | Emitting $k_1..k_4$ would build one vendor's constraint into the model's shape | low | exporter manifold detection, spec `ff-params-01` §5 |
| A3 | Charges from an EEM/QEq KKT solve rather than an additive graph correction | Charges respond to the solved electrostatic problem; the polarizable successor (ByteFF-Pol) is the same direction | medium — one dense solve per molecule | `tests/test_molpot/test_charge/` |
| A4 | Fixed EEM capacity `eem_max_atoms` instead of `counts.max()` | Data-dependent shape is a graph break on a hot path | low | compile test, 0 graph breaks |
| A5 | Container assembled once per run; only topology re-bound per batch | 22 graph breaks measured when both halves share a compiled region, none of them physics | low | compile test |
| A6 | Label columns saved and restored around the derivative op | `Derivative` writes in place onto `graphs.energy` / `atoms.forces`, which is also where the DFT labels live; without this the loss compares a prediction with itself and the training signal is **silently** zero | high if regressed | a test asserting `\|pred − ref\| \ne 0` on a labelled batch |
| A8 | An objective may restrict which parameter tables it trains (`RondoPotential(trainable=…)` for an energy fit, `ForceFieldPredictions(trainable=…)` for the partial-Hessian fit) | A single combined objective under-determines a Class-I parameter set (R2); ByteFF isolates per loss with `set_trainable_terms` | low | `tests/test_molzoo/test_rondo/test_potential.py::TestRondoPotentialTrainable` |
| A7 | Graph-automorphism orbit averaging is **not** implemented | Message passing is automorphism-equivariant by construction, so it is redundant for this backbone | **becomes high if the backbone changes** — an attention backbone with structural encodings loses the guarantee | a test asserting equal parameters for chemically equivalent atoms, required before any backbone swap |
| A10 | Stage 3 fits QM forces directly, not ByteFF's force-to-zero | ByteFF's `Force_MSE` drives the MM force to zero, which is an objective for *equilibrium* geometries: it says the QM minimum is also an MM stationary point, and it is what fits `Bond_length` / `Angle_theta`. `hessian-relax` frames are off-equilibrium, where the QM force is non-zero and published, so the informative residual is against that force rather than against zero | low — same quantity, stronger supervision | `force_loss(outlier_norm=…)`, and the stage-3 smoke asserting both halves contribute |
| A9 | ByteFF's per-table gradient clamp (`set_grad_max`) is **not** implemented; Rondo clips globally (`GradClipHook`) | The clamp rescales one predicted table's gradient to a max infinity-norm before it reaches the shared encoder, which global clipping cannot express — but ByteFF defaults it to `finfo().max`, i.e. off, for every trainable term, and wiring it would duplicate the detach rule across both the energy and the prediction path. Taken as a detail, not algorithm (operator ruling 2026-09-21: alignment is functional and algorithmic) | low — it is a safety valve, and the global clip covers the failure it guards | revisit if one table's gradient is observed to dominate the encoder update |

### 6.1 Open methodological risks

These are recorded because they are known, not because they are solved.
Naming them is required by the repository's no-silent-debt rule.

| ID | Risk | Detail |
|----|------|--------|
| R1 | **Learned non-bonded parameters carry no training signal** | LJ $\sigma$ / $\varepsilon$ and the long-range part of the Coulomb term are free parameters, but the training data is isolated gas-phase molecules — there are no intermolecular configurations, so the loss is nearly flat along those directions. ByteFF meets the same gap and answers it by freezing GAFF-2.2 (D2). Rondo currently leaves them free. The downstream symptoms are already visible in the loss design: $(\sigma/r)^{12}$ spikes needing a Huber, and force outliers above 200 kcal mol⁻¹ Å⁻¹ being dropped. Those are analgesics, not a fix. |
| R2 | **One objective driving every parameter leaves the families degenerate** | Bond, angle, torsion, LJ and charge parameters fitted against one combined $E + F$ objective leave the classic MM degeneracy in which a 1-4 LJ term and a torsion barrier compensate each other. **Addressed** by `RondoPotential(trainable=…)`: an objective keeps a gradient path only to the namespaces it constrains, every other predicted table being detached after prediction and before the energy — so the parameters still contribute their values and only their gradient is cut. This is ByteFF's `set_trainable_terms`, and it is not the same as excluding parameters from the optimizer: the encoder is shared, so detaching a table removes *that path's* contribution to the encoder gradient while leaving the others. The *schedule* is now ByteFF's own plus a charge stage it has no counterpart for, declared in `scripts.mm_param_learning.configs.CURRICULUM`: MBIS charges on `themol/mbis`, partial Hessian on `themol/hessian` training bonds / angles / impropers, torsion profiles on `themol/torsion-scan` training propers, then off-equilibrium $E$/$F$ on `themol/hessian-relax` training everything, each stage carrying the previous checkpoint. **No stage supervises an absolute energy**: every one declares `dressing: False`, the two energy stages being relative to a profile's own minimum, so the $\varepsilon[Z]$ table and the $\sim 2.3\times10^5$ kcal/mol per-molecule constant it left are out of the curriculum. What remains open is whether four stages suffice — not which objective drives which table. |

## 7. Validation Contract

### 7.1 Research Reproduction

Not claimed, and not claimable. Rondo is not a reproduction of Espaloma or
ByteFF: the charge model, the non-bonded treatment, the torsion
parameterisation and the training data all differ (§1.3, §5). No published
metric from either paper is a valid target, and quoting one would be a
category error rather than a stretch goal.

### 7.2 Symmetry and Shape Tests

| Claim | Test path | Tolerance |
|-------|-----------|-----------|
| Bond parameters invariant under $(i,j) \to (j,i)$ | `tests/test_molrep/test_chem/test_context.py` | exact (same tensor) |
| Angle parameters invariant under $(i,j,k) \to (k,j,i)$ | `tests/test_molrep/test_chem/test_context.py` | exact |
| Proper parameters invariant under $(i,j,k,l) \to (l,k,j,i)$ | `tests/test_molrep/test_chem/test_context.py` | exact |
| Improper parameters invariant under $S_3$ on outer legs | `tests/test_molrep/test_chem/test_context.py` | exact |
| Encoder reads no coordinate | `tests/test_molzoo/test_rondo/test_encoder.py` — static check | n/a |
| $\sum_i q_i = Q$ per molecule | `tests/test_molzoo/test_rondo/test_potential.py` | 1e-6 |
| Energy invariant under global translation and rotation | `tests/test_molzoo/test_rondo/test_potential.py` | dtype-appropriate |
| Ghost rows contribute zero energy | `tests/test_molzoo/test_rondo/test_potential.py` | exact |
| Derivative does not destroy label columns (A6) | `tests/test_molzoo/test_rondo/test_potential.py` | $\ne 0$ |
| Traced half compiles with 0 graph breaks, for `trainable=None` **and** a named subset | `tests/test_molzoo/test_rondo/test_compile.py` (`slow`) | `graph_count == 1` **and** `graph_break_count == 0` |

### 7.3 Engineering Benchmark

The driver and its job script live in the **molab workspace**, not in this
repo — a training run is an experiment, not library code:

```
/home/jicli594/work/lab-new/projects/rondo/
  train/fit_torsion_profile.py      # the driver
  train/run_fit_torsion.sbatch      # its submission
  experiments/torsion-profile/runs/ # one record per objective
```

One THEMol torsion scan (24 conformers of one molecule, 33 atoms, 7.014
kcal/mol barrier), 16 train / 8 val, `hidden=64 n_blocks=3 n_torsion_terms=4`.

**Measure on a compute node.** The login node of this cluster runs at load
~28 on 4 cores; wall-clock there reports contention, not code — the same step
measured 2700 ms there and 39 ms on an allocated node, a factor of 69.

**The numbers live in §7.4 and only there.** This section used to carry its own
copy of the two MAEs, and the copy drifted: it read 0.361 / 0.634 for the Huber
row against §7.4's 2.09 / 1.95, with no date, commit or step count on either to
reconcile them. The table was removed on 2026-09-18 rather than guessed at —
§7.4 rows carry their own provenance, so read the objective comparison there.

What the comparison shows is unchanged and is a fact about the *objective*, not
the model: per-group centering removes the constant offset the model would
otherwise have to learn, and the Parsley weight stops grid points the functional
form cannot reproduce from dominating the fit. No model code differs between the
rows.

Step time 35 ms (GH200, job 2616174); the `wrt=("atoms.pos",)` force graph
accounts for ~6 ms of it — 34 ms/step with it against 28 ms/step at `wrt=()` —
whether or not the objective reads the forces.

### 7.4 Run Log

| run_id | date | commit | dirty | dataset | config | steps | train_mae | val_mae | fwd_ms | bwd_ms | compiled | note |
|--------|------|--------|-------|---------|--------|-------|-----------|---------|--------|--------|----------|------|
| 1 | 2026-09-17 | uncommitted | 1 | themol/torsion-scan scan 0 (24 conformers, 33 atoms, 7.014 kcal/mol barrier), 16 train / 8 val | hidden=64 n_blocks=3 n_torsion_terms=4 eem_max_atoms=64 | 400 | 2.09 | 1.95 | — | — | no | Huber on the residual energy, unweighted. First light. |
| 2 | 2026-09-17 | uncommitted | 1 | as run 1 | as run 1 | 400 | **0.027** | **0.091** | 30 | 7 | no | `TorsionProfileLoss` + `ParsleyEnergyWeight`. Same model code as run 1: the 7x is entirely per-group centering plus the energy weight. Signal spread 1.57 kcal/mol. GH200 node (login node reports 69x this step time under load ~28 on 4 cores). |
| 3 | 2026-09-18 | uncommitted | 1 | as run 1 | as run 2 | 400 | 0.0271 | 0.0909 | 34 | — | no | Reproduction of run 2 from its new home, `lab-new/projects/rondo/train/fit_torsion_profile.py` (GH200 job 2616174, node n454): same numbers to the digit, residual std 1.5695, 35 ms/step. Not a new result — the check that relocating the driver into the molab workspace changed nothing. |

## 8. System Boundary

| Concern | Owner | Contract |
|---------|-------|----------|
| Covalent-graph perception | `molrep.chem` | Reads topology only, never a coordinate — so predicted parameters are geometry-independent and deployed forces are exactly `-dE/dpos` |
| Parameter tables | `molpot.Parameterizer` | Returns a filled `ForceField`; has no energy API |
| Class-I energy | `molpot.forcefield.Potentials` | Owns cutoffs; assembled once per run, topology bound per batch |
| Derivatives | `molix.autograd.Derivative` | The single force/virial op; no second path |
| Neighbour list radius | `RondoForceField.neighbor_list()` | The field owns the radius it prices; a list built from a second, equal-looking field is a number tracked by hand twice |
| Static shapes / padding | `molix.data.MoleculePadder` | Capacity-first; the molecule count per batch floats |
| Datasets, splits, units | MolHub | MolNex never hard-codes a dataset coordinate |
| Training configuration and evidence | MolExp / MolAb | |

## 9. Version Pinning

| Item | Value |
|------|-------|
| Paper | Not a reproduction; lineage and contrast cited in the header |
| Reference repository | `choderalab/espaloma` (lineage), `bytedance/byteff` (contrast) |
| Reference commit | `413eb55074037540e1414eb365da7304bd5d72be` (espaloma, 2026-02-10), `90ef54a7c38e58f162adc23f3e8f457aaf922d46` (byteff, 2025-02-26) |
| Related | `ByteDance-Seed/byteff2@d87249d360491b66ff7d84ac9be04837100328a7` — ByteFF-Pol, the polarizable successor; relevant to D1 |
| Dependencies | `molrep.chem`, `molpot.forcefield`, `molpot.composition`, `molpot.charge`, `molix.autograd`, `molix.data` |
| Public docs mirror | `docs/molzoo/specs/rondo.md` |

## 10. Drift Policy

### 10.1 Triggers

Re-run `/molzoo-spec rondo` (update mode) or the `molzoo-auditor` agent when:

- a parameter head is added, removed, or changes its emitted keys;
- the backbone changes (this invalidates **A7** — see §10.4);
- the template's declared styles, cutoff, or exclusion scales change;
- the charge model or its kernel changes;
- a §7.4 row shows an MAE regression above 10 % against the previous row;
- the compile seam moves (host half vs traced half).

### 10.2 Breaking changes

Adding, removing, or renaming a section of this document is a breaking
change. The ten-section structure is immutable.

### 10.3 Code-before-spec

Code authored against a `draft`-status spec is a process violation, not
merely drift. This spec is `partial`: §2 / §3 / §5 are filled, so encoder
code may be written against them.

### 10.4 Backbone-change gate

Adaptation **A7** records that graph-automorphism orbit averaging is omitted
because message passing provides it for free. Any backbone that adds
structural or positional encodings voids that argument. Such a change may not
land until a test asserting equal parameters for chemically equivalent atoms
exists and passes.

## Appendix A. Maintenance Log

- 2026-09-17: Filled §2 / §3 / §4 / §5 / §6 / §7.2 / §10 from the code under
  `scripts/mm_param_learning/` plus the Espaloma and ByteFF papers; status
  `draft` → `partial`. Recorded risks R1 (non-bonded parameters have no
  training signal) and R2 (single-stage fit leaves families degenerate), and
  gate §10.4 on adaptation A7.
- 2026-09-17: Created (create mode). Rondo names the learnable Class-I force
  field that had been living unnamed in `scripts/mm_param_learning/` as
  `espaloma_like_template()` / `build_espaloma_like_model()` /
  `ForceAwareModel`. Reference pins resolved for espaloma and byteff.
  Status `draft`; §3 / §4 / §5 / §6 await update mode.
