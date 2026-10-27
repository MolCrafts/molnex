# Allegro User Guide

`molzoo.allegro` is the MolNex port of the mainline Allegro energy model
(`nequip-allegro` 0.8.3 on `nequip` 0.19.1). It has three public names:

| Name | What it is |
|---|---|
| `AllegroSpec` | torch-free, validated configuration; field names are upstream's |
| `AllegroEncoder(spec)` | the strictly local feature body: writes per-pair features |
| `AllegroPotential(spec, wrt=...)` | the body plus upstream's energy heads; `AllegroPotential.load(...)` reads an official `.nequip.zip` |

## Sources of Truth

- Paper: [Musaelian et al., "Learning local equivariant representations for large-scale atomistic dynamics", Nat. Commun. 14, 579 (2023)](https://arxiv.org/abs/2204.05249)
- Official docs: [Allegro Model](https://nequip.readthedocs.io/projects/allegro/en/latest/guide/allegro_model.html)
- MolNex spec: `MolZoo -> Spec -> Allegro Spec` (`src/molzoo/specs/allegro.md`)
- MolNex code: `src/molzoo/allegro/` (`spec.py`, `encoder.py`, `layers.py`,
  `potential.py`, `utils.py`, `checkpoint.py`)

The paper explains the architecture. The spec is the authority for the exact
mathematics, key table and deliberate deviations; where this page and the spec
disagree, the spec wins.

## 1. What Allegro Is Solving

An interatomic potential is a scalar energy
$E(\mathbf{r}_1, \ldots, \mathbf{r}_N)$ whose negative gradient gives the
forces, $\mathbf{F}_i = -\nabla_i E$. It must be accurate, differentiable,
symmetry-aware and scalable. Atom-centred message passing grows the receptive
field with every layer; Allegro keeps it fixed at the cutoff `r_max`:

```text
Allegro updates directed pair representations, never atom states.
```

A directed pair `(i, j)` has centre (source) `i`, neighbour (target) `j` and
vector $\mathbf{r}_{ij} = \mathbf{r}_j - \mathbf{r}_i$. In MolNex:

```text
pair_index[:, 0] = i          # source / centre
pair_index[:, 1] = j          # target / neighbour
pair_diff        = pos[j] - pos[i]   (+ lattice shift when periodic)
```

Because no atom state is passed between layers, the energy of atom `i` depends
only on atoms within `r_max` of `i` — the property that makes Allegro
spatially parallel.

## 2. Energy Decomposition

The total energy is a sum of per-atom energies, each a scaled and shifted sum of
ordered pair energies:

$$
E = \sum_i E_i, \qquad
E_i = s_{t_i}\,\frac{1}{\sqrt{2}\sqrt{\langle N\rangle}}\sum_{j\in\mathcal{N}(i)} e_{ij} + b_{t_i}.
$$

In general $e_{ij} \ne e_{ji}$: $e_{ij}$ depends on the environment of the
centre `i`. The full (bidirectional) neighbour list is therefore required.

| Concept | MolNex owner |
|---|---|
| directed neighbour list | `molix.neighbor.NeighborList(symmetry=True)` |
| per-pair features | `AllegroEncoder` |
| pair readout, edgewise reduce, scale/shift, ZBL | `AllegroPotential` |
| forces / virial | `molix.autograd.Derivative` (built by `AllegroPotential(wrt=...)`) |
| losses, training | `molix` |

## 3. Equivariance, Irreps, and Tensor Products

The energy is invariant under rotations, translations and permutations;
internal features may be *equivariant*, $f(D_X(g)x) = D_Y(g)f(x)$. Allegro
organises its tensor features by O(3) irreps `(l, p)`: scalars are `0e`,
vectors `1o`, and so on. The core operation is the Clebsch–Gordan product

$$
(\mathbf{x}\otimes\mathbf{y})_{l_3 m_3} =
\sum_{m_1 m_2} C^{l_3 m_3}_{l_1 m_1, l_2 m_2}\, x_{l_1 m_1}\, y_{l_2 m_2},
\qquad |l_1-l_2| \le l_3 \le l_1+l_2,\; p_3 = p_1 p_2 .
$$

Raising `l_max` multiplies tensor-product *paths*, not just feature length,
which is why it is the most expensive knob. MolNex runs every product through
cuEquivariance; its `O3` Clebsch–Gordan coefficients equal upstream's
`wigner_3j · √(2 l_out + 1)` exactly, and tensor features use the `ir_mul`
layout `(E, D·u)`.

## 4. Notation and Configuration

| Symbol | Meaning | `AllegroSpec` field (default) |
|---|---|---|
| `T` | atom types; type `t` is `atomic_numbers[t]` | `atomic_numbers` (required) |
| `r_max` | strict local cutoff, Å | `r_max` (required) |
| `N_b`, `p` | Bessel basis size, envelope exponent | `num_bessels` (8), `polynomial_cutoff_p` (6) |
| `D` | two-body type-embedding width | `radial_chemical_embed_dim` (`None` → `F`; must be even) |
| `F` | scalar track width | `num_scalar_features` (64) |
| `u` | tensor channels per irrep | `num_tensor_features` (16) |
| `L` | Allegro layers | `num_layers` (2) |
| `l_max` | maximum angular order | `l_max` (2), `parity` (True) |
| `⟨N⟩` | average directed neighbour count | `avg_num_neighbors` (`None` → no normalisation) |
| `s_t`, `b_t` | per-type energy scale / shift, eV | `per_type_energy_scales` / `_shifts` (`None` → 1 / 0) |

MLP shapes use upstream names: `scalar_embed_mlp_hidden_layers_{depth,width}`,
`allegro_mlp_hidden_layers_{depth,width}`,
`readout_mlp_hidden_layers_{depth,width}` and the matching `*_nonlinearity`.
`zbl=True` adds the ZBL repulsion. The one MolNex-only field is
`expose_tensor_track`. The tensor products run on molnex's own kernels
(`molrep.equivariance`) on CUDA and on their PyTorch reference on CPU.

The pair-feature width is `F (L + 1)` (`encoder.output_dim`): the initial
scalar slice plus one per layer.

## 5. The Forward Pass

Per directed pair `(i → j)`, $r = \lVert\mathbf{r}_{ij}\rVert$, $x = r/r_{max}$,
types $t_i, t_j$ (spec §5):

1. **Radial basis** (`radial_features`, `SincBesselRBF × PolynomialCutoff`):
   $b_n = \operatorname{sinc}(x w_n)\, w_n \cdot u_p(x)$, $n = 1..N_b$, with
   $w_n$ initialised to $n$ and $\operatorname{sinc}(z) = \sin(\pi z)/(\pi z)$.
   $u_p$ is the polynomial envelope, zero at and beyond `r_max`.
2. **Two-body type embedding** (`ProductTypeEmbedding`):
   $h^0 = [E_c(t_i) \,\|\, E_n(t_j)] \odot (W_b\, b)$, width `D`. Centre and
   neighbour have separate tables; the product is elementwise, not a
   concatenation with the basis.
3. **Two-body MLP** (`scalar_embed_mlp`, `ScalarMLPFunction`):
   $h = \mathrm{MLP}_s(h^0)$, width `F` (SiLU, bias-free, nequip
   forward-normalised init). Steps 2–3 are `twobody_features`.
4. **Initial tensor track** (`angular_features`, `WeightedChannels`):
   $Y = Y(\hat{\mathbf{r}}_{ij})$ up to `l_max` (component normalisation);
   $V_0 = \mathrm{WeightedChannels}(Y, L_{env}(h))$ — `u` channels per irrep,
   weights laid out `[u, R]` (channel-major).
5. **First projection**: $[s_0 \,\|\, w_0] = L_{proj}(h)$ — `F` scalars and
   `u·R` environment weights.
6. **Layers** $\ell = 0..L-1$ (`layer_features`):
   - environment, summed into the **centre** with $1/\sqrt{\langle N\rangle}$:
     $\Omega_i = \frac{1}{\sqrt{\langle N\rangle}} \sum_k
     \mathrm{WeightedChannels}(Y_{ik}, w_{\ell,ik})$ — one cuEquivariance
     segmented polynomial with indexed output, so no `(E, D·u)` message is
     built;
   - tensor product (`Contracter`): $V_{\ell+1} = \mathrm{Contracter}_\ell(V_\ell,
     \Omega_{i})$, with path-channel weights $W_\ell \in \mathbb{R}^{u\times P_\ell}$
     and Clebsch–Gordan paths enumerated `(out, in1, in2)`; $\Omega_i$ is
     gathered by the pair's source index inside the kernel;
   - DenseNet latent update:
     $[s_{\ell+1} \,\|\, w_{\ell+1}] = \mathrm{MLP}_\ell([s_0 \ldots s_\ell \,\|\, V_{\ell+1}^{0e}])$
     (the last layer emits no `w`).

   Tensor irreps follow upstream pruning: forward from the allowed set
   (`l ≤ l_max`, both parities when `parity`), last layer scalars only, then
   backward pruning (the encoder's private `_layer_irreps` schedule).
7. **Energy heads** (`AllegroPotential.energy`):
   $e_{ij} = \mathrm{MLP}_r([s_0 \ldots s_L])$ (`edge_readout`), then
   EdgewiseReduce onto the centre and the per-type scale/shift **in float64**:
   $E_i = s_{t_i}\big(\sum_j e_{ij} / \sqrt{\langle N\rangle} / \sqrt{2}\big) + b_{t_i}$,
   plus, with `zbl=True`, the LAMMPS `pair_zbl` repulsion
   $\sum_j \tfrac12 k_e Z_i Z_j / r \cdot \phi(Z, r) \cdot u_p(x)$ (`UniversalZBL`).
8. **Graph sum**: $E_g = \sum_{i\in g} m_i E_i$, `m` the real-atom mask
   (`atoms.mask`, when present).

The encoder's `forward` stops after step 6 and writes `pairs.edge_features`
`(E, F·(L+1))`; with `expose_tensor_track=True` it also writes
`pairs.edge_tensor_features` `(E, D_out·u)` in `ir_mul` layout.

## 6. Hands-On: A Tiny Batch

Models read a post-collate nested `TensorDict`. A water molecule, built by hand:

```python
import torch
from tensordict import TensorDict

pos = torch.tensor([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2390, 0.9270, 0.0]])
Z = torch.tensor([8, 1, 1])
pair_index = torch.tensor([[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]])  # (E, 2), both directions
pair_diff = pos[pair_index[:, 1]] - pos[pair_index[:, 0]]


def water() -> TensorDict:
    return TensorDict(
        atoms=TensorDict(Z=Z, pos=pos, batch=torch.zeros(3, dtype=torch.long), batch_size=[3]),
        pairs=TensorDict(pair_index=pair_index, pair_diff=pair_diff, batch_size=[6]),
        graphs=TensorDict({}, batch_size=[1]),
        batch_size=[],
    )
```

In production the pairs come from `molix.neighbor.NeighborList` (full list,
`symmetry=True` is the default), run as a batch task on the collated batch:

```python
from molix.neighbor import NeighborList

batch = NeighborList(cutoff=4.0).execute(water())
print(batch["pairs", "pair_index"].shape)  # torch.Size([6, 2])
```

## 7. Feature Body: `AllegroEncoder`

Use the encoder when a head reads pair features from the batch (for example
`molpot.heads.PermMultipoleHead` or `molpot.composition.Sonata`).

```python
from molzoo.allegro import AllegroEncoder, AllegroSpec

spec = AllegroSpec(
    atomic_numbers=[1, 8],
    r_max=4.0,
    num_layers=2,
    l_max=2,
    num_scalar_features=16,
    num_tensor_features=4,
    avg_num_neighbors=2.0,
)
encoder = AllegroEncoder(spec, dtype=torch.float64)
batch = encoder(water())
print(batch["pairs", "edge_features"].shape, encoder.output_dim)  # torch.Size([6, 48]) 48
print([tp.num_paths for tp in encoder.tps])                       # [11, 3]
```

`forward(td)` reads the batch and calls `edge_features(pos, Z, pair_index,
edge_diff=...)`, which composes the public primitives, each one named step of
§5 (`AllegroPotential.energy` shares that composition). Call them directly to
inspect an intermediate:

```python
from molrep.embedding.geometry import edge_lengths, edge_vectors

td = water()
encoder.validate_elements(td["atoms", "Z"])  # eager check, raises on unknown Z
types = encoder.node_types(td["atoms", "Z"])
vectors = edge_vectors(td["atoms", "pos"], td["pairs", "pair_index"])
lengths = edge_lengths(vectors)
radial = encoder.radial_features(lengths)                          # (E, num_bessels)
twobody = encoder.twobody_features(types, td["pairs", "pair_index"], radial)  # (E, F)
sh = encoder.angular_features(vectors, lengths)                    # (E, (l_max + 1)^2)
features, tensor = encoder.layer_features(
    twobody=twobody, sh=sh, edge_index=td["pairs", "pair_index"], num_atoms=3
)
print(features.shape, tensor.shape)  # torch.Size([6, 48]) torch.Size([6, 4])
```

The type table follows `atomic_numbers` order (upstream `type_names`), not
sorted `Z`. An atomic number outside the table raises once, on the first eager
call; it is never re-checked inside a compiled region.

`expose_tensor_track=True` keeps the last layer's tensor irreps instead of
collapsing to scalars and writes them for equivariant heads:

```python
tensor_encoder = AllegroEncoder(
    spec.model_copy(update={"expose_tensor_track": True}), dtype=torch.float64
)
batch = tensor_encoder(water())
print(tensor_encoder.tensor_track_irreps)            # 4x0e+4x0o+4x1e+4x1o+4x2e+4x2o
print(batch["pairs", "edge_tensor_features"].shape)  # torch.Size([6, 72])
```

Such an encoder has weights no official package carries, so it is an encoder
option only; `AllegroPotential` raises on a spec with `expose_tensor_track=True`.

## 8. Energy Model: `AllegroPotential`

`AllegroPotential` is a `molix.autograd.Potential` over the encoder: it
subclasses the body and registers upstream's `edge_readout`,
`per_type_energy_scale_shift` and optional `pair_potential` (ZBL) at the
module root, so its `state_dict` is flat. It takes the spec, `wrt` and the build `dtype`:

```python
from molzoo.allegro import AllegroPotential, AllegroSpec

model = AllegroPotential(
    AllegroSpec(
        atomic_numbers=[1, 8],
        r_max=4.0,
        num_scalar_features=16,
        num_tensor_features=4,
        avg_num_neighbors=2.0,
        per_type_energy_shifts=[-13.6, -2041.0],
    ),
    wrt=("atoms.pos",),
    dtype=torch.float64,
)
out = model(water())
print(out["graphs", "energy"].dtype, out["graphs", "energy"].shape)  # torch.float64 torch.Size([1])
print(out["atoms", "energy"].shape, out["atoms", "forces"].shape)    # torch.Size([3]) torch.Size([3, 3])
```

Two public methods, the potential contract:

| Method | Reads / writes |
|---|---|
| `energy(frame)` | reads `atoms.pos`, `atoms.Z`, `pairs.pair_index` and, when present, `pairs.pair_diff` / `atoms.mask`; writes `atoms.energy` `(N,)` and `graphs.energy` `(B,)` in float64 eV; detaches nothing |
| `forward(frame)` | `Derivative(self.energy, wrt=wrt)`, fixed at construction |

`B` is `frame["graphs"].batch_size[0]`, so the `graphs` namespace is required
(collate, the MD bind path and the deploy `build_batch` all write it).

`wrt=("atoms.pos",)` writes `atoms.forces` $= -\partial E/\partial\mathbf{r}$;
adding `"graphs.strain"` (one graph with `graphs.cell`) writes `graphs.virial`
$= +\partial E/\partial\varepsilon$ `(1, 3, 3)`. **nequip's virial is
$-\partial E/\partial\varepsilon$**, so `graphs.virial` is minus upstream's.

```python
cell = torch.eye(3, dtype=pos.dtype) * 10.0
periodic = water()
periodic["graphs", "cell"] = cell.view(1, 3, 3)
stress_model = AllegroPotential(
    AllegroSpec(atomic_numbers=[1, 8], r_max=4.0, num_scalar_features=16, num_tensor_features=4),
    wrt=("atoms.pos", "graphs.strain"),
    dtype=torch.float64,
)
print(stress_model(periodic)["graphs", "virial"].shape)  # torch.Size([1, 3, 3])
```

`forward` and `Derivative(AllegroPotential(spec, wrt=()).energy, wrt=...)` are the same
code path. The model writes `graphs.energy` / `atoms.forces` in place, so keep
reference labels in other columns (`graphs.energy_ref`, `atoms.forces_ref`).

**Compile.** `forward` traces to one graph with no break under
`torch.no_grad()` after one eager call (which spends the element check).
Compile with `molix.compile.Compiler`.

## 9. Loading Official Packages

`AllegroPotential.load` builds the model an official `.nequip.zip` describes
and loads its weights. Nothing upstream is imported — no `nequip`, `allegro`
or their dependencies: the package's pickled module tree is read through the
shared `molzoo.checkpoint.StubUnpickler` (torch / stdlib / numpy /
cuEquivariance resolved, every upstream class stubbed), and the
hyper-parameters are read off it.

```python
from molzoo.allegro import AllegroPotential

model = AllegroPotential.load(
    "Allegro-OAM-L-0.1.nequip.zip", wrt=("atoms.pos",), dtype=torch.float64
)
```

For example `Allegro-OAM-L:0.1` (Zenodo record
[16980200](https://zenodo.org/records/16980200); 89 elements, `r_max = 7 Å`,
4 layers, `l_max = 3`). Notes:

- The architecture is the package's — `NequipPackage.read` fills a
  `NequipConfig` from the module tree and `AllegroSpec.from_package` builds
  the spec from it; `load` takes no spec field.
- Build precision is the `dtype` argument: an fp32 package loaded with
  `dtype=torch.float64` evaluates its fp32 weights in fp64.
- molnex leaves PyTorch's fp32 matmul policy at its default (`"highest"`,
  no TF32). Do not lower it for Allegro-OAM-L: TF32 matmuls put it off by
  ~4e-3 eV/atom and ~0.1 eV/Å on a periodic cell (GH200, 2026-10-15).
- Official parity against upstream fp64 goldens (an open molecule and a
  triclinic cell with self-images, spec §7.3): the small test packages agree
  to ~1e-13 (gate 1e-8 eV/atom, 1e-7 eV/Å); Allegro-OAM-L:0.1, an fp32
  package evaluated in fp64, agrees to fp32 rounding (~5e-7 eV/atom,
  ~4e-5 eV/Å).
- The key table is `molzoo.allegro.checkpoint.allegro_keys`. The reader
  normalises upstream's layout once (per-type `(T, 1)` → `(T,)`, single-path
  `(u,)` → `(u, 1)`, the `w3j` constants dropped); the Clebsch–Gordan tables
  and `1/r_max` are rebuilt, not loaded.

**Unsupported upstream options** — `load` raises, naming the option:
`per_edge_type_cutoff`; per-type `avg_num_neighbors`;
`forward_normalize=False`; deep-linear or biased MLPs;
`tp_path_channel_coupling=False`; `weight_individual_irreps=False`; a
non-integer cutoff exponent; a radial embedding other than
`TwoBodyBesselScalarEmbed`; a pair potential other than ZBL (in LAMMPS
`metal` units).

## 10. Hyperparameter Guidance

| Field | Role | Starting point |
|---|---|---|
| `r_max` | strict local cutoff | chemistry prior; more edges as it grows |
| `avg_num_neighbors` | environment and edge-sum normalisation | compute once from the training set at the same cutoff, full list; never per batch |
| `l_max` | angular resolution | `2`; `1`–`3`, cost grows quickly |
| `num_layers` | tensor-product depth | `2`; `1`–`3` |
| `num_scalar_features` | scalar latent width `F` | `64`–`256`; sets `output_dim = F(L+1)` |
| `num_tensor_features` | tensor channels `u` | `16`–`64` |
| `per_type_energy_shifts` / `_scales` | per-type reference energy / spread | from a linear fit of energies on composition |

## 11. Spec Crosswalk

| Question | Spec | Answer |
|---|---|---|
| What maps to what upstream? | §2 | `SincBesselRBF`, `ProductTypeEmbedding`, `ScalarMLPFunction`, `WeightedChannels`, `Contracter`, `PerSpeciesScaleShift`, `UniversalZBL`, `Derivative` |
| Edge and type conventions | §3.2 | upstream `edge_index[0]` is `pair_index[:, 0]`; types index `atomic_numbers` |
| Deviations and refused options | §4 | element-table check, tensor-track exposure, layout adapters, float64 heads |
| Exact equations | §5 | §5 of this page |
| Field names | §6 | `AllegroSpec` |
| Validation | §7 | invariance, finite-difference forces, padded masking, compile, official parity |

## 12. Common Failure Modes

- **Wrong edge direction**: `pair_diff` must be `pos[target] - pos[source]`.
- **Half list**: `NeighborList(symmetry=False)` drops one direction; Allegro
  needs both, since `e_ij ≠ e_ji`.
- **`(2, E)` pair index**: `pair_index` is `(E, 2)` everywhere.
- **Labels overwritten**: the model writes `graphs.energy` / `atoms.forces`;
  a loss against those same columns is identically zero.
- **Batch-dependent normalisation**: `avg_num_neighbors` is a dataset
  constant, part of the model; an official package carries its own.
- **Virial sign**: compare `graphs.virial` with `-virial` from nequip.
