# Allegro User Guide

This page is the MolNex guide for `molzoo.Allegro`. It combines the theory,
the paper-to-code mapping, a runnable tutorial path, and the local spec
contract.

Allegro in MolNex is an encoder. It produces edge features. Energy readout,
force derivation, losses, and training live outside `molzoo`.

## Sources of Truth

Use these together:

- Paper: [Musaelian et al., "Learning local equivariant representations for large-scale atomistic dynamics"](https://www.nature.com/articles/s41467-023-36329-y)
- Official docs: [Allegro Model](https://nequip.readthedocs.io/projects/allegro/en/latest/guide/allegro_model.html)
- MolNex spec: `MolZoo -> Spec -> Allegro Spec`
- MolNex code: `src/molzoo/allegro.py`

The paper explains the architecture. The official docs explain public
hyperparameters. The MolNex spec is the authority for exact tensor paths,
shape contracts, and deliberate deviations from the reference implementation.

## 1. What Allegro Is Solving

Atomistic machine-learning models need a scalar potential energy

$$
E_\mathrm{system}(\mathbf{r}_1, \ldots, \mathbf{r}_N)
$$

whose negative coordinate gradient gives forces:

$$
\mathbf{F}_i = -\nabla_i E_\mathrm{system}.
$$

The model therefore needs to be accurate, differentiable, and symmetry-aware.
It also needs to scale to large systems. Ordinary atom-centered message passing
can be accurate, but every layer expands the effective receptive field. Allegro
keeps the receptive field fixed at the cutoff radius `r_max`.

The key design choice is:

```text
Allegro updates directed edge-pair representations, not atom states.
```

A directed edge `(i, j)` has source or center atom `i`, neighbor atom `j`, and
edge vector:

$$
\mathbf{r}_{ij} = \mathbf{r}_j - \mathbf{r}_i.
$$

MolNex stores this as:

```text
edge_index[:, 0] = i
edge_index[:, 1] = j
bond_diff        = pos[j] - pos[i]
bond_dist        = ||bond_diff||
```

## 2. Energy Decomposition

The paper starts from an atomwise energy decomposition:

$$
E_\mathrm{system}
= \sum_i \sigma_{Z_i} E_i + \mu_{Z_i}.
$$

Then each atom energy is decomposed into ordered pair contributions:

$$
E_i
= \sum_{j \in \mathcal{N}(i)} \sigma_{Z_i,Z_j} E_{ij}.
$$

Important consequence:

$$
E_{ij} \ne E_{ji}
$$

in general, because `E_ij` is allowed to depend on the local environment of
source atom `i`, while `E_ji` depends on the local environment of source atom
`j`.

MolNex keeps this structure, but splits responsibilities:

| Paper / reference model concept | MolNex owner |
|---|---|
| directed local environment | `molix.data.NeighborList(symmetry=True)` |
| edge representation | `molzoo.Allegro` |
| edge energy readout | `molpot.heads.EdgeEnergyHead` |
| force derivation and losses | downstream model / `molix` training path |

This split is recorded in the spec adaptation ledger.

## 3. Equivariance, Irreps, and Tensor Products

The potential energy is invariant under translations and rotations. Internal
features may be equivariant: if the input geometry rotates, those features
rotate by the corresponding representation.

A function is equivariant when:

$$
f(D_X(g)x) = D_Y(g)f(x)
$$

for every group action `g`. If `D_Y(g)` is the identity, the output is
invariant.

Allegro organizes geometric tensors by irreducible representations, indexed by
angular order `l` and parity `p`. Scalar features are `l = 0`. Directional
features use `l > 0`.

The core operation is the Clebsch-Gordan tensor product:

$$
(\mathbf{x} \otimes \mathbf{y})_{l_\mathrm{out},m_\mathrm{out}}
=
\sum_{m_1,m_2}
C^{l_\mathrm{out},m_\mathrm{out}}_{l_1,m_1,l_2,m_2}
\mathbf{x}_{l_1,m_1}
\mathbf{y}_{l_2,m_2}.
$$

Valid paths satisfy:

$$
|l_1 - l_2| \le l_\mathrm{out} \le l_1 + l_2,
\qquad
p_\mathrm{out} = p_1 p_2.
$$

This is why the official Allegro docs warn that increasing `l_max` is
expensive. It increases tensor-product paths, not just feature length.

## 4. Notation Used in MolNex

| Symbol | Meaning | MolNex parameter or tensor |
|---|---|---|
| `N` | number of atoms | `batch["atoms", "Z"].shape[0]` |
| `E` | number of directed edges | `batch["edges", "edge_index"].shape[0]` |
| `L` | Allegro layer count | `num_layers` |
| `F` | scalar track width | `num_scalar_features` |
| `U` | tensor channel multiplicity | `num_tensor_features` |
| `r_max` | strict local cutoff | `r_max` |
| `l_max` | maximum angular order | `l_max` |
| `B` | Bessel radial basis | `num_bessel` |
| `p` | polynomial cutoff exponent | `poly_p` |
| `⟨|N|⟩` | average directed neighbor count | `avg_num_neighbors` |

The output feature width is:

$$
\texttt{encoder.output\_dim} = F(L + 1).
$$

The `+1` is the initial two-body scalar slice.

## 5. Initial Two-Body Embedding

Each directed edge starts from distance and atom-type information.

### 5.1 Radial Basis and Cutoff

MolNex follows the spec's Bessel edge encoding:

$$
B_n(r_{ij})
=
\operatorname{sinc}\!\left(\frac{n r_{ij}}{r_\mathrm{max}}\right)n,
\qquad
n = 1,\ldots,N_\mathrm{bessel}.
$$

The smooth polynomial cutoff is:

$$
u(r) =
1
- \frac{(p+1)(p+2)}{2}\left(\frac{r}{r_\mathrm{max}}\right)^p
+ p(p+2)\left(\frac{r}{r_\mathrm{max}}\right)^{p+1}
- \frac{p(p+1)}{2}\left(\frac{r}{r_\mathrm{max}}\right)^{p+2}
$$

for `r <= r_max`, and zero outside.

MolNex multiplies the basis by the cutoff once:

$$
\tilde{B}_n(r_{ij}) = B_n(r_{ij}) u(r_{ij}).
$$

The paper writes a cutoff factor in the latent update equation as well. The
MolNex spec follows the reference implementation detail: the cutoff enters at
the radial basis, and the following linear/MLP layers are bias-free, so zero at
the cutoff remains zero.

### 5.2 Center-Neighbor Type Embedding

Allegro is ordered-pair based, so it embeds source and target atom types
separately:

$$
\mathbf{t}_{ij}
=
\left[
e_\mathrm{center}(Z_i);
e_\mathrm{neighbor}(Z_j)
\right].
$$

The radial projection and type embedding are multiplied elementwise:

$$
\mathbf{s}_{ij}
=
\mathbf{t}_{ij}
\odot
W_\mathrm{basis}\tilde{\mathbf{B}}(r_{ij}).
$$

This is the `ProductTypeEmbedding` behavior called out in the spec. It is a
Hadamard product, not a concatenation.

In code:

```python
type_embed = torch.cat(
    [self.center_embed(Z[src]), self.neighbor_embed(Z[dst])],
    dim=-1,
)
twobody_scalar_embed = type_embed * self.basis_linear(edge_radial)
```

## 6. Initial Tensor Track

The paper initializes equivariant features by weighting spherical harmonics:

$$
\mathbf{V}^{ij,0}_{n,l,p}
=
w^{ij,0}_{n,l,p}
\mathbf{Y}^{ij}_{l,p}.
$$

The weights are predicted from the initial scalar edge embedding:

$$
w^{ij,0}_{n,l,p}
=
\operatorname{MLP}^{0}_\mathrm{embed}
(\mathbf{x}^{ij,0})_{n,l,p}.
$$

MolNex implements this as:

```python
tensor_basis = self.spherical_harmonics(bond_diff)
v0_weights = self.env_embed_linear(twobody_scalar_embed)
tensor_features = _make_weighted_channels(
    tensor_basis,
    v0_weights,
    mul=self.num_tensor_features,
    sh_irrep_dims=self._sh_irrep_dims,
)
```

The helper `_make_weighted_channels` is the spec's port of Allegro's
`MakeWeightedChannels(weight_individual_irreps=True)`.

## 7. Allegro Layer

Each layer combines the current edge tensor track with a learned embedding of
the source atom's local environment.

### 7.1 Learned Embedded Environment

For source atom `i`, the environment is a learned weighted sum over directed
edges `(i, k)`:

$$
\mathbf{A}^{i,L}_{n,l,p}
=
\sum_{k \in \mathcal{N}(i)}
w^{ik,L}_{n,l,p}
\mathbf{Y}^{ik}_{l,p}.
$$

The paper's environment weights are functions of scalar latent edge features:

$$
w^{ik,L}_{n,l,p}
=
\operatorname{MLP}^{L}_\mathrm{embed}
(\mathbf{x}^{ik,L-1})_{n,l,p}.
$$

MolNex computes the same object in two steps:

```python
env_w_edges = _make_weighted_channels(tensor_basis, env_w, ...)
env_w_scatter.scatter_add_(0, src.unsqueeze(-1).expand_as(env_w_edges), env_w_edges)
env_w_scatter = env_w_scatter * (1.0 / math.sqrt(avg_num_neighbors))
```

The normalization constant is dataset-level `avg_num_neighbors`.

### 7.2 Tensor Product Update

The paper uses bilinearity to avoid doing one tensor product for every
neighbor separately. In compact MolNex notation:

$$
\mathbf{V}^{ij,L}
=
\operatorname{TP}
\left(
\mathbf{V}^{ij,L-1},
\mathbf{A}^{i,L}
\right).
$$

The long paper form corresponds to Eqs. 11-13: sum over neighbors, move the
sum inside the second tensor-product argument, then evaluate a single tensor
product against the embedded environment.

MolNex code:

```python
new_tensor = tp(tensor_features, env_w_scatter, indices_2=src)
```

The spec maps this to `_allegro_uuu_descriptor`, whose descriptor has
subscripts:

```text
u, iu, ju, ku + ijk
```

That means per-channel tensor product with Clebsch-Gordan paths fused into the
shared output segment for each output irrep.

### 7.3 Scalar Feedback

Allegro then extracts scalar outputs from the tensor product and feeds them
back into the invariant scalar track:

$$
\mathbf{x}^{ij,L}
=
\operatorname{MLP}^{L}_\mathrm{latent}
\left(
\mathbf{x}^{ij,L-1}
\parallel
\bigoplus_{\mathrm{scalar\ paths}}
\mathbf{V}^{ij,L}_{\mathrm{scalar}}
\right).
$$

MolNex follows the reference implementation's DenseNet-style scalar stack:

```python
scalars = new_tensor[..., :n_scalar]
latents_out = latent(torch.cat(accumulated + [scalars], dim=-1))
new_scalar = latents_out[..., : self.num_scalar_features]
accumulated.append(new_scalar)
```

The final output is:

$$
\mathrm{edge\_features}_{ij}
=
[\mathbf{x}^{ij,0}; \mathbf{x}^{ij,1}; \ldots; \mathbf{x}^{ij,L}]
\in
\mathbb{R}^{F(L+1)}.
$$

This is the most important shape fact in practical use.

## 8. From Edge Features to Energy

The paper predicts pair energy with an output MLP:

$$
E_{ij}
=
\operatorname{MLP}_\mathrm{output}
(\mathbf{x}^{ij,L}).
$$

MolNex factors this out of `molzoo`:

```python
from molpot.heads import EdgeEnergyHead

readout = EdgeEnergyHead(
    input_dim=encoder.output_dim,
    avg_num_neighbors=avg_num_neighbors,
)
```

This is the deliberate adaptation recorded in `Allegro Spec`, Section 6:
`molzoo.Allegro` is the encoder; `molpot.heads.EdgeEnergyHead` owns readout and
aggregation.

## 9. Hands-On: Build a Tiny Allegro Batch

This section constructs a tiny directed graph by hand. In production, let
`molix.data.NeighborList(symmetry=True)` build the edge tensors.

```python
import torch
from molix.data.types import AtomData, EdgeData, GraphBatch, GraphData

pos = torch.tensor(
    [
        [0.0000, 0.0000, 0.0000],
        [0.9572, 0.0000, 0.0000],
        [-0.2390, 0.9270, 0.0000],
    ],
    dtype=torch.float32,
)
Z = torch.tensor([8, 1, 1], dtype=torch.long)

edge_index = torch.tensor(
    [
        [0, 1],
        [1, 0],
        [0, 2],
        [2, 0],
    ],
    dtype=torch.long,
)

src = edge_index[:, 0]
dst = edge_index[:, 1]
bond_diff = pos[dst] - pos[src]
bond_dist = bond_diff.norm(dim=-1)

batch = GraphBatch(
    atoms=AtomData(
        Z=Z,
        pos=pos,
        batch=torch.zeros(len(Z), dtype=torch.long),
        batch_size=[len(Z)],
    ),
    edges=EdgeData(
        edge_index=edge_index,
        bond_diff=bond_diff,
        bond_dist=bond_dist,
        batch_size=[len(edge_index)],
    ),
    graphs=GraphData(
        num_atoms=torch.tensor([len(Z)]),
        batch_size=[1],
    ),
    batch_size=[],
)
```

The directed average neighbor count for this toy batch is:

```python
avg_num_neighbors = len(edge_index) / len(Z)
```

## 10. Run the Encoder

```python
from molzoo import Allegro

encoder = Allegro(
    num_elements=119,
    r_max=5.0,
    avg_num_neighbors=avg_num_neighbors,
    l_max=2,
    num_layers=2,
    num_scalar_features=64,
    num_tensor_features=16,
)

batch = encoder(batch)
edge_features = batch["edges", "edge_features"]
print(edge_features.shape)
print(encoder.output_dim)
```

Expected:

```text
torch.Size([4, 192])
192
```

because `E = 4`, `F = 64`, and `L + 1 = 3`.

## 11. Add the Energy Readout

```python
from molpot.heads import EdgeEnergyHead

readout = EdgeEnergyHead(
    input_dim=encoder.output_dim,
    avg_num_neighbors=avg_num_neighbors,
)

pred = readout(batch)
print(pred["energy"].shape)
```

Expected:

```text
torch.Size([1])
```

## 12. Wrap Encoder and Readout for Training

```python
import torch.nn as nn


class AllegroEnergyModel(nn.Module):
    def __init__(self, encoder, readout):
        super().__init__()
        self.encoder = encoder
        self.readout = readout

    def forward(self, batch):
        batch = self.encoder(batch)
        return self.readout(batch)


model = AllegroEnergyModel(encoder, readout)
```

With graph-level energy labels:

```python
import torch.nn.functional as F


def loss_fn(pred, batch):
    return F.mse_loss(pred["energy"], batch["graphs", "energy"])
```

Then train with `molix.core.trainer.Trainer`.

## 13. Hyperparameter Guidance

| Parameter | Theory role | Practical starting point |
|---|---|---|
| `r_max` | strict local cutoff | Choose from chemistry/materials prior; larger means more edges. |
| `avg_num_neighbors` | environment and readout normalization | Compute from training data using the same cutoff and directed-edge setting. |
| `l_max` | angular resolution | Start with `2`; official docs suggest `1`, `2`, or `3`, with cost growing quickly. |
| `num_layers` | tensor-product depth / body order | Start with `2`; common range is `1` to `3`. |
| `num_scalar_features` | scalar latent capacity | Start with `64` or `128`. This multiplies final output width. |
| `num_tensor_features` | equivariant tensor channels | Start with `16`; increase only if tensor capacity is the bottleneck. |
| `num_bessel` | radial resolution | Default `8`, matching the spec and official guide. |
| `type_embed_dim` | center-neighbor chemical embedding | Must be even in MolNex. Default `64`. |
| `latent_mlp_width` | hidden width in layer MLPs | Default `128`; official docs recommend multiples of 16 or 32 for performance. |

## 14. Spec Crosswalk

| Question | Spec section | MolNex answer |
|---|---|---|
| What does Allegro read/write? | Section 2 | Reads atom type and edge geometry; writes `edges.edge_features`, and optionally `edges.edge_tensor_features`. |
| Why no energy head in `molzoo`? | Section 6, A1 | Readout is factored into `molpot.heads.EdgeEnergyHead`. |
| Where is ProductTypeEmbedding? | Section 3.2 | `type_embed * basis_linear(edge_radial)`. |
| Where is the embedded environment? | Section 3.5 | `_make_weighted_channels` plus `scatter_add_` to source atoms. |
| Where is the tensor product? | Section 3.5 | `_allegro_uuu_descriptor` and `EquivariantPolynomialTP`; no post-TP linear. |
| Why output width `F(L+1)`? | Section 3.6 | DenseNet scalar stack. |
| How is optional tensor output exposed? | Section 2.2 and A2 | `expose_tensor_track=True` writes `edges.edge_tensor_features`. |
| What validates reference alignment? | Sections 5-7 | Reference crosswalk, adaptation ledger, and validation contract. |

## 15. Common Failure Modes

Wrong edge direction:

```text
bond_diff = pos[source] - pos[target]  # wrong for MolNex Allegro
```

Correct:

```text
bond_diff = pos[target] - pos[source]
```

Using half edges:

```python
NeighborList(cutoff=5.0, symmetry=False)  # usually wrong for Allegro
```

Use:

```python
NeighborList(cutoff=5.0, symmetry=True)
```

Mismatched readout input:

```python
EdgeEnergyHead(input_dim=num_scalar_features)  # wrong when num_layers > 0
```

Use:

```python
EdgeEnergyHead(input_dim=encoder.output_dim)
```

Batch-dependent normalization:

```python
avg_num_neighbors = edges_in_this_batch / atoms_in_this_batch  # unstable
```

Compute it once from the training set and keep it fixed.
