# Allegro Specification

This page is the implementation contract for `molzoo.Allegro`. It is not a
tutorial; use the MolZoo user guide for the theory narrative and worked
examples. The purpose of this spec is to make the encoder's public behavior,
mathematics, reference alignment, and validation obligations explicit.

| Field | Value |
|-------|-------|
| Module | `molzoo.allegro` |
| Entry point | `Allegro` (config `AllegroSpec`) |
| Paper | Musaelian et al., *Learning Local Equivariant Representations for Large-Scale Atomistic Dynamics*, Nature Communications 14, 579 (2023) |
| arXiv | https://arxiv.org/abs/2204.05249 |
| DOI | https://doi.org/10.1038/s41467-023-36329-y |
| Reference implementation | `mir-group/allegro::Allegro_Module` plus `mir-group/nequip` edge/MLP primitives |
| Spec status | stable for the MolNex API; upstream commit pin still needs a fresh audit |

## 1. Scope

`molzoo.Allegro` is a strictly local equivariant encoder. It reads atom types
and directed edge geometry, then writes scalar edge features. Its receptive
field is exactly the neighbor-list cutoff `r_max`; increasing `num_layers`
increases tensor-product depth, not the spatial radius.

It does not own:

- neighbor-list construction
- graph energy prediction
- pair-energy scale and shift
- force or stress gradients
- training loops or losses

Those are owned by `molix` and `molpot`.

The optional tensor-track output is still representation data, not a physical
observable. It is used by downstream equivariant heads such as multipole
readouts.

## 2. Public Contract

### 2.1 Required Inputs

| Direction | TensorDict path | Shape | Dtype | Contract |
|-----------|------------------|-------|-------|----------|
| In | `("atoms", "Z")` | `(N,)` | `int64` | Atomic numbers used by the center and neighbor embedding tables. |
| In | `("edges", "edge_index")` | `(E, 2)` | `int64` | Directed edges; column 0 is source/center `i`, column 1 is neighbor `j`. |
| In | `("edges", "bond_diff")` | `(E, 3)` | float | Edge vector `pos[j] - pos[i]`. |
| In | `("edges", "bond_dist")` | `(E,)` | float | Edge distance `||bond_diff||`. |

### 2.2 Outputs

| Direction | TensorDict path | Shape | Written when | Contract |
|-----------|------------------|-------|--------------|----------|
| Out | `("edges", "edge_features")` | `(E, F * (L + 1))` | always | DenseNet scalar stack `[f_0; f_1; ...; f_L]`. |
| Out | `("edges", "edge_tensor_features")` | `(E, tensor_track_irreps.dim)` | `expose_tensor_track=True` | Final equivariant tensor track in `ir_mul` layout. |

Here `F = num_scalar_features` and `L = num_layers`. The encoder exposes
`output_dim = F * (L + 1)` for readout construction.

The module may add the output keys above. It must not overwrite input keys,
write `("graphs", *)`, compute gradients, or attach losses.

## 3. Forward Contract

### 3.1 Notation

| Symbol | Meaning | Code anchor |
|--------|---------|-------------|
| `N` | number of atoms | `td["atoms", "Z"].shape[0]` |
| `E` | number of directed edges | `td["edges", "edge_index"].shape[0]` |
| `F` | scalar feature width | `num_scalar_features` |
| `U` | tensor channel multiplicity | `num_tensor_features` |
| `L` | Allegro layer count | `num_layers` |
| `B` | number of Bessel radial functions | `num_bessel` |
| `l_max` | maximum angular order | `l_max` |
| `r_cut` | radial cutoff | `r_max` |
| `d_type` | two-body type embedding width | `type_embed_dim` |
| `n_sh` | number of spherical-harmonic irreps | `l_max + 1` |
| `d_sh` | spherical-harmonic vector width | `sum_l (2l + 1)` |
| `W` | environment-weight width | `n_sh * U` |
| `N(i)` | directed neighbors of source atom `i` | `{k : (i, k) in edge_index}` |

### 3.2 Radial and Type Embedding

For each directed edge `(i, j)`, the radial basis is

$$
B_n(r_{ij}) =
\operatorname{sinc}\!\left(\frac{n r_{ij}}{r_\mathrm{cut}}\right) n,
\qquad n = 1,\ldots,B .
$$

The polynomial cutoff envelope is

$$
u(r) =
1
- \frac{(p+1)(p+2)}{2}\left(\frac{r}{r_\mathrm{cut}}\right)^p
+ p(p+2)\left(\frac{r}{r_\mathrm{cut}}\right)^{p+1}
- \frac{p(p+1)}{2}\left(\frac{r}{r_\mathrm{cut}}\right)^{p+2},
$$

for `r <= r_cut`, and zero outside. MolNex applies the cutoff once:

$$
\tilde{\mathbf{B}}(r_{ij}) =
\mathbf{B}(r_{ij}) u(r_{ij}) .
$$

The ordered center-neighbor type embedding is

$$
\mathbf{t}_{ij} =
\left[
e_\mathrm{center}(Z_i);
e_\mathrm{neighbor}(Z_j)
\right]
\in \mathbb{R}^{d_\mathrm{type}} .
$$

The complete two-body scalar embedding is the Hadamard product

$$
\mathbf{h}_{ij} =
\mathbf{t}_{ij}
\odot
W_\mathrm{basis}\tilde{\mathbf{B}}(r_{ij})
\in \mathbb{R}^{d_\mathrm{type}} .
$$

There is no extra scalar MLP between `ProductTypeEmbedding` and the Allegro
layer loop. `h_ij` feeds both the initial tensor-track projection and the
first scalar/environment projection.

| Quantity | Shape | Code anchor |
|----------|-------|-------------|
| `B(r)` | `(E, B)` | `Allegro.forward`, Bessel block |
| `u(r)` | `(E,)` | `PolynomialCutoff` |
| `type_embed` | `(E, d_type)` | `center_embed`, `neighbor_embed` |
| `h_ij` | `(E, d_type)` | `twobody_scalar_embed` |

### 3.3 Initial Tensor Track

Single-channel spherical harmonics are computed from the edge vector:

$$
\mathbf{Y}_{ij} = Y(\hat{\mathbf{r}}_{ij}) \in \mathbb{R}^{d_\mathrm{sh}} .
$$

The initial tensor-track weights are predicted directly from `h_ij`:

$$
\mathbf{w}^{(0)}_{ij}
=
W_{\mathrm{env},0}\mathbf{h}_{ij}
\in \mathbb{R}^{W}.
$$

`MakeWeightedChannels` broadcasts these weights over the spherical-harmonic
basis:

$$
V^{(0)}_{ij,(l,m,c)}
=
Y^l_m(\hat{\mathbf{r}}_{ij})
w^{(0),(l,c)}_{ij},
\qquad c=1,\ldots,U .
$$

The tensor track uses `ir_mul` layout: for each irrep, the `m` axis is outer
and channel is inner-fast.

| Quantity | Shape | Code anchor |
|----------|-------|-------------|
| `Y(r_hat)` | `(E, d_sh)` | `spherical_harmonics` |
| `w^(0)` | `(E, W)` | `env_embed_linear` |
| `V^(0)` | `(E, d_sh * U)` | `_make_weighted_channels` |

### 3.4 First Scalar and Environment Weights

The first projection emits the initial scalar feature slice and the
environment weights for layer 1 in one bias-free linear map:

$$
\left[
\mathbf{f}^{(0)}_{ij};
\mathbf{w}^{(1)}_{ij}
\right]
=
W_\mathrm{first}\mathbf{h}_{ij},
\qquad
\mathbf{f}^{(0)}_{ij}\in\mathbb{R}^{F},
\quad
\mathbf{w}^{(1)}_{ij}\in\mathbb{R}^{W}.
$$

The accumulated scalar list starts as

$$
\mathrm{accumulated}_{ij} =
\left[\mathbf{f}^{(0)}_{ij}\right].
$$

| Quantity | Shape | Code anchor |
|----------|-------|-------------|
| `f^(0)` | `(E, F)` | `first_layer_env_embed_projection[..., :F]` |
| `w^(1)` | `(E, W)` | `first_layer_env_embed_projection[..., F:]` |

### 3.5 Allegro Layer Update

For layer `ell = 1, ..., L`, current per-edge environment weights generate a
weighted spherical-harmonic edge embedding:

$$
\mathbf{E}^{(\ell)}_{ij}
=
\operatorname{MakeWeightedChannels}
\left(
Y(\hat{\mathbf{r}}_{ij}),
\mathbf{w}^{(\ell)}_{ij}
\right)
\in \mathbb{R}^{d_\mathrm{sh}U}.
$$

The source environment is a normalized sum over directed edges out of `i`:

$$
\bar{\mathbf{E}}^{(\ell)}_i
=
\frac{1}{\sqrt{\overline{|N|}}}
\sum_{k\in N(i)}
\mathbf{E}^{(\ell)}_{ik}.
$$

The tensor track is updated by one per-channel Clebsch-Gordan tensor product:

$$
V^{(\ell)}_{ij}
=
\operatorname{TP}_\ell
\left(
V^{(\ell-1)}_{ij},
\bar{\mathbf{E}}^{(\ell)}_i
\right).
$$

The descriptor is the Allegro `u,iu,ju,ku+ijk` contraction. It directly targets
the layer's pruned output irreps. There is no post-TP equivariant linear layer.

Valid Clebsch-Gordan paths satisfy

$$
|l_1 - l_2| \le l_3 \le l_1 + l_2,
\qquad
p_3 = p_1 p_2 .
$$

The invariant feedback slice is the scalar block of the tensor product:

$$
\mathbf{I}^{(\ell)}_{ij}
=
\operatorname{scalar}
\left(
V^{(\ell)}_{ij}
\right)
\in \mathbb{R}^{U}.
$$

The latent MLP consumes the DenseNet scalar history plus the new invariants:

$$
\left[
\mathbf{f}^{(\ell)}_{ij};
\mathbf{w}^{(\ell+1)}_{ij}
\right]
=
\operatorname{MLP}^{(\ell)}
\left(
\left[
\mathbf{f}^{(0)}_{ij};
\ldots;
\mathbf{f}^{(\ell-1)}_{ij};
\mathbf{I}^{(\ell)}_{ij}
\right]
\right),
$$

for `ell < L`. On the last layer, the MLP emits only
`\mathbf{f}^{(L)}_{ij}` because no next environment weights are needed.

| Quantity | Shape | Code anchor |
|----------|-------|-------------|
| `E_ij^(ell)` | `(E, d_sh * U)` | `_make_weighted_channels(tensor_basis, env_w, ...)` |
| `Ebar_i^(ell)` | `(N, d_sh * U)` | `scatter_add_` followed by `avg_num_neighbors_inv_sqrt` |
| `V^(ell)` | `(E, dim(ir_out_ell))` | `EquivariantPolynomialTP` |
| `I^(ell)` | `(E, U)` | first scalar block of `new_tensor` |
| `f^(ell)` | `(E, F)` | `latents[layer][..., :F]` |
| `w^(ell+1)` | `(E, W)` | `latents[layer][..., F:]`, omitted on last layer |

### 3.6 Final Outputs

The scalar encoder output is

$$
\mathrm{edge\_features}_{ij}
=
\left[
\mathbf{f}^{(0)}_{ij};
\mathbf{f}^{(1)}_{ij};
\ldots;
\mathbf{f}^{(L)}_{ij}
\right]
\in
\mathbb{R}^{F(L+1)}.
$$

When `expose_tensor_track=False`, the last layer is pruned to scalar output
irreps because no downstream equivariant tensor output is requested. When
`expose_tensor_track=True`, the final layer keeps the tensor irreps and writes
that final tensor track to `("edges", "edge_tensor_features")` without changing
`edge_features`.

| Quantity | Shape | Code anchor |
|----------|-------|-------------|
| `edge_features` | `(E, F * (L + 1))` | `torch.cat(accumulated, dim=-1)` |
| `output_dim` | `F * (L + 1)` | `self.output_dim` |
| `edge_tensor_features` | `(E, tensor_track_irreps.dim)` | `expose_tensor_track=True` branch |

### 3.7 Properties

| Transformation | Property |
|----------------|----------|
| Translation `r_i -> r_i + t` | invariant, because only edge vectors and distances are used. |
| Rotation/reflection `r_i -> R r_i`, `R in O(3)` | scalar output is invariant; internal tensor track is equivariant. |
| Atom relabeling | equivariant with edge endpoints; graph-level readouts must aggregate consistently. |
| Cutoff limit | `edge_features -> 0` and first distance derivative vanishes at `r_cut` for polynomial exponent `p >= 2`. |

## 4. Configuration Contract

| `AllegroSpec` field | Meaning | Default | Constraint / note |
|---------------------|---------|---------|-------------------|
| `num_elements` | atomic-number embedding table size | required | `> 0` |
| `num_scalar_features` | scalar width `F` | `64` | `> 0` |
| `num_tensor_features` | tensor channel multiplicity `U` | `16` | `> 0` |
| `r_max` | radial cutoff `r_cut` | required | `> 0` |
| `num_bessel` | radial basis count `B` | `8` | `> 0` |
| `l_max` | maximum angular order | `2` | `>= 0` |
| `num_layers` | Allegro depth `L` | `2` | `> 0` |
| `poly_p` | polynomial cutoff exponent | `6` | `>= 1` |
| `type_embed_dim` | width of `h_ij` | `64` | must be even; split center/neighbor |
| `latent_mlp_depth` | hidden depth in each layer latent MLP | `2` | `>= 0` |
| `latent_mlp_width` | hidden width in each layer latent MLP | `128` | `> 0` |
| `latent_activation` | latent MLP nonlinearity | `nn.SiLU` | `None` means deep linear |
| `avg_num_neighbors` | dataset-level average neighbor count | required | compute once from training data |
| `expose_tensor_track` | additionally write final equivariant tensor track | `False` | required by equivariant downstream heads |

## 5. Reference Crosswalk

| Concept | Reference implementation | MolNex anchor | Status |
|---------|--------------------------|---------------|--------|
| Bessel edge length encoding | `nequip` Bessel edge encoding | `Allegro.forward`, radial block | matched |
| Polynomial cutoff | `nequip` polynomial cutoff | `molrep.embedding.cutoff.PolynomialCutoff` | matched |
| Product type embedding | Allegro `ProductTypeEmbedding` | `center_embed`, `neighbor_embed`, `basis_linear` | matched |
| No extra scalar MLP after product type embedding | `TwoBodyBesselScalarEmbed` output feeds Allegro loop | `twobody_scalar_embed` feeds both first projections directly | matched |
| Spherical harmonics | `SphericalHarmonicEdgeAttrs` | `molrep.embedding.angular.SphericalHarmonics` | matched |
| Weighted channels | `MakeWeightedChannels(weight_individual_irreps=True)` | `_make_weighted_channels` | matched |
| Initial tensor track | `TwoBodySphericalHarmonicTensorEmbed` | `env_embed_linear` plus `_make_weighted_channels` | matched |
| First layer projection | `first_layer_env_embed_projection` | `first_layer_env_embed_projection` | matched |
| Source environment aggregation | Allegro layer scatter to center atom | `scatter_add_` over `src` | matched |
| Average neighbor normalization | `avg_num_neighbors_norm` | `avg_num_neighbors_inv_sqrt` | matched |
| Irrep pruning | Allegro two-pass pruning | `_build_layer_irreps` | matched |
| Per-channel tensor product | Allegro `Contracter` / `u,iu,ju,ku+ijk` | `_allegro_uuu_descriptor`, `EquivariantPolynomialTP` | matched |
| No post-TP equivariant linear | Contracter directly targets `irreps_out` | no `tp_linears`; `new_tensor = tp(...)` | matched |
| Scalar feedback | first scalar block of tensor output | `scalars = new_tensor[..., :n_scalar]` | matched |
| DenseNet scalar accumulation | accumulated scalar list | `torch.cat(accumulated, dim=-1)` | matched |
| Pair-energy readout | bundled in reference Allegro/NequIP stack | factored to `molpot.heads.EdgeEnergyHead` | adapted (A1) |
| Per-species energy scale/shift | NequIP post-processing | owned outside `molzoo` | adapted (A1) |
| Optional tensor-track exposure | reference optional tensor forwarding | `expose_tensor_track=True` writes TensorDict key | adapted (A2) |

## 6. MolNex Adaptations

| ID | Adaptation | Reason | Risk | Validation |
|----|------------|--------|------|------------|
| A1 | Pair-energy readout, energy aggregation, and per-species scale/shift are outside `molzoo`. | MolZoo encoders emit representations; MolPot owns physical readouts. | low | `tests/test_molzoo/test_allegro.py::TestEnergyInvariants`, `tests/test_molzoo/test_allegro.py::TestOverfitSingleBatch`, `tests/test_molpot/test_heads/test_edge_energy.py` |
| A2 | Optional tensor-track exposure uses `("edges", "edge_tensor_features")` instead of a reference return flag. | MolNex modules communicate through TensorDict keys. Scalar output remains unchanged. | low | `tests/test_molpot/test_heads/test_multipole_symmetry.py`, `tests/test_molpot/test_heads/test_multipole_energy_kernels.py` |

## 7. Validation Contract

### 7.1 Research Reproduction

This spec does not currently claim paper-level reproduction of QM9, 3BPA, or
revMD17 metrics. Add those rows only when the repository contains the matching
dataset recipe, training configuration, and run log.

### 7.2 Symmetry and Shape Tests

| Claim | Test path | Tolerance |
|-------|-----------|-----------|
| translation invariance | `tests/test_molzoo/test_symmetry.py::TestTranslationInvariance` | relative <= `1e-5` |
| rotation invariance/equivariance | `tests/test_molzoo/test_symmetry.py::TestRotationEquivariance` | relative <= `1e-4` |
| permutation equivariance | `tests/test_molzoo/test_symmetry.py::TestPermutationEquivariance` | relative <= `1e-5` |
| energy translation invariance | `tests/test_molzoo/test_allegro.py::TestEnergyInvariants::test_translation_invariance` | relative <= `1e-4` |
| energy rotation invariance | `tests/test_molzoo/test_allegro.py::TestEnergyInvariants::test_rotation_invariance` | relative <= `1e-4` |
| energy permutation invariance | `tests/test_molzoo/test_allegro.py::TestEnergyInvariants::test_permutation_invariance` | relative <= `1e-4` |
| cutoff vanishing | `tests/test_molzoo/test_allegro.py::TestEnergyInvariants::test_cutoff_vanishing` | relative <= `1e-5` |
| output width `F * (L + 1)` | `tests/test_molzoo/test_allegro.py::TestAllegroEncoder::test_output_dim_is_densenet_stack` | exact |
| edge feature write | `tests/test_molzoo/test_allegro.py::TestAllegroEncoder::test_forward_writes_edge_features` | exact |
| single-batch overfit | `tests/test_molzoo/test_allegro.py::TestOverfitSingleBatch` | loss < `1e-3` |
| tensor-track consumer path | `tests/test_molpot/test_heads/test_multipole_symmetry.py` | test-specific |

### 7.3 Engineering Benchmark

Driver:

```bash
python benchmarks/molzoo/bm_allegro.py --device cuda
```

Reference local measurement, taken on 2026-04-26 with Tesla T4, `bs=32`,
average atom count `18`, `L=3`, `l_max=2`, `F=128`, `U=32`, and
`num_bessel=8`:

| Quantity | Measured |
|----------|----------|
| forward time / batch | `5.70 +/- 0.01 ms` |
| forward+backward time / batch | `13.53 +/- 1.08 ms` |
| forward peak memory | `97.4 MiB` |
| forward+backward peak memory | `199.6 MiB` |

### 7.4 Run Log

| run_id | date | commit | dirty | dataset | config | steps | train_mae | val_mae | fwd_ms | bwd_ms | compiled | note |
|--------|------|--------|-------|---------|--------|-------|-----------|---------|--------|--------|----------|------|
| 1 | 2026-04-26 | local | 1 | synthetic-qm9-shape | `L=3 l_max=2 F=128 U=32 T4` | n/a | n/a | n/a | 5.70 | 13.53 | 0 | engineering bench after strict reference-alignment cleanup |

## 8. System Boundary

| Concern | Owner | Contract |
|---------|-------|----------|
| edge construction | `molix.data.NeighborList` | creates full directed edges with `symmetry=True`; `bond_diff = pos[dst] - pos[src]` |
| scalar encoder | `molzoo.Allegro` | reads §2.1 and writes `("edges", "edge_features")` |
| optional tensor encoder output | `molzoo.Allegro` | writes `("edges", "edge_tensor_features")` only when `expose_tensor_track=True` |
| pair-energy readout | `molpot.heads.EdgeEnergyHead` | consumes scalar edge features |
| equivariant multipole readout | `molpot.heads.PermMultipoleHead` and related heads | may consume tensor edge features |
| graph energy aggregation | `molpot` | writes graph-level energies |
| force and stress gradients | `molpot` | owns autograd over coordinates or strain |
| training loop | `molix.core.Trainer` | owns state, hooks, optimization, and losses |

Hard rules:

- `molzoo.Allegro` must not write graph-level physical outputs.
- `molzoo.Allegro` must not call `torch.autograd.grad`.
- `molzoo.Allegro` must not overwrite its input TensorDict keys.
- `edge_features` must keep shape `(E, F * (L + 1))` regardless of
  `expose_tensor_track`.

## 9. Version Pinning

| Item | Value |
|------|-------|
| Paper | Musaelian et al. 2023, arXiv:2204.05249 |
| Reference repository | `mir-group/allegro` and `mir-group/nequip` |
| Reference commit | not recorded in the current repo; next upstream audit must pin exact commits |
| PyTorch | `>=2.6` |
| cuequivariance / cuequivariance-torch | as pinned by project dependencies |
| MolNex implementation anchor | `src/molzoo/allegro.py` |
| Public docs mirror | `docs/molzoo/specs/allegro.md` must remain identical to this file |

## 10. Drift Policy

Update this spec when any of the following change:

- any input or output key
- `output_dim`
- the formulas in §3
- the reference alignment in §5
- the MolNex adaptation list in §6
- validation paths or tolerances in §7
- ownership boundaries in §8
- dependency or upstream reference pins in §9

For public documentation, do not use fragile code line numbers as the primary
contract. Prefer stable anchors such as class names, method names, TensorDict
keys, and test paths. If exact line numbers are included during an audit, verify
them in the same change.

The source spec and public docs copy must stay identical:

```bash
diff -u src/molzoo/specs/allegro.md docs/molzoo/specs/allegro.md
```

## Appendix A. Maintenance Log

- 2026-04-25: initial canonical 10-section spec.
- 2026-04-26: aligned implementation with the Allegro reference layer loop,
  DenseNet scalar accumulation, ProductTypeEmbedding wiring, and strict average
  neighbor normalization.
- 2026-04-26: removed stale fallback behavior in `EdgeEnergyHead`; readout now
  consumes the flat DenseNet scalar stack.
- 2026-04-27: rewrote the spec as a readable public contract, removed stale
  two-body-MLP and post-TP-linear descriptions, and documented optional
  tensor-track exposure.
