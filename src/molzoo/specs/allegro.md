# Allegro Specification

Implementation contract for `molzoo.allegro`: the MolNex port of the mainline
Allegro energy model (`nequip` ≥ 0.7 framework) that loads official
`.nequip.zip` packages.

| Field | Value |
|-------|-------|
| Module | `molzoo.allegro` |
| Entry points | `AllegroEncoder(spec)` (body), `AllegroPotential(spec, wrt=())` (energy model), `AllegroPotential.load(path)` |
| Config | `AllegroSpec` (torch-free) |
| Paper | Musaelian et al., *Learning Local Equivariant Representations for Large-Scale Atomistic Dynamics*, Nat. Commun. 14, 579 (2023) — https://arxiv.org/abs/2204.05249 |
| Reference implementation | `nequip-allegro` 0.8.3 + `nequip` 0.19.1 (PyPI); `Allegro-OAM-L:0.1` built with allegro 0.7.1 (same mathematics) |
| Spec status | partial — §2/§3/§5 transcribed from the reference; official parity measured (§7.4) |

## 1. Scope

`AllegroEncoder` is the strictly local equivariant body: atom types and
directed pair geometry in, per-pair scalar features (and optionally the last
tensor track) out. Its receptive field is exactly `r_max`.

`AllegroPotential` (a `molix.autograd.Potential`) subclasses the body and adds
upstream's energy heads (edge readout MLP, edgewise reduce, per-type
scale/shift, optional ZBL) with the potential contract: `energy(frame)` writes
`atoms.energy` / `graphs.energy`, `forward = Derivative(energy, wrt)`, and
`load(path)` for `.nequip.zip` packages.

Not owned: neighbour-list construction (`molix.neighbor.NeighborList`), force
and virial derivation (`molix.autograd.Derivative`), training.

## 2. Paper↔Code Mapping

| Upstream module | Role | MolNex |
|---|---|---|
| `EdgeLengthNormalizer` | `x = r / r_max` | `x = lengths * _rmax_recip` (float64 buffer) |
| `BesselEdgeLengthEncoding` | `sinc(x w) w · u_p(x)` | `molrep.embedding.radial.SincBesselRBF` × `PolynomialCutoff(r_max, p)` |
| `ProductTypeEmbedding` | `[E_c(t_i) ‖ E_n(t_j)] ⊙ W b` | `molzoo.allegro.layers.ProductTypeEmbedding` |
| `ScalarMLP` (`scalar_embed_mlp`) | two-body MLP `D → F` | `ScalarMLPFunction` |
| `SphericalHarmonics` (upstream, component, normalised) | `Y(r̂)` | `molrep.embedding.angular.SphericalHarmonics` (identical basis, l ≤ 3 verified) |
| `MakeWeightedChannels` | `V_0 = w ⊙ Y` | `molzoo.allegro.layers.WeightedChannels` |
| `Allegro_Module` layers | scatter, `Contracter`, latent DenseNet | `AllegroEncoder.layer_features`, `molzoo.allegro.layers.Contracter` |
| `ScalarMLP` (`edge_readout`) | `F(L+1) → 1` | `ScalarMLPFunction` |
| `EdgewiseReduce` | `Σ_j e_ij / √⟨N⟩ / √2` onto the centre | `AllegroPotential.energy` |
| `PerTypeScaleShift` | `s_t e + b_t` in float64 | `molpot.heads.rescale.PerSpeciesScaleShift(dtype=float64)` |
| `ZBL` | LAMMPS `pair_zbl`, `½` per directed pair | `molpot.potentials.repulsion.UniversalZBL` |
| `ForceStressOutput` | `F = −∂E/∂r`, `W = −∂E/∂ε` | `molix.autograd.Derivative` (`graphs.virial = +∂E/∂ε`, opposite sign) |

## 3. Reference Alignment

### 3.1 Reference implementation

- `allegro/model/allegro_models.py:33-283` (`AllegroModel`, `FullAllegroModel`)
- `allegro/nn/_allegro.py:16-323`, `allegro/nn/_strided/_contract.py`,
  `allegro/nn/_strided/_cueq_contracter.py`, `allegro/nn/_strided/_channels.py`
- `allegro/nn/_edgeembed.py`, `allegro/nn/scalarembed.py`, `allegro/nn/tensorembed.py`,
  `allegro/nn/edgewise.py`
- `nequip/nn/embedding/_edge.py`, `nequip/nn/mlp.py`, `nequip/nn/atomwise.py`,
  `nequip/nn/pair_potential.py`

### 3.2 Convention mapping

- Upstream `edge_index[0]` (centre) is MolNex `pair_index[:, 0]` (source); the
  full list carries both directions, and periodic self-images `(i, i, S ≠ 0)`
  are ordinary pairs.
- `vec = pos[edge_index[1]] - pos[edge_index[0]] + S @ cell` is
  `molrep.embedding.geometry.edge_vectors` (imaged value from `pairs.pair_diff`).
- Atom types are indices into `atomic_numbers` (upstream `type_names` order,
  not necessarily sorted; Allegro-OAM-L skips Po–Ra).
- Tensor features are carried in cuEquivariance `ir_mul` layout `(E, D·u)`;
  upstream's strided `(E, u, D)` is transposed only inside its own CUDA path.
- cuEquivariance `O3` Clebsch–Gordan coefficients equal upstream's stored
  `wigner_3j · √(2 l_out + 1)` exactly (factor 1.000000000000 on every path).

## 4. Adaptation Ledger

- **Element table.** `atomic_numbers` (any order, unique) maps `Z → type`
  through a dense lookup; an atomic number outside the table is rejected once,
  eagerly (`validate_elements`).
- **Tensor-track exposure (MolNex-only).** `expose_tensor_track=True` keeps
  the last layer's tensor irreps (forward-pruned, no scalar-only collapse) and
  writes `pairs.edge_tensor_features` for equivariant heads
  (`molpot.heads.PermMultipoleHead`, `Sonata`). Such a model has more weights
  than upstream's and cannot load an official package.
- **Unsupported upstream options** (rejected by `load` with the option named):
  `per_edge_type_cutoff`, per-type `avg_num_neighbors`,
  `forward_normalize=False`, deep-linear MLPs, `tp_path_channel_coupling=False`,
  `weight_individual_irreps=False`, a non-integer cutoff exponent, a radial
  embedding other than `TwoBodyBesselScalarEmbed`.
- **Layout adapters in `load`.** Upstream stores per-type scales/shifts as
  `(T, 1)` (MolNex `(T,)`) and a single-path `Contracter` weight as `(u,)`
  (MolNex `(u, 1)`); the reshapes are explicit, per key. `tps.*.w3j` and
  `edge_norm._rmax_recip` are rebuilt, not loaded.
- **Precision.** Bessel weights, `1/r_max`, scales/shifts and the ZBL prefactor
  are float64 as upstream; per-atom energies leave the scale/shift in float64.

## 5. Mathematical Contract

Per directed pair `(i → j)`, `r = ‖r_ij‖`, `x = r / r_max`, types `t_i, t_j`:

1. `b_n = sinc(x w_n) w_n · u_p(x)`, `n = 1..N_b`, `w_n` initialised to `n`
   (`sinc(z) = sin(πz)/(πz)`), `u_p` the polynomial envelope.
2. `h⁰ = [E_c(t_i) ‖ E_n(t_j)] ⊙ (W_b b)` of width `D`
   (`D = radial_chemical_embed_dim`, default `F`).
3. `h = MLP_s(h⁰)` of width `F` (SiLU, bias-free, nequip init).
4. `Y = Y(r̂_ij)` up to `l_max` (component normalisation);
   `V_0 = WeightedChannels(Y, L_env(h))`, `u` channels per irrep,
   weights laid out `[u, R]` (channel-major).
5. `[s_0 ‖ w_0] = L_proj(h)` (`F` scalars, `u·R` environment weights).
6. Layer `ℓ = 0..L-1`:
   `Ω_i = (1/√⟨N⟩) Σ_k WeightedChannels(Y_ik, w_ℓ,ik)` (scatter to the centre),
   `V_{ℓ+1} = Contracter_ℓ(V_ℓ, Ω_{i(pair)})` with path-channel weights
   `W_ℓ ∈ ℝ^{u × P_ℓ}`, paths enumerated `(out, in1, in2)`;
   `[s_{ℓ+1} ‖ w_{ℓ+1}] = MLP_ℓ([s_0 … s_ℓ ‖ V_{ℓ+1}^{0e}])`.
   Tensor irreps: forward pruning from the allowed set (`l ≤ l_max`, both
   parities when `parity`), last layer scalars only, then backward pruning.
7. `e_ij = MLP_r([s_0 … s_L])`, `E_i = s_{t_i} (Σ_j e_ij / √⟨N⟩ / √2) + b_{t_i}`
   (+ `Σ_j ½ k_e Z_i Z_j / r · φ(Z, r) · u_p(x)` with ZBL).
8. `E_g = Σ_{i∈g} m_i E_i` (`m` the real-atom mask).

Encoder `forward` writes `pairs.edge_features` `(E, F·(L+1))` and, with
`expose_tensor_track`, `pairs.edge_tensor_features` `(E, D_out·u)` `ir_mul`.

## 6. Config Mapping

`AllegroSpec` fields use upstream names: `atomic_numbers` [`type_names`],
`r_max`, `num_bessels`, `bessel_trainable`, `polynomial_cutoff_p`,
`radial_chemical_embed_dim`, `scalar_embed_mlp_hidden_layers_{depth,width}`,
`scalar_embed_mlp_nonlinearity`, `l_max`, `parity`, `num_layers`,
`num_scalar_features`, `num_tensor_features`,
`allegro_mlp_hidden_layers_{depth,width}`, `allegro_mlp_nonlinearity`,
`readout_mlp_hidden_layers_{depth,width}`, `readout_mlp_nonlinearity`,
`avg_num_neighbors`, `per_type_energy_scales`, `per_type_energy_shifts`,
`per_type_energy_{scales,shifts}_trainable`, `zbl`; MolNex-only:
`expose_tensor_track`, `tp_method`. Defaults are upstream's
(`l_max` 2 aside, which upstream requires).

## 7. Validation Contract

### 7.1 Research Reproduction

No paper benchmark is reproduced in-tree. Official parity (below) is the
reproduction claim.

### 7.2 Symmetry and Shape Tests

`tests/test_molzoo/test_allegro/`: element table, irreps schedule against
upstream's path counts, translation / rotation / permutation invariance,
cutoff vanishing, forces against finite differences, padded-batch masking,
`slow` compile (inference `forward` → 1 graph / 0 breaks).

### 7.3 Official Parity

`test_checkpoint.py` (skipped without `MOLNEX_ALLEGRO_WEIGHTS_DIR`): load the
official package, evaluate the golden structures (open molecule, triclinic cell
with self-images), compare energy / forces / virial with upstream goldens
(fp64: 1e-8 eV/atom, 1e-7 eV/Å).

### 7.4 Run Log

| run_id | date | command | note |
|---|---|---|---|

## 8. System Boundary

`molrep` supplies the generic pieces (`SincBesselRBF`, `ScalarMLPFunction`,
`PolynomialCutoff`, `SphericalHarmonics`, edge geometry). `molpot` owns
`PerSpeciesScaleShift` and `UniversalZBL`. `molzoo.allegro` owns the spec, the
Allegro layers (`ProductTypeEmbedding`, `WeightedChannels`, `Contracter`), the
body, the energy model, the package reader and the key table. `molzoo.checkpoint`
owns the shared stub unpickler and `CheckpointRemap`.

## 9. Version Pinning

`nequip` 0.19.1, `nequip-allegro` 0.8.3 (PyPI; out-of-tree oracle venv, never a
molnex dependency). Official model: `Allegro-OAM-L:0.1`
(Zenodo record 16980200).

## 10. Drift Policy

Any change to the Allegro mathematics, key table, edge convention or output
keys updates this spec first; a new upstream release is re-checked against the
goldens before the pin moves.
