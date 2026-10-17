# PiNet2 Spec

Status: partial

## 1. Scope & Boundary

This spec covers the MolNex PyTorch port of Teoroo-CMC/PiNN `PiNet2`
(`pinn/networks/pinet2.py`) and its energy model (`pinn/models/potential.py`):

| Class | Role |
|---|---|
| `molzoo.pinet.PiNetSpec` | torch-free validated configuration |
| `molzoo.pinet.PiNetEncoder(spec, dtype=)` | feature body: element table, basis, `GCBlock` stack, residual updates; primitives + the encoder `forward(td)` |
| `molzoo.pinet.PiNetPotential(spec, wrt=(), dtype=)` | `PiNetEncoder` + per-block `OutLayer` energy readout, atomic dress, scale; `energy` / `forward = Derivative(energy, wrt)` |
| `molzoo.pinet.PiNetDipole(spec, ...)` / `PiNetPolarizability(spec, ...)` | `PiNetEncoder` + a `molpot.heads` dipole / charge-response head |

The PiNN layers are model-specific and live in `molzoo.pinet.layers`; only the
generic pieces come from `molrep` (`molrep.embedding.mlp.FFLayer`, cutoffs,
radial bases, `molrep.embedding.geometry`).

Out of scope: TensorFlow checkpoint import (operator ruling 2026-10-15), PiNN
YAML compatibility, BPNN, legacy PiNet (v1), PiNNAcLe, PiNNwall wrappers.

## 2. Paper↔Code Mapping

PiNet2 carries invariant scalar `P1` features and optional equivariant
Cartesian `P3` (`(N, 3, C)`) and `P5` (`(N, 5, C)`) tracks. Each
graph-convolution block computes scalar `PI-II-IP-PP`, uses scalar interaction
chunks to gate the vector/tensor tracks, mixes `[P1, dot(P3), dot(P5)]` through
a `PP` layer whose chunks rescale the equivariant tracks, and residual-updates
all enabled tracks.

| Paper / PiNN | MolNex |
|---|---|
| `FFLayer` (Keras `Dense` stack, activation `None` unless passed) | `molrep.embedding.mlp.FFLayer(activation=None)` |
| `PILayer`, `IPLayer` | `molzoo.pinet.layers.PILayer`, `IPLayer` |
| `PIXLayer`, `ScaleLayer`, `DotLayer` | `molzoo.pinet.layers.*` |
| `InvarLayer`, `EquivarLayer`, `GCBlock` | `molzoo.pinet.layers.*` (`GCBlock` returns `GCOutput`) |
| `ResUpdate` | `molzoo.pinet.layers.ResUpdate` |
| `OutLayer(out_nodes, out_units)` | `molzoo.pinet.layers.OutLayer(activation=None)` |
| `CutoffFunc` `f1` / `f2` / `hip` | `CosineCutoff` / `TanhCutoff` / `HalfCosineCutoff` |
| `PolynomialBasis`, `GaussianBasis` | `molrep.embedding.radial.PolynomialBasis`, `GaussianBasis` |
| `AtomicOnehot(atom_types)` | `PiNetEncoder.node_attrs` over the sorted element table |
| `pot_model`: `pred / e_scale + e_dress` | `PiNetPotential`: `scale · Σ out + Σ atomic_energies[Z]` |

## 3. Reference Alignment

### 3.1 Reference implementation

- `Teoroo-CMC/PiNN@b592996c4ec2d19d6fe9ffedffb38c1ab998f77b`
  (`pinn/networks/pinet2.py`, `pinn/networks/pinet.py`, `pinn/layers/basis.py`,
  `pinn/layers/misc.py`, `pinn/models/potential.py`). The local checkout's
  HEAD `6c930d1e` has identical copies of these files.

### 3.2 Convention mapping

- PiNN's `ind_2[:, 0]` is MolNex `edge_index[:, 0]` (source); `IPLayer` sums
  onto the source atom.
- PiNN's `diff = coord[j] - coord[i]` is `molrep.embedding.geometry.edge_vectors`
  (`pos[target] - pos[source]`, imaged value from `pairs.pair_diff` when present,
  gradient always through `pos`).
- PiNN's neighbour list stays outside the model; MolNex receives collated pairs.

## 4. Adaptation Ledger

- **Elements.** `atomic_numbers` (strictly ascending) replaces PiNN's
  `atom_types`; the one-hot index is `searchsorted(z_table, Z)`. An atomic
  number outside the table is rejected once, eagerly (`validate_elements`),
  instead of reaching `one_hot` as `-1`.
- **Energy units.** PiNN divides the network output by `e_scale`; MolNex
  multiplies by `scale = 1 / e_scale`. `atomic_energies` is PiNN's `e_dress` in
  table order (default all zero). `e_unit` is not modelled (MolNex energies are
  eV in and out).
- **Unused PiNN layer.** PiNN's `EquivarLayer` builds an `ii_layer` its `call`
  never uses; MolNex does not build it.
- **Out-layer activation.** `out_activation` defaults to `None`, PiNN's
  behaviour. MolNex before 2026-10-15 hard-coded `tanh`; reproduce such a model
  with `out_activation="tanh"`.
- `out_extra` readouts are not embedded in the encoder. `PiNetEncoder.forward`
  writes `i1/i3/i5` and `p3/p5` tracks and `PiNetDipole` /
  `PiNetPolarizability` pool them per block.
- `emit_property_features` (no PiNN counterpart) gates only the stacking and
  TensorDict write of the per-edge `i*` and `p3/p5` histories in
  `PiNetEncoder.forward`; `PiNetPotential` never builds them into the batch.
- Cutoff functions are zeroed outside `r_max` because MolNex pairs may come from
  arbitrary preprocessing.
- Ghost atoms of a padded batch are killed by multiplying `atoms.mask` into the
  per-atom energy before the graph sum (never a slice).

## 5. Mathematical Contract

For every pair `(i, j)`: `r_ij = r_j - r_i`, `d_ij = ‖r_ij‖` (clamped at
`1e-8`), `d3 = r_ij / d_ij`, and for rank 5
`d5 = [⅔x²−⅓y²−⅓z², ⅔y²−⅓x²−⅓z², xy, xz, yz]` of `d3`. The basis is
`e_ijb = f_c(d_ij) · φ_b(d_ij)`.

Block `t` (`t = 1..depth`), with `P1⁰ = onehot(Z)`, `P3⁰ = 0 (N,3,1)`,
`P5⁰ = 0 (N,5,1)`:

1. `I1 = II(PI(P1, e))` — width `C · n_props`, split into chunks `I1_k`.
2. `P1' = PP(IP(I1))`.
3. For `x ∈ {3, 5}` enabled by `rank`: `Ix = (PIX(Px) + dx) · I1_k`,
   `Px' = PPx(IP(Ix))` (bias-free linear), `Dx = dot(Px')`.
4. `[Q1, Q3, Q5] = split(PP_mix([P1', D3, D5]))`; block outputs
   `p1_t = Q1`, `p3_t = P3' · Q3`, `p5_t = P5' · Q5`.
5. `P1ᵗ = Res(P1ᵗ⁻¹) + p1_t`, likewise for `P3`, `P5`.

Energy: `o_0 = 0`, `o_t = W_t FF_t(p1_t) + o_{t-1}`;
`E_g = scale · Σ_{i∈g} m_i o_depth,i + Σ_{i∈g} m_i e0(Z_i)` with `m_i` the
real-atom mask. Forces and virial come from `molix.autograd.Derivative`.

Encoder `forward` writes `atoms.node_features` `(N, depth, C)` (the residual
`P1ᵗ`), `atoms.p1_block_outputs` `(N, depth, C)` (`p1_t`) and, with
`emit_property_features`, `atoms.p3_features` `(N, depth, 3, C)`,
`atoms.p5_features` `(N, depth, 5, C)`, `pairs.i1_features`
`(E, depth, C · n_props)`, `pairs.i3_features`, `pairs.i5_features`.

## 6. Config Mapping

`PiNetSpec` fields (PiNN name in brackets): `atomic_numbers` [`atom_types`],
`r_max` [`rc`], `cutoff_type`, `basis_type`, `n_basis`, `gamma`, `center`,
`pp_nodes`, `pi_nodes`, `ii_nodes`, `depth`, `activation` [`act`], `weighted`,
`rank`, `out_nodes`, `out_activation`, `atomic_energies` [`e_dress`], `scale`
[`1 / e_scale`], plus the MolNex-only `emit_property_features` (§4).
Validated at construction: ascending unique `atomic_numbers`,
`pp_nodes[-1] == ii_nodes[-1]`, `len(atomic_energies) == len(atomic_numbers)`,
non-empty `out_nodes`.

## 7. Benchmark Contract

Unit coverage: block shapes per rank, translation invariance, scalar rotation
invariance, vector rotation equivariance, permutation equivariance, forces
against finite differences, padded-batch masking, and a `slow` compile test
(inference `forward` → 1 graph / 0 breaks).

### 7.4 Run Log

| run_id | date | command | note |
|---|---|---|---|

## 8. System Boundary

`molzoo.pinet` owns the spec, the PiNN layers, encoder, potential and the two
property products; `molrep` supplies only generic pieces (`FFLayer`, cutoffs,
radial bases, edge geometry). `molpot.heads` owns `DipoleHead`,
`ChargeResponseHead`, `AtomicReferenceEnergy`, `GlobalRescale`. `molix` owns
`Derivative`, the geometry helpers' consumers and the losses.

## 9. Version Pinning

Reference repository: `Teoroo-CMC/PiNN`, commit
`b592996c4ec2d19d6fe9ffedffb38c1ab998f77b`.

## 10. Spec Drift Policy

Any future change to PiNet2 math, edge convention, or output keys must update
this spec before code changes land.
