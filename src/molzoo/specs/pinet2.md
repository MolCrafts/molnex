# PiNet2 Spec

Status: partial

## 1. Scope & Boundary

This spec covers the MolNex PyTorch port of Teoroo-CMC/PiNN `PiNet2` at
`pinn/networks/pinet2.py`, limited to the encoder. Potential energy, dipole,
charge-response, and polarizability models are downstream `molpot` modules.

Out of scope: TensorFlow checkpoint import, PiNN YAML compatibility, BPNN,
legacy PiNet, PiNNAcLe, PiNNwall, and ASE calculator wrappers.

## 2. Paper↔Code Mapping

PiNet2 carries invariant scalar `P1` features and optional equivariant `P3`
and `P5` tracks. Each graph-convolution block computes scalar `PI-II-IP-PP`,
uses scalar interactions to update vector/tensor tracks, forms invariant dot
products from those tracks, and residual-updates all enabled tracks.

## 3. Reference Alignment

Reference implementation:

- `Teoroo-CMC/PiNN@b592996c4ec2d19d6fe9ffedffb38c1ab998f77b`
- `pinn/networks/pinet2.py`
- `pinn/networks/pinet.py`
- `pinn/layers/basis.py`
- `pinn/layers/misc.py`

MolNex adaptation:

- PiNN's `ind_2[:, 0]` maps to MolNex `edge_index[:, 0]` source atom.
- PiNN's `diff = coord[j] - coord[i]` maps to MolNex `bond_diff`.
- PiNN's neighbor-list ownership stays outside the encoder; MolNex receives
  already-collated edges.

## 4. Adaptation Ledger

- PiNN's Keras `out_extra` readouts are not embedded in the encoder. MolNex
  writes raw `i1/i3/i5` representation tracks and downstream `molpot` heads
  perform task-specific projections.
- PiNN cutoff functions are zeroed outside `r_max` because MolNex edges may be
  supplied by arbitrary preprocessing tasks.
- The encoder writes TensorDict keys in place rather than returning TensorFlow
  estimator outputs.

## 5. Mathematical Contract

For every edge `(i, j)`, `bond_diff = r_j - r_i` and `bond_dist = ||r_j-r_i||`.
`d3` is the normalized edge direction. `d5` follows PiNN's five-component
rank-5 basis:

`[2/3 x^2 - 1/3 y^2 - 1/3 z^2, 2/3 y^2 - 1/3 x^2 - 1/3 z^2, xy, xz, yz]`.

The encoder outputs per-block scalar states `(N, depth, D)`, optional vector
states `(N, depth, 3, D)`, optional rank-5 states `(N, depth, 5, D)`, and
per-edge interaction tracks.

## 6. Config Mapping

`PiNet2Spec` mirrors PiNN constructor arguments: `atom_types`, `r_max`,
`cutoff_type`, `basis_type`, `n_basis`, `gamma`, `center`, `pp_nodes`,
`pi_nodes`, `ii_nodes`, `depth`, `activation`, `weighted`, and `rank`.

## 7. Benchmark Contract

Unit coverage must include shape, translation invariance, scalar rotation
invariance, vector rotation equivariance, permutation equivariance, and
backward propagation through a small downstream task.

### 7.4 Run Log

| run_id | date | command | note |
|---|---|---|---|

## 8. System Boundary

`molrep` owns reusable PiNN layers. `molzoo` owns the public `PiNet2` encoder.
`molpot` owns task heads and composers. `molix` owns GraphBatch-aware losses.

## 9. Version Pinning

Reference repository: `Teoroo-CMC/PiNN`, commit
`b592996c4ec2d19d6fe9ffedffb38c1ab998f77b`.

## 10. Spec Drift Policy

Any future change to PiNet2 math, edge convention, or output keys must update
this spec before code changes land.
