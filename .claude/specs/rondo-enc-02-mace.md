# Rondo MACE conformer encoder

| Field | Value |
|---|---|
| Opened | 2026-10-28 |
| Chain | rondo-enc (charter `rondo-enc-00-charter.md`) |
| Status | draft; starts after 01's parity gate |

## Contract

`MaceConformerEncoder(spec, *, dtype)` in `src/molzoo/rondo/encoder.py` returns
`EncoderFeatures(atom, bond)` at the canonical width (256, ByteFF-26's).

- **Backbone.** `molzoo.mace.encoder.MACEEncoder` building blocks (`element_index`,
  `initial_node_features`, `angular_features`, `radial_features`, `layer_features`), never its
  `forward`, which raises on `graphs.total_charge` for an unconditioned spec
  (`molzoo/mace/encoder.py:371-378`). Spec shaped like `MACEOffSpec` (H C N O F P S Cl Br I);
  trained from scratch. A MACE-OFF24-M warm start is a Phase-3 decision.
- **Positions.** Reads `atoms.pos.detach()` and builds its own edge vectors on
  `pairs.pair_index`; never reads `pairs.pair_diff` (the `Derivative` op rebuilds it from the live
  leaf). No ∂parameter/∂R term enters any force. The 10 Å Rondo list is reused; the polynomial
  cutoff zeroes every pair beyond r_max.
- **Atom features.** l=0 scalars of every layer `(N, L, F)` → flatten → `nn.Linear` → 256.
- **Bond features.** `½(MLP[h_i‖h_j] + MLP[h_j‖h_i])` plus bond order / in-ring embeddings.
- **Equivalence.** Mean over `atoms.equi_index` / `bonds.equi_index`, as for ByteFF-26: a single
  conformer breaks topological symmetry, the contract restores it.
- **Reference conformer.** Batch task `molix.data.ReferenceConformer` writes `atoms.ref_atom`: for
  each atom, the index of the same atom in the first graph of its profile (frames of one row share
  atom order). Features are gathered through it, so parameters are static per molecule with no
  host sync. One frame (MD set-up): `ref_atom = arange`. Running on every conformer gives the
  cross-conformer spread for free (Phase 4 metric). Check `molpot.classical.tables.ParameterMap`
  before writing the task.

## Tests

Rotation, translation and permutation invariance; no gradient reaches `atoms.pos`; conformers of
one profile gather identical features; `features()` evaluates no head; 0 graph breaks (slow);
bounded smoke on a THEMol hessian-relax subset (≤ 200 steps, one GH200, ≤ 45 min).
