# MACE Specification

This page is the implementation contract for `molzoo.MACE` (encoder-only). It is
not a tutorial; use the MolZoo user guide for narrative and worked examples.

| Field | Value |
|-------|-------|
| Module | `molzoo.mace` |
| Entry point | `MACE` (config `MACESpec`) |
| Paper | Batatia et al., *MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields*, NeurIPS 2022 |
| arXiv | https://arxiv.org/abs/2206.07697 |
| DOI | not applicable (NeurIPS proceedings) |
| Reference implementation | `ACEsuit/mace` (ScaleShiftMACE / RealAgnosticResidualInteractionBlock family) |
| Spec status | partial |

**Related modules (out of scope for this file):**

| Module | Role |
|--------|------|
| `molzoo.mace_omol.MACEOMol` | Full energy/force model (OMOL weights); see `mace_omol.md` |
| `molzoo.mace_matpes.MACEMatpes` | Full energy/force model (MatPES/MP weights); see `mace_matpes.md` |
| `molpot.composition` / heads / derivation | Energy readout, forces, composition |
| `molix.data.NeighborList` | Cutoff graph construction |

## 1. Scope

`molzoo.MACE` is an **encoder-only** equivariant message-passing module. It owns:

- node and edge embeddings (Bessel RBF, spherical harmonics, cosine cutoff)
- multi-layer equivariant interactions (`ConvTP` + product / element update)
- writing per-layer node features under `atoms.node_features` as
  `(N, num_layers, feature_dim)`

It does **not** own:

- neighbor-list construction
- graph energy / atomic energy heads
- force or stress derivation
- training loops or losses
- OMOL weight import (that is `MACEOMol`)

Those are owned by `molix` and `molpot` (or `mace_omol` for the full-model path).

## 2. Public Contract

### 2.1 Required Inputs

| Direction | TensorDict path / kwarg | Shape | Dtype | Contract |
|-----------|-------------------------|-------|-------|----------|
| In | `("atoms", "Z")` or `Z=` | `(N,)` | int64 | Atomic numbers |
| In | `("edges", "edge_index")` or `edge_index=` | `(E, 2)` | int64 | Col 0 = source, col 1 = target |
| In | `("edges", "edge_diff")` or `edge_diff=` | `(E, 3)` | float | `pos[target] - pos[source]` |
| In | `("edges", "edge_dist")` or `edge_dist=` | `(E,)` | float | `‖edge_diff‖` |

### 2.2 Outputs

| Direction | TensorDict path / return | Shape | Contract |
|-----------|--------------------------|-------|----------|
| Out | `("atoms", "node_features")` / return tensor | `(N, num_layers, F)` | Per-layer node features |

## 3. Forward Contract

### 3.1 Notation

| Symbol | Meaning | Code anchor |
|--------|---------|-------------|
| \(N, E\) | atoms, edges | batch sizes |
| \(L\) | `num_layers` | `MACESpec` / interaction stack |
| \(F\) | `num_features` | scalar channel multiplicity at \(\ell=0\) |
| \(\ell_{\max}\) | `l_max` | spherical harmonics / TP |

### 3.2 Pipeline

1. **Embed** — `JointEmbedding(Z)` + `BesselRBF(edge_dist)` +
   `SphericalHarmonics(edge_diff/‖·‖)` + `CosineCutoff`.
2. **Interact** — each layer: equivariant linear → `ConvTP` (node ⊗ \(Y_\ell\)) →
   scatter to nodes → product / element residual.
3. **Stack** — concatenate or stack per-layer features to `(N, L, F)`.

Default cuEq `use_fallback=True` on `ConvTP` so a downstream
`ForceDerivation(method="functorch")` path remains traceable. Callers that only
use `method="autograd"` may construct blocks with `use_fallback=False` for fused
kernels (see `MACEOMol`).

## 4. Configuration Contract

| `MACESpec` / ctor field | Meaning | Constraint |
|-------------------------|---------|------------|
| `node_attr_specs` | Discrete/continuous embeddings (e.g. Z) | non-empty |
| `num_elements` | Species table size | > 0 |
| `num_features` | Channel multiplicity | > 0 |
| `r_max` | Radial cutoff (Å) | > 0 |
| `num_layers` | Message-passing depth | ≥ 1 |
| `l_max` | Angular momentum | ≥ 0 |
| `num_bessel` | Radial basis size | > 0 |

## 5. Reference Crosswalk

| Concept | Reference | MolNex anchor | Status |
|---------|-----------|---------------|--------|
| Higher-order equivariant messages | MACE paper Eq. product basis | `ConvTP` + product path | partial |
| Radial Bessel + cutoff | ACEsuit/mace | `BesselRBF` + `CosineCutoff` | matched |
| Edge convention source→target | molnex CLAUDE | `edge_index[:,0]` source | matched |
| Full energy/force model | ScaleShiftMACE | **not this module** — `MACEOMol` | out of scope |

## 6. MolNex Adaptations

| ID | Adaptation | Reason | Risk | Validation |
|----|------------|--------|------|------------|
| A1 | Encoder-only surface (no energy head) | industrial package split | low | unit tests `tests/test_molzoo/` |
| A2 | TensorDict + raw kwargs dual API | batch contract | low | forward smoke |
| A3 | Default `use_fallback=True` on TP | functorch force compatibility | medium | force-path tests on composed models |

## 7. Validation Contract

### 7.1 Research Reproduction

Not claimed for the encoder-only port; energy/force MAE lives with composed
`molpot` heads or `MACEOMol`.

### 7.2 Symmetry and Shape Tests

| Claim | Test path | Tolerance |
|-------|-----------|-----------|
| Output shape `(N, L, F)` | `tests/test_molzoo/` MACE tests | exact |
| Rotation equivariance of features | symmetry helpers / MACE tests | project default |

### 7.3 Engineering Benchmark

No dedicated `bench_mace` yet (tracked as perf work queue).

### 7.4 Run Log

| run_id | date | commit | dirty | dataset | config | steps | train_mae | val_mae | fwd_ms | bwd_ms | compiled | note |
|--------|------|--------|-------|---------|--------|-------|-----------|---------|--------|--------|----------|------|

## 8. System Boundary

| Concern | Owner | Contract |
|---------|-------|----------|
| Neighbor list | `molix.data.tasks.NeighborList` | cutoff graph, edge convention |
| Encoder features | `molzoo.MACE` | this spec |
| Energy / force | `molpot` or `molzoo.MACEOMol` | not this module |
| Training | `molix.Trainer` | TrainState namespaces |

## 9. Version Pinning

| Item | Value |
|------|-------|
| Paper | Batatia et al., NeurIPS 2022 |
| Reference repository | `ACEsuit/mace` (pin TBD on next audit) |
| Dependencies | `cuequivariance`, `cuequivariance_torch`, `torch>=2.10` |
| Public docs mirror | `docs/molzoo/` (encoder docs partial) |

## 10. Drift Policy

Any change to message irreps, edge convention, or feature layout **must** update
this file in the same PR. Encoder-only vs `MACEOMol` boundary changes are
breaking for specs. Force backend / `use_fallback` defaults require a note in
§6 and a validation row in §7.

## Appendix A. Maintenance Log

- 2026-07-29: Scaffolded from template; encoder-only contract filled from
  `src/molzoo/mace.py` + industrial layout notes.
