# MaceOmol Specification

This page is the implementation contract for `molzoo.mace.potential.MaceOmol`.
It is not a tutorial; use the MolZoo user guide for theory narrative and worked
examples.

| Field | Value |
|-------|-------|
| Module | `molzoo.mace.potential` (exported as `molzoo.mace.MaceOmol`) |
| Entry point | `MaceOmol(spec: MACEOMolSpec, *, wrt=(), energy_dtype=torch.float64, input_dtype=None)`. Weights: `MaceOmol.load(path, config=None, *, wrt, energy_dtype, input_dtype)`, which reads the archive into a typed `MaceOfficialConfig`, builds `MACEOMolSpec.from_official(cfg)` and renames through the `omol_keys` table (`molzoo.mace.checkpoint`) with `molzoo.checkpoint.CheckpointRemap(omol_keys, on_unexpected="return", skip=MACE_SKIP)`. |
| Paper | Batatia et al., "MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields", NeurIPS 2022. OMol25 / MACE-omol-0 foundation model. |
| arXiv | https://arxiv.org/abs/2206.07697 |
| DOI | not applicable |
| Reference implementation | `ACEsuit/mace@v0.3.16` (sha not pinned — see §9) |
| Spec status | partial |

## 1. Scope

`MaceOmol` is a native MolNex (cuEquivariance) reimplementation of the official
**MACE-omol-0** base model (`ScaleShiftMACE`: 1024 channels, `r_max=6.0`,
`l_max=3`, 3 residual interactions, product `correlation=2`, 83 elements, single
`omol` head, with `total_charge` / `total_spin` conditioning, 52.7M parameters).

It **owns**: the OMOL preset of the full energy/force forward — the block graph
in `molzoo.mace.encoder.MACEEncoder`, the energy pipeline in
`molzoo.mace.potential.MacePotential`, and the `omol_keys` key dialect in
`molzoo.mace.checkpoint` that imports official weights (after
MACE's official cuEquivariance converter, run out of tree), reached through `MaceOmol.load`. The MACE blocks themselves live beside it in
`molzoo.mace.layers` (`ResidualInteraction`, `EquivariantProductBasis`) and
`molzoo.mace.heads` (`NonLinearBiasReadout`, `JointFeatureEmbedding`).

It does **not** own: the building blocks themselves (they live in `molrep` /
`molpot` and are reused by other models), neighbor-list construction, dataset /
collation, the training loop, the derivative op (`molix.autograd.Derivative`),
or multi-head / pair-repulsion outputs (out of scope — see §6 and the
`mace-omol-port` specs in `.claude/specs/`).

Unlike the other MolZoo encoders, `MaceOmol` is a full **energy/force model**,
not an encoder-only feature extractor: its `forward` writes `graphs.energy` and,
when built with `wrt=("atoms.pos",)`, `atoms.forces` — not `atoms.node_features`.

## 2. Public Contract

### 2.1 Required Inputs

| Direction | TensorDict path | Shape | Dtype | Contract |
|-----------|------------------|-------|-------|----------|
| In | `atoms.Z` | `(N,)` | long | Atomic numbers; must be present in the construction `atomic_numbers` table |
| In | `atoms.pos` | `(N, 3)` | float (the model's `dtype`) | Cartesian positions; forces are `-∂E/∂pos` |
| In | `atoms.batch` | `(N,)` | long | Graph membership index `0..B-1` |
| In | `pairs.pair_index` | `(E, 2)` | long | `[:,0]`=source/sender, `[:,1]`=target/receiver (MolNex convention) |
| In | `graphs.total_charge` | `(B,)` | long | Optional; per-graph total charge in units of `e`. Absent → neutral (`0`) |
| In | `graphs.total_spin` | `(B,)` | long | Optional; per-graph spin channel. Absent → `1` (singlet multiplicity `2S+1 = 1`, all electrons paired). **Not `0`** — `MacePotential._condition_charge_spin` fills `torch.ones(...)`, and spin `0` would index an untrained embedding row (`spin_offset = 0`) and return garbage |
| In | `atoms.mask` | `(N,)` | bool | Optional; padded-batch real-atom mask, multiplied into every per-atom energy |

`pairs.pair_diff` `(E, 3)` is required; `energy` raises `ValueError` without it
and does not rebuild it from `pairs.shifts`.
`pairs.pair_dist` is not an input: the body recomputes lengths from `pair_diff`.

### 2.2 Outputs

| Direction | TensorDict path | Shape | Written when | Contract |
|-----------|------------------|-------|--------------|----------|
| Out | `graphs.energy` | `(B,)` | always | Per-graph total energy = E0 + scale·shift(readout) |
| Out | `atoms.forces` | `(N, 3)` | `wrt` contains `"atoms.pos"` | `F = -∂E/∂pos` via `molix.autograd.Derivative` |
| Out | `graphs.virial` | `(1, 3, 3)` | `wrt` also contains `"graphs.strain"` | one-graph periodic batch |

`forward = Derivative(self.energy, wrt=...)`, fixed at construction; it mutates
the batch in place and returns the same object. `energy(frame)` is the one
tensor-level method: it reads the columns above (the `graphs` namespace at
`batch_size=[B]` is required), casts the geometry to `input_dtype`, and writes
`atoms.energy` `(N,)` and `graphs.energy` `(B,)`. A missing
`graphs.total_charge` / `total_spin` defaults to the neutral singlet through
`MACEEncoder.conditioning` (the one semantics the encoder's `forward` shares).

## 3. Forward Contract

### 3.1 Notation

| Symbol | Meaning | Code anchor |
|--------|---------|-------------|
| $N, E, B$ | atoms, edges, graphs | `molzoo.mace.potential.MacePotential.energy` |
| $Z_i$ | atomic number of atom $i$ | `atoms.Z` |
| $\mathbf r_i$ | position of atom $i$ | `atoms.pos` |
| $s,t$ | sender / receiver of an edge | `edge_index[:,0]`, `edge_index[:,1]` |
| $\mathbf v_e=\mathbf r_t-\mathbf r_s$ | edge vector | `molrep.embedding.geometry.edge_vectors` |
| $h_i$ | node features (irreps, `cue.ir_mul`) | `node_feats` |
| $E_0$ | per-element reference + charge/spin readout | `e0` |
| $q,\sigma$ | per-graph total charge / spin | `total_charge`, `total_spin` |

### 3.2 Reference energy and conditioning

$$
E_0 = \sum_{i} \varepsilon(Z_i) + \sum_i \mathrm{ro}\big(h_i^{(0)}\big),\qquad
h_i^{(0)} = W_{\text{emb}}\,\mathrm{onehot}(Z_i) + J(q_{b(i)}, \sigma_{b(i)})
$$

| Quantity | Shape | Code anchor |
|----------|-------|-------------|
| $\varepsilon(Z)$ | `(N,)` | `AtomicReferenceEnergy` (`molpot.heads.energy`) |
| $J(q,\sigma)$ | `(N, C)` | `JointFeatureEmbedding` (`molzoo.mace.heads`) |
| $\mathrm{ro}$ | `(N,)` | `embedding_readout` (`cuet.Linear` → `1x0e`) |

### 3.3 Edge embedding

$$
\mathbf{Y}_e = Y_{l\le l_{\max}}(\mathbf v_e),\quad
b_e = \mathrm{Bessel}(\lVert\mathbf v_e\rVert),\quad
c_e = \mathrm{PolyCutoff}(\lVert\mathbf v_e\rVert)
$$

| Quantity | Shape | Code anchor |
|----------|-------|-------------|
| $\mathbf Y_e$ | `(E, (l_max+1)^2)` | `SphericalHarmonics` |
| $b_e$ | `(E, num_bessel)` | `BesselRBF(trainable=True, normalize=False, eps=0)` |
| $c_e$ | `(E, 1)` | `PolynomialCutoff(exponent=num_polynomial_cutoff)` |

### 3.4 Residual interaction + product (× `num_interactions`)

$$
(h^{\text{msg}}, h^{\text{sc}}) = \mathrm{Interaction}_i(a, h, \mathbf Y, b, \mathrm{idx}, c),\qquad
h \leftarrow \mathrm{Product}_i(h^{\text{msg}}, h^{\text{sc}}, a)
$$

where $a=\mathrm{onehot}(Z)$. `Interaction` is the residual non-linear block
(`molzoo.mace.layers.ResidualInteraction`): source/target node-attr embeddings
are concatenated onto the radial features driving a channel-wise TP
$h\otimes Y$, neighbour-scattered, density-normalised $m/(\rho\beta+\alpha)$,
plus a residual path and a gated nonlinearity; `Product` is the symmetric
contraction (`molzoo.mace.layers.EquivariantProductBasis`, `degree=2`,
`num_elements=1`).

### 3.5 Readout and total energy

$$
E = E_0 + \sum_i \mathrm{scale\_shift}\big(\mathrm{readout}(h_i^{\text{last}})\big),\qquad
\mathbf F = -\,\partial E/\partial \mathbf r
$$

| Quantity | Shape | Code anchor |
|----------|-------|-------------|
| readout | `(N,)` | `NonLinearBiasReadout` (`molzoo.mace.heads`) |
| scale_shift | `(N,)` | `GlobalRescale` (`molpot.heads.rescale`) |
| $\mathbf F$ | `(N,3)` | `molix.autograd.Derivative` (`forward`) — one `torch.autograd.grad` |

## 4. Configuration Contract

`MaceOmol` takes a `molzoo.mace.spec.MACEOMolSpec`; every default below lives
on the spec (`wrt` / `energy_dtype` / `input_dtype` are the constructor's own
keyword arguments).

| Spec field | Meaning | Default | Constraint / note |
|-----------------|---------|---------|-------------------|
| `atomic_numbers` | element table (Z order) | — | `list[int]`; OMOL = 83 elements |
| `atomic_energies` | per-element E0 | — | tensor aligned to `atomic_numbers` |
| `r_max` | radial cutoff (Å) | `6.0` | OMOL value |
| `num_bessel` | Bessel basis count | `8` | |
| `num_polynomial_cutoff` | poly cutoff exponent | `5` | |
| `l_max` | max angular order | `3` | ≥ 1 (`MACEOMolSpec` validator); see A7 |
| `num_features` | channel multiplicity | `1024` | OMOL value |
| `num_interactions` | residual layers | `3` | schedule from `MACEOMolSpec.body_schedule()` |
| `correlation` | product body order | `2` | |
| `mlp_dim` | readout MLP width | `16` | |
| `edge_channels` | mid-layer edge irreps / radial-MLP width | `128` | OMOL value |
| `charge_classes` / `charge_offset` | charge embedding table | `201` / `100` | charge `q` → index `q+offset` |
| `spin_classes` / `spin_offset` | spin embedding table | `101` / `0` | |
| `scale` / `shift` | global rescale | `1.0` / `0.0` | |
| `wrt` | derivative keys for `forward` | `()` | `("atoms.pos",)` adds forces |
| `energy_dtype` | graph-energy accumulation dtype | `torch.float64` | `torch.float32` or `None` (model dtype) |

## 5. Reference Crosswalk

| Concept | Reference (`ACEsuit/mace`) | MolNex anchor | Status |
|---------|----------------------------|---------------|--------|
| Bessel basis | `modules.radial.BesselBasis` | `molrep.embedding.BesselRBF` | matched |
| Polynomial cutoff | `modules.radial.PolynomialCutoff` | `molrep.embedding.PolynomialCutoff` | matched |
| Residual interaction | `RealAgnosticResidualNonLinearInteractionBlock` | `molzoo.mace.layers.ResidualInteraction` | matched |
| Radial MLP | `modules.radial.RadialMLP` | `molzoo.mace.layers.RadialMLP` | matched |
| Gated nonlinearity | MACE gated nonlinearity (MACE's numerics) | `molzoo.mace.layers.GatedNonlinearity` | matched |
| Product basis | `EquivariantProductBasisBlock` (`original_mace=True`) | `molzoo.mace.layers.EquivariantProductBasis` | matched |
| Non-linear readout | `NonLinearBiasReadoutBlock` | `molzoo.mace.heads.NonLinearBiasReadout` | matched |
| Per-element E0 | `AtomicEnergiesBlock` | `molpot.heads.AtomicReferenceEnergy` | matched |
| Scale/shift | `ScaleShiftBlock` | `molpot.heads.GlobalRescale` | matched |
| Charge/spin embed | `GenericJointEmbedding` | `molzoo.mace.heads.JointFeatureEmbedding` | matched |
| Forces | `autograd.grad` | `molix.autograd.Derivative` (`torch.autograd.grad`; cuEq's fused ops are legacy `autograd.Function`s that `torch.func.grad` rejects) | matched |
| Weight import | official MACE checkpoint `state_dict` (pre-conversion) | `omol_keys` (`molzoo.mace.checkpoint`) via `molzoo.checkpoint.CheckpointRemap` in `MaceOmol.load` (after MACE's official cuEquivariance converter, run out of tree) | adapted |

## 6. MolNex Adaptations

| ID | Adaptation | Reason | Risk | Validation |
|----|------------|--------|------|------------|
| A1 | PolynomialCutoff + trainable un-normalised Bessel (`eps=0`) | OMOL variant vs standard MACE | low | `scripts/omol_port/verify_radial.py` (7e-15) |
| A2 | charge/spin via `JointFeatureEmbedding` added to node feats + into E0 | OMOL conditioning | low | `scripts/omol_port/verify_joint_embed.py` (0) |
| A3 | cuEquivariance `"O3"` group everywhere (not the MACE-compatible `O3` variant) | weights are converted into the cue-O3 twin, so O3 is the native target; the two O3 variants' CG differ only ~1.4e-8/op and the O3 twin already matches the official checkpoint (pre-conversion) to 1.5e-8 — the MACE-compatible variant would not reduce the residual and would add a dependency outside cuEquivariance | low | residual 7e-7 eV / 4.3e-6 eV/Å vs official (reimplementation accumulation, not a convention diff) — inside the 1e-4 bar |
| A4 | `forward` is `molix.autograd.Derivative(self.energy, wrt=...)`, fixed at construction — one `torch.autograd.grad`; the earlier `torch.func.grad` plan was dropped because cuEquivariance's fused ops register legacy `autograd.Function`s without `setup_context` | compile-friendly molnex contract | low | `tests/test_molzoo/test_mace/test_potential.py::TestMacePotential::test_forward_is_the_same_op_composed_outside`, `tests/test_molzoo/test_mace/test_potential.py::TestMaceOmol::test_force_loss_through_forward_reaches_the_parameters` |
| A5 | edge convention `v=pos[t]-pos[s]`, `pair_index (E,2)` (read as `edge_index`) end to end (upstream's `(2,E)` transposed away at the port boundary) | MolNex collate schema | low | `tests/test_molrep/test_embedding/test_geometry.py::TestEdgeVectors`, `tests/test_molzoo/test_mace/test_potential.py::TestMaceOmol` |
| A6 | `RadialMLP` is built at the constructor's `dtype` | fp64 without `.double()` | low | `tests/test_molzoo/test_mace/test_potential.py::TestMaceOmol` (`conftest.DTYPE`) |
| A7 | per-layer irreps from `MACEOMolSpec.body_schedule()`: hidden order `l_max-1`, mid-layer edge irreps and radial MLP at `edge_channels` (default 128), scalar last layer, one shared product | faithful OMOL weight load | medium | `tests/test_molzoo/test_mace/test_checkpoint.py::TestCheckpointRemap::test_roundtrip_restores_every_parameter_omol`, `tests/test_molzoo/test_mace/test_spec.py::TestMACEOMolSpec::test_scalar_only_angular_order_is_rejected` |

## 7. Validation Contract

### 7.1 Research Reproduction

The accepted accuracy bar is E/F within **1e-4** of official OMOL (operator
decision, 2026-06-21).

**Historical record (2026-06-21; oracles deleted in `b85d12f`, not
reproducible in-tree — Appendix A).** Full model with official weights vs the
official cueq OMOL twin, charged molecule: **7.0e-7 eV / 4.3e-6 eV/Å**
(`verify_e2e.py`, RESULT: PASS); the cueq twin vs the official OMOL checkpoint (pre-conversion): 1.5e-8 eV /
3.2e-8 eV/Å (`verify_omol_cueq_equiv.py`). Three to four orders inside the
bar. The run predates the 2026-08-07 loader fix (official `bessel_weights`
silently dropped, `bessel.freqs` left at init), so the 7e-7 eV includes that
~2.2e-7 Å⁻¹ perturbation; it is otherwise molnex's own reimplementation
accumulation, **not** a CG-convention difference (cuEquivariance O3 vs the MACE-compatible O3 variant CG
~1.4e-8/op, A3). Re-measuring upstream parity needs an out-of-tree oracle
(route per Appendix A, 2026-08-09).

**Current in-tree verification (2026-10-17).** `MOLNEX_MACE_WEIGHTS_DIR`-gated
`tests/test_molzoo/test_mace/test_checkpoint.py::TestOfficialOMolWeights`:
strict 104-parameter load through `omol_keys` (`CheckpointRemap`;
`test_official_omol_load_fills_every_learnable`) plus **upstream parity** on a
five-atom cluster (`test_official_omol_energy_and_forces_match_upstream`): the
energy and the whole force array against `mace-torch` 0.3.16's own evaluation
of the same checkpoint on the same conditioned cluster (fp64, CPU;
`omol_cluster_reference.pt` beside the weights, produced out of tree by the
asset's `reference.py`), measured ΔE 5.4e-8 eV and max |ΔF| 2.2e-7 eV/Å
against tolerances of 1e-6 eV / 1e-5 eV/Å (the ~5e-8 eV gap is the order of
the CPU thread-count scatter). The weights dump is regenerated offline by
`scripts/omol_port/convert_omol_to_cueq_state.py` (no `mace` or other upstream
import).

**Bessel frequencies (measured 2026-08-09).** The official `bessel_weights`
are bit-for-bit the fp32 evaluation of the analytic init `nπ/r_max` (upcast to
fp64); the offset from the fp64 analytic values (max 2.2120e-7 Å⁻¹ at n=7,
≤ 1 fp32 ulp per entry) is fp32 rounding of an untrained parameter, **not**
fitted drift. Doctrine unchanged: `bessel.freqs` is an `nn.Parameter` and must
be filled from the checkpoint — bit-exactness against the official surface
requires the checkpoint's fp32-rounded values, not the fp64 re-derivation.

### 7.2 Symmetry and Shape Tests

Variant-level claims live in
`tests/test_molzoo/test_mace/test_potential.py::TestMaceOmol` (the `::…` rows
below are relative to it); the shared pipeline is covered by
`tests/test_molzoo/test_mace/test_potential.py::TestMacePotential`.

| Claim | Test path | Tolerance |
|-------|-----------|-----------|
| `forward` is the op over `energy` (shared pipeline, MatPES arm) | `tests/test_molzoo/test_mace/test_potential.py::TestMacePotential::test_forward_is_the_same_op_composed_outside` | exact |
| net force ≈ 0 (shared pipeline, MatPES arm) | `tests/test_molzoo/test_mace/test_potential.py::TestMacePotential::test_net_force_vanishes` | 1e-10 |
| the charge/spin columns reach the embedding | `::test_the_total_charge_moves_the_energy` | > 1e-6 eV |
| neutral singlet default when `graphs.*` absent (spin `1`, not `0`) | `::test_missing_charge_spin_defaults_to_the_neutral_singlet`, `tests/test_molzoo/test_mace/test_encoder.py::TestMACEEncoder::test_conditioning_defaults_to_the_neutral_singlet` | 1e-10 eV |
| per-graph conditioning routes to each graph | `::test_per_graph_conditioning_routes_to_each_graph` | 1e-10 eV |
| force loss reaches parameters through `forward` | `::test_force_loss_through_forward_reaches_the_parameters` | exact |
| unfilled parameter rejected under both unexpected-key policies | `tests/test_molzoo/test_checkpoint.py::TestCheckpointRemap::test_preflight_rejection_does_not_write_parameters` | raises |
| `omol_keys` roundtrip restores every parameter | `tests/test_molzoo/test_mace/test_checkpoint.py::TestCheckpointRemap::test_roundtrip_restores_every_parameter_omol` | exact |
| edge vectors / lengths under PBC images | `tests/test_molrep/test_embedding/test_geometry.py` | exact |
| block-level vs cueq (radial/mlp/e0/joint/interaction/product/readout) | `scripts/omol_port/verify_*.py` (deleted in `b85d12f`; historical, Appendix A) | 0–7e-15 |

### 7.3 Engineering Benchmark

Not claimed. The model runs on cuEquivariance's kernels only (`molrep.cueq`;
no CPU path); throughput validation is out of scope for this spec (tracked in
`scripts/omol_port/SPEC.md`).

### 7.4 Run Log

| run_id | date | commit | dirty | dataset | config | steps | train_mae | val_mae | fwd_ms | bwd_ms | compiled | note |
|--------|------|--------|-------|---------|--------|-------|-----------|---------|--------|--------|----------|------|
| 1 | 2026-06-21 | e87a415 | 1 | n/a | l_max=1 nf=64 fp64 | 0 | n/a | n/a | n/a | n/a | no | integration test green (forward↔energy_forces 1e-8); test_molzoo 134 passed |

## 8. System Boundary

| Concern | Owner | Contract |
|---------|-------|----------|
| Neighbor list / pairs | collate / `molix.neighbor.NeighborList` | populates `pairs.pair_index` `(E,2)` and `pairs.pair_diff` `(E,3)` |
| Edge displacements | `molrep.embedding.geometry` | `edge_vectors` / `edge_lengths`; differentiable w.r.t. `pos` |
| Charge/spin inputs | dataset / collate | `graphs.total_charge`, `graphs.total_spin` (optional) |
| Building blocks | generic `molrep` / `molpot` | reused; not owned here. The MACE-only blocks (`molzoo.mace.layers`, `molzoo.mace.heads`) are owned here |
| Model graph | `molzoo.mace.encoder.MACEEncoder` | blocks + wiring; no energy, no forces |
| Energy / forces | `molzoo.mace.potential.MacePotential` | `energy(frame)` writes `atoms.energy` / `graphs.energy`; `forward = molix.autograd.Derivative(self.energy, wrt=...)` |
| Configuration | `molzoo.mace.spec.MACEOMolSpec` | torch-free pydantic preset |
| Weight import | `MaceOmol.load` → `molzoo.mace.utils.MaceArchive.read` (typed `MaceOfficialConfig`) → `MACEOMolSpec.from_official` → `molzoo.checkpoint.CheckpointRemap(omol_keys, on_unexpected="return", skip=MACE_SKIP)`, after MACE's official cuEquivariance converter (out of tree) | cueq `state_dict` → `MaceOmol`; strict on missing `nn.Parameter`s since 2026-08-07; unhoused keys returned |
| Package export | `molzoo/mace/__init__` | eager `from molzoo.mace import MaceOmol`; nothing is exported from `molzoo/__init__` |

## 9. Version Pinning

| Item | Value |
|------|-------|
| Paper | Batatia et al., NeurIPS 2022 (arXiv:2206.07697); OMol25 foundation model |
| Reference repository | `ACEsuit/mace` |
| Reference commit | not pinned to sha; `mace==0.3.16` (PyPI) used for conversion + verify. Follow-up audit to pin exact sha. |
| Dependencies | `torch==2.12.1`, `cuequivariance==0.10.0`, `cuequivariance_torch==0.10.0`, `tensordict==0.13.0` |
| Module relocation | `mace-subpackage-restructure` chain, commits `1ddd5ff..e825a51` (merged 2026-08-09): `src/molzoo/mace_omol.py` was retired into the `src/molzoo/mace/` package — config in `spec.py`, blocks in `encoder.py`, energy/forces in `potential.py`, key remap in `checkpoint.py`, the `MACEOMol` alias in `variants.py`. MACE-only `molrep` blocks moved to `molrep/interaction/mace/{conv,block,density}.py`, `molrep/readout/mace.py`, `molrep/embedding/mace.py`. Tests moved to `tests/test_molzoo/test_mace/`. Weights, hyper-parameters and numerics unchanged (§7.1 not re-run). Later superseded: `variants` and `geometry` are gone (`MaceOmol` lives in `potential.py`; edge geometry is `molrep.embedding.geometry`; tests in `test_potential.py` and `tests/test_molrep/test_embedding/test_geometry.py`), the `molrep` MACE-only modules were folded into generic ones (`molrep/interaction/{conv,residual,product_basis}.py`, `molrep/readout/scalar.py`), and `CheckpointRemap` moved to `molzoo.checkpoint`. 2026-10-16: those blocks left `molrep` again for `molzoo/mace/layers.py` and `molzoo/mace/heads.py` (a block serving one model lives in that model's package). |
| Public docs mirror | `docs/molzoo/specs/mace_omol.md` — byte-identical copy of this file; re-sync both halves on every edit |

## 10. Drift Policy

Triggers a `molzoo-auditor` pass when: (a) `MaceOmol.load` refuses a missing
learnable key, or `TestOfficialOMolWeights` sees a new unhoused key; (b) `verify_e2e.py` E/F residual regresses > 10×;
(c) cuEquivariance is bumped (cue-O3 CG basis may shift, A3);
(d) the per-layer irreps schedule (A7) changes. §6/§7 are the enforcement
surface; block-level `verify_*.py` are the source of truth for §5 `matched`
rows.

## Appendix A. Maintenance Log

- 2026-06-21: created from paper + `ACEsuit/mace@v0.3.16`; filled §1–§9 from the
  `mace-omol-port-01/02` implementation (status draft → partial). §5 rows
  `matched` per `scripts/omol_port/verify_*.py`.
- 2026-06-21: accuracy bar set to 1e-4 (operator); cue O3 meets it at
  7e-7 eV / 4.3e-6 eV/Å. Verified; chain done.
- 2026-06-21: measured cuEquivariance O3 vs the MACE-compatible O3 variant CG = 1.4e-8/op → the 7e-7 residual is
  reimplementation accumulation, not a convention diff. Dropped the MACE-compatible-O3
  pursuit entirely and removed the dead `MACEOMol(group=)` hook from
  MACEOMol / ResidualInteraction / EquivariantProductBasis (always cue O3).
- 2026-08-09: Anchors re-pointed for the `mace-subpackage-restructure` chain
  (`1ddd5ff..e825a51`) — header, §1 ownership, §2.2 / §3.1 / §3.5 code anchors,
  §5 crosswalk, §6 A4/A5/A6 verification paths, §7.2 test paths, §8 boundary,
  §9 pinning row. Three content corrections found while re-pointing:
  (a) `graphs.total_spin` absent defaults to **`1`** (closed-shell singlet
  multiplicity `2S+1 = 1`), not `0` — `MACEPotential._condition_charge_spin`
  fills `torch.ones(...)`, and `0` would index an untrained embedding row;
  (b) force units written `eV·Å` now read `eV/Å` (§6 A3, §7.1, §7.2 and this
  log); (c) forces are `torch.autograd.grad`
  (`molpot.derivation.kernels.grad_force_pass` /
  `molpot.derivation.force.autograd_forces_from_energy`), never
  `ForceDerivation(method="functorch")` / `torch.func.grad` — cuEquivariance's
  fused ops are legacy `autograd.Function`s that `torch.func.grad` rejects.
  No section added, removed or renamed; §7.4 rows untouched; no numerical
  claim changed. `docs/molzoo/specs/mace_omol.md` re-synced from this file
  (the two copies had drifted on §6 A5, §7.1 and §8).
- 2026-08-09: **Dangling anchors, not fixed here.** Every
  `scripts/omol_port/verify_*.py` cited by §5 (`matched` source of truth), §6
  A1/A2, §7.1, §7.2, §7.3 and §10 was deleted in commit `b85d12f`; only
  `README.md` and `SPEC.md` remain in that directory. The recorded numbers
  (7.0e-7 eV / 4.3e-6 eV/Å, the 0–7e-15 block-level residuals) are therefore
  no longer reproducible in-tree, and §10's drift trigger (b) — "`verify_e2e.py`
  E/F residual regresses > 10×" — cannot fire. Restoring the oracles (or
  re-homing them under `regressions/` with hard-coded goldens) is out of scope
  for `mace-subpackage-restructure-07-cleanup`, which is anchor-refresh only.
  Route: `/mol:fix` or a follow-up spec. Combined with the 2026-08-07 caveat
  in §7.1 (the parity run predates the trainable-Bessel loader fix), §7.1
  should be treated as **stale, pending re-measurement**.
- 2026-08-09: §7.1 rewritten (molzoo-auditor, operator-directed). (a) The
  mace-torch parity figures are now labelled a dated **historical record**
  (oracles deleted in `b85d12f`), and the current in-tree surface is named:
  `MOLNEX_MACE_WEIGHTS_DIR`-gated `TestOfficialOMolWeights` (strict 104-param
  load + E/F stability goldens) with the dump regenerable via
  `scripts/omol_port/convert_omol_to_cueq_state.py`. (b) The "fitted drift"
  reading of `bessel.freqs` is corrected to measurement: the official
  `bessel_weights` are **bit-for-bit** the fp32 evaluation of the analytic
  `nπ/r_max` init upcast to fp64 (max 2.2120e-7 Å⁻¹ from the fp64 values at
  n=7, ≤ 1 fp32 ulp per entry) — storage rounding of an untrained parameter,
  not training drift; the strict-loading doctrine is unchanged. A ⚠️ was
  printed (not applied) against `src/molzoo/mace/checkpoint.py`'s docstring
  ("fitted like any other weight" / "the fitted frequencies were dropped").
  No section added, removed or renamed; §7.4 untouched; §6 A1/A2 and §7.2's
  dangling `verify_*.py` anchors remain covered by the entry above.
- 2026-10-15: Refreshed to the current API, no numerical claim changed:
  `MACEOMol` / `MACEPotential` / `from_checkpoint` / `load_omol_state_dict` /
  `OMOL_REMAP` / `energy_forces` / `grad_force_pass` → `MaceOmol` /
  `MacePotential` / `MaceOmol.load` / `omol_keys` +
  `molzoo.checkpoint.CheckpointRemap` / `molix.autograd.Derivative`. §2.2 forces
  are a construction choice (`wrt=`). §4 corrected (`MACEOMolSpec` exists and is built inside; added
  `edge_channels` / `wrt` / `use_fallback` / `energy_dtype`); §6 A7 rewritten to
  the configurable `body_schedule()`. §6 / §7.2 cite tests that exist
  (`test_potential.py::TestMaceOmol` / `TestMacePotential`, `test_checkpoint.py`,
  `tests/test_molrep/test_embedding/test_geometry.py`); the old alien-checkpoint
  and OMOL net-force rows had no surviving test and were replaced. §8 `pairs`
  namespace and `molix.neighbor.NeighborList`. docs copy re-synced.
- 2026-10-17: §7.1 "current in-tree verification" is an upstream parity record
  again: `TestOfficialOMolWeights::test_official_omol_energy_and_forces_match_upstream`
  compares against `mace-torch`'s evaluation of the checkpoint on the fixture
  cluster (`omol_cluster_reference.pt`, out-of-tree `reference.py`), replacing
  the self-measured E/F goldens (`OMOL_GOLDEN_*`, test
  `test_official_omol_energy_and_forces_match_the_goldens`). Measured ΔE
  5.4e-8 eV, max |ΔF| 2.2e-7 eV/Å. The 2026-08-09 "stale, pending
  re-measurement" caveat on §7.1 is closed. docs copy re-synced.
