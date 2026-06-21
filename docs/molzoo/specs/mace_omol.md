# MACEOMol Specification

This page is the implementation contract for `molzoo.mace_omol`. It is not a
tutorial; use the MolZoo user guide for theory narrative and worked examples.

| Field | Value |
|-------|-------|
| Module | `molzoo.mace_omol` |
| Entry point | `MACEOMol` (plain `nn.Module`; constructor kwargs, no pydantic Spec) |
| Paper | Batatia et al., "MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields", NeurIPS 2022. OMol25 / MACE-omol-0 foundation model. |
| arXiv | https://arxiv.org/abs/2206.07697 |
| DOI | not applicable |
| Reference implementation | `ACEsuit/mace@v0.3.16` (sha not pinned — see §9) |
| Spec status | partial |

## 1. Scope

`MACEOMol` is a native MolNex (cuEquivariance) reimplementation of the official
**MACE-omol-0** base model (`ScaleShiftMACE`: 1024 channels, `r_max=6.0`,
`l_max=3`, 3 residual interactions, product `correlation=2`, 83 elements, single
`omol` head, with `total_charge` / `total_spin` conditioning, 52.7M parameters).

It **owns**: the full energy/force forward of MACE-OMOL, assembled from generic
`molrep` / `molpot` blocks (no MACE prefix), and the `load_omol_state_dict`
converter that imports official weights (after `mace.cli.convert_e3nn_cueq`).

It does **not** own: the building blocks themselves (they live in `molrep` /
`molpot` and are reused by other models), neighbor-list construction, dataset /
collation, the training loop, or multi-head / pair-repulsion / stress outputs
(out of scope — see §6 and the `mace-omol-port` specs in `.claude/specs/`).

Unlike the other MolZoo encoders, `MACEOMol` is a full **energy/force model**,
not an encoder-only feature extractor: its `forward` writes `graphs.energy` and
`atoms.forces`, not `atoms.node_features`.

## 2. Public Contract

### 2.1 Required Inputs

| Direction | TensorDict path | Shape | Dtype | Contract |
|-----------|------------------|-------|-------|----------|
| In | `atoms.Z` | `(N,)` | long | Atomic numbers; must be present in the construction `atomic_numbers` table |
| In | `atoms.pos` | `(N, 3)` | float (`config.ftype`) | Cartesian positions; forces are `-∂E/∂pos` |
| In | `atoms.batch` | `(N,)` | long | Graph membership index `0..B-1` |
| In | `edges.edge_index` | `(E, 2)` | long | `[:,0]`=source/sender, `[:,1]`=target/receiver (MolNex convention) |
| In | `graphs.total_charge` | `(B,)` | long | Optional; per-graph total charge. Absent → neutral (0) |
| In | `graphs.total_spin` | `(B,)` | long | Optional; per-graph spin. Absent → singlet (0) |

`edges.bond_diff` / `edges.bond_dist` are **not** consumed: `forward` recomputes
edge vectors from `atoms.pos` so the energy is differentiable w.r.t. positions.

### 2.2 Outputs

| Direction | TensorDict path | Shape | Written when | Contract |
|-----------|------------------|-------|--------------|----------|
| Out | `graphs.energy` | `(B,)` | always | Per-graph total energy = E0 + scale·shift(readout) |
| Out | `atoms.forces` | `(N, 3)` | always | `F = -∂E/∂pos` via `molpot.derivation.ForceDerivation` |

`forward` mutates `td` in place (creating the `graphs` sub-dict if absent) and
returns the same object. The raw-tensor entry point `MACEOMol.energy_forces(...)`
returns `{"energy", "forces"}` and is used by the `scripts/omol_port/verify_*.py`
block/E2E checks.

## 3. Forward Contract

### 3.1 Notation

| Symbol | Meaning | Code anchor |
|--------|---------|-------------|
| $N, E, B$ | atoms, edges, graphs | `MACEOMol._compute_energy` |
| $Z_i$ | atomic number of atom $i$ | `atoms.Z` |
| $\mathbf r_i$ | position of atom $i$ | `atoms.pos` |
| $s,t$ | sender / receiver of an edge | `edge_index[0]`, `edge_index[1]` |
| $\mathbf v_e=\mathbf r_t-\mathbf r_s$ | edge vector | `_compute_energy` |
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
| $J(q,\sigma)$ | `(N, C)` | `JointFeatureEmbedding` (`molrep.embedding.node`) |
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
(`molrep.interaction.ResidualInteraction`): source/target node-attr embeddings
are concatenated onto the radial features driving a channel-wise TP
$h\otimes Y$, neighbour-scattered, density-normalised $m/(\rho\beta+\alpha)$,
plus a residual path and a gated nonlinearity; `Product` is the symmetric
contraction (`molrep.interaction.EquivariantProductBasis`, `degree=2`,
`num_elements=1`).

### 3.5 Readout and total energy

$$
E = E_0 + \sum_i \mathrm{scale\_shift}\big(\mathrm{readout}(h_i^{\text{last}})\big),\qquad
\mathbf F = -\,\partial E/\partial \mathbf r
$$

| Quantity | Shape | Code anchor |
|----------|-------|-------------|
| readout | `(N,)` | `NonLinearBiasReadout` (`molrep.readout.scalar`) |
| scale_shift | `(N,)` | `GlobalRescale` (`molpot.heads.rescale`) |
| $\mathbf F$ | `(N,3)` | `ForceDerivation` (`forward`) / `autograd.grad` (`energy_forces`) |

## 4. Configuration Contract

`MACEOMol` takes keyword-only constructor args (there is no `MACEOMolSpec`).

| Constructor arg | Meaning | Default | Constraint / note |
|-----------------|---------|---------|-------------------|
| `atomic_numbers` | element table (Z order) | — | `list[int]`; OMOL = 83 elements |
| `atomic_energies` | per-element E0 | — | tensor aligned to `atomic_numbers` |
| `r_max` | radial cutoff (Å) | `6.0` | OMOL value |
| `num_bessel` | Bessel basis count | `8` | |
| `num_polynomial_cutoff` | poly cutoff exponent | `5` | |
| `l_max` | max angular order | `3` | non-OMOL `l_max≥2` + small `num_features` unsupported (see A7) |
| `num_features` | channel multiplicity | `1024` | OMOL value |
| `num_interactions` | residual layers | `3` | per-layer irreps schedule hardcoded for ≤3 |
| `correlation` | product body order | `2` | |
| `mlp_dim` | readout MLP width | `16` | |
| `charge_classes` / `charge_offset` | charge embedding table | `201` / `100` | charge `q` → index `q+offset` |
| `spin_classes` / `spin_offset` | spin embedding table | `101` / `0` | |
| `scale` / `shift` | global rescale | `1.0` / `0.0` | |

## 5. Reference Crosswalk

| Concept | Reference (`ACEsuit/mace`) | MolNex anchor | Status |
|---------|----------------------------|---------------|--------|
| Bessel basis | `modules.radial.BesselBasis` | `molrep.embedding.BesselRBF` | matched |
| Polynomial cutoff | `modules.radial.PolynomialCutoff` | `molrep.embedding.PolynomialCutoff` | matched |
| Residual interaction | `RealAgnosticResidualNonLinearInteractionBlock` | `molrep.interaction.ResidualInteraction` | matched |
| Radial MLP | `modules.radial.RadialMLP` | `molrep.interaction.RadialMLP` | matched |
| Gated nonlinearity | e3nn `Gate` | `molrep.interaction.GatedNonlinearity` | matched |
| Product basis | `EquivariantProductBasisBlock` (`original_mace=True`) | `molrep.interaction.EquivariantProductBasis` | matched |
| Non-linear readout | `NonLinearBiasReadoutBlock` | `molrep.readout.NonLinearBiasReadout` | matched |
| Per-element E0 | `AtomicEnergiesBlock` | `molpot.heads.AtomicReferenceEnergy` | matched |
| Scale/shift | `ScaleShiftBlock` | `molpot.heads.GlobalRescale` | matched |
| Charge/spin embed | `GenericJointEmbedding` | `molrep.embedding.JointFeatureEmbedding` | matched |
| Forces | `autograd.grad` | `molpot.derivation.ForceDerivation` (`torch.func.grad`) | adapted |
| Weight import | e3nn `state_dict` | `load_omol_state_dict` (after `convert_e3nn_cueq`) | adapted |

## 6. MolNex Adaptations

| ID | Adaptation | Reason | Risk | Validation |
|----|------------|--------|------|------------|
| A1 | PolynomialCutoff + trainable un-normalised Bessel (`eps=0`) | OMOL variant vs standard MACE | low | `scripts/omol_port/verify_radial.py` (7e-15) |
| A2 | charge/spin via `JointFeatureEmbedding` added to node feats + into E0 | OMOL conditioning | low | `scripts/omol_port/verify_joint_embed.py` (0) |
| A3 | cue `"O3"` group everywhere (no `O3_e3nn`) | weights are converted into the cue-O3 twin, so O3 is the native target; O3 vs O3_e3nn CG differ only ~1.4e-8/op and the O3 twin already matches e3nn to 1.5e-8 — O3_e3nn would not reduce the residual and would add an `e3nn` dep | low | residual 7e-7 eV / 4.3e-6 eV·Å vs official (reimplementation accumulation, not a convention diff) — inside the 1e-4 bar (`mace-omol-port-02` ac-003) |
| A4 | TensorDict `forward` forces via `ForceDerivation` (`func.grad`); `energy_forces` via `autograd.grad` | compile-friendly molnex contract | low | `tests/test_molzoo/test_mace_omol.py` (1e-8 vs autograd) |
| A5 | edge convention `v=pos[t]-pos[s]`, `edge_index (E,2)→(2,E)` | MolNex collate schema | low | `tests/test_molzoo/test_mace_omol.py` |
| A6 | `RadialMLP` honors `config.ftype` | fp64-via-config without `.double()` | low | `tests/test_molzoo/test_mace_omol.py` (fp64) |
| A7 | per-layer irreps + edge-mid (128) hardcoded to OMOL dims | faithful OMOL weight load | medium | non-OMOL `l_max≥2`+small `num_features` unsupported; tracked in `mace-omol-port-02` |

## 7. Validation Contract

### 7.1 Research Reproduction

The accepted accuracy bar is E/F within **1e-4** of official OMOL (operator
decision, 2026-06-21). The full model with official OMOL weights reproduces the
official cueq OMOL twin on a charged molecule to **7.0e-7 eV / 4.3e-6 eV·Å**
(`scripts/omol_port/verify_e2e.py`, RESULT: PASS) — three to four orders inside
the bar; the cueq twin itself matches e3nn OMOL to 1.5e-8 eV / 3.2e-8 eV·Å
(`scripts/omol_port/verify_omol_cueq_equiv.py`). The 7e-7 residual is molnex's
own reimplementation accumulation, **not** a CG-convention difference: O3 vs
O3_e3nn Clebsch-Gordan differ only ~1.4e-8/op and the O3 twin already aligns
with e3nn to 1.5e-8, so the e3nn-convention group is neither used nor needed
(A3).

### 7.2 Symmetry and Shape Tests

| Claim | Test path | Tolerance |
|-------|-----------|-----------|
| `forward(td)` energy == `energy_forces` | `tests/test_molzoo/test_mace_omol.py::test_forward_matches_energy_forces` | 1e-9 eV |
| `forward(td)` forces == `energy_forces` | same | 1e-8 eV·Å |
| net force ≈ 0 (translation invariance) | `::test_forces_translation_invariant` | 1e-7 |
| neutral default when `graphs.*` absent | `::test_missing_charge_spin_defaults_to_neutral` | 1e-9 |
| per-graph batching | `::test_batched_graphs` | 1e-9 |
| block-level vs cueq (radial/mlp/e0/joint/interaction/product/readout) | `scripts/omol_port/verify_*.py` | 0–7e-15 |

### 7.3 Engineering Benchmark

Not claimed. CPU runs fall back to the naive cuEquivariance kernels
(`cuequivariance_ops_torch` absent); GPU/throughput validation is out of scope
for this spec (tracked in `scripts/omol_port/SPEC.md`).

### 7.4 Run Log

| run_id | date | commit | dirty | dataset | config | steps | train_mae | val_mae | fwd_ms | bwd_ms | compiled | note |
|--------|------|--------|-------|---------|--------|-------|-----------|---------|--------|--------|----------|------|
| 1 | 2026-06-21 | e87a415 | 1 | n/a | l_max=1 nf=64 fp64 | 0 | n/a | n/a | n/a | n/a | no | integration test green (forward↔energy_forces 1e-8); test_molzoo 134 passed |

## 8. System Boundary

| Concern | Owner | Contract |
|---------|-------|----------|
| Neighbor list / edges | collate / `NeighborList` | populates `edges.edge_index` `(E,2)` |
| Charge/spin inputs | dataset / collate | `graphs.total_charge`, `graphs.total_spin` (optional) |
| Building blocks | `molrep` / `molpot` | reused; not owned here |
| Forces | `molpot.derivation.ForceDerivation` | `F=-∂E/∂pos` from energy closure |
| Weight import | `load_omol_state_dict` + `mace.cli.convert_e3nn_cueq` | cueq `state_dict` → `MACEOMol` |
| Lazy export | `molzoo/__init__` | PEP 562 `__getattr__`; no eager cueq import |

## 9. Version Pinning

| Item | Value |
|------|-------|
| Paper | Batatia et al., NeurIPS 2022 (arXiv:2206.07697); OMol25 foundation model |
| Reference repository | `ACEsuit/mace` |
| Reference commit | not pinned to sha; `mace==0.3.16` (PyPI) used for conversion + verify. Follow-up audit to pin exact sha. |
| Dependencies | `torch==2.12.1`, `cuequivariance==0.10.0`, `cuequivariance_torch==0.10.0`, `tensordict==0.13.0` |
| Public docs mirror | `docs/molzoo/specs/mace_omol.md` |

## 10. Drift Policy

Triggers a `molzoo-auditor` pass when: (a) `load_omol_state_dict` reports
missing learnable keys; (b) `verify_e2e.py` E/F residual regresses > 10×;
(c) cuEquivariance is bumped (cue-O3 CG basis may shift, A3);
(d) the per-layer irreps schedule (A7) changes. §6/§7 are the enforcement
surface; block-level `verify_*.py` are the source of truth for §5 `matched`
rows.

## Appendix A. Maintenance Log

- 2026-06-21: created from paper + `ACEsuit/mace@v0.3.16`; filled §1–§9 from the
  `mace-omol-port-01/02` implementation (status draft → partial). §5 rows
  `matched` per `scripts/omol_port/verify_*.py`.
- 2026-06-21: accuracy bar set to 1e-4 (operator); cue O3 meets it at
  7e-7 eV / 4.3e-6 eV·Å. `mace-omol-port-02` ac-003 verified; chain done.
- 2026-06-21: measured O3 vs O3_e3nn CG = 1.4e-8/op → the 7e-7 residual is
  reimplementation accumulation, not a convention diff. Dropped the O3_e3nn
  pursuit entirely and removed the dead `MACEOMol(group=)` hook from
  MACEOMol / ResidualInteraction / EquivariantProductBasis (always cue O3).
