# MACEOMol Specification

This page is the implementation contract for `molzoo.mace.variants.MACEOMol`.
It is not a tutorial; use the MolZoo user guide for theory narrative and worked
examples.

| Field | Value |
|-------|-------|
| Module | `molzoo.mace.variants` (in the `molzoo.mace` package) |
| Entry point | `MACEOMol` — a thin, keyword-compatible alias over `molzoo.mace.potential.MACEPotential`. The constructor still takes the OMOL keywords directly; internally it builds a `molzoo.mace.spec.MACEOMolSpec`. Weights: `MACEPotential.from_checkpoint` with the `molzoo.mace.checkpoint.OMOL_REMAP` preset; `load_omol_state_dict` stays as a back-compat free function in `molzoo.mace.variants`. |
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

It **owns**: the OMOL preset of the full energy/force forward — the block graph
in `molzoo.mace.encoder`, the energy/force pipeline in
`molzoo.mace.potential`, and the `OMOL_REMAP` key dialect in
`molzoo.mace.checkpoint` that imports official weights (after
`mace.cli.convert_e3nn_cueq`), reached through `load_omol_state_dict` or
`MACEPotential.from_checkpoint`. The blocks themselves come from `molrep` /
`molpot` — some generic (`ResidualInteraction`, `EquivariantProductBasis`),
some in the MACE-only namespaces (`molrep.readout.mace`).

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
| In | `graphs.total_charge` | `(B,)` | long | Optional; per-graph total charge in units of `e`. Absent → neutral (`0`) |
| In | `graphs.total_spin` | `(B,)` | long | Optional; per-graph spin channel. Absent → `1` (singlet multiplicity `2S+1 = 1`, all electrons paired). **Not `0`** — `MACEPotential._condition_charge_spin` fills `torch.ones(...)`, and spin `0` would index an untrained embedding row (`spin_offset = 0`) and return garbage |

`edges.edge_diff` / `edges.edge_dist` are **not** consumed: `forward` recomputes
edge vectors from `atoms.pos` so the energy is differentiable w.r.t. positions.

### 2.2 Outputs

| Direction | TensorDict path | Shape | Written when | Contract |
|-----------|------------------|-------|--------------|----------|
| Out | `graphs.energy` | `(B,)` | always | Per-graph total energy = E0 + scale·shift(readout) |
| Out | `atoms.forces` | `(N, 3)` | always | `F = -∂E/∂pos` via `molpot.derivation.kernels.grad_force_pass` |

`forward` mutates `td` in place (creating the `graphs` sub-dict if absent) and
returns the same object. The raw-tensor entry point `MACEOMol.energy_forces(...)`
returns `{"energy", "forces"}` and is used by the `scripts/omol_port/verify_*.py`
block/E2E checks; the differentiable energy alone is
`MACEPotential.energy_core(positions, Z, edge_index, batch, num_graphs,
shifts=None, total_charge=None, total_spin=None)`.

## 3. Forward Contract

### 3.1 Notation

| Symbol | Meaning | Code anchor |
|--------|---------|-------------|
| $N, E, B$ | atoms, edges, graphs | `molzoo.mace.potential.MACEPotential.energy_core` |
| $Z_i$ | atomic number of atom $i$ | `atoms.Z` |
| $\mathbf r_i$ | position of atom $i$ | `atoms.pos` |
| $s,t$ | sender / receiver of an edge | `edge_index[:,0]`, `edge_index[:,1]` |
| $\mathbf v_e=\mathbf r_t-\mathbf r_s$ | edge vector | `molzoo.mace.geometry.edge_vectors` |
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
| readout | `(N,)` | `NonLinearBiasReadout` (`molrep.readout.mace`) |
| scale_shift | `(N,)` | `GlobalRescale` (`molpot.heads.rescale`) |
| $\mathbf F$ | `(N,3)` | `molpot.derivation.kernels.grad_force_pass` (`forward`) / `molpot.derivation.force.autograd_forces_from_energy` (`energy_forces`) — both `torch.autograd.grad` |

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
| Non-linear readout | `NonLinearBiasReadoutBlock` | `molrep.readout.mace.NonLinearBiasReadout` (re-exported as `molrep.readout.NonLinearBiasReadout`) | matched |
| Per-element E0 | `AtomicEnergiesBlock` | `molpot.heads.AtomicReferenceEnergy` | matched |
| Scale/shift | `ScaleShiftBlock` | `molpot.heads.GlobalRescale` | matched |
| Charge/spin embed | `GenericJointEmbedding` | `molrep.embedding.JointFeatureEmbedding` | matched |
| Forces | `autograd.grad` | `molpot.derivation.kernels.grad_force_pass` / `molpot.derivation.force.autograd_forces_from_energy` (both `torch.autograd.grad`; cuEq's fused ops are legacy `autograd.Function`s that `torch.func.grad` rejects) | matched |
| Weight import | e3nn `state_dict` | `molzoo.mace.checkpoint.OMOL_REMAP` via `load_omol_state_dict` / `MACEPotential.from_checkpoint` (after `convert_e3nn_cueq`) | adapted |

## 6. MolNex Adaptations

| ID | Adaptation | Reason | Risk | Validation |
|----|------------|--------|------|------------|
| A1 | PolynomialCutoff + trainable un-normalised Bessel (`eps=0`) | OMOL variant vs standard MACE | low | `scripts/omol_port/verify_radial.py` (7e-15) |
| A2 | charge/spin via `JointFeatureEmbedding` added to node feats + into E0 | OMOL conditioning | low | `scripts/omol_port/verify_joint_embed.py` (0) |
| A3 | cue `"O3"` group everywhere (no `O3_e3nn`) | weights are converted into the cue-O3 twin, so O3 is the native target; O3 vs O3_e3nn CG differ only ~1.4e-8/op and the O3 twin already matches e3nn to 1.5e-8 — O3_e3nn would not reduce the residual and would add an `e3nn` dep | low | residual 7e-7 eV / 4.3e-6 eV/Å vs official (reimplementation accumulation, not a convention diff) — inside the 1e-4 bar (`mace-omol-port-02` ac-003) |
| A4 | TensorDict `forward` forces via `molpot.derivation.kernels.grad_force_pass`; `energy_forces` via `molpot.derivation.force.autograd_forces_from_energy`. Both are `torch.autograd.grad`; the earlier `torch.func.grad` plan was dropped because cuEquivariance's fused ops register legacy `autograd.Function`s without `setup_context` | compile-friendly molnex contract | low | `tests/test_molzoo/test_mace/test_variants.py::TestMACEOMol::test_forward_matches_energy_forces` (1e-8 vs autograd) |
| A5 | edge convention `v=pos[t]-pos[s]`, `edge_index (E,2)` end to end (upstream's `(2,E)` transposed away at the port boundary) | MolNex collate schema | low | `tests/test_molzoo/test_mace/test_geometry.py::TestEdgeVectors`, `tests/test_molzoo/test_mace/test_variants.py::TestMACEOMol` |
| A6 | `RadialMLP` honors `config.ftype` | fp64-via-config without `.double()` | low | `tests/test_molzoo/test_mace/test_variants.py::TestMACEOMol` (fp64 fixtures) |
| A7 | per-layer irreps + edge-mid (128) hardcoded to OMOL dims | faithful OMOL weight load | medium | non-OMOL `l_max≥2`+small `num_features` unsupported; tracked in `mace-omol-port-02` |

## 7. Validation Contract

### 7.1 Research Reproduction

The accepted accuracy bar is E/F within **1e-4** of official OMOL (operator
decision, 2026-06-21).

**Historical record (2026-06-21; oracles deleted in `b85d12f`, not
reproducible in-tree — Appendix A).** Full model with official weights vs the
official cueq OMOL twin, charged molecule: **7.0e-7 eV / 4.3e-6 eV/Å**
(`verify_e2e.py`, RESULT: PASS); the cueq twin vs e3nn OMOL: 1.5e-8 eV /
3.2e-8 eV/Å (`verify_omol_cueq_equiv.py`). Three to four orders inside the
bar. The run predates the 2026-08-07 loader fix (official `bessel_weights`
silently dropped, `bessel.freqs` left at init), so the 7e-7 eV includes that
~2.2e-7 Å⁻¹ perturbation; it is otherwise molnex's own reimplementation
accumulation, **not** a CG-convention difference (O3 vs O3_e3nn CG
~1.4e-8/op, A3). Re-measuring upstream parity needs an out-of-tree oracle
(route per Appendix A, 2026-08-09).

**Current in-tree verification (2026-08-09).** `MOLNEX_MACE_WEIGHTS_DIR`-gated
`tests/test_molzoo/test_mace/test_checkpoint.py::TestOfficialOMolWeights`:
strict 104-parameter load through `OMOL_REMAP` plus E/F stability goldens on a
five-atom cluster; the weights dump is regenerated offline by
`scripts/omol_port/convert_omol_to_cueq_state.py` (no `mace`/`e3nn`). This is
a stability lock on this machine's own output — not an upstream parity claim.

**Bessel frequencies (measured 2026-08-09).** The official `bessel_weights`
are bit-for-bit the fp32 evaluation of the analytic init `nπ/r_max` (upcast to
fp64); the offset from the fp64 analytic values (max 2.2120e-7 Å⁻¹ at n=7,
≤ 1 fp32 ulp per entry) is fp32 rounding of an untrained parameter, **not**
fitted drift. Doctrine unchanged: `bessel.freqs` is an `nn.Parameter` and must
be filled from the checkpoint — bit-exactness against the official surface
requires the checkpoint's fp32-rounded values, not the fp64 re-derivation.

### 7.2 Symmetry and Shape Tests

Variant-level claims live in
`tests/test_molzoo/test_mace/test_variants.py::TestMACEOMol` (the `::…` rows
below are relative to it); the shared pipeline it aliases is covered by
`tests/test_molzoo/test_mace/test_potential.py::TestMACEPotential`.

| Claim | Test path | Tolerance |
|-------|-----------|-----------|
| `forward(td)` energy == `energy_forces` | `tests/test_molzoo/test_mace/test_variants.py::TestMACEOMol::test_forward_matches_energy_forces` | 1e-9 eV |
| `forward(td)` forces == `energy_forces` | same | 1e-8 eV/Å |
| net force ≈ 0 (translation invariance) | `::test_net_force_vanishes_on_an_isolated_molecule` | 1e-7 |
| neutral default when `graphs.*` absent | `::test_missing_charge_spin_defaults_to_neutral` | 1e-9 |
| default spin is the closed-shell singlet `1`, not `0` | `tests/test_molzoo/test_mace/test_potential.py::TestMACEPotential::test_omol_defaults_to_a_neutral_closed_shell_singlet` | exact |
| per-graph batching | `::test_batched_graphs` | 1e-9 |
| force loss reaches parameters (eval mode / through `forward`) | `::test_force_loss_reaches_parameters_in_eval_mode`, `::test_force_loss_through_forward_reaches_parameters` | exact |
| alien checkpoint rejected | `::test_load_omol_state_dict_refuses_an_alien_checkpoint` | raises |
| `OMOL_REMAP` roundtrip restores every parameter | `tests/test_molzoo/test_mace/test_checkpoint.py::TestCheckpointRemap::test_roundtrip_restores_every_parameter_omol` | exact |
| edge vectors / lengths under PBC shifts | `tests/test_molzoo/test_mace/test_geometry.py` | exact |
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
| Neighbor list / edges | collate / `molix.data.tasks.NeighborList` | populates `edges.edge_index` `(E,2)` |
| Edge displacements | `molzoo.mace.geometry` | `edge_vectors` / `edge_lengths`; differentiable w.r.t. `pos` |
| Charge/spin inputs | dataset / collate | `graphs.total_charge`, `graphs.total_spin` (optional) |
| Building blocks | `molrep.readout.mace` (+ generic `molrep` / `molpot`) | reused; not owned here |
| Model graph | `molzoo.mace.encoder.MACEEncoder` | blocks + wiring; no energy, no forces |
| Energy / forces | `molzoo.mace.potential.MACEPotential` | `energy_core` (public, compile seam) + `molpot.derivation.kernels.grad_force_pass` |
| Configuration | `molzoo.mace.spec.MACEOMolSpec` | torch-free pydantic preset |
| Weight import | `molzoo.mace.checkpoint.OMOL_REMAP` via `load_omol_state_dict` / `MACEPotential.from_checkpoint`, after `mace.cli.convert_e3nn_cueq` | cueq `state_dict` → `MACEOMol`; strict on missing `nn.Parameter`s since 2026-08-07 |
| Lazy export | `molzoo/__init__` + `molzoo/mace/__init__` | PEP 562 `__getattr__`; no eager cueq import |

## 9. Version Pinning

| Item | Value |
|------|-------|
| Paper | Batatia et al., NeurIPS 2022 (arXiv:2206.07697); OMol25 foundation model |
| Reference repository | `ACEsuit/mace` |
| Reference commit | not pinned to sha; `mace==0.3.16` (PyPI) used for conversion + verify. Follow-up audit to pin exact sha. |
| Dependencies | `torch==2.12.1`, `cuequivariance==0.10.0`, `cuequivariance_torch==0.10.0`, `tensordict==0.13.0` |
| Module relocation | `mace-subpackage-restructure` chain, commits `1ddd5ff..e825a51` (merged 2026-08-09): `src/molzoo/mace_omol.py` was retired into the `src/molzoo/mace/` package — config in `spec.py`, blocks in `encoder.py`, energy/forces in `potential.py`, key remap in `checkpoint.py`, the `MACEOMol` alias in `variants.py`. MACE-only `molrep` blocks moved to `molrep/interaction/mace/{conv,block,density}.py`, `molrep/readout/mace.py`, `molrep/embedding/mace.py`. Tests moved to `tests/test_molzoo/test_mace/`. Weights, hyper-parameters and numerics unchanged (§7.1 not re-run). |
| Public docs mirror | `docs/molzoo/specs/mace_omol.md` — byte-identical copy of this file; re-sync both halves on every edit |

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
  7e-7 eV / 4.3e-6 eV/Å. `mace-omol-port-02` ac-003 verified; chain done.
- 2026-06-21: measured O3 vs O3_e3nn CG = 1.4e-8/op → the 7e-7 residual is
  reimplementation accumulation, not a convention diff. Dropped the O3_e3nn
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
