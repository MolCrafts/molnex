# MACE-OFF / MACE-LES Specification

This page is the implementation contract for `molzoo.mace.MaceOff` — a native
reimplementation of the official MACE-OFF organic force fields — and its
long-range extension `molzoo.mace.MaceLes` (MACE-OFF + Latent Ewald Summation). It
is not a tutorial; use the MolZoo user guide for narrative and worked examples.

| Field | Value |
|-------|-------|
| Module | `molzoo.mace.potential` |
| Entry point | `MaceOff(spec: MACEOffSpec, *, wrt=(), energy_dtype=torch.float64, input_dtype=None)` and `MaceLes(spec: MACELesSpec, ...)`; `.load(path, config=None, *, wrt, energy_dtype, input_dtype)` reads the archive into a typed `MaceOfficialConfig`, builds the spec with `from_official` (the family's rejection rules: `RealAgnostic` stack, no ZBL / transform, charge-only LES) and renames the cuEq-converted state through `off_keys` / `les_keys` (`molzoo.mace.checkpoint`) with `molzoo.checkpoint.CheckpointRemap(keys, skip=MACE_SKIP)` |
| Paper | Kovács et al., *MACE-OFF: Short-Range Transferable Machine Learning Force Fields for Organic Molecules*, JACS 2025; Cheng, *Latent Ewald summation for machine learning of long-range interactions*, npj Comput. Mater. 11, 80 (2025) |
| arXiv / DOI | https://arxiv.org/abs/2312.15211 · https://doi.org/10.1038/s41524-025-01577-7 |
| Reference implementation | `ACEsuit/mace` v0.3.17 (`ScaleShiftMACE`, `MACELES`); `ChengUCB/les` 0.2.0 (`les/module/ewald.py`) |
| Reference checkpoints | `MACE-OFF24_medium.model`; `MACELES-OFF_small.model` |
| Spec status | partial (§7.1 filled) |

**Related modules (out of scope for this file):**

| Module | Role |
|--------|------|
| `molzoo.mace.encoder.MACEEncoder` | Shared block graph every MACE product inherits |
| `molzoo.mace.potential.MaceMatpes` / `MaceOmol` | Density-normalised / OMOL products; see `mace_matpes.md`, `mace_omol.md` |
| `molpot.potentials.elec.les.LatentEwaldEnergy` | The LES energy of latent charges (isolated pair sum or reciprocal sum) |
| `molpot.potentials.gto.kspace.KSpaceGrid` | Fixed-capacity reciprocal half-box shared with MACE-POLAR |
| `molix.autograd.Derivative` | Forces / virial (`forward`) |

## 1. Scope

`MaceOff` is a **full energy/force model**. It owns the embedding →
interaction → product → readout stack of `ScaleShiftMACE` with interactions
normalised by the training set's average neighbour count, the frozen `E0`
table and the global scale/shift. `MaceLes` adds one latent-charge readout per
layer and the LES energy of those charges — a graph-level term added to
`graphs.energy` after the per-atom scatter in its own `energy`, so
`atoms.energy` holds the short-range part alone.

Neither owns neighbour-list construction, the derivative op, the
official → cueq weight conversion (out of tree, MACE's converter), training or
MD integration. Only the charge-only LES is ported: latent dipoles,
quadrupoles, induced charges/dipoles, the atomwise LES head, fixed charges and
ε_r scaling are refused at load time.

## 2. Public Contract

### 2.1 Required Inputs

| Direction | Path / argument | Shape | Contract |
|-----------|-----------------|-------|----------|
| In | `("atoms", "Z")` / `Z=` | `(N,)` int64 | inside the element table |
| In | `("atoms", "pos")` / `positions=` | `(N, 3)` | Å |
| In | `("atoms", "batch")` / `batch=` | `(N,)` int64 | graph id |
| In | `("pairs", "pair_index")` / `edge_index=` | `(E, 2)` | `[:, 0]` source, `[:, 1]` target |
| In | `("pairs", "pair_diff")` / `edge_diff=` | `(E, 3)` | imaged `pos[t]-pos[s]` (+ image), Å; required |
| In | `("graphs", "cell")` / `cell=` | `(1, 3, 3)` | `MaceLes` only: selects the periodic LES sum (single system; `bind_cell` first) |
| In | `("atoms", "mask")` / `atom_mask=` | `(N,)` | optional; ghost atoms get zero energy **and** zero latent charge |

### 2.2 Outputs

| Path / return | Shape | Contract |
|---------------|-------|----------|
| `("graphs", "energy")` | `(B,)` | eV; `MaceLes` includes the LES term |
| `("atoms", "energy")` | `(N,)` | eV; the short-range per-atom part |
| `("atoms", "forces")` | `(N, 3)` | `wrt=("atoms.pos",)` |
| `("graphs", "virial")` | `(1, 3, 3)` | `wrt=("atoms.pos", "graphs.strain")`; the LES cell dependence is differentiated too |

## 3. Forward Contract

### 3.1 Notation

| Symbol | Meaning | Code anchor |
|--------|---------|-------------|
| \(\bar n\) | `avg_num_neighbors` | `molzoo.mace.layers._AverageNormalised` |
| \(q_i\) | latent charge, e | `MaceLes.energy` (`_sum_readouts(charge_readouts) * les_output_scale`) |
| \(\sigma, \Delta, c\) | LES width, grid spacing, `norm_factor` | `LatentEwaldEnergy` |

### 3.2 Pipeline

```
E = E0[Z] + scale · Σ_l readout_l(h_l) + shift·N  [+ E_LES(q) for MaceLes]

edge_feats = Bessel(r) · u(r)           8 trainable Bessel channels, PolynomialCutoff(p=5)
edge_attrs = Y_l(r̂)                     l = 0..3
h_0 = Linear(one_hot(Z))
layer 0  AgnosticInteraction (OFF24)  |  AgnosticResidualInteraction (OFF23, LES-OFF)
layer i  AgnosticResidualInteraction
         m_i ← Linear(Σ_j TP(h_j, Y(r̂_ij); R(r_ij))) / n̄
readouts LinearReadout (layers 0..L-2), NonLinearReadout(MLP 16) on the last
```

LES (charge-only, `les/module/ewald.py`):

```
q_i = les_output_scale · Σ_l charge_readout_l(h_l)     (same readout kinds as the energy)
isolated:  E = ½ Σ_{i≠j} q_i q_j erf(r_ij/(√2σ))/r_ij · c/2π
periodic:  E = c/Ω Σ_{k≠0,|k|<2π/Δ} exp(-σ²k²/2)/k² |S(k)|² − c Σ q_i²/(σ(2π)^{3/2})
```

## 4. Configuration Contract

| Ctor field | OFF24-M | LES-OFF-S | Meaning |
|------------|---------|-----------|---------|
| `r_max` | 6.0 | 4.5 | Å |
| `num_features` / `max_hidden_l` | 128 / 1 | 192 / 1 | node state `F x0e + F x1o` |
| `num_interactions` | 2 | 2 | |
| `residual_first` | `False` | `True` | first block residual |
| `correlation` | 3 | from checkpoint | body order |
| `avg_num_neighbors` | from checkpoint | 15.652 | message divisor |
| `les_sigma` / `les_dl` / `les_norm_factor` | — | 1.0 / 2.0 / 90.0474 | LES Ewald |
| `les_output_scale` | — | 1.0 (absent on the dump → LES default) | |

## 5. Reference Crosswalk

| Reference component | MolNex anchor | Status |
|---------------------|---------------|--------|
| `RealAgnosticInteractionBlock` | `molzoo.mace.layers.AgnosticInteraction` | matched |
| `RealAgnosticResidualInteractionBlock` | `molzoo.mace.layers.AgnosticResidualInteraction` | matched |
| `EquivariantProductBasisBlock` (`use_sc` iff residual first or i > 0) | `molzoo.mace.layers.EquivariantProductBasis` | matched |
| `LinearReadoutBlock` / `NonLinearReadoutBlock` | `molzoo.mace.heads` | matched |
| `MACELES.les_readouts` | `MaceLes.charge_readouts` (`les_keys`) | matched |
| `les.module.ewald.Ewald` (legacy, charges) | `molpot.potentials.elec.les.LatentEwaldEnergy` | adapted (A2, A3) |

## 6. MolNex Adaptations

| ID | Adaptation | Reason | Risk | Validation |
|----|------------|--------|------|------------|
| A1 | The four MACE interaction blocks share one base; normalisation (density head or `1/n̄`) and skip placement are orthogonal mixins | one owner per concern | low | `tests/test_molrep/test_interaction/test_density.py::TestAgnosticInteraction`, `::TestAgnosticResidualInteraction` |
| A2 | Periodic LES sums a fixed-capacity half box (`KSpaceGrid`), sized once by `bind_cell` | LES legacy sizes the k-grid from `cell.item()` per call — a host sync and a data-dependent shape | none (same k-set) | `tests/test_molpot/test_potentials/test_elec/test_les.py` |
| A3 | Strict `|k| < 2π/Δ` and no extra per-axis box | the ball implies LES's `|n_i| ≤ ⌊‖a_i‖/Δ⌋`; boundary equality is measure-zero | negligible | §7.1 |
| A4 | LES energy inside `energy` (the one method the op differentiates) | one graph for E, F, virial | low | `tests/test_molzoo/test_mace/test_potential.py::TestMaceLes::test_les_long_range_lands_on_graphs_energy` |

## 7. Validation Contract

### 7.1 Research Reproduction

Native (GH200, fp64, cuEquivariance kernels) vs official (mace-torch 0.3.17,
fp64), same structure — a 96-atom periodic PEO subset of the rho363 start cell —
molnex `molix.md.NeighborList` against MACE's own neighbour list
(oracle scripts out of tree, `.build-alloc/zoo-parity/offparity/`):

| Model | ΔE/atom (eV) | max ΔF (eV/Å) | max Δσ (eV/Å³) |
|---|---|---|---|
| MACE-OFF24-M | 1.5e-13 | 7.9e-15 | 1.1e-18 |
| MACE-LES-OFF-S | 1.8e-8 | 1.5e-7 | 4.3e-11 |

At 4600 atoms in fp32 on GH200 (rho363 probe) forces and stress agree with the
official model to fp32 rounding (max |F| 4.083197 vs 4.083196 eV/Å); the
8–9 eV total-energy gap there is the official model's fp32 accumulation
(molnex sums the graph energy in fp64), not a model difference.

In tree (2026-10-17; `MOLNEX_MACE_WEIGHTS_DIR`-gated, `slow`):
`tests/test_molzoo/test_mace/test_checkpoint.py::TestOfficialOffWeights` and
`::TestOfficialLesWeights`. `MaceOff.load` / `MaceLes.load` read upstream's own
config json (`extract_config_mace_model` output — `interaction_cls_first` /
`interaction_cls`, no block list — `off24_medium_config.json` /
`les_off_small_config.json`, written out of tree by the asset's
`convert_config.py` beside the cueq dumps) with strict key coverage
(`test_official_off_load_fills_every_learnable`,
`test_official_les_load_fills_every_learnable`), and the loaded surface is
compared with `mace-torch`'s evaluation of the same checkpoint on the five-atom
fixture cluster, energy and whole force array (`off24_cluster_reference.pt`,
`les_cluster_reference.pt`; `test_official_off_energy_and_forces_match_upstream`,
`test_official_les_energy_and_forces_match_upstream`); CPU, fp64,
`OMP_NUM_THREADS=1`:

| Model | ΔE (eV) | max ΔF (eV/Å) | tolerance (eV, eV/Å) |
|---|---|---|---|
| MACE-OFF24-M | 0 | 5.5e-15 | 1e-6 / 1e-5 |
| MACE-LES-OFF-S | 2.4e-7 | 2.0e-6 | 1e-6 / 1e-5 |

LES-OFF-S is fp32-trained: upstream evaluates its `.double()`-ed pickle, whose
non-learnable constants keep their fp32 rounding (Bessel prefactor, e3nn's
Clebsch–Gordan and contraction tables) where molnex rebuilds them in fp64. An
fp64 re-conversion of the cueq dump reproduces the gap to the digit, and
copying upstream's fp32-rounded prefactor alone moves it (ΔE 1.2e-7, max ΔF
4.0e-6) without removing it — the sum of several such constants, the same
order as the 96-atom row above.

### 7.2 Symmetry and Shape Tests

`tests/test_molzoo/test_mace/test_potential.py::TestShortRangeAssembly` (block
types, readouts, LES = short range + LES of the summed charges);
`tests/test_molpot/test_potentials/test_elec/test_les.py` (isolated pair sum,
large-box limit, lattice-translation invariance, compile).

### 7.3 Engineering Benchmark

rho363 (4600-atom PEO, 363 K NPT, GH200) — see §7.4.

### 7.4 Run Log

| run_id | date | commit | dirty | dataset | config | steps | train_mae | val_mae | fwd_ms | bwd_ms | compiled | note |
|--------|------|--------|-------|---------|--------|-------|-----------|---------|--------|--------|----------|------|

## 8. System Boundary

| Concern | Owner |
|---------|-------|
| Neighbour list | caller / `molix.neighbor.NeighborList` / `molix.md.NeighborList` |
| Blocks | `molzoo.mace.layers`, `molzoo.mace.heads` |
| LES energy | `molpot.potentials.elec.les.LatentEwaldEnergy` |
| Weight conversion | MACE's cuEquivariance converter (out of tree) |
| Weight import | `MaceArchive.read` → `MACEOffSpec` / `MACELesSpec.from_official` → `CheckpointRemap(off_keys / les_keys, skip=MACE_SKIP)` |

## 9. Version Pinning

| Item | Value |
|------|-------|
| Reference repository | `ACEsuit/mace` v0.3.17 (oracle venv out of tree: `.build-alloc/mace-weights/.venv-les`, 0.3.16 wheels + 0.3.17 source overlay); `les` 0.2.0 |
| Checkpoints | `MACE-OFF24_medium.model`, `MACELES-OFF_small.model` (`_converted` variant) |
| Public docs mirror | `docs/molzoo/specs/mace_off.md` — byte-identical copy |

## 10. Drift Policy

A change to the interaction normalisation, the first-layer block kind, the
readout placement, the LES charge readouts or the LES k-set must update this
file in the same change and re-run §7.1.

## Appendix A. Maintenance Log

- 2026-10-15: Created with the MACE-OFF official loader and the MACE-LES port
  (charge-only LES, compile-safe reciprocal sum).
- 2026-10-17: §7.1 gains the in-tree official-weight tests
  (`TestOfficialOffWeights`, `TestOfficialLesWeights`): strict `load` from
  upstream's config json and upstream parity on the fixture cluster. Loading
  that json exposed that `MaceOfficialConfig` read only the pickled-module
  block list (`interaction_blocks`); it now also reads upstream's
  `interaction_cls_first` / `interaction_cls` pair. docs copy re-synced.
