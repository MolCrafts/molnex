# MACE-MatPES Specification

This page is the implementation contract for `molzoo.MACEMatpes` — a native
reimplementation of the official `MACE-matpes-*-0` foundation models. It is not a
tutorial; use the MolZoo user guide for narrative and worked examples.

| Field | Value |
|-------|-------|
| Module | `molzoo.mace.variants` (in the `molzoo.mace` package) |
| Entry point | `MACEMatpes` — a thin, keyword-compatible alias over `molzoo.mace.potential.MACEPotential` configured by `molzoo.mace.spec.MACEMatpesSpec`. Weights: `MACEPotential.from_checkpoint` with the `molzoo.mace.checkpoint.MATPES_REMAP` preset; `load_matpes_state_dict` stays as a back-compat free function in `molzoo.mace.variants`. |
| Paper | Batatia et al., *MACE*, NeurIPS 2022; Batatia et al., *A foundation model for atomistic materials chemistry* (MACE-MP-0) |
| arXiv | https://arxiv.org/abs/2206.07697 · https://arxiv.org/abs/2401.00096 |
| Dataset paper | Kaplan et al., *MatPES* — https://arxiv.org/abs/2503.04070 |
| Reference implementation | `ACEsuit/mace` v0.3.16 (`ScaleShiftMACE`) |
| Reference checkpoint | `MACE-matpes-r2scan-omat-ft.model` (ACEsuit/mace-foundations, tag `mace_matpes_0`, ASL licence) |
| Spec status | partial |

**Related modules (out of scope for this file):**

| Module | Role |
|--------|------|
| `molzoo.mace.research.MACE` | Encoder-only trainable MACE; see `mace.md` |
| `molzoo.mace.variants.MACEOMol` | OMOL foundation model (charge/spin conditioned); see `mace_omol.md` |
| `molzoo.mace.encoder.MACEEncoder` | Shared block graph both variants inherit (`MACEPotential` subclasses it) |
| `molix.data.tasks.NeighborList` | Cutoff graph construction (PBC minimum image) |

## 1. Scope

`molzoo.MACEMatpes` is a **full energy/force model**. It owns:

- the embedding → interaction → product → readout stack of `ScaleShiftMACE`
  with density-normalised interactions,
- the frozen per-element reference energy `E0` and the global scale/shift,
- the ZBL pair-repulsion term,
- strict import of an official cueq-converted `state_dict`.

It does **not** own:

- neighbour-list construction or PBC shift vectors (caller supplies both),
- the e3nn → cueq weight conversion (out of tree, `mace.cli.convert_e3nn_cueq`),
- training loops, losses, or MD integration.

## 2. Public Contract

### 2.1 Required Inputs

| Direction | Path / argument | Shape | Dtype | Contract |
|-----------|-----------------|-------|-------|----------|
| In | `("atoms", "Z")` / `Z=` | `(N,)` | int64 | Atomic numbers; must all be in the model's z-table |
| In | `("atoms", "pos")` / `positions=` | `(N, 3)` | float | Cartesian positions, Å |
| In | `("atoms", "batch")` / `batch=` | `(N,)` | int64 | Graph membership |
| In | `("edges", "edge_index")` | `(E, 2)` | int64 | Col 0 = source, col 1 = target |
| In | `edge_index=` (raw API) | `(E, 2)` | int64 | `[:, 0]` = source, `[:, 1]` = target (repo edge convention) |
| In | `("edges", "shifts")` / `shifts=` | `(E, 3)` | float | Optional PBC shift `unit_shifts @ cell` |

### 2.2 Outputs

| Direction | Path / return | Shape | Contract |
|-----------|---------------|-------|----------|
| Out | `("graphs", "energy")` / `["energy"]` | `(B,)` | Total energy, eV |
| Out | `("atoms", "forces")` / `["forces"]` | `(N, 3)` | `-∂E/∂pos`, eV/Å |

## 3. Forward Contract

### 3.1 Notation

| Symbol | Meaning | Code anchor |
|--------|---------|-------------|
| \(N, E, B\) | atoms, edges, graphs | batch sizes |
| \(F\) | `num_features` = 128 | `hidden_irreps` 0e multiplicity (`molzoo.mace.spec.MACEMatpesSpec`) |
| \(\ell_{\max}\) | `l_max` = 3 | spherical harmonics / TP |
| \(\nu\) | `correlation` = 3 | symmetric-contraction degree |
| \(\rho_i\) | learned edge density | `molrep.interaction.mace.density.DensityInteraction._message` |
| \(E(\mathbf r)\) | per-graph energy `(B,)`, eV | `molzoo.mace.potential.MACEPotential.energy_core` |
| \(\mathbf v_e\) | edge displacement (with PBC shifts) | `molzoo.mace.geometry.edge_vectors` / `edge_lengths` |

### 3.2 Pipeline

```
E = E0[Z] + scale · ( V_ZBL + Σ_l readout_l(h_l) )

r_ij, cutoff u(r_ij)                     PolynomialCutoff(p=5) on the RAW r
r̃_ij = Agnesi(r_ij; Z_i, Z_j)            element-pair radial transform
edge_feats = Bessel(r̃_ij) · u(r_ij)      10 trainable Bessel channels
edge_attrs = Y_l(r_ij vector)            l = 0..3

h_0 = Linear(one_hot(Z))                                    (89x0e -> 128x0e)
layer 0  DensityInteraction          -> EquivariantProductBasis(use_sc=False)
         readout_0 = LinearReadout(128x0e+128x1o)
layer 1  DensityResidualInteraction  -> EquivariantProductBasis(use_sc=True)
         readout_1 = NonLinearReadout(128x0e, MLP 16)
```

Density normalisation (the MatPES/MP-family divergence from stock MACE):

\[
\rho_i = \sum_{j \in \mathcal{N}(i)} \tanh\!\big(\mathrm{MLP}(\mathbf{e}_{ij})^2\big),
\qquad
\mathbf{m}_i \leftarrow \frac{\mathrm{Linear}(\mathbf{m}_i)}{\rho_i + 1}
\]

replacing MACE's fixed `avg_num_neighbors` divisor. `apply_cutoff=True` in the
reference config, so the envelope is folded into `edge_feats` and the
interaction receives `cutoff=None`.

**Ordering that matters:** the cutoff is evaluated on the **untransformed**
distance and the Bessel basis on the **transformed** one. Swapping them silently
changes every edge feature.

## 4. Configuration Contract

| Ctor field | MatPES-r2SCAN value | Meaning |
|------------|--------------------|---------|
| `atomic_numbers` | 89 elements | z-table, checkpoint order |
| `atomic_energies` | 89 floats | frozen `E0`, same order |
| `r_max` | 6.0 | radial cutoff, Å |
| `num_bessel` | 10 | trainable Bessel channels |
| `num_polynomial_cutoff` | 5 | envelope exponent (also ZBL's) |
| `l_max` | 3 | spherical-harmonics order |
| `num_features` | 128 | scalar multiplicity |
| `max_hidden_l` | 1 | node state is `128x0e+128x1o` |
| `num_interactions` | 2 | ≥ 2 (first + last layer differ) |
| `correlation` | 3 | body order |
| `mlp_dim` | 16 | `MLP_irreps` |
| `radial_mlp` | `[64, 64, 64]` | conv-weight MLP widths |
| `scale` / `shift` | 0.7735790334431056 / 0.0 | `atomic_inter_scale/shift` |
| `use_fallback` | `False` (default; fused kernels) — pass `True` on CPU / without the `cuequivariance-ops-torch` wheel | forces are always autograd here, so the functorch reason for the fallback never applies; fused is ~36x faster per MD step |

## 5. Reference Crosswalk

| Reference component | MolNex anchor | Status |
|---------------------|---------------|--------|
| `LinearNodeEmbeddingBlock` | `molzoo.mace.encoder.MACEEncoder.node_embedding` (`cuet.Linear`) | matched |
| `RadialEmbeddingBlock.bessel_fn` | `molrep.embedding.radial.BesselRBF(normalize=False, eps=0, trainable=True)` | matched |
| `AgnesiTransform` | `molrep.embedding.radial.AgnesiTransform` | matched |
| `PolynomialCutoff` | `molrep.embedding.cutoff.PolynomialCutoff` | matched |
| `ZBLBasis` | `molpot.potentials.repulsion.ZBLRepulsion` | matched |
| `AtomicEnergiesBlock` | `molpot.heads.energy.AtomicReferenceEnergy` (Z-indexed) | adapted (A1) |
| `RealAgnosticDensityInteractionBlock` | `molrep.interaction.mace.density.DensityInteraction` | matched |
| `RealAgnosticDensityResidualInteractionBlock` | `molrep.interaction.mace.density.DensityResidualInteraction` | matched |
| `EquivariantProductBasisBlock` | `molrep.interaction.product_basis.EquivariantProductBasis` (shared with non-MACE models — **not** moved into the `mace` namespace) | matched |
| `LinearReadoutBlock` | `molrep.readout.mace.LinearReadout` | matched |
| `NonLinearReadoutBlock` | `molrep.readout.mace.NonLinearReadout` | matched |
| `e3nn.nn.FullyConnectedNet` | `molrep.embedding.mlp.MomentNormalizedMLP` (generic; not moved) | matched |
| `ScaleShiftBlock` | `molpot.heads.rescale.GlobalRescale` | matched |
| `get_outputs` force path | autograd, two entries: `molpot.derivation.kernels.grad_force_pass` on the batch path (`MACEPotential.forward`) and `molpot.derivation.force.autograd_forces_from_energy` on the raw path (`MACEMatpes.energy_forces`) | matched |

`molrep.interaction.density` and `molrep.readout.scalar` still exist as
deprecated re-export shims (`molrep.readout.product` likewise). New code must
import from the `mace` namespaces above; the shims are scheduled for removal.

## 6. MolNex Adaptations

| ID | Adaptation | Reason | Risk | Validation |
|----|------------|--------|------|------------|
| A1 | `E0` stored Z-indexed, not element-table-indexed | encoder takes raw `Z` | low | §7.1 parity |
| A2 | Covalent radii inlined (Cordero 2008) rather than read from `molpy.Element` | keeps a core `molrep` block free of the compiled molrs extension; molpy stores radii in fp32 | low | table asserted equal to the checkpoint buffer to 0.0 |
| A3 | cuEq `O3` group, not MACE's `O3_e3nn` | same route the OMOL port validated; CG conventions differ by ~1e-8/op | low | §7.1 parity |
| A4 | Weight import consumes a cueq-converted `state_dict`; conversion runs out of tree | molnex must not import `mace-torch` | low | strict loader, §7.2 |
| A5 | `skip_tp` is rebuilt when the module dtype changes | `cuet.FullyConnectedTensorProduct` bakes its precision in at construction, so `.double()` otherwise raises | low | `tests/test_molrep/test_interaction/test_mace/test_density.py::TestDensityInteraction::test_skip_tp_weight_survives_dtype_change` |
| A6 | `skip_tp` pinned to cuEq `method="naive"` (`molrep.interaction.mace.density.SKIP_TP_METHOD`) | cuEq's default `fused_tp` is 19x slower on a one-hot `89x0e` second operand (31.9 vs 1.8 ms); `naive` is also what MACE's own converted cueq model runs | low | output identical to `fused_tp` at 1.1e-15; §7.1 |
| A7 | Energy and forces come from one forward differentiated in place, not a re-run closure | MACE's `get_outputs` shape; the closure form cost two full forwards per step | none (same math) | §7.2 forward↔energy_forces tests |

**Known limitation.** cuEquivariance freezes `math_dtype` at construction, and
`EquivariantProductBasis` (shared with `MACEOMol`) is not rebuilt on `.double()`.
A model built under the default fp32 and then `.double()`-d therefore still
contracts in float32 (~1e-8 eV noise). For fp64 work call
`molix.config.set_precision("fp64")` **before** construction.

## 7. Validation Contract

### 7.1 Research Reproduction

Native model vs the official `MACE-matpes-r2scan-0` (e3nn, fp64), identical
neighbour lists, on CPU:

| System | ΔE/atom (eV) | max ΔF (eV/Å) |
|---|---|---|
| Si diamond 2×1×1 (pbc) | 3.4e-08 | 6.6e-08 |
| NaCl rocksalt (pbc) | 3.6e-08 | 8.6e-10 |
| Fe bcc (pbc) | 4.3e-07 | 2.7e-08 |
| H₂O cluster (no pbc) | 3.2e-08 | 2.0e-06 |
| TiO₂ rutile (pbc) | 1.0e-07 | 8.1e-08 |

Worst case **4.3e-07 eV/atom, 2.0e-06 eV/Å** — three orders inside the 1e-4 bar
the OMOL port set, and the same magnitude as that port's residual (7e-7 eV /
4.3e-6 eV/Å). The residual is float64 accumulation order between the e3nn and
cueq contraction paths, not a modelling difference.

**Along a real NVE trajectory** (`mace-r2san-gh200`, 193-atom water/H3O+ box,
200 steps of 0.5 fs, fp64, fused cuEq kernels on GH200), against the official
model driven through the *same* integrator, neighbour list and initial
conditions (`mace-r2san-torch-gh200`):

| Quantity | Worst over 200 steps |
|---|---|
| ΔE_pot/atom (both arms' own energies) | 1.8e-07 eV |
| Δ\|F\| | 6.4e-06 eV/Å |
| Δ position | 9.5e-07 Å (= float32 trajectory-storage resolution) |
| E_tot drift | -0.0119 meV/atom, **identical** in both arms |
| Mean T | 321.7 K, range 281.3-353.8 K, **identical** in both arms |

Re-evaluating the official model on molnex's own frames gives 2.3e-07 eV/atom
and 5.0e-05 eV/Å; that force figure is dominated by the trajectory's float32
position storage (≈ Hessian × 1e-6 Å), not by the model.

**Against MACE's own cueq model** — the same kernels rather than the e3nn
reference — agreement is essentially bit-level, which isolates the numbers above
as e3nn-vs-cueq contraction order rather than a modelling difference:

| Comparison (193-atom box, fp64, GH200) | ΔE | max ΔF |
|---|---|---|
| molnex vs `convert_e3nn_cueq(official)` | 4.5e-13 eV | 3.3e-14 eV/Å |
| `skip_tp` naive vs fused_tp, whole model | 2.3e-13 eV | 5.6e-15 eV/Å |

### 7.2 Symmetry and Shape Tests

Variant-level claims live in
`tests/test_molzoo/test_mace/test_variants.py::TestMACEMatpes` (the `::…` rows
below are relative to it); the shared pipeline they alias is covered by
`tests/test_molzoo/test_mace/test_potential.py::TestMACEPotential`.

| Claim | Test path | Tolerance |
|-------|-----------|-----------|
| `forward(td)` == `energy_forces` | `tests/test_molzoo/test_mace/test_variants.py::TestMACEMatpes::test_forward_matches_energy_forces` | 1e-9 eV / 1e-8 eV/Å |
| energy rotation invariance | `::test_energy_is_rotation_invariant` | 1e-9 eV |
| force equivariance `F(Rx) = R F(x)` | `::test_forces_rotate_with_the_system` | 1e-8 |
| `F = -dE/dpos` vs finite differences | `::test_forces_match_finite_differences` | 1e-5 |
| net force ≈ 0 | `::test_net_force_vanishes_on_an_isolated_cluster` | 1e-8 |
| energy extensive over separated graphs | `::test_energy_is_extensive_over_separated_graphs` | 1e-9 |
| out-of-table `Z` rejected | `::test_rejects_atomic_numbers_outside_the_table`, `::test_forward_rejects_an_element_outside_the_table` | raises |
| alien checkpoint rejected | `::test_load_matpes_state_dict_refuses_an_alien_checkpoint` | raises |
| `energy_core` == `forward` energy | `tests/test_molzoo/test_mace/test_potential.py::TestMACEPotential::test_energy_core_agrees_with_the_forward_energy` | exact |
| loader is strict (unknown / missing / mis-shaped) | `tests/test_molzoo/test_mace/test_checkpoint.py::TestLoadMatpesStateDict`, `::TestCheckpointRemap` | raises |
| `from_checkpoint` builds from config + weights | `tests/test_molzoo/test_mace/test_checkpoint.py::TestFromCheckpoint` | exact |
| density normalisation, skip placement | `tests/test_molrep/test_interaction/test_mace/test_density.py` | exact |
| ZBL sign, decay, envelope, halving | `tests/test_molpot/test_potentials/test_repulsion.py` | exact |
| Agnesi monotonicity, pair symmetry | `tests/test_molrep/test_embedding/test_radial.py::TestAgnesiTransform` | exact |
| covalent table == checkpoint buffer | `tests/test_molrep/test_embedding/test_covalent.py::TestCovalentRadii` | 0.0 |

### 7.3 Engineering Benchmark

NVE on `wat64_h3o+` (193 atoms, 12.432 Å cubic, r_max 6.0, 17344 directed
edges) through `molix.md` on one GH200. Energy + forces, fp64, eager:

| Configuration | ms/step |
|---|---|
| initial port (two forwards, `skip_tp` on cuEq's default `fused_tp`) | 166.0 |
| one forward differentiated in place (A7) | 117.2 |
| + `skip_tp` on `naive` (A6) | **36.9** |
| MACE's own cueq model, same graph | 36.0 |

Block breakdown that located it (fp64, parts sum to the layer): `interaction[0]`
34.4 ms of a 53.3 ms forward, of which `skip_tp` 33.0 ms and the actual message
passing `conv_tp` 0.7 ms.

The measurements are at 193 atoms; `fused_tp` may win at much larger node
counts, which is why `skip_tp_method` is a constructor argument rather than a
constant.

#### `torch.compile` strategy

Eager is launch-bound at this size, so collapsing the launches is the remaining
lever. Compiling `MACEPotential.energy_core` (the pure ``positions -> energy``
function; measured under its former private name `_compute_energy`, which
survives as a name alias on `MACEMatpes`) and taking ``autograd.grad``
**outside** the compiled region:

| | fp64 ms (step/s) | fp32 ms (step/s) |
|---|---|---|
| eager, fused cuEq | 35.7 (28.0) | 35.0 (28.6) |
| `torch.compile` default inductor | 19.0 (52.5) | 19.6 (50.9) |
| **inductor + `reduce-overhead`** | **4.18 (239)** | **2.40 (417)** |
| + `fullgraph=True` | 4.19 (239) | 2.40 (417) |

**Recommended: `molix.compile.Compiler(cuda_graphs=True)`** — molnex's existing
preset (inductor + `reduce-overhead` + `dynamic=False` + `fullgraph=True`)
transfers to MACE unchanged. Requires static shapes, which the frozen
neighbour list of an MD run already provides.

Two facts worth keeping:

* **cuEquivariance traces cleanly**: ``graph_breaks=0, graphs=1, ops=381``. The
  fused cuEq kernels are custom autograd Functions and might have forced a graph
  break; they do not, so ``fullgraph=True`` is free (identical timing — there
  were no breaks to close).
* **Precision only starts mattering once the launches are gone.** Eager fp32 and
  fp64 are indistinguishable (35.0 vs 35.7 ms) because the step is latency-bound;
  under CUDA graphs fp32 is 1.75x fp64 (2.40 vs 4.18 ms). Compiling in fp64 is
  numerically free (ΔE = 0, ΔF = 2.6e-14 vs eager). fp32 costs ~2e-3 eV and
  ~1.6e-3 eV/Å against fp32 eager — that is float32 reassociation under inductor
  fusion, not a compile defect, and it is the usual precision MACE foundation
  models are run at for MD.

### 7.4 Run Log

| run_id | date | commit | dirty | dataset | config | steps | train_mae | val_mae | fwd_ms | bwd_ms | compiled | note |
|--------|------|--------|-------|---------|--------|-------|-----------|---------|--------|--------|----------|------|
| 1 | 2026-08-07 | 82c3091 | 1 | 5 crystals + H2O | official weights, fp64, CPU | 0 | n/a | n/a | n/a | n/a | no | single-point parity vs official: 4.3e-7 eV/atom, 2.0e-6 eV/Å (§7.1) |
| 2 | 2026-08-07 | 82c3091 | 1 | wat64_h3o+ | official weights, fp64, CPU, 1 step | 1 | n/a | n/a | n/a | n/a | no | `mace-r2san-cpu` smoke: E=-1015.900312 eV, \|F\|max=3.287 eV/Å, T=306 K, ~32 s/step (loaded login node) |
| 3 | 2026-08-07 | 82c3091 | 1 | wat64_h3o+ | official weights, fp64, GH200, fused cuEq | 200 | n/a | n/a | 170 | incl. | no | `mace-r2san-gh200` NVE dt=0.5 fs: same single point as CPU to all printed digits; E_tot drift -0.0119 meV/atom over 100 fs; T 281-354 K; 0.17 s/step |
| 4 | 2026-08-07 | 82c3091 | 1 | wat64_h3o+ | official mace-torch, fp64, GH200 | 200 | n/a | n/a | 35 | incl. | no | `mace-r2san-torch-gh200` control arm: identical driver/NL/ICs; E=-1015.900277 eV, same drift and T range; 0.035 s/step |
| 5 | 2026-08-07 | 82c3091 | 1 | wat64_h3o+ | official weights, fp64, GH200, after A6+A7 | 0 | n/a | n/a | 36.9 | incl. | no | energy+forces 166 -> 36.9 ms/step (4.5x), vs 36.0 ms for MACE's own cueq model; ΔE vs that model 4.5e-13 eV |
| 6 | 2026-08-07 | 82c3091 | 1 | wat64_h3o+ | fp64, GH200, inductor reduce-overhead | 0 | n/a | n/a | 4.18 | incl. | yes | 239 step/s, 8.5x eager; ΔE=0 ΔF=2.6e-14 vs eager; graph_breaks=0 |
| 7 | 2026-08-07 | 82c3091 | 1 | wat64_h3o+ | fp32, GH200, inductor reduce-overhead | 0 | n/a | n/a | 2.40 | incl. | yes | 417 step/s, 14.6x eager; fp32 reassociation ~2e-3 eV vs fp32 eager |
| 8 | 2026-08-07 | 82c3091 | 1 | wat64_h3o+ | fp64, GH200, full NVE loop `--compile` | 200 | n/a | n/a | 5.0 | incl. | yes | end-to-end MD 37.5 -> 5.0 ms/step (7.5x incl. integrator+hook); trajectory bit-identical to eager (max \|dPos\|=0, max \|dF\|=0, ΔE/atom 1.9e-12 meV); one-off compile ~60 s |
| 9 | 2026-08-08 | 82c3091 | 1 | wat64_h3o+ | launch gate: rebuild_every=5 + compile, 3 precisions | 2000 | n/a | n/a | 14.9/11.8/13.2 | incl. | yes | dead-edge dE≤1.1e-13 (atomicAdd reorder only); rebuild-vs-fresh dE=0; drift fp64 0.03 / fp32 0.04 / bf16 ~2 meV/atom/ps (bf16 heats — expected physics); bf16 needs autocast INSIDE the compiled callable (52->13.2 ms) |
| 10 | 2026-08-08 | 82c3091 | 1 | wat64_h3o+ | production 5 ns × 3 precisions (jobs 978984/978985/979020) | 10M | n/a | n/a | — | — | yes | dt=0.5 fs, stride 2000, rebuild_every=5, checkpoint 100k, auto-resume; dirs `mace-r2san-5ns-{fp64,fp32,bf16}` |

## 8. System Boundary

| Concern | Owner | Contract |
|---------|-------|----------|
| Neighbour list + PBC shifts | caller / `molix.data.tasks.NeighborList` | `(E,2)` edges; `shifts = mic_diff - (pos[t]-pos[s])` |
| Edge displacements | `molzoo.mace.geometry` | `edge_vectors` / `edge_lengths`; differentiable w.r.t. `pos` |
| Building blocks | `molrep.interaction.mace` / `molrep.readout.mace` / `molrep.embedding.mace` (+ generic `molrep` / `molpot`) | reused; not owned here |
| Model graph | `molzoo.mace.encoder.MACEEncoder` | blocks + wiring; no energy, no forces |
| Energy / forces | `molzoo.mace.potential.MACEPotential` | `energy_core` (public, compile seam) + `molpot.derivation.kernels.grad_force_pass` |
| Configuration | `molzoo.mace.spec.MACEMatpesSpec` | torch-free pydantic preset |
| Weight conversion | `mace.cli.convert_e3nn_cueq` (out of tree) | e3nn `.model` → cueq `state_dict` |
| Weight import | `molzoo.mace.checkpoint.CheckpointRemap` (`MATPES_REMAP`) via `MACEPotential.from_checkpoint`; `load_matpes_state_dict` back-compat wrapper | strict; raises on any unmapped or unfilled tensor |
| MD | `molix.md` | frozen neighbour list, `gamma=0` → NVE |
| Compile seam | `molix.compile.Compiler` | wraps `energy_core`; `autograd.grad` stays outside |
| Lazy export | `molzoo/__init__` + `molzoo/mace/__init__` | PEP 562 `__getattr__`; no eager cueq import |

## 9. Version Pinning

| Item | Value |
|------|-------|
| Reference repository | `ACEsuit/mace` v0.3.16 |
| Checkpoint | `MACE-matpes-r2scan-omat-ft.model`, `mace_matpes_0` release |
| Checkpoint config | `correlation=3`, `use_reduced_cg=False`, `use_agnostic_product=False`, `apply_cutoff=True`, `pair_repulsion=True`, `distance_transform=Agnesi`, `heads=['default']` |
| Dependencies | `cuequivariance` 0.10.0, `cuequivariance_torch`, `torch>=2.10` |
| Conversion oracle | e3nn 0.4.4 + mace-torch 0.3.16 in an out-of-tree venv |
| Module relocation | `mace-subpackage-restructure` chain, commits `1ddd5ff..e825a51` (merged 2026-08-09): `src/molzoo/mace_matpes.py` was retired into the `src/molzoo/mace/` package — config in `spec.py`, blocks in `encoder.py`, energy/forces in `potential.py`, key remap in `checkpoint.py`, the `MACEMatpes` alias in `variants.py`. MACE-only `molrep` blocks moved to `molrep/interaction/mace/{conv,block,density}.py`, `molrep/readout/mace.py`, `molrep/embedding/mace.py`; `molrep.interaction.density` / `molrep.readout.scalar` / `molrep.readout.product` remain as deprecated shims. Tests moved to `tests/test_molzoo/test_mace/`. Weights, hyper-parameters and numerics unchanged (§7.1 not re-run). |

## 10. Drift Policy

Any change to the interaction schedule, the readout placement, the density
normalisation, or the checkpoint key map **must** update this file in the same
PR and re-run §7.1. `use_reduced_cg` / `original_mace` and the `apply_cutoff`
ordering are load-bearing for weight compatibility — changing either silently
produces a model that runs and is wrong.

## Appendix A. Maintenance Log

- 2026-08-07: Created alongside the native port; §2/§3/§5 filled from the paper
  and `ACEsuit/mace` v0.3.16; §7.1 filled from the out-of-tree parity oracle.
- 2026-08-09: Anchors re-pointed for the `mace-subpackage-restructure` chain
  (`1ddd5ff..e825a51`) — header, §3.1 code anchors, §5 crosswalk, §6 A5/A6, §7.2
  test paths, §7.3 compile target, §8 boundary, §9 pinning row. Two content
  corrections found while re-pointing: the §5 force row named
  `ForceDerivation(method="autograd")`, but the MACE path actually runs
  `molpot.derivation.kernels.grad_force_pass` (batch) and
  `molpot.derivation.force.autograd_forces_from_energy` (raw) — same autograd
  math, different symbol; and one force tolerance in §7.1/§7.2 was written
  `eV·Å` instead of `eV/Å`. No section added, removed or renamed; §7.4 rows
  untouched; no numerical claim changed.
