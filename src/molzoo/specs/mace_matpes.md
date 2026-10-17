# MACE-MatPES Specification

This page is the implementation contract for `molzoo.mace.MaceMatpes` — a native
reimplementation of the official `MACE-matpes-*-0` foundation models. It is not a
tutorial; use the MolZoo user guide for narrative and worked examples.

| Field | Value |
|-------|-------|
| Module | `molzoo.mace.potential` |
| Entry point | `MaceMatpes(spec: MACEMatpesSpec, *, wrt=(), energy_dtype=torch.float64, input_dtype=None)`; `MaceMatpes.load(path, config=None, *, wrt, energy_dtype, input_dtype)` reads the archive (`MaceArchive.read -> (state, MaceOfficialConfig)`), builds `MACEMatpesSpec.from_official(cfg)` and renames the cuEq-converted state through `matpes_keys` (`molzoo.mace.checkpoint`) with `molzoo.checkpoint.CheckpointRemap(matpes_keys, skip=MACE_SKIP)`. No in-tree basis converter. |
| Paper | Batatia et al., *MACE*, NeurIPS 2022; Batatia et al., *A foundation model for atomistic materials chemistry* (MACE-MP-0) |
| arXiv | https://arxiv.org/abs/2206.07697 · https://arxiv.org/abs/2401.00096 |
| Dataset paper | Kaplan et al., *MatPES* — https://arxiv.org/abs/2503.04070 |
| Reference implementation | `ACEsuit/mace` v0.3.16 (`ScaleShiftMACE`) |
| Reference checkpoint | `MACE-matpes-r2scan-omat-ft.model` (ACEsuit/mace-foundations, tag `mace_matpes_0`, ASL licence) |
| Spec status | partial |

**Related modules (out of scope for this file):**

| Module | Role |
|--------|------|
| `molzoo.mace.potential.MaceOmol` | OMOL foundation model (charge/spin conditioned); see `mace_omol.md` |
| `molzoo.mace.encoder.MACEEncoder` | Shared block graph both variants inherit (`MacePotential` subclasses it) |
| `molix.neighbor.NeighborList` | Cutoff graph construction (PBC minimum image) |
| `molix.autograd.Derivative` | Forces / virial (`forward`) |

## 1. Scope

`molzoo.mace.MaceMatpes` is a **full energy/force model**. It owns:

- the embedding → interaction → product → readout stack of `ScaleShiftMACE`
  with density-normalised interactions,
- the frozen per-element reference energy `E0` and the global scale/shift,
- the ZBL pair-repulsion term,
- strict import of an official cueq-converted `state_dict`.

It does **not** own:

- neighbour-list construction or PBC imaging (caller supplies `pairs.pair_index`
  and the already-imaged `pairs.pair_diff`),
- the derivative op (`molix.autograd.Derivative`),
- the official-checkpoint → cueq weight conversion (out of tree, MACE's official cuEquivariance converter),
- training loops, losses, or MD integration.

## 2. Public Contract

### 2.1 Required Inputs

| Direction | Path / argument | Shape | Dtype | Contract |
|-----------|-----------------|-------|-------|----------|
| In | `("atoms", "Z")` / `Z=` | `(N,)` | int64 | Atomic numbers; must all be in the model's z-table |
| In | `("atoms", "pos")` / `positions=` | `(N, 3)` | float | Cartesian positions, Å |
| In | `("atoms", "batch")` / `batch=` | `(N,)` | int64 | Graph membership |
| In | `("pairs", "pair_index")` / `edge_index=` | `(E, 2)` | int64 | `[:, 0]` = source, `[:, 1]` = target (repo edge convention) |
| In | `("pairs", "pair_diff")` | `(E, 3)` | float | Already-imaged `pos[t]-pos[s]` (+ lattice image), Å; used when present, else derived from `pos` |
| In | `graphs` namespace | `batch_size=[B]` | — | Required: `B = frame["graphs"].batch_size[0]`, a static int (no host sync) |
| In | `("atoms", "mask")` | `(N,)` | bool | Optional padded-batch real-atom mask; every per-atom term is multiplied by it before the graph sum |

`energy(frame)` is the one tensor-level method: it reads these columns, casts
the geometry to `input_dtype`, runs the encoder primitives and the readouts
and writes `atoms.energy` `(N,)` and `graphs.energy` `(B,)`. Element
validation is the encoder's eager precondition, spent on the first eager call.

### 2.2 Outputs

| Direction | Path / return | Shape | Contract |
|-----------|---------------|-------|----------|
| Out | `("graphs", "energy")` | `(B,)` | Total energy, eV |
| Out | `("atoms", "energy")` | `(N,)` | Per-atom energy, eV; sums to `graphs.energy` |
| Out | `("atoms", "forces")` | `(N, 3)` | `-∂E/∂pos`, eV/Å; written when built with `wrt=("atoms.pos",)` |
| Out | `("graphs", "virial")` | `(1, 3, 3)` | eV; written when `wrt=("atoms.pos", "graphs.strain")`, one-graph periodic batch |

`forward = Derivative(self.energy, wrt=...)`, fixed at construction; the default
`wrt=()` writes the energy alone.

## 3. Forward Contract

### 3.1 Notation

| Symbol | Meaning | Code anchor |
|--------|---------|-------------|
| \(N, E, B\) | atoms, edges, graphs | batch sizes |
| \(F\) | `num_features` = 128 | `hidden_irreps` 0e multiplicity (`molzoo.mace.spec.MACEMatpesSpec`) |
| \(\ell_{\max}\) | `l_max` = 3 | spherical harmonics / TP |
| \(\nu\) | `correlation` = 3 | symmetric-contraction degree |
| \(\rho_i\) | learned edge density | `molzoo.mace.layers.DensityInteraction._message` |
| \(E(\mathbf r)\) | per-graph energy `(B,)`, eV | `molzoo.mace.potential.MacePotential.energy` |
| \(\mathbf v_e\) | edge displacement (already imaged `pair_diff`) | `molrep.embedding.geometry.edge_vectors` / `edge_lengths` |

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
| `wrt` | `()` | derivative keys for `forward`; `("atoms.pos",)` adds forces |
| `energy_dtype` | `torch.float64` | graph accumulation dtype; `torch.float32` or `None` (model dtype) are explicit alternatives; does not change parameter keys |

## 5. Reference Crosswalk

| Reference component | MolNex anchor | Status |
|---------------------|---------------|--------|
| `LinearNodeEmbeddingBlock` | `molzoo.mace.encoder.MACEEncoder.node_embedding` (`cuet.Linear`) | matched |
| `RadialEmbeddingBlock.bessel_fn` | `molrep.embedding.radial.BesselRBF(normalize=False, eps=0, trainable=True)` | matched |
| `AgnesiTransform` | `molrep.embedding.radial.AgnesiTransform` | matched |
| `PolynomialCutoff` | `molrep.embedding.cutoff.PolynomialCutoff` | matched |
| `ZBLBasis` | `molpot.potentials.repulsion.ZBLRepulsion` | matched |
| `AtomicEnergiesBlock` | `molpot.heads.energy.AtomicReferenceEnergy` (Z-indexed) | adapted (A1) |
| `RealAgnosticDensityInteractionBlock` | `molzoo.mace.layers.DensityInteraction` | matched |
| `RealAgnosticDensityResidualInteractionBlock` | `molzoo.mace.layers.DensityResidualInteraction` | matched |
| `EquivariantProductBasisBlock` | `molzoo.mace.layers.EquivariantProductBasis` | matched |
| `LinearReadoutBlock` | `molzoo.mace.heads.LinearReadout` | matched |
| `NonLinearReadoutBlock` | `molzoo.mace.heads.NonLinearReadout` | matched |
| MACE radial MLP (moment-normalised SiLU, `normalize2mom`) | `molrep.embedding.mlp.MomentNormalizedMLP` (generic; not moved) | matched |
| `ScaleShiftBlock` | `molpot.heads.rescale.GlobalRescale` | matched |
| `get_outputs` force path | autograd: `molix.autograd.Derivative` over `MacePotential.energy` (`forward`), one `torch.autograd.grad` | matched |

`molzoo.mace.layers` and `molzoo.mace.heads` are the modules.
There is no `mace` block namespace.

## 6. MolNex Adaptations

| ID | Adaptation | Reason | Risk | Validation |
|----|------------|--------|------|------------|
| A1 | `E0` stored Z-indexed, not element-table-indexed | encoder takes raw `Z` | low | §7.1 parity |
| A2 | Covalent radii inlined (Cordero 2008) rather than read from `molpy.Element` | keeps a core `molrep` block free of the compiled molrs extension; molpy stores radii in fp32 | low | table asserted equal to the checkpoint buffer to 0.0 |
| A3 | cuEq `O3` group, not the MACE-compatible `O3` variant | same route the OMOL port validated; CG conventions differ by ~1e-8/op | low | §7.1 parity |
| A4 | Weight import consumes a cueq-converted `state_dict`; conversion runs out of tree | molnex must not import `mace-torch` | low | strict loader, §7.2 |
| A5 | `skip_tp`'s weight follows the module dtype with no rebuild | it is a plain parameter of `ElementLinear` (A6); the product it replaced baked its precision in at construction | low | `ElementLinear.weight` is a plain `nn.Parameter`: no kernel submodule to rebuild |
| A6 | `skip_tp` as a per-element linear, `molzoo.mace.layers.ElementLinear`: the weight is held element-major, `(C, Σ_p u_p·w_p)` — row `e` is element `e`'s `(u, w)` block of every path — so a call gathers every atom's row once (`index_select` on the `ElementIndex`) and runs one `bmm` `(N, d, u) @ (N, u, w)` per path, on every device (2026-10-17 native layout + `bmm`; earlier on 10-17 the gather ran over the product's own flat `(u, C, w)` blocks with a per-call `einsum` transpose; `naive` until 2026-10-15, `fused_tp` on 10-15, cue's `indexed_linear_B` kernel on 10-16) | the product against a one-hot is linear in the element-selected weight block, coefficient `1/sqrt(u·C)` per path (read from cue's descriptor). The product's layout — one flat row of path-major `(u, C, w)` blocks — exists only in the official checkpoint and is converted once at that boundary: `molzoo.mace.checkpoint.element_linear_weight`, applied by `CheckpointRemap.load` by target type (`molzoo.checkpoint.ConvertedWeight`), refusing any other shape of equal numel; `node_embedding` and the residual source / target embeddings likewise (`OneHotLinear`, `(C, F)`, `one_hot_weight`). cue's `fused_tp` runs the dense product on a fallback kernel: 791 ms vs 3.7 ms for 4600 atoms (forward + input gradient, GH200), 90% of a MACE-MP force call. cue 0.12.0's `indexed_linear_B` (atoms sorted by element, one call per path) reads the counts on the host — one sync per call — and invalidates a CUDA-graph capture (`cudaErrorStreamCaptureInvalidated`, GH200 job 3972523), so `Compiler(cuda_graphs=True)` cannot run it; against it the gather is 1.1–1.3× slower forward, 1–2× faster forward + input gradient, and holds `u·w` floats per atom and path (4096 atoms, 89 elements, 128 channels, fp32: 0.79 vs 0.45 GiB peak) | low (equal to the product: ≤3.3e-16 fp64 CPU; CUDA graphs = eager to 0.0 eV with the 10-16 gather, job 3972523 — the `bmm` form awaits its GH200 run) | `tests/test_molzoo/test_mace/test_layers.py::TestElementLinear`, `::TestOneHotLinear`; `test_checkpoint.py::TestCheckpointRemap` round trips through the boundary conversion |
| A7 | Energy and forces come from one forward differentiated in place, not a re-run closure | MACE's `get_outputs` shape; the closure form cost two full forwards per step | none (same math) | §7.2 `forward` ↔ `energy` row |
| A8 | Shared configurable graph accumulation and differentiable model-input dtype boundary | fp64 MD state may use strict fp32 model calculation without fp32 baseline summation | numerical policy is not a long-time drift guarantee | `tests/test_molzoo/test_mace/test_potential.py::TestMacePotential::test_energy_dtype_selects_the_accumulation`, `::test_forward_applies_the_input_dtype`; official-weight 193-atom CPU/GPU three-step smoke, job 2063791 |

Precision/loading controls and current acceptance limits are documented in
`docs/molzoo/user-guide/mace.md`. The parameter dtype is the constructor's
`dtype` argument, fixed at construction and before compilation; molnex leaves
the fp32 matmul policy at PyTorch's default. Official loading accepts converted cuEq weights, not a raw
unconverted (pre-cuEquivariance) contraction concatenation. Shared scalar/Linear singleton ranks are the
only checkpoint reshapes; no in-tree basis conversion is implemented.

**Known limitation.** cuEquivariance freezes `math_dtype` at construction, and
`EquivariantProductBasis` (shared with `MaceOmol`) is not rebuilt on `.double()`.
A model built at fp32 and then `.double()`-d therefore still contracts in
float32 (~1e-8 eV noise). For fp64 work construct with `dtype=torch.float64`.

## 7. Validation Contract

### 7.1 Research Reproduction

Native model vs the official `MACE-matpes-r2scan-0` (pre-conversion checkpoint, fp64), identical
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
4.3e-6 eV/Å). The residual is float64 accumulation order between the official (pre-conversion) and
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

**Against MACE's own cueq model** — the same kernels rather than the pre-conversion
reference — agreement is essentially bit-level, which isolates the numbers above
as official-vs-cueq contraction order rather than a modelling difference:

| Comparison (193-atom box, fp64, GH200) | ΔE | max ΔF |
|---|---|---|
| molnex vs official (cueq-converted by MACE's converter) | 4.5e-13 eV | 3.3e-14 eV/Å |
| `skip_tp` naive vs fused_tp, whole model | 2.3e-13 eV | 5.6e-15 eV/Å |

### 7.2 Symmetry and Shape Tests

Variant-level claims live in
`tests/test_molzoo/test_mace/test_potential.py::TestMaceMatpes` (the `::…` rows
below are relative to it); the shared pipeline is covered by
`tests/test_molzoo/test_mace/test_potential.py::TestMacePotential` (written
`TestMacePotential::…` below; its physics rows run on the MatPES arm).

| Claim | Test path | Tolerance |
|-------|-----------|-----------|
| `energy(frame)` writes `atoms.energy` + `graphs.energy`; `forward` leaves `graphs.energy` equal and adds the derivatives | `TestMacePotential::test_energy_writes_both_columns_and_forward_leaves_graph_energy`, `::test_forward_is_the_same_op_composed_outside` | 1e-10 eV |
| feature rotation / translation invariance | `tests/test_molzoo/test_mace/test_encoder.py::TestMACEEncoder::test_forward_is_rotation_invariant`, `::test_last_layer_features_are_translation_invariant` | 1e-10 |
| force equivariance `F(Rx) = R F(x)` | `TestMacePotential::test_forces_rotate_with_the_system` | 1e-10 |
| `F = -dE/dpos` vs central differences | `TestMacePotential::test_forces_match_central_differences` | 1e-6 |
| virial vs central differences of the strained energy | `TestMacePotential::test_virial_matches_central_differences_of_the_strained_energy` | see test |
| net force ≈ 0 | `TestMacePotential::test_net_force_vanishes` | 1e-10 |
| energy extensive over separated graphs | `TestMacePotential::test_energy_is_extensive_over_separated_graphs` | 1e-10 |
| out-of-table `Z` rejected | `tests/test_molzoo/test_mace/test_encoder.py::TestMACEEncoder::test_validate_elements_lists_the_out_of_table_numbers` | raises |
| padded ghosts add nothing | `TestMacePotential::test_a_padded_batch_returns_the_unpadded_energy_and_forces` | 1e-10 |
| `forward` op compiles to one graph, 0 breaks in inference (`slow`) | `TestMacePotential::test_forward_compiles_to_one_graph_in_inference` | exact |
| loader is strict (unknown / missing / mis-shaped) | `tests/test_molzoo/test_checkpoint.py::TestCheckpointRemap` | raises |
| `matpes_keys` round trip | `tests/test_molzoo/test_mace/test_checkpoint.py::TestCheckpointRemap::test_roundtrip_restores_every_parameter_matpes` | exact |
| `load` builds from config + weights; `from_official` rejection rules | `tests/test_molzoo/test_mace/test_potential.py::TestLoad`, `tests/test_molzoo/test_mace/test_spec.py::TestMACEMatpesSpec` | exact / raises |
| density normalisation, skip placement | `tests/test_molzoo/test_mace/test_layers.py` | exact |
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

The measurements are at 193 atoms. Since 2026-10-15 `naive` is not an option
(kernels only); since 2026-10-17 `skip_tp` is the per-atom gather of A6: the
weight held element-major, one row gather per call and one `bmm` per path, no
per-call transpose — the product's flat layout exists only in the official
checkpoint and is converted at load (cue's `indexed_linear` kernel, run for a
day in between, is 3.7 vs 4.3 ms against `naive` at 4600 atoms but cannot be
captured into a CUDA graph).

#### `torch.compile` strategy

Eager is launch-bound at this size, so collapsing the launches is the remaining
lever. The table below was measured compiling the pure ``positions -> energy``
function (the former public flat seam, deleted 2026-10-16) with
``autograd.grad`` **outside** the compiled region. Today the whole `forward` op
(`molix.autograd.Derivative` over `energy`) compiles as one region under
`Compiler` after one eager forward:

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
| 11 | 2026-08-09 | 43cd33d | 0 | wat64_h3o+ | official weights, fp64, GH200, full NVE `--compile`, post mace-subpackage-restructure + fix batch (dev unified) | 200 | n/a | n/a | 8.6 | incl. | yes | `mace-nve-validate-1017194` post-restructure gate: single point E=-1015.900312 eV, \|F\|max=3.2866 eV/Å — matches rows 2/3 to all printed digits; E_tot drift -0.0146 meV/atom over 0.10 ps (-0.146 meV/atom/ps, order of row 3); T 283.7-353.7 K; rebuilds 41 (~40 expected at rebuild_every=5); sbatch work/mace-nve/run_gh200_validation.sbatch, job 1017194 |
| 12 | 2026-09-08 | c6be5652 (dp25_n200 at d4adeac, 2026-09-09) | 0 | PEO -OCH3 melts (lab-v3 peo-tg/mace-scaling-bench, GAFF-cooled, 7400/8850/17700/21400/35400 atoms) | official weights, fp64 + fp32, GH200, eager + fullgraph-Inductor core | 0 | n/a | n/a | 41 (7400, fp32 compiled E+F) | incl. | yes | fp64 parity vs official mace-torch 0.3.16 cueq: dE 8e-16 eV/atom (at |E|·ε), max|dF| 4e-14 eV/Å, max|dW| 5e-12 eV — machine precision, E/F/W; virial convention W = −W_official element-wise. fp32 scaling 7400→35400: molnex compiled 5.5→5.4 µs/atom (α ≈ 1.0), official 7.1→6.7 µs/atom; at 35400 atoms (3.67 M edges) molnex compiled 0.190 s / 56 GiB vs official 0.236 s / 57 GiB, official E+F+W 1.17 s (5× its E+F), molnex E+F+W = E+F; peak ≈1.65 MiB/atom both engines. Report `lab-v3/projects/peo-tg/experiments/mace-scaling-bench/report.md`, `parity.md` |

## 8. System Boundary

| Concern | Owner | Contract |
|---------|-------|----------|
| Neighbour list + PBC imaging | caller / `molix.neighbor.NeighborList` | `pairs.pair_index` `(E,2)`; imaged `pairs.pair_diff` `(E,3)` |
| Edge displacements | `molrep.embedding.geometry` | `edge_vectors` / `edge_lengths`; imaged value, gradient from raw `pos` |
| Building blocks | `molrep.embedding` (`angular`, `cutoff`, `radial`, `mlp`), `molrep.interaction.product` (+ `molpot.heads`, `molpot.potentials.repulsion`) | reused; not owned here. The MACE-only blocks (`molzoo.mace.layers`, `molzoo.mace.heads`) are owned here |
| Model graph | `molzoo.mace.encoder.MACEEncoder` | blocks + wiring; no energy, no forces |
| Energy / forces | `molzoo.mace.potential.MacePotential` | `energy(frame)` writes `atoms.energy` / `graphs.energy`; `forward = molix.autograd.Derivative(self.energy, wrt=...)` |
| Configuration | `molzoo.mace.spec.MACEMatpesSpec` | torch-free pydantic preset |
| Weight conversion | MACE's official cuEquivariance converter (out of tree) | official MACE checkpoint (pre-conversion) `.model` → cueq `state_dict` |
| Weight import | `MaceMatpes.load` → `molzoo.mace.utils.MaceArchive.read` (typed `MaceOfficialConfig`) → `MACEMatpesSpec.from_official` → `molzoo.checkpoint.CheckpointRemap(matpes_keys, skip=MACE_SKIP)` | preflights names, shapes and parameter coverage; official cuEq-converted weights required |
| MD | `molix.md` | frozen neighbour list, `gamma=0` → NVE |
| Compile seam | `molix.compile.Compiler` | wraps the `forward` op after one eager forward (element gate) |
| Package export | `molzoo/mace/__init__` | eager `from molzoo.mace import MaceMatpes`; nothing is exported from `molzoo/__init__` |

## 9. Version Pinning

| Item | Value |
|------|-------|
| Reference repository | `ACEsuit/mace` v0.3.16 |
| Checkpoint | `MACE-matpes-r2scan-omat-ft.model`, `mace_matpes_0` release |
| Checkpoint config | `correlation=3`, `use_reduced_cg=False`, `use_agnostic_product=False`, `apply_cutoff=True`, `pair_repulsion=True`, `distance_transform=Agnesi`, `heads=['default']` |
| Dependencies | `cuequivariance` 0.10.0 at the §7.1 runs; current floor `cuequivariance>=0.11`, `cuequivariance_torch>=0.11`, `torch>=2.14` (`pyproject.toml`) |
| Conversion oracle | mace-torch 0.3.16, in an out-of-tree venv |
| Module relocation | `mace-subpackage-restructure` chain, commits `1ddd5ff..e825a51` (merged 2026-08-09): `src/molzoo/mace_matpes.py` was retired into the `src/molzoo/mace/` package — config in `spec.py`, blocks in `encoder.py`, energy/forces in `potential.py`, key remap in `checkpoint.py`, the `MaceMatpes` alias in `variants.py`. MACE-only `molrep` blocks moved to `molrep/interaction/mace/{conv,block,density}.py`, `molrep/readout/mace.py`, `molrep/embedding/mace.py`; `molrep.interaction.density` / `molrep.readout.scalar` / `molrep.readout.product` remain as deprecated shims. Tests moved to `tests/test_molzoo/test_mace/`. Weights, hyper-parameters and numerics unchanged (§7.1 not re-run). Later superseded: `variants` and `geometry` are gone (`MaceMatpes` lives in `potential.py`; edge geometry is `molrep.embedding.geometry`), `molrep/interaction/mace/` and the `molrep/{readout,embedding}/mace.py` modules no longer exist (`molrep.interaction.density` / `molrep.readout.scalar` were the real modules, not shims), and `CheckpointRemap` moved to `molzoo.checkpoint`. 2026-10-16: the MACE-only blocks left `molrep` for `molzoo/mace/layers.py` (interactions, product basis, `ElementLinear`, `RadialMLP`, `GatedNonlinearity`) and `molzoo/mace/heads.py` (readouts, `JointFeatureEmbedding`); tests in `tests/test_molzoo/test_mace/test_{layers,heads}.py`. |
| Public docs mirror | `docs/molzoo/specs/mace_matpes.md` — byte-identical copy of this file; re-sync both halves on every edit |

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
  corrections found while re-pointing: the §5 force row named a symbol that is
  not the force path — forces are `molix.autograd.Derivative(self.energy, wrt)`
  (`forward`), one `torch.autograd.grad`; and one force tolerance in §7.1/§7.2 was written
  `eV·Å` instead of `eV/Å`. No section added, removed or renamed; §7.4 rows
  untouched; no numerical claim changed.
- 2026-10-15: Refreshed to the current API, no numerical claim changed: §2
  drops `shifts=` / `energy_forces` (`pairs.pair_diff` is the PBC
  channel; forces and virial are a `wrt=`
  construction choice via `molix.autograd.Derivative`); §3.1 / §8 point at
  `molrep.embedding.geometry`, `molix.neighbor.NeighborList` and
  `molzoo.checkpoint.CheckpointRemap` + `matpes_keys`; §5 force row and §6
  A7/A8 validation re-pointed; §7.2 cites tests that exist
  (`test_potential.py::TestMaceMatpes` / `TestMacePotential`,
  `test_checkpoint.py::TestLoad`); §7.3 notes that today the whole `forward`
  op compiles; §9 notes the post-restructure removals and adds the docs mirror;
  header related-modules table header fixed.
- 2026-10-17: §6 A6 and §7.3 — `ElementLinear` / `OneHotLinear` store their
  weights in the layout the forward reads (`(C, Σ u·w)` / `(C, F)`); cue's
  flat layout is converted once at the official boundary
  (`molzoo.mace.checkpoint.element_linear_weight` / `one_hot_weight`, applied
  by target type through `molzoo.checkpoint.ConvertedWeight`). No numerical
  claim changed (CPU equality to the product re-measured at ≤3.3e-16); the
  seeded tiny-model reference `benchmarks/_gh200_verify/step7_ref.pt` was
  regenerated, since the parameter shapes move the init draw.
