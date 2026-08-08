# Changelog

All notable changes to MolNex are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/) (public APIs may still change between
minor releases while < 1.0).

## [Unreleased]

Work accumulated on top of the initial scaffold while the package version stays
`0.1.0` — `molix` grows a full execution/analysis stack, the data pipeline moves
to a packed on-disk cache, and `molzoo` gains the PiNet and Sonata reference
models.

### Added

**molix — training & execution**
- `torch.compile` / CUDA-graph capture and AOT-Inductor model export
  (`molix.compile`, `molix.export`).
- In-process Langevin velocity-Verlet MD driver (`molix.md`).
- Trajectory diagnostics and a thermal-noise verdict (`molix.analysis`), plus
  weight-quantization tooling with paired force-Δ and effective-temperature
  scalars (`molix.quant`).
- Zarr-backed async `JournalHook` for metric persistence; on-device metrics and
  an eval-phase hook lifecycle.
- Token-budget dynamic batch sampler and a packed-aware collate fast path.
- `MolRecSource` for labeled-configuration datasets; bulk-water RPBE-D3 loader.
- `ModuleProfiler` / `TrainerProfiler` / `DataLoaderProfiler` for hotspot
  breakdowns.

**molzoo — reference models**
- PiNet encoder with functorch forces and fixed-shape ghost padding for
  CUDA-graph-compatible energy+force training.
- Sonata long-range encoder and the `build_sonata` composer.
- Per-encoder paper-traceable spec workflow (`src/molzoo/specs/`) with a
  `molzoo-auditor` agent.

**molpot — potentials**
- LES screened-Coulomb electrostatics and multipole QM/MM energy kernels.
- Sonata composer with scope refusal and Ewald finite-difference stress hook.

**interface**
- C++ AOT model runtime library for serving exported models.

**infrastructure**
- GitHub Actions CI + pre-commit (ruff / ty), channel-based logging via `mollog`.

### Changed
- MACE family performance + convention fixes (review-driven).
  `MACEMatpes(use_fallback=)` now defaults to `False` (fused cuEq kernels —
  measured ~36x faster per MD step; forces are always autograd there, so the
  functorch reason for the fallback never applies; pass `True` on CPU), and
  `MACE` / `ProductHead` expose `use_fallback` instead of hardwiring the
  pure-torch path. The element-table check moved off the per-step path
  (`MACEMatpes.validate_elements`, run once per instance). The MACE-side
  `(2, E)` edge layout was removed everywhere — `DensityInteraction` /
  `ResidualInteraction` / `ZBLRepulsion` and both model cores now take the
  repo-wide `(E, 2)` `[:, 0]`=source convention, eliminating the per-step
  `.t().contiguous()` copy and restoring the `bond_index` anti-alias guard.
  `scatter_sum_compile_safe` defaults to `index_add_`; the bit-exact one-hot
  GEMM (2.4–3.3x slower at every profiled shape) is opt-in via
  `MOLNEX_SCATTER_ONEHOT=1`. `MACEOMol` gained the `num_interactions` guard
  and the strict state-dict loader already written for MatPES (raises on
  unfilled learnables/shape mismatches instead of silently dropping weights),
  plus an `edge_channels` parameter replacing hardcoded `128x{l}` tables.
  New `cueq-cu12` / `cueq-cu13` extras declare the fused-kernel wheel;
  `run_nve.py` now reports actual fused-kernel availability, not the request.
- `molix.md` public contract reworked. `MD` is the single documented entry
  point; `MD(dtype=)` now governs the **MD side only** (state, integrator
  constants, mass) with the potential's precision set independently via
  `MD.set_potential_dtype` — the integrator casts force output back to the
  state dtype at the boundary. `MD(integrator=)` accepts any constructed
  `Integrator` (the ABC now declares `advance` / `advance_n` / `rollout` /
  `removed_dof`). `MDRunner` speaks its own `MDHook` protocol
  (`on_run_start` / `on_step_start` / `on_step_end` / `on_run_end` with typed
  `MDObservables`) instead of impersonating `Trainer` hooks, and returns a
  typed `MDState`. Renames: `MDState.force` → `forces`, `molix.md`'s
  `CheckpointHook` → `MDCheckpointHook`; velocity sampling moved off `MD` to
  `MaxwellBoltzmann`; `LangevinVerletIntegrator.run` and the migrated-out
  `dynamics` study layer (`run_trajectory`, `build_paired_trajectory`,
  `TrajectoryArtifact`) were removed (they live in the `pinet-quant`/`csmd`
  project).
- `PotentialForceField`'s potential contract is monomorphic: `forward(td)`
  writes `graphs.energy` / `atoms.forces` per `molix.schema`; force derivation
  is fixed at potential construction (`compute_forces=True`), no longer a
  per-call flag, and `calc_energy` is no longer cheaper than a full
  evaluation. New adapters: `PeriodicPotentialForceField` (rebuilding
  fixed-capacity neighbour list bound by reference) and `CallableForceField`
  (any `pos -> (energy, forces)` callable, e.g. an AOTI `.pt2`).
- `PeriodicNeighborList.edge_index` is now `(capacity, 2)` per the repo-wide
  edge convention (was `(2, capacity)`); `to()` accepts positional
  device/dtype. The batch-schema keys (`ENERGY_KEY`, `FORCES_KEY`, …) moved to
  the new `molix.schema` (re-exported by `molpot.derivation.protocol`), and
  physical constants (`KB_EV_PER_K`, `EV_PER_AMU_A2_FS2`, `KB_AMU_A_FS`) to
  the new `molix.units` — both breaking the latent `molix → molpot` import
  cycle.
- On-disk cache is now the single-file `PackedCache` (`.pt` with packed
  per-atom/edge/graph buckets, `mmap` loads) — replaces per-sample memmap dirs to
  stay within HPC inode budgets.
- Post-collate batch is a plain nested `TensorDict` (`atoms` / `edges` /
  `graphs`); the `AtomData` / `EdgeData` / `GraphData` subclasses were removed.
- `TrainState` enforces a fixed namespace layout (nested writes only; slash- and
  tuple-path reads) to prevent train/eval metric collisions.
- Edge convention standardized repo-wide: `edge_index[:,0]` = source,
  `[:,1]` = target, `edge_diff = pos[target] - pos[source]`.

### Fixed
- `RevMD17Source` downloads directly from the figshare REST API using only the
  stdlib — `molix` no longer imports the sibling `molhub` package (their coupling
  is protocol-based via the `DataSource` protocol).
- AMP-safe PiNet `IPLayer` scatter; per-epoch reshuffle and eval-starvation
  warnings; CUDA-graph force-training gradient correctness.

## [0.1.0]

- Initial four-package scaffold (`molix`, `molrep`, `molpot`, `molzoo`).
