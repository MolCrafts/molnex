# Changelog

All notable changes to MolNex are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/) (public APIs may still change between
minor releases while < 1.0).

## [0.2.0] - 2026-06-14

The first substantial release after the initial scaffold — `molix` grows a full
execution/analysis stack, the data pipeline moves to a packed on-disk cache, and
`molzoo` gains the PiNet and Sonata reference models.

### Added

**molix — training & execution**
- `torch.compile` / CUDA-graph capture and AOT-Inductor model export
  (`molix.compile`, `molix.export`).
- In-process Langevin velocity-Verlet MD driver (`molix.md`) with an ASE shim.
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
- On-disk cache is now the single-file `PackedCache` (`.pt` with packed
  per-atom/edge/graph buckets, `mmap` loads) — replaces per-sample memmap dirs to
  stay within HPC inode budgets.
- Post-collate batch is a plain nested `TensorDict` (`atoms` / `edges` /
  `graphs`); the `AtomData` / `EdgeData` / `GraphData` subclasses were removed.
- `TrainState` enforces a fixed namespace layout (nested writes only; slash- and
  tuple-path reads) to prevent train/eval metric collisions.
- Edge convention standardized repo-wide: `edge_index[:,0]` = source,
  `[:,1]` = target, `bond_diff = pos[target] - pos[source]`.

### Fixed
- `RevMD17Source` downloads directly from the figshare REST API using only the
  stdlib — `molix` no longer imports the sibling `molhub` package (their coupling
  is protocol-based via the `DataSource` protocol).
- AMP-safe PiNet `IPLayer` scatter; per-epoch reshuffle and eval-starvation
  warnings; CUDA-graph force-training gradient correctness.

## [0.1.0]

- Initial four-package scaffold (`molix`, `molrep`, `molpot`, `molzoo`).

[0.2.0]: https://github.com/MolCrafts/molnex/releases/tag/v0.2.0
