---
title: MD compile seam and potential precision policy (F28)
status: in-progress
created: 2026-09-08
chain: composable-a0-debt
---

# composable-a0-debt-03-md-seam

## Summary
Give `molix.md.MD` the compile seam and the precision policy its benchmarks were written for: `MD.set_compiler(compiler)` installs a compiled build of the classical energy/force core through `Potentials.compile_core` (re-applied on every `set_*` rebuild), and `MD(dtype, *, potential_dtype=)` evaluates the potential in a second dtype with the integrator state kept in the MD dtype. Closes F28.

## Domain basis
"Everything compiles" iron law: the analytic classical step is a hot path and must run under `torch.compile` with the compiled region as large as possible — today `Potentials.compile_core` serves only an mlip, so a classical MD rollout never compiles. Split precision (fp64 integrator, fp32 potential) is the quantization-as-thermal-noise study's operating point (`molix.quant`); the benchmark's `--potential-precision` arm has been failing closed since d9043cee.

## Design
- `Potentials.compile_core(compiler)`: for a classical field (no mlip) compiles the analytic core — a bound method `_energy_forces_core(work, pos) -> (energy, forces)` that `calc_energy_forces` routes through when installed (the seam is the one autograd break at `derivation/force.py` for the periodic Coulomb term, unchanged); for an mlip the existing delegation stays. `RuntimeError` only when there is nothing to compile.
- `MD.set_compiler(compiler: Callable[[Callable], Callable] | None) -> MD`; `_wire()` applies it after building `Potentials` (pots are rebuilt per setter). `MD(dtype, *, potential_dtype: torch.dtype | None = None)`: `_wire` moves the potential's parameter tables to `potential_dtype`; `_TensorDictForce.__call__` casts `pos` (and the cell) to it before evaluation and returns energy/forces in the MD dtype. `MD.compute_force` honours both.
- `benchmarks/verify_md_ljcut_nve.py`: `--compile/--no-compile`, `--cuda-graphs` (via `molix.compile.Compiler(cuda_graphs=True)`) and `--potential-precision` restored on the seams; the fail-closed exit and the F28 NOTE are deleted. `benchmarks/verify_md_lj_nve.py` may use the seam too; its private `MD._wire()` reach-through is replaced by the public setters.

## Files
- `src/molpot/forcefield/potentials.py`, `src/molix/md/driver.py`, `benchmarks/verify_md_ljcut_nve.py`, `benchmarks/verify_md_lj_nve.py`
- `tests/test_molpot/test_forcefield/test_potentials.py`, `tests/test_molix/test_md/test_driver.py`

## Tasks
- [x] RED tests (classical compile_core installs and matches eager; set_compiler re-applied after a rebuild; potential_dtype casts tables and returns MD-dtype forces; slow: the installed core traces with breaks only at the force seam)
- [x] `Potentials` classical core + `MD.set_compiler` + `potential_dtype` (+ `ForceField.to_dtype`, bind-time graph count, `position_leaf` seam); benchmarks on the seams (eager 2 → compile 7 → cuda-graphs 589 steps/s on the 256-atom smoke, bit-identical drift)

## Testing
Fast: a recording "compiler" (identity wrapper that counts installs) proves `set_compiler` is applied at `set_forcefield` and again after `set_integrator`; energies/forces through the installed core equal eager to 1e-12; `MD(torch.float64, potential_dtype=torch.float32)` returns fp64 forces whose values equal the fp32 potential's forces cast up. Slow: `Compiler.explain` on the classical core of the LJ-only fixture → 0 breaks; on the periodic Coulomb fixture → breaks only in `derivation/force.py`.

## Out of scope
CUDA-graph capture tests (no GPU on this box); the quantization study itself.
