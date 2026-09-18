---
title: Static shapes end to end — full-capacity edges on the MD path, dead-edge padding for training, atom mask in MACE
status: draft (ready to launch after the PME fix)
created: 2026-09-11
chain: derivation-head
depends: derivation-head-04-molix-derivative
---

## Summary

Operator directive (2026-09-11): 静态形状也要. Measured on GH200 (matpes fused, fp64,
`.scratch/compile-seam/repro/probe_cudagraph.py`, job 2293350): the E+F step is
launch-bound — eager ≈ 20 ms and default inductor ≈ 5.5 ms flat from 128 to 1024
atoms — and `Compiler(cuda_graphs=True)` on a fixed-capacity edge buffer gives
0.41 / 0.41 / 0.72 ms (128 / 256 / 1024 atoms), forces identical to 1e-14; +35 %
dead edges cost nothing. Today three things forbid it: the MD driver slices the
capacity buffer to the live prefix on every op call (`src/molix/md/driver.py`
`_TensorDictForce.__call__` → `_expose_live_edges` / `_restore_edges`, fed by
`n_live` set in `MD.compute_force`), `PadMolecularBatch` pads edges as `r = 0`
self-loops (live for MACE, `1/r` in ZBL), and MACE ignores `atoms.mask` (ghost
atoms add `e0` + readout). After spec 04 the driver composes ONE
`Derivative(pots, wrt=(POS_KEY[, STRAIN_KEY]))` in `bind_work`, runs it once
eagerly and then installs `compiler(op)` — that op is what CUDA graphs capture.

## Design

1. **MD: hand the op the full capacity buffer.** `_expose_live_edges` /
   `_restore_edges` and `n_live` go; `NeighborList` already pads dead rows as
   `(0, 0)` self-loops with `shifts = [10·r_cut, 0, 0]` and recomputes
   `pair_diff` / `pair_dist` over the whole capacity (`neighbors.py _write`,
   `_write_geometry`), so every op call sees a static `E = capacity`. The
   `potential_dtype` re-cast of the float edge columns moves to the list's
   `_write_geometry` (once per rebuild, not per step). The op itself rebuilds
   `pair_diff` from its leaf over all capacity rows — dead rows stay inert
   because their shift is 10·r_cut and their `pos[t] − pos[s]` is 0.
2. **Dead-edge convention is one constant, one place.** `molix.schema` owns
   `DEAD_EDGE_SHIFT(cutoff)`; `NeighborList._dead_shift`, `StaticForward.pad_len`
   and `PadMolecularBatch` all use it. `PadMolecularBatch` pads edges as
   `(ghost0, ghost0)` with `pair_diff = [10·r_cut, 0, 0]`, `pair_dist = 10·r_cut`,
   and takes `cutoff` at construction.
3. **Atom mask in the energy.** `MacePotential.energy_core(..., atom_mask=None)`
   multiplies the per-atom energies (`e0` and readout, before `_sum_energy`)
   by the mask; `MacePotential.energy(frame)` forwards `atoms.mask` when
   present (always as a multiply, never a slice). Same in PiNet (already does)
   and the classical `Potentials.energy` (per-term atom sums).
   Losses: `energy_force_mse` divides the force MSE by the real atom count
   (`atoms.mask.sum()`), not `max_atoms`.
4. **CUDA-graph step contract.** `_TensorDictForce.__call__` clones the
   outputs it hands to the integrator and calls
   `torch.compiler.cudagraph_mark_step_begin()` before the compiled op when
   the installed compiler is the CUDA-graph preset (`Compiler.cuda_graphs`);
   the neighbour-list rebuild stays eager, outside the captured region.
5. `md.md` / `throughput-and-compilation.md`: replace the "`dynamic=True` on the
   MD path" prescription with the static one; keep `dynamic=True` documented
   only for ragged training without padding.

## Files

`src/molix/md/driver.py`, `src/molix/md/neighbors.py`, `src/molix/data/pad.py`,
`src/molix/schema.py`, `src/molix/engine/static.py`, `src/molzoo/mace/potential.py`,
`src/molzoo/mace/polar.py`, `src/molpot/forcefield/potentials.py`,
`src/molix/core/losses/molecular.py`, docs above; mirrored tests.

## Testing

- `test_driver.py`: the potential is handed `edges` with `batch_size == [capacity]`
  every step; forces equal the live-only forces to 1e-12.
- `test_pad.py`: padded edges have `pair_dist == 10·cutoff`; a MACE tiny model on a
  padded batch returns the unpadded energy/forces to 1e-12 (`test_mace/test_potential.py`).
- `test_neighbors.py`: no per-step dtype cast; `_write_geometry` casts once.
- slow: `Compiler.explain` of the composed op on a bound system with the full
  capacity buffer → 1 graph / 0 breaks (the rebuild is outside); GH200
  benchmark row in `compile-no-graph-breaks` (default inductor vs
  `cuda_graphs=True`; the 2026-09-11 probe numbers are the target: 0.41 ms at
  256 atoms).

## Out of scope

The neighbour-list rebuild itself (host-side, data-dependent by nature);
`molix.engine` adapters (schema v3 already static).
