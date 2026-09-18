---
title: One derivative head — DerivativeHead replaces the func/grad kernels, GradMode and VirialDerivation
status: active
created: 2026-09-11
chain: derivation-head
---

# One derivative head (`DerivativeHead`)

## Summary

Operator directive (2026-09-11): 重新统一 Force、Virial 这些 Head 的设计，在整个项目中
统一用一套 head，不要有两条路线. Today `F = -∂E/∂pos` and `W = ∂E/∂ε` reach a batch
through four different bodies — `kernels.grad_force_pass` (autograd, MACE),
`kernels.func_force_pass` (functorch, PiNet), `modes.GradMode` / `FuncMode`
(the sequential `EnergyReadout(backward=True)` → `ForceReadout` session) and
`virial.VirialDerivation` (functorch + `geometry.rebuild_geometry`, NPT). Each
has its own leaf policy, its own geometry trick (STE, freeze, rebuild) and its
own compile story. This spec collapses them into **one** `nn.Module`,
`molpot.derivation.head.DerivativeHead`, which every call site composes.

## Domain basis

`F_i = -∂E/∂r_i`. Virial by the symmetric-strain derivative: positions, lattice
vectors and per-edge lattice remainders are all mapped by `v ↦ v (I + ε_sym)ᵀ`
at `ε = 0`, `W = ∂E/∂ε` (Cauchy: `σ = W / V`, shear not doubled —
`molpot.derivation.stress.symmetrize_strain_grad`), after Thompson, Plimpton &
Mattson, *J. Chem. Phys.* **131**, 154107 (2009). Nothing new; the physics is
the one already pinned by `autograd_forces_and_virial`.

## Design

`DerivativeHead(energy, *, virial=False)` — `energy(frame) -> Tensor (B,)`
reads `atoms.pos`, `edges.pair_index` / `pair_diff` (live, imaged),
`graphs.cell` (virial) off the frame it is handed and may write `atoms.energy`.
What is derived is fixed at construction: no `method=`, no per-call switches.

`forward(frame) -> frame` (in place):

1. `create_graph` = ambient `torch.is_grad_enabled()` at entry.
2. The head owns the position leaf: **the leaf is created inside the
   derivative primitive the head calls, never by a caller, and never left in
   the frame.** Strained geometry — `atoms.pos`, `graphs.cell`,
   `edges.pair_diff = pos_s[t] − pos_s[s] + shifts_s`, `edges.pair_dist`,
   `edges.shifts` — is rebuilt from that leaf and written into the frame for
   the energy call, so `∂/∂pos` and `∂/∂ε` flow through the geometry (this
   replaces the STE / freeze / rebuild tricks). `shifts` is `edges.shifts`
   when present, else `edge_diff_buf − (pos[t] − pos[s])`, detached.
3. One `torch.autograd.grad` — `force.autograd_energy_forces` (new, F only) or
   `force.autograd_forces_and_virial` (F and W). `force.py` stays the only
   module with `torch.autograd.grad`.
4. The caller's `atoms.pos` / `graphs.cell` / `edges.*` tensor objects go
   back; `graphs.energy` (reshaped to the `graphs` batch size; detached iff
   `create_graph` is False, `atoms.energy` likewise), `atoms.forces`, and
   `graphs.virial` are written. Straight-line restore, no `try/finally`.
5. `compile_core(compiler)` installs `compiler(self._eager_forward)` once;
   `forward` dispatches through it. The head is the compile boundary.

Call sites: `MacePotential.head` / `PiNetPotential.head` (`forward` dispatches
to it; `compute_forces=False` keeps the energy-only pipeline;
`MacePotential.energy_forces` is a raw-tensor adapter over the head;
`MacePotential.compile_core` compiles `energy_core` — the energy seam the
container route reaches through `call_energy` — and the head), `ForceReadout`
(= `EnergyReadout` + head), `Potentials._force_head` / `_virial_head` over
`calc_energy` (`calc_forces`, the mlip branch of `calc_energy_forces`,
`calc_energy_forces_virial`; the analytic `_classical_core` stays the classical
NVE fast path; `compile_core` compiles the heads).

`ForceDerivation` survives as the plain autograd tensor-level type
(`BasePotential`, `Sonata`, `PotentialComposer` need `F = -dE/dpos` on raw
tensors). `HessianDerivation` keeps its own leaf and composes
`autograd_forces_from_energy` directly.

## Files

Create: `src/molpot/derivation/head.py`, `tests/test_molpot/test_derivation/test_head.py`.

Delete: `src/molpot/derivation/{kernels,virial,geometry,derivative}.py`,
`src/molpot/derivation/modes/` (`func.py`, `grad.py`, `__init__.py`),
`tests/test_molpot/test_derivation/{test_kernels,test_virial}.py`.

Modify: `src/molpot/derivation/{__init__,force,protocol,energy_readout,force_readout,hessian}.py`,
`src/molpot/{__init__,README.md}`, `src/molpot/potentials/base.py`,
`src/molpot/composition/sonata.py`, `src/molpot/forcefield/potentials.py`,
`src/molzoo/mace/potential.py`, `src/molzoo/pinet/{potential,geometry}.py`,
`docs/molpot/user-guide/gradients.md`, `docs/molix/user-guide/md.md`,
`benchmarks/{pinet_compile_probe,bench_pinet_rank3_speed,bench_pinet_force_compile}.py`,
tests: `test_molpot/test_derivation/{conftest,test_force,test_readouts}.py`,
`test_molpot/test_forcefield/{test_potentials,test_predictions}.py`,
`test_molpot/test_readout/test_force.py`, `test_molzoo/test_mace/test_potential.py`,
`test_molzoo/test_pinet/{test_potential,test_geometry}.py`, `test_molix/test_md/test_driver.py`.

## Tasks

1. `head.py` + `force.autograd_energy_forces`; RED tests in `test_head.py`.
2. Delete the second route (kernels / modes / virial / geometry / sessions);
   `ForceReadout` runs the head; `HessianDerivation.graph_blocks` inlines its
   one energy forward.
3. molzoo: MACE and PiNet own a head; PiNet `method` gone; PiNet geometry
   trusts a live `pair_diff` like MACE's.
4. molpot.forcefield: `Potentials` heads; `compile_core` compiles them.
5. Docs + benchmarks + exports.

## Testing

Mirrored unit tests via Slurm only (`.scratch/molnex_tests/run_job.sbatch`),
`-m ""` so the slow explain tests run: inference `Compiler.explain(head, frame)`
→ 1 graph / 0 breaks for `virial=False` and `virial=True`; training-mode trace
→ `graph_break_count <= 1`, the break at the leaf (`force.py`).

## Out of scope

`molix.engine` (`EngineForward` / `StaticForward`) keep their own leaf for the
LAMMPS / AOTI export — they should become `DerivativeHead` users in a follow-up.
`StressDerivation` (tensor-level functorch stress, no in-tree producer) is left
as is. molzoo spec tables (`src/molzoo/specs/mace_*.md`) that name
`grad_force_pass` are the auditor's to patch. `.claude/notes` /
CLAUDE.md "Force derivation (dual backends)" need a `/mol:note` sync.
