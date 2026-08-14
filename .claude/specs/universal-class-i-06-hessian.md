---
title: Cartesian Hessian producer for Class-I force-constant supervision
status: code-complete
created: 2026-08-14
chain: universal-class-i
grilled: true
---

# Cartesian Hessian producer for Class-I force-constant supervision

## Summary

`ProjectedInternalHessianLoss` (`src/molix/core/losses/molecular.py:239`) already
projects a Cartesian Hessian onto internal coordinates, but nothing in the repo
produces that Hessian — the loss is currently only exercisable on hand-built
matrices (`regressions/universal-class-i-03-losses.py`). This spec adds one
derivation type, `HessianDerivation`, in a new `src/molpot/derivation/hessian.py`
(sibling of `stress.py:20`), which turns any Class-I energy graph into
`H = ∂²E/∂pos²` in the exact Cartesian layout the loss consumes: `(3N, 3N)`,
row/column order `x0, y0, z0, x1, …`, units kcal·mol⁻¹·Å⁻² for kcal/mol energies
and Å positions. It works per molecule inside a packed batch by returning one
Hessian block per graph, so graph identities are never mixed, and it keeps the
Hessian connected to model parameters when grad is enabled so a Hessian loss can
train a parameter predictor.

## Domain basis

Cartesian Hessian of the potential energy surface:

```
H_ij = ∂²E / ∂x_i ∂x_j = − ∂F_j / ∂x_i ,     i, j ∈ [0, 3N)
```

with the flat Cartesian index `i = 3a + c` for atom `a`, component `c ∈ {x,y,z}`
— i.e. the `x0,y0,z0,x1,…` ordering already fixed by
`molix.data.pes_bundle.PesBundle` (`graphs.hessian`, shape `(C, 3N, 3N)`).

Invariants used as test oracles:

- **Symmetry** (Schwarz/Clairaut): `H = Hᵀ` for any twice-differentiable `E`.
- **Translational invariance**: `H · t = 0` for a uniform translation `t`, hence
  every row and column of an exact Hessian sums to zero (per Cartesian
  component).
- **Graph block-diagonality**: for a packed batch of non-interacting molecules
  (the collate contract — `edge_index` / `bond_index` are rebased per sample, so
  no term couples two graphs), `E_total = Σ_g E_g` gives
  `∂²E_total/∂x_i∂x_j = ∂²E_g/∂x_i∂x_j` for `i, j ∈ g` and exactly `0` for
  `i ∈ g, j ∈ g′ ≠ g`. This is what makes per-graph blocks of a single batch-wide
  differentiation exact rather than an approximation.

Closed-form golden — harmonic bond dimer, `E = ½k(r − r₀)²`,
`d = x₂ − x₁`, `r = ‖d‖`, `u = d/r` (matches `molpot.potentials.BondHarmonic`):

```
∂²E/∂d∂d = k u uᵀ + k (r − r₀)/r · (I − u uᵀ)
H_11 = H_22 = ∂²E/∂d∂d ,   H_12 = H_21 = − ∂²E/∂d∂d
```

At equilibrium (`r = r₀`) the transverse term vanishes and the Hessian is the
rank-1 form `H = k BᵀB` with the Wilson row `B = ∂r/∂x = (−u, u)` — numerically
identical to the `H_x = k BᵀB, k = 2.0` golden already hard-coded in
`regressions/universal-class-i-03-losses.py`, so this spec's golden chains
directly into spec 03's.

Units: energy kcal/mol, positions Å ⇒ Hessian kcal·mol⁻¹·Å⁻². Like
`molpot.derivation.kernels`, the producer is unit-agnostic — it differentiates
whatever the energy core emits; keeping the core on kcal/mol is the caller's job.

Reference:

- Wilson, E. B. *J. Chem. Phys.* **9**, 76 (1941). https://doi.org/10.1063/1.1750829
- Wilson, Decius & Cross, *Molecular Vibrations*, McGraw-Hill (1955), ch. 2
  (Cartesian force-constant matrix and its zero modes).

## Design

**One new type, autograd double-backward only.**
`HessianDerivation(nn.Module)` in `src/molpot/derivation/hessian.py`, shaped like
its siblings `StressDerivation` (`stress.py:20`) and `ForceDerivation`
(`force.py:133`): constructed with no arguments, `forward(energy_fn, pos)` as the
tensor-level entry. Unlike `ForceDerivation` it deliberately has **no `method`
dispatch** — a functorch (`torch.func.jacrev ∘ grad`) backend is not offered
because the cuEquivariance double-backward story is owned by the active
`cuet-force-doublebackward` spec; this spec targets pure-torch Class-I energy
graphs and states that scope in the module docstring.

**No third differentiation path.** The *first* derivative is never re-derived
here: `forward` composes `autograd_forces_from_energy` (`force.py:99`) and
`graph_blocks` composes `grad_force_pass` (`kernels.py:74`) with
`create_graph=True, detach_energy=False`. The only new differentiation call in
this spec is the *second* `torch.autograd.grad` that walks the rows of
`∂E/∂pos`. Finite differences are a test oracle at most, never a production
path.

**Two primitive public methods** (construct → one concern → read result):

1. `forward(energy_fn: Callable[[Tensor], Tensor], pos: Tensor) -> Tensor` —
   dense `(3N, 3N)` Hessian of a scalar `energy_fn(pos)`. Takes a private
   `requires_grad` leaf from `pos.detach()`, evaluates the energy inside
   `torch.enable_grad()`, gets `g = ∂E/∂pos` from `autograd_forces_from_energy`
   (negated), then differentiates each of the `3N` entries of `g.reshape(-1)`
   with `torch.autograd.grad(..., retain_graph=True, create_graph=…)` and stacks
   the rows. Returned as computed — **not** symmetrized, so a broken energy graph
   shows up as asymmetry instead of being silently averaged away.
2. `graph_blocks(energy_core: EnergyCore, batch: TensorDict) -> list[Tensor]` —
   one `(3Nᵢ, 3Nᵢ)` block per graph, in graph order. Runs **one** energy forward
   via `grad_force_pass`, then for each global row `i` differentiates
   `(-forces).reshape(-1)[i]` w.r.t. the position leaf and keeps only the columns
   of `i`'s own graph, so peak memory is `Σ (3Nᵢ)²` rather than `(3N_total)²`.
   Graph extents come from `("atoms", "batch")` alone (single source of truth);
   a `batch` vector that is not non-decreasing raises `ValueError` — the guard
   that makes "never mix graph identities" a checked property rather than a
   comment. `graph_blocks` mutates the batch exactly as
   `grad_force_pass` documents (position leaf + `atoms.forces` written in place)
   and writes **no** hessian key itself: batch-key writes and readout modes stay
   with `kernels.py` / a future readout (spec 08).

**Differentiability policy** mirrors `force.py`: `create_graph` resolves to the
ambient `torch.is_grad_enabled()` at entry, so training keeps
`∂³E/∂pos²∂θ` alive (a `ProjectedInternalHessianLoss` can reach a Class-I
parameter head) while `torch.no_grad()` yields a detached Hessian for reporting.

**Ownership / lifecycle.** Stateless `nn.Module` (no buffers, no parameters, no
session side-channel), reusable across batches, exported from
`molpot.derivation` and the `molpot` top-level namespace next to
`ForceDerivation` / `StressDerivation`. Google docstrings with an explicit Units
line and the Wilson reference, per `doc.style: google`.

### Reuse decision

No `librarian_report` was supplied for this sub-spec (blueprint refresh
deferred); the candidates below come from a direct survey of
`src/molpot/derivation/`, `src/molix/core/losses/`, `src/molix/data/` and
`tests/test_molpot/test_derivation/`.

- `reuse` `molpot.derivation.force.autograd_forces_from_energy` (`force.py:99`)
  — first derivative for the tensor-level `forward`.
- `reuse` `molpot.derivation.kernels.grad_force_pass` (`kernels.py:74`) —
  position-leaf ownership + single energy forward + first derivative for
  `graph_blocks`; no leaf logic is re-implemented.
- `reuse` `molpot.derivation.protocol` keys/helpers (`POS_KEY`, `FORCES_KEY`,
  `EnergyCore` shape) — no new key constants.
- `reuse` `molix.data.pes_bundle` `graphs.hessian` layout contract
  (`(3N, 3N)`, `x0,y0,z0,…`, energy/Å²) as the output convention; nothing is
  redefined.
- `reuse` `molix.core.losses.ProjectedInternalHessianLoss.project`
  (`molecular.py:278`) as the consumer in the regression example — the produced
  `H` is fed to the existing projection, not to a new one.
- `reuse` `tests/test_molpot/test_derivation/conftest.py` `KernelToyPotential`
  / `kernel_toy_batch` as the packed-batch unit fixture (`E = s Σ‖r‖²` ⇒
  `H = 2s·I₃ₙ`, block-diagonal by construction). No new fixture module.
- `pattern` `StressDerivation` (`stress.py:20`) and `ForceDerivation`
  (`force.py:133`) — naming (`<Quantity>Derivation`), `nn.Module` base,
  `forward(energy_fn, …)` signature shape, `ValueError` on contract violations.
- `new — HessianDerivation` itself: no existing symbol produces a second
  derivative of the energy; `ForceDerivation` is first-order only and
  generalizing it to arbitrary order would force a `method`/`order` switch onto
  the hottest path in the repo (explicitly forbidden by the design defaults).
- `new — HessianDerivation.graph_blocks`: no existing helper slices a
  per-atom-square quantity into per-graph blocks; `molix.F.scatter` is reduction
  only.

## Files to create or modify

- `src/molpot/derivation/hessian.py` (new) — `HessianDerivation` with
  `forward` and `graph_blocks`.
- `src/molpot/derivation/__init__.py` — import + `__all__` entry for
  `HessianDerivation`.
- `src/molpot/__init__.py` — re-export `HessianDerivation` alongside
  `ForceDerivation` / `StressDerivation` (line 29 import, `__all__` near
  line 107).
- `tests/test_molpot/test_derivation/test_hessian.py` (new) — `TestHessianDerivation`.
- `regressions/universal-class-i-06-hessian.py` (new) — public-API regression
  example with hard-coded analytic goldens.

## Tasks

- [x] Write failing unit tests for HessianDerivation (tests/test_molpot/test_derivation/test_hessian.py → TestHessianDerivation): analytic dimer goldens, invariants, per-graph blocks, contiguity guard, grad policy
- [x] Implement `HessianDerivation.forward` in src/molpot/derivation/hessian.py, composing `autograd_forces_from_energy` for the first derivative (no functorch, no finite differences)
- [x] Implement `HessianDerivation.graph_blocks` in src/molpot/derivation/hessian.py, composing `grad_force_pass`, slicing per-graph columns from `("atoms","batch")` and raising `ValueError` on a non-monotonic batch vector
- [x] Export `HessianDerivation` from src/molpot/derivation/__init__.py and src/molpot/__init__.py (import + `__all__` in both)
- [x] Add Google docstrings to src/molpot/derivation/hessian.py with Units (kcal·mol⁻¹·Å⁻²), Cartesian `x0,y0,z0,…` layout, autograd-only scope note, and the Wilson 1941 reference
- [x] Verify the closed-form harmonic-dimer Hessians (equilibrium and stretched), symmetry, zero row sums, and cross-graph zero coupling in tests/test_molpot/test_derivation/test_hessian.py
- [x] Add regression example regressions/universal-class-i-06-hessian.py (public API only; hard-coded goldens, no third-party runtime)
- [x] Run full check + test suite

## Testing strategy

Unit tests live only in `tests/test_molpot/test_derivation/test_hessian.py`
(mirror of `src/molpot/derivation/hessian.py`), class `TestHessianDerivation`,
each test targeting a single method, float64, reusing the existing
`conftest.py` fixtures.

`forward` — happy path and domain validation (hard-coded values):

- Quadratic toy `E = s Σ‖r‖²` via `KernelToyPotential.total_energy` with
  `s = 1.0`: `H == 2.0 · I₃ₙ` exactly (atol 1e-12).
- **Equilibrium harmonic dimer** (`BondHarmonic(k=[2.0], r0=[1.5])`, atoms at
  `(0,0,0)` and `(1.5,0,0)`): the only non-zero entries are
  `H[0,0] = H[3,3] = 2.0`, `H[0,3] = H[3,0] = −2.0` (atol 1e-10) — equals
  `k BᵀB` with `B = [[−1,0,0,1,0,0]]`, chaining spec 03's golden.
- **Stretched harmonic dimer** (same `k`, `r0 = 1.5`, atoms at `(0,0,0)` and
  `(2.0,0,0)`): longitudinal `H[0,0] = H[3,3] = 2.0`, `H[0,3] = −2.0`;
  transverse `H[1,1] = H[2,2] = H[4,4] = H[5,5] = k(r−r₀)/r = 0.5`,
  `H[1,4] = H[2,5] = −0.5`; all other entries `0` (atol 1e-10). This case is the
  one that fails if the `(I − uuᵀ)` term is dropped.
- Invariants on the stretched case: `max|H − Hᵀ| ≤ 1e-10` and
  `max|H.sum(dim=0)| ≤ 1e-10` (translational invariance).
- Edge cases: single-atom system → `(3, 3)` zero matrix for a translation-only
  energy; `pos` that is not `(N, 3)` → `ValueError`.

`graph_blocks` — batch behaviour:

- `kernel_toy_batch(n_atoms=4, n_graphs=2)` with `KernelToyPotential(scale=1.5)`:
  returns a 2-element list, each `(6, 6)` and equal to `3.0 · I₆` (atol 1e-12).
- Block-diagonality: the dense `forward` Hessian of the same batch has the two
  diagonal blocks equal to `graph_blocks`' output (atol 1e-12) and cross-graph
  off-diagonal blocks exactly `0.0`.
- Contiguity guard: a batch whose `("atoms","batch")` is shuffled
  (e.g. `[0,1,0,1]`) raises `ValueError` naming `atoms.batch`.
- One energy forward only: `KernelToyPotential.n_forward == 1` after a
  `graph_blocks` call.

Grad policy:

- Under grad, `loss = blocks[0].sum()`; `loss.backward()` gives a non-zero
  `KernelToyPotential.scale.grad` (Hessian reaches parameters — the property
  spec 08 depends on).
- Under `torch.no_grad()`, the returned Hessian has `requires_grad is False`.

**Regression example** — `regressions/universal-class-i-06-hessian.py`
(repo-root `regressions/`, never under `tests/`): builds the equilibrium
harmonic dimer through the public API (`molpot.BondHarmonic`,
`molpot.HessianDerivation`), asserts the 6×6 Hessian against the hard-coded
golden above (atol 1e-10), then chains it into the existing consumer —
`ProjectedInternalHessianLoss.project(H, B)` with `B = [[−1,0,0,1,0,0]]` must
give `[[2.0]]` (atol 1e-6) and `ProjectedInternalHessianLoss()` against
`diagonal = [2.0]` must give `0` (≤ 1e-12). All goldens are analytic
(Wilson closed form, 2026-08-14); no third-party oracle is imported or
subprocessed at runtime. Prints one summary line and exits 0.

## Out of scope

- **Wilson `B` matrix construction** — owned by sibling spec
  `universal-class-i-07-wilson`; this spec only guarantees a layout `B` can
  multiply.
- **Writing `graphs.hessian` onto the batch / a `HessianReadout` mode / loss
  wiring** — owned by sibling spec `universal-class-i-08-adapter`.
  `graph_blocks` returns values and writes no hessian key.
- **A functorch (`torch.func.hessian` / `jacrev ∘ grad`) backend and any
  cuEquivariance energy graph** — blocked on the active
  `cuet-force-doublebackward` spec; deliberately not a `method=` switch here.
- **Finite-difference Hessians as a production path** — considered and rejected:
  `6N²` energy evaluations, no parameter gradient, and the analytic goldens make
  it unnecessary even as an oracle.
- **Vectorized row batching** (`torch.autograd.grad(..., is_grads_batched=True)`)
  and any other throughput work — correctness first; a follow-up perf spec can
  swap the row loop behind the same two methods.
- **Mass-weighted Hessians, normal modes, frequencies, projecting out
  translations/rotations, partial/blocked Hessians of a subset of atoms.**
- **Periodic or otherwise graph-coupled systems**: `graph_blocks` assumes the
  collate contract of independent graphs and does not detect cross-graph terms
  beyond the block-diagonality test.
- **Dataset-side hessian targets and any training run** (Iron law: no training
  before verified optimization).
