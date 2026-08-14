---
title: Wilson B-matrix builder for Class-I internal coordinates
status: code-complete
created: 2026-08-14
grilled: true
---

# Wilson B-matrix builder for Class-I internal coordinates

## Summary

`ProjectedInternalHessianLoss` (`src/molix/core/losses/molecular.py:314`) already consumes a
Wilson `B = ∂q/∂x` matrix as `target["B"]`, but nothing in the library produces one — today the
matrix is hand-typed in a unit test (`tests/test_molix/test_core/test_losses_molecular.py:168`)
and in a regression script (`regressions/universal-class-i-03-losses.py:33`). This spec adds a
single molpot type, `WilsonBMatrix`, that builds `B` analytically (standard Wilson/Decius/Cross
s-vectors) for the two Class-I internal coordinates that project: bond lengths `r` over the
`bonds` topology and bend angles `θ` over the `angles` topology. It emits the exact
`(n_q, 3N)` layout the loss expects, per molecule, so a packed multi-graph batch yields one
independent block per graph and never mixes atoms across graphs. Users get exact derivatives
(no autograd, no finite differences) at whatever geometry they hand in; nothing about a
stationary point is assumed or exploited.

## Domain basis

Internal coordinate `q`, Cartesian coordinates `x ∈ R^{3N}` flattened as `pos.reshape(-1)`
(atom-major, xyz-minor, so column `3a + c` is component `c` of atom `a`). Row `m` of `B` is
`∂q_m/∂x`, non-zero only on the atoms of that term (Wilson's *s-vectors*).

**Stretch** `q = r_ij = ‖x_j − x_i‖`, with `û = (x_j − x_i)/r_ij`:

```
∂r/∂x_i = −û        ∂r/∂x_j = +û        (all other atoms: 0)
```

Row units: **dimensionless** (direction cosines, `Å/Å`).

**Bend** `q = θ_ijk`, apex atom `j` (the batch convention: `angles.atomj` is the central atom,
matching `AngleHarmonic`, `src/molpot/potentials/angles/harmonic.py:98`). With
`u = (x_i − x_j)/r_ij`, `v = (x_k − x_j)/r_kj`, `cos θ = u·v`, `sin θ = ‖u × v‖`:

```
∂θ/∂x_i = (cos θ · u − v) / (r_ij · sin θ)
∂θ/∂x_k = (cos θ · v − u) / (r_kj · sin θ)
∂θ/∂x_j = −( ∂θ/∂x_i + ∂θ/∂x_k )
```

Row units: **rad Å⁻¹ ≡ Å⁻¹**. Consequently `H_q = (B⁺)ᵀ H_x B⁺` carries kcal mol⁻¹ Å⁻² on bond
rows and kcal mol⁻¹ rad⁻² on angle rows, which is exactly what
`ProjectedInternalHessianLoss` documents (`molecular.py:248`).

Exact invariants used as tests, not as tolerances:

- Translation: `Σ_a s_a = 0` for every row (a rigid shift changes no internal coordinate).
- Rotation: under `x → R x` each s-vector maps `s_a → R s_a` (covariance, `R ∈ SO(3)`).
- GF consistency for one coordinate: `H_x = k BᵀB ⇒ diag(H_q) = k` exactly.

**Domain constraint (binding): no Seminario projection.** The Seminario recipe
`k = ûᵀ(−H_AB)û` (per-bond eigen projection of the interaction sub-block) is exact **only at a
QM stationary point**; off-minimum it silently absorbs the residual gradient into the force
constant. This builder therefore never eigen-decomposes a Hessian sub-block and never assumes
a minimum: it returns `B` at the geometry it is handed, and the projection to `H_q` stays in
the existing loss. Equilibrium values `r0` / `θ0` are fitted by energy losses elsewhere.

References:

- Wilson, E. B.; Decius, J. C.; Cross, P. C. *Molecular Vibrations: The Theory of Infrared and
  Raman Vibrational Spectra*, McGraw-Hill, 1955, Ch. 4 (s-vector formulas for stretch and bend).
- Wilson, E. B. *J. Chem. Phys.* **9**, 76 (1941). https://doi.org/10.1063/1.1750829
- Seminario, J. M. *Int. J. Quantum Chem.* **60**, 1271 (1996).
  https://doi.org/10.1002/(SICI)1097-461X(1996)60:7<1271::AID-QUA8>3.0.CO;2-W
  (cited for the minimum-geometry restriction this spec deliberately does not rely on).

## Design

**Placement.** `src/molpot/forcefield/wilson.py`. Internal-coordinate definitions are Class-I
physics and must not live in molix; `molpot.ir` is the *parameter* IR (bags / units / scaling),
not geometry; `molpot/forcefield/` already owns the Class-I bonded surface plus a stateless
geometry builder of the same shape (`NeighborListBuilder`, `neighbor_list.py:15`). No new
top-level concept, no new subpackage.

**Type.** One plain class (not `nn.Module` — no parameters, no buffers), constructed directly:

```python
class WilsonBMatrix:
    def __init__(self, *, sin_tol: float = 1e-6, length_tol: float = 1e-9) -> None: ...
    def stretch_rows(self, pos: Tensor, bond_index: Tensor) -> Tensor:  # (n_b, 3N)
    def bend_rows(self, pos: Tensor, angle_index: Tensor) -> Tensor:    # (n_a, 3N)
    def graph_blocks(
        self, pos: Tensor, atom_batch: Tensor, *,
        bond_index: Tensor | None = None, angle_index: Tensor | None = None,
    ) -> list[Tensor]:                                                  # per graph (n_q_g, 3N_g)
```

Three primitive public methods, one concern each: bond rows, angle rows, per-molecule blocks.
No `make_*` factory, no `compute_all`, no config object.

**Index conventions (identical to the existing potentials, so a caller cannot mis-wire).**
`bond_index` is COO `(2, n_bonds)`; `angle_index` is COO `(3, n_angles)` with row 1 = apex.
A tensor whose leading dim is not 2 (resp. 3) raises `ValueError` naming the
`bond_index` vs `edge_index` distinction, mirroring `BondHarmonic`
(`src/molpot/potentials/bonds/harmonic.py:109`) — a cutoff edge is not a bond, and a wrong-shape
tensor must never silently produce a plausible matrix.

**Narrow inputs, not the batch blob.** `WilsonBMatrix` takes tensors, never a batch
`TensorDict`; callers convert the nested namespaces (`bonds.bond_index` or
`bonds.atomi/atomj`; `angles.atomi/atomj/atomk`) to COO at the call site, exactly as
`Potentials._bond_index` / `_angle_index` (`forcefield/potentials.py:517,531`) already do. This
keeps the new type free of a duplicated namespace reader and free of an ambient context blob.

**Row order and per-graph blocks.** Within one block, rows are *all bond rows in the given
input order, then all angle rows in the given input order* — deterministic, so a caller's
`target["diagonal"]` force-constant vector (`k_r…` then `k_θ…`) aligns by construction.
`graph_blocks` splits terms by `atom_batch`, requires each term's atoms to share one graph
(else `ValueError` naming the offending term index — this is the structural guarantee that
packed batches never mix graphs), requires `atom_batch` to be non-decreasing (packed collate's
`repeat_interleave` layout), rebases global atom ids to graph-local ids, and builds each block
against that graph's own positions. Block `g` is `(n_q_g, 3 N_g)`; graphs with no terms give
`(0, 3 N_g)`.

**Numerics.** Pure torch ops on the input dtype/device (float32 and float64 both supported, and
`B` stays differentiable w.r.t. `pos` for free, though nothing in this spec needs that).
Analytic s-vectors instead of `torch.autograd.functional.jacobian`: exact to machine precision,
one pass, no graph retention, and no dependence on `pos.requires_grad`. Degenerate geometry
fails loud rather than being clamped: `r < length_tol` raises, and `|sin θ| < sin_tol`
(near-linear bend, where the coordinate itself is undefined) raises naming the row index —
clamping there would emit a huge wrong row and poison the projected force constants.

### Reuse decision

No `librarian_report` was supplied with this task; the candidate set below comes from a direct
scan of `src/molpot/` and `src/molix/core/losses/`. Every candidate is resolved:

- `ProjectedInternalHessianLoss.project` (`src/molix/core/losses/molecular.py:278`) — **reuse**.
  It is the consumer of `B`; this spec adds no projection, no `pinv`, no redundant-coordinate
  handling of its own.
- `Potentials._bond_index` / `Potentials._angle_index` (`forcefield/potentials.py:517,531`) —
  **reuse** at call sites (batch namespace → COO). `WilsonBMatrix` accepts that COO form so the
  reader is not duplicated; they stay private and unchanged (no generalization task needed
  because the new type does not read a batch).
- `BondHarmonic` / `AngleHarmonic` (`potentials/bonds/harmonic.py`, `potentials/angles/harmonic.py`)
  — **reuse** of convention and of the loud shape guard (COO layouts, apex = row 1). Their
  energy math is not derivative math, so the s-vector code is new.
- `NeighborListBuilder` (`forcefield/neighbor_list.py:15`) — **pattern**. `WilsonBMatrix` copies
  its shape: stateless builder class in the same package, keyword-only tensor arguments, public
  method + private helpers, direct construction.
- `molix.units` — **new (none needed)**: rows are dimensionless or Å⁻¹, so there is no
  conversion factor to restate; units are documented, never hard-coded as floats.
- `torch.autograd.functional.jacobian` — **new**: rejected, see *Numerics* above.

## Files to create or modify

- src/molpot/forcefield/wilson.py (new)
- src/molpot/forcefield/__init__.py
- src/molpot/__init__.py
- tests/test_molpot/test_forcefield/test_wilson.py (new)
- regressions/universal-class-i-07-wilson.py (new)

## Tasks

- [x] Write failing unit tests for WilsonBMatrix stretch/bend rows (tests/test_molpot/test_forcefield/test_wilson.py → TestWilsonBMatrix): diatomic golden row, right-angle bend s-vector golden, water finite-difference agreement, translation/rotation invariants, empty inputs, and loud failures for (E,2)-shaped bond_index, zero-length bond, near-linear angle
- [x] Implement WilsonBMatrix.stretch_rows in src/molpot/forcefield/wilson.py (new) with analytic bond s-vectors and the COO shape guard
- [x] Implement WilsonBMatrix.bend_rows in src/molpot/forcefield/wilson.py with analytic bend s-vectors (apex = angle_index row 1) and the sin_tol guard
- [x] Write failing unit tests for WilsonBMatrix.graph_blocks (tests/test_molpot/test_forcefield/test_wilson.py → TestWilsonBMatrixGraphBlocks): per-graph (n_q_g, 3N_g) shapes, bonds-then-angles row order, block equality against standalone single-molecule builds, cross-graph term and non-monotonic atom_batch raising ValueError
- [x] Implement WilsonBMatrix.graph_blocks in src/molpot/forcefield/wilson.py (split by atom_batch, local index rebase, cross-graph and contiguity guards)
- [x] Export WilsonBMatrix from src/molpot/forcefield/__init__.py and src/molpot/__init__.py
- [x] Add Google-style docstrings per doc.style to src/molpot/forcefield/wilson.py with tensor shapes, per-row units (bond rows dimensionless, angle rows Å⁻¹), the Wilson/Decius/Cross 1955 and Wilson 1941 references, and the explicit no-Seminario-projection caveat
- [x] Add regression example regressions/universal-class-i-07-wilson.py (public API only; hard-coded goldens, no third-party runtime)
- [x] Run full check + test suite

## Testing strategy

Unit tests live at `tests/test_molpot/test_forcefield/test_wilson.py`, mirroring
`src/molpot/forcefield/wilson.py`; classes `TestWilsonBMatrix` (row builders) and
`TestWilsonBMatrixGraphBlocks` (packed batches) mirror the type. Each test targets a single
method. All numeric tests run in float64 with hard-coded geometries and hard-coded expected
values; no third-party oracle is imported anywhere.

**Happy path (hard-coded analytic goldens).**

- `stretch_rows` on `pos = [[0,0,0],[1,0,0]]`, `bond_index = [[0],[1]]` equals
  `[[-1,0,0, 1,0,0]]` — the same golden already asserted at
  `tests/test_molix/test_core/test_losses_molecular.py:168` and
  `regressions/universal-class-i-03-losses.py:33`, now produced instead of typed.
- `bend_rows` on `pos = [[1,0,0],[0,0,0],[0,1,0]]`, `angle_index = [[0],[1],[2]]` (apex atom 1,
  θ = 90°, r = 1 Å) equals `[[0,-1,0, 1,1,0, -1,0,0]]` from the closed-form s-vectors.
- Multi-term stacking: two bonds and one angle produce three rows in bonds-then-angles order,
  each row non-zero only in its own atoms' columns.

**Domain validation.**

- Water at the experimental geometry (O `(0,0,0)`, H `(0.9572,0,0)`, H `(-0.23921,0.92695,0)`;
  r = 0.9572 Å, θ = 104.52°): every entry of `stretch_rows` and `bend_rows` matches a central
  finite difference of `r` and `θ = acos(û_ji·û_jk)` with `h = 1e-5` Å to `atol 1e-7`.
- Translation invariance: per-row sum of s-vectors is zero (`atol 1e-12`), and a global shift of
  `pos` leaves `B` bitwise-comparable (`atol 1e-12`).
- Rotation covariance: for a fixed `R ∈ SO(3)`, each per-atom 3-vector block of the rotated
  build equals `R @ s` of the unrotated build (`atol 1e-10`).
- GF consistency: with `H_x = k BᵀB` (`k = 2.0`), `torch.linalg.pinv`-based projection returns
  `diag(H_q) = [2.0]` for the diatomic (`atol 1e-6`), matching the loss's documented identity.

**Edge cases.**

- Empty `bond_index` `(2,0)` / `angle_index` `(3,0)` return `(0, 3N)` with `pos`'s dtype and
  device; a graph with no terms yields a `(0, 3N_g)` block.
- `bond_index` passed as `(E,2)` raises `ValueError` whose message names the
  `bond_index` vs `edge_index` distinction; `angle_index` with leading dim ≠ 3 raises.
- Zero-length bond (`r < length_tol`) raises `ValueError` naming the term index.
- Near-linear bend (θ = 179.9999°) raises `ValueError` naming the row index; θ = 179° succeeds
  and still matches finite differences.
- float32 input returns float32 rows; dtype/device of `pos` is preserved.

**Packed batches.**

- Two-molecule batch (water then H₂, `atom_batch = [0,0,0,1,1]`) gives two blocks of shapes
  `(3, 9)` and `(1, 6)`, each exactly equal to the standalone single-molecule build with locally
  rebased indices — cross-graph coupling is structurally impossible because each block is only
  `3 N_g` wide.
- A term whose atoms span two graphs, and a non-monotonic `atom_batch`, each raise `ValueError`.

**Regression example.** `regressions/universal-class-i-07-wilson.py` — minimal public-API script
(`from molpot import WilsonBMatrix`, plus
`from molix.core.losses import ProjectedInternalHessianLoss` for the projection step). It
asserts, against literals hard-coded in the file with a provenance comment (analytic
Wilson/Decius/Cross s-vectors, hand-derived 2026-08-14, no tool, no third party at runtime):
the diatomic row `[[-1,0,0,1,0,0]]`, the right-angle bend row `[0,-1,0, 1,1,0, -1,0,0]`,
`ProjectedInternalHessianLoss.project(2·BᵀB, B) == [[2.0]]` (`atol 1e-6`), and the two-molecule
block shapes `(3,9)` / `(1,6)` with equality to the standalone builds. Exits 0 on success.

## Out of scope

- Producing the Cartesian Hessian `H_x` itself — that is the sibling spec
  `universal-class-i-06-hessian`. This spec only builds `B`.
- Torsion and improper rows: `n_q` covers stretch and bend only (those are the coordinates the
  projected-Hessian loss consumes today).
- Seminario per-bond eigen projection `ûᵀ(−H_AB)û → k` and any other minimum-geometry recipe —
  rejected on physics grounds, see Domain basis. Considered and refused, not merely omitted.
- Fitting `r0` / `θ0`: equilibrium values come from the energy losses in other chain links.
- Redundant-coordinate machinery beyond the `pinv` already inside the loss: no G-matrix, no
  Eckart frame, no normal-mode frequencies, no internal-coordinate transformation utilities.
- Dataset / target plumbing that writes `target["B"]` into a batch or cache — a later chain link.
- Sparse or block-diagonal storage, `torch.compile` / CUDA-graph capture, and any performance
  work; the dense per-molecule `(n_q, 3N)` block is the deliverable.
- Pre-existing debt noticed while surveying the package but **not** touched here:
  `src/molpot/forcefield/neighbor_list.py:82` does `import molrs`, which violates the
  molpy-only rule in CLAUDE.md. Flagged for a separate `/mol:fix`; this spec's new module
  imports neither `molrs` nor `molpy`.
