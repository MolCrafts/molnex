---
title: Pure-torch binned (cell-list) O(N) build path for molix.md.NeighborList
status: approved
created: 2026-08-09
grilled: true
chain: md-neighborlist-skin
---

# Pure-torch binned (cell-list) O(N) build path for molix.md.NeighborList

## Summary

`molix.md.NeighborList` builds its list by handing the whole system to the
compiled O(N²) pair kernel: every rebuild enumerates `N(N-1)/2` candidate pairs,
so a 4 000-atom condensed-phase box spends ~8 million distance evaluations per
rebuild to find ~10⁵ edges. That cost is the wall the TODO at
`src/molix/data/tasks/neighbor.py:116` names ("the kernel is O(N²); large
condensed-phase systems need a cell-list/Verlet neighbour algorithm (not yet in
`molix.op`)") and the same gap recorded in `.claude/notes/architecture.md`. This
link adds a second, pure-torch build backend selected by a new keyword-only
constructor argument `bin`: atoms are sorted into a periodic bin grid derived
from the cell and `r_build`, and only a fixed stencil of neighbouring bins is
searched, making the build linear in `N` at fixed density and GPU-capable with
no per-atom Python loop. `bin=None` (the default) leaves the existing kernel
path bit-for-bit unchanged — it stays the production default *and* becomes the
equivalence oracle the binned path is tested against: for every fixture the two
paths must return the identical set of `(source, target, shift)` triples. The
compiled C++/CUDA kernel is not modified, and no consumer of the list has to
change: both paths feed the same `_write`, the same fixed-capacity buffers, the
same dead-edge padding, and the same link-04 `update()` policy.

## Domain basis

Units: `bin`, `r_build`, `cutoff`, `skin`, positions, cell entries, bin
thicknesses and perpendicular widths in Å; bin counts `n_i`, stencil half-widths
`k_i` and edge counts dimensionless. Cell vectors `a_1, a_2, a_3` are the **rows**
of the `(3, 3)` cell, `V = |det(cell)|` (Å³), perpendicular widths
`w_i = V / ‖a_j × a_k‖` (link 01), fractional coordinates `s = pos · cell⁻¹`.

**Stencil completeness (why a bounded stencil is exact).** Bins are cubes in
*fractional* space: atom `A` sits in bin `p_i = ⌊s_i n_i⌋` along axis `i`. If two
atoms are `m` bins apart along axis `i` (minimal modular difference), then
`s_B ≥ (p+m)/n_i` while `s_A < (p+1)/n_i`, so `|Δs_i| > (m−1)/n_i` **strictly**.
The perpendicular component of the displacement along axis `i` is `|Δs_i| · w_i`,
and `‖d‖ ≥ |d · n̂_i| = |Δs_i| w_i`, hence with the *effective* perpendicular bin
thickness `b_i = w_i / n_i`:

```
‖d‖ > (m − 1) · b_i
```

So a stencil half-width `k_i` with `k_i · b_i ≥ r_build` is **complete**: any pair
more than `k_i` bins apart on some axis has `‖d‖ > k_i b_i ≥ r_build` and is
correctly excluded by the `‖d‖ ≤ r_build` filter. The strict inequality is what
makes the classic textbook statement exact at the boundary: `b_i = r_build`
⇒ `k_i = 1` ⇒ the 27-cell (3×3×3) stencil is complete, with no epsilon fudge.

**Bin-size choice.** Candidate volume for half-bins `b = r_build / k` is
`(2 + 1/k)³ · r_build³`, decreasing monotonically in `k`: `k = 1` → `27 r³`
(6.4× the `4.19 r³` sphere actually needed), `k = 2` → `15.6 r³` (3.7×), against a
sorting/gather cost that grows with the bin count. LAMMPS resolves this at
`k = 2` (`src/nbin_standard.cpp`, `binsize_optimal = 0.5 * cutneighmax`), which is
the auto default adopted here: `bin = 0.0` ⇒ requested `b = r_build / 2`.

**Minimum image by fractional rounding is exact inside the link-04 guard.**
Write a periodic displacement as `d = Σ_i f_i a_i`. Its perpendicular component
along axis `i` is `|f_i| w_i = |d · n̂_i| ≤ ‖d‖`. Therefore, if
`‖d‖ ≤ r_build ≤ min_i w_i / 2 ≤ w_i / 2` (exactly the link-01/04 guard), then
`|f_i| ≤ 1/2` for every `i` — i.e. **any** in-range image is already the one
selected by `f ← f − round(f)`. Fractional rounding and the kernel's *sequential*
diagonal reduction therefore return the same, unique minimum image everywhere the
guard admits, which is what makes edge-set equality between the two paths a
theorem rather than a coincidence. (Ties `|f_i| = 1/2` are reachable only when
`r_build = min_i w_i / 2` exactly *and* a pair sits exactly on the boundary, where
`torch.round`'s half-to-even and C's half-away-from-zero can differ; measure-zero,
documented, and avoided by the fixtures, which keep `r_build` strictly inside the
guard.)

**Filter parity with the kernel.** The compiled backends accept a pair iff
`0 < r ≤ r_cut` (`src/molix/op/src/neighbors/get_neighbor_pairs.cpp:52`
`(distances <= cutoff) & (distances > 0)`; `.cu:67` drops `distance2 > cutoff2 ||
distance2 == 0`). The binned path must use the same closed upper bound *and* the
same `r > 0` rejection, so coincident atoms — and pairs separated by exactly one
lattice vector, whose minimum image is the zero vector — are dropped identically.

**Edge order is not part of the contract.** The binned path emits edges in
bin-sorted order, the kernel in upper-triangle index order. The binding contract
is **set equality of canonical `(min(i,j), max(i,j), oriented shift)` triples plus
equal edge counts**; downstream consumers aggregate with order-independent
scatter/`index_add_` reductions, and the shift buffers are consumed positionally
alongside `edge_index`, never by index-order assumption.

**References.**

- Allen, M. P.; Tildesley, D. J. *Computer Simulation of Liquids*, 2nd ed.;
  Oxford University Press, 2017. DOI:
  [10.1093/oso/9780198803195.001.0001](https://doi.org/10.1093/oso/9780198803195.001.0001)
  — cell (link-cell) lists, the 27-cell stencil, and minimum-image validity.
- Thompson, A. P. et al. "LAMMPS — a flexible simulation tool…", *Comput. Phys.
  Commun.* **271** (2022) 108171. DOI:
  [10.1016/j.cpc.2021.108171](https://doi.org/10.1016/j.cpc.2021.108171); binning
  policy in `lammps/lammps` develop `src/nbin_standard.cpp`
  (`binsize_optimal = 0.5 * cutneighmax`) and https://docs.lammps.org/neighbor.html.
- Quentrec, B.; Brot, C. *J. Comput. Phys.* **13** (1973) 430. DOI:
  [10.1016/0021-9991(73)90046-6](https://doi.org/10.1016/0021-9991(73)90046-6)
  — the original cell method. **Caveat:** paywalled and *not* re-verified for this
  spec; cited for attribution only. Every relation above is derived in-line and
  cross-checked against Allen & Tildesley and the LAMMPS source.

## Design

Preconditions: link 03 (class is `molix.md.NeighborList`; the pipeline task is
imported as `NeighborListTask`) and link 04 (`skin`, `r_build = cutoff + skin`,
`update()` policy, kernel invoked at `cutoff=r_build`, half-width guard evaluated
on `r_build`). Both are assumed landed; nothing here changes their semantics.

**Constructor.** One new keyword-only argument, appended to the link-04 signature:

```python
NeighborList(*, cell, cutoff, positions, skin=0.0, every=1, delay=0,
             check=True, capacity_factor=1.35, bin=None, device=None)
```

- `bin=None` (default) — the existing compiled O(N²) path, unchanged.
- `bin=0.0` — automatic size, requested `b = r_build / 2` (LAMMPS `nbin_standard`).
- `bin=<float > 0>` — explicit requested perpendicular bin thickness in Å.

The name `bin` is LAMMPS's and is what a user searching for this knob will type;
it shadows the builtin only inside `__init__`'s scope, the builtin is unused in
this module, and ruff's flake8-builtins rules are not enabled
(`pyproject.toml [tool.ruff.lint] select = ["E", "F", "W", "I"]`). Stored as
`self.bin: float | None`.

**Grid derivation — `NeighborList._configure_bins()`**, a private method called
once from `__init__` (single call site; a private method on the owning type is
what the repo shape rule prescribes — no free function, no factory). Given the
requested `b`:

1. `w = _perpendicular_widths(cell)` → `(3,)` Å (see Reuse decision).
2. `n_i = max(1, floor(w_i / b))` — sizing on the **perpendicular** width, not the
   row norm, so the requested thickness means the same thing on triclinic cells.
3. `b_i = w_i / n_i` — the *effective* thickness, `≥ b` except where the `max(1, …)`
   clamp applies (cell thinner than one requested bin).
4. `k_i = ceil(r_build / b_i)`, bumped by one while `k_i · b_i < r_build` (a
   float-rounding exactness guard that fires at most once).
5. Per-axis offsets `o_i = unique((arange(-k_i, k_i + 1)) mod n_i)`. Taking
   **distinct residues** is load-bearing: when `2k_i + 1 > n_i` (e.g. `n_i = 4`,
   `k_i = 2` — the ordinary 12 Å-cube case) the raw stencil wraps onto the same
   bin twice and would emit each pair twice. Distinct residues make every ordered
   candidate pair appear exactly once, so no pair-level deduplication is needed.
6. The 3-D stencil is the Cartesian product of the three residue sets, stored once
   as a `(S, 3)` long tensor `self._stencil` on the list's device;
   `self._inv_cell = inverse(cell)` is cached alongside. Both are recomputed by
   `to()`. The grid depends only on the (fixed) cell and `r_build`, so it is never
   recomputed per rebuild.

Public diagnostic `self.n_bins: tuple[int, int, int] | None` (None when
`bin is None`) — the observable surface the tests assert on, mirroring how
`r_build` / `ndanger` were exposed in link 04. `k_i`, `b_i` and the stencil stay
private and are reachable in error messages only.

**Construction validation** (all `ValueError`, measured numbers first, repo style):

1. `bin < 0` → refuse, naming the value and stating that `0.0` selects the
   automatic `r_build / 2` size.
2. `k_i > 8` on any axis (`_MAX_STENCIL_HALF_WIDTH = 8`, i.e. a stencil above
   `17³ = 4913` bin offsets) → refuse, naming `bin`, `b_i`, `k_i`, the cap, the
   resulting offset count, and the automatic alternative. This is the reachable
   guard against an absurdly small explicit `bin` (e.g. `bin=0.05` at
   `r_build = 5 Å` implies `k = 100` and a 8·10⁶-offset stencil).
3. Completeness invariant `k_i · b_i ≥ r_build` on every axis → `ValueError`
   naming `bin`, `n_i`, `b_i`, `k_i`, `r_build`. Construction (step 4) guarantees
   it; the check is a fail-loud tripwire per the iron law, not a user knob — an
   incomplete stencil is a silently wrong energy, which is exactly what must never
   pass quietly.

**Build — `NeighborList._build_binned(positions) -> (source, target, shifts)`**, a
private method with the same return contract as the existing `_compute`, so
`_write` (capacity, overflow `RuntimeError`, dead-edge padding) and everything
downstream are shared verbatim. Steps, all vectorised over atoms; the only Python
loop runs over the `S ≤ 4913` stencil offsets and is independent of `N`:

1. `frac = pos.detach() @ self._inv_cell`; `frac_w = frac - floor(frac)` — the wrap
   is a **build-time indexing device only**: stored positions stay unwrapped, and
   the wrapped copy is never used for displacements.
2. `bin_ijk = (frac_w * n).floor().long().clamp_(min=0, max=n-1)` (the clamp
   absorbs `frac_w` rounding to exactly 1.0); flat id
   `bin_id = (bin_ijk[:,0]*n_1 + bin_ijk[:,1])*n_2 + bin_ijk[:,2]`.
3. `order = argsort(bin_id, stable=True)`; `counts = bincount(bin_id,
   minlength=n_total)`; `starts = cumsum(counts, 0) - counts`.
4. Per stencil offset `o`: neighbour-bin flat ids `nb` for all atoms at once;
   `cnt = counts[nb]`; ragged expansion
   `seg = repeat_interleave(arange(N), cnt)`,
   `cand = order[starts[nb][seg] + (arange(total) - (cumsum(cnt,0)-cnt)[seg])]`.
5. Keep `seg < cand` (half pairs), compute `df = frac[cand] - frac[seg]`,
   `d = (df - round(df)) @ cell`, `r = ‖d‖`, keep `(r <= r_build) & (r > 0)`.
6. `shift = d - (pos[cand] - pos[seg])` — identical definition to `_compute`'s
   `edge_diff - (pos[target] - pos[source])`, so the two paths' shifts are directly
   comparable and reconstruct `edge_diff = pos[t] - pos[s] + shift` against the
   stored unwrapped positions.
7. Concatenate the per-offset half pairs, then symmetry-expand
   (`(s,t,Δ) → (t,s,−Δ)`) to the full bidirectional list, matching
   `NeighborListTask(symmetry=True)`.

Every intermediate tensor is created with an explicit `device=` / `dtype=` taken
from `positions` — no bare `torch.arange` / `torch.tensor` — so the path runs on
CUDA unchanged. The build stays **eager** (called between force evaluations,
exactly like `_compute` and `update()`); nothing here is traced or compiled.

**Dispatch.** A private `_build_pairs(positions)` returns
`self._build_binned(positions) if self.bin is not None else self._compute(positions)`;
`__init__`'s initial sizing build and `rebuild()` both call it. `_compute` keeps
its name and body — the kernel path is untouched, which is what makes it a trustworthy
oracle. `NeighborStrategy` is **unchanged**: the build backend is an implementation
detail, and forcing `bin` / `n_bins` onto every strategy would be protocol creep.

**Degeneration is graceful, not an error.** When `2k_i + 1 ≥ n_i` on every axis the
residue sets cover the whole grid and the binned path enumerates all pairs — correct,
just not faster. The O(N) win appears once `n_i > 2k_i + 1`, i.e. cells wider than
about `5 r_build / 2` per axis at the auto bin size. The docstring says so plainly so
nobody reads a small-cell timing as a regression.

**Reuse decision.** No `librarian_report` was supplied with this task — the caller
should note the missing blueprint advisory. The survey below is from a direct read
of `src/molix/md/`, `src/molix/data/`, `src/molix/op/` and a repo-wide grep for
`bucketize|bincount|cell.?list|linked.?cell|n_bins` (zero cell-list prior art
outside CMake build artefacts, confirming this is genuinely new code):

- `molix.data.tasks.neighbor.NeighborList` (imported as `NeighborListTask`) —
  **reuse**, unmodified, as the `bin=None` default path *and* as the test oracle.
- `NeighborList._write` / `_dead_shift` / `capacity` / `capacity_factor` overflow
  contract — **reuse**; the binned path returns the same `(source, target, shifts)`
  triple and shares all padding/capacity logic. No second padding implementation.
- `NeighborList.update` / `ago` / `ndanger` (link 04) — **reuse** unchanged; the
  policy decides *when* to build, `bin` decides *how*. Their independence is pinned
  by a test.
- `_min_perpendicular_width` (link 01) — **generalize** to
  `_perpendicular_widths(cell) -> torch.Tensor` `(3,)` Å: the guards need
  `min_i w_i`, the bin grid needs all three. Same float64 computation, same
  `(3,3)`-shape and degenerate-cell `ValueError`s; the link-01/04 guards become
  `float(_perpendicular_widths(cell).min())` with their messages — including the
  `4.000 A` substring contract link 01 pinned — byte-identical. This is the second
  real call site, which is exactly the repo's extraction trigger.
- `molix.units.DEAD_EDGE_CUTOFF_FACTOR` — **reuse** via the shared `_write`; the
  binned path restates nothing.
- `molix.data.collate._gather_indices` (`src/molix/data/collate.py:249`) —
  **pattern**, not imported: its `counts → repeat_interleave(seg) → cumsum offsets
  → starts[seg] + arange − offsets[seg]` ragged-gather idiom is followed verbatim
  in step 4 above. It is not called because it is private to the packed-collate
  fast path, keyed to the `PackedCache` `ptr` contract, returns four values of
  which two are needed, and — decisively — creates both its `torch.arange`s
  **without `device=`**, i.e. it is CPU-pinned for DataLoader workers. Promoting it
  to a device-aware shared helper would edit a hot, separately-tested collate path
  in another subpackage for one new caller; if a third call site appears (a data-side
  cell list, or link 07), promote it then.
- `molix.F.locality.get_neighbor_pairs` and everything under `src/molix/op/` —
  **not touched**. The C++/CUDA kernel is out of scope by construction.
- `NeighborList.bin` / `n_bins` / `_configure_bins` / `_build_binned` — **new**: no
  in-tree symbol builds a spatial bin grid. Naming (`_`-private methods on the
  owning type), keyword-only construction, `ValueError`-at-construction /
  `RuntimeError`-at-runtime error handling and measured-numbers-first messages all
  follow the closest pattern, the existing `NeighborList.__init__` guards and
  `_write` overflow check.

**Shape check.** Every new symbol is an attribute or a method on `NeighborList`; no
factory function, no context blob, no all-in-one façade; the two single-call-site
helpers stay private methods on the owning type; the one new module-level function
is pure cell geometry generalised from an existing one; `bin` is keyword-only.

## Files to create or modify

- `src/molix/md/neighbors.py` — `bin` keyword-only ctor arg and `self.bin`;
  public `n_bins`; `_perpendicular_widths` (generalised from
  `_min_perpendicular_width`); `_configure_bins`; `_build_binned`; `_build_pairs`
  dispatch used by `__init__` and `rebuild`; `to()` extended to `_stencil` /
  `_inv_cell`; module + constructor docstrings.
- `tests/test_molix/test_md/test_neighbors.py` — new `TestNeighborListBinned`,
  a module-level canonical-edge-key helper, and the jittered / triclinic /
  unwrapped / coincident fixtures.
- `regressions/md-neighborlist-skin-06-bins.py` (new) — public-API oracle-in-repo
  regression: binned vs kernel edge sets on hard-coded systems, golden edge counts.
- `docs/molix/user-guide/md.md` — document the `bin=` knob (auto vs explicit,
  when it wins) in the periodic-MD section, with the advisory timing note.

## Tasks

- [ ] Write failing unit tests for the bin-grid derivation and validation in `tests/test_molix/test_md/test_neighbors.py` (`TestNeighborListBinned`: `n_bins` goldens `(4,4,4)` / `(9,9,9)` / `(4,4,5)` / `(2,2,2)` / `(1,1,1)`, `bin=None` leaves `n_bins is None` and the kernel path untouched, `bin < 0` and tiny-`bin` `ValueError`s)
- [ ] Write failing unit tests for binned-vs-kernel edge-set equality in `tests/test_molix/test_md/test_neighbors.py` (cubic-aliasing, jittered 512-atom, triclinic, whole-cell-translated unwrapped and coincident-atom fixtures; `skin` 0.0/1.5 counts; duplicate-free `num_edges`; bin-size independence; `update()` parity; `to(torch.float32)` rebuild)
- [ ] Generalize `_min_perpendicular_width` in `src/molix/md/neighbors.py` to `_perpendicular_widths(cell) -> torch.Tensor` `(3,)` serving both the link-01/04 half-width guards (via `.min()`, messages and the `4.000 A` substring unchanged) and the per-axis bin sizing
- [ ] Implement the keyword-only `bin` argument, the `n_bins` diagnostic and `NeighborList._configure_bins` in `src/molix/md/neighbors.py` (auto `r_build/2`; `n_i = max(1, floor(w_i / bin))`; `b_i = w_i / n_i`; `k_i = ceil(r_build / b_i)` with the exactness bump; unique-residue stencil; the three construction validations)
- [ ] Implement `NeighborList._build_binned` and the `_build_pairs` dispatch in `src/molix/md/neighbors.py` (wrapped-fractional binning, stable argsort + bincount/cumsum ragged gather, fractional-rounding minimum image, `0 < r <= r_build` filter, half pairs then symmetry expansion, shared `_write`, explicit `device=`/`dtype=` on every tensor, `to()` extended to `_stencil`/`_inv_cell`)
- [ ] Add google-style docstrings with units for `bin` / `n_bins` / `_configure_bins` / `_build_binned` and extend the `src/molix/md/neighbors.py` module docstring with the stencil-completeness and `|f_i| ≤ 1/2` minimum-image lemmas, the `0 < r ≤ r_build` filter parity, the edge-order caveat, the graceful small-cell degeneration, and the LAMMPS / Allen–Tildesley / Quentrec–Brot references with the paywall caveat
- [ ] Measure the binned vs O(N²) build wall clock at N = 4000 and record it as an advisory note (device, torch version, date, explicit "no threshold is asserted anywhere") in the `bin` docstring and in `docs/molix/user-guide/md.md`
- [ ] Add regression example `regressions/md-neighborlist-skin-06-bins.py` (public API only; hard-coded goldens, no third-party runtime)
- [ ] Run full check + test suite

## Testing strategy

Unit tests only, in the source mirror
`src/molix/md/neighbors.py` → `tests/test_molix/test_md/test_neighbors.py`, in a
new `TestNeighborListBinned` class alongside the existing ones. Everything is
deterministic CPU float64, each test targeting one behaviour of the binned build.
A module-level helper (test-module local — no `helpers.py`, per the repo test
layout rule) builds the canonical comparison key:

```
key(edge) = (min(s,t), max(s,t), round(shift oriented low→high, 6 decimals))
```

with the shift negated when `s > t`, so the key is orientation-free while still
distinguishing periodic images. Comparison is always **both** `set(keys_binned)
== set(keys_kernel)` **and** `num_edges_binned == num_edges_kernel` plus
`len(set(keys)) == num_edges`: set equality alone cannot see a duplicated edge,
and duplicate emission is precisely the failure mode of a wrapped stencil.
Every equivalence fixture also asserts its precondition that no pair lies within
`1e-9 Å` of `r_build`, so a float tie can never make the comparison flaky.

**Fixtures.**

- **(a) Cubic lattice, stencil-aliasing regime.** `_lattice(n_side=4,
  spacing=3.0)` — 64 atoms, 12 Å cube, half-width 6.0 Å; `cutoff=3.5`,
  `skin=1.5` ⇒ `r_build = 5.0`, auto bin ⇒ `n_bins == (4,4,4)`, `b_i = 3.0`,
  `k_i = 2`, so `2k+1 = 5 > 4` and the raw stencil wraps onto itself. Goldens are
  crystallographic: 18 neighbours per atom (6 at 3.0 Å, 12 at 3√2 = 4.2426 Å;
  3√3 = 5.196 Å is outside) ⇒ `num_edges == 1152`, and `384` at `skin=0.0`.
- **(b) Jittered dense system, pruning regime.** `_lattice(n_side=8,
  spacing=3.0)` (512 atoms, 24 Å cube) displaced by `±0.4 Å` uniform under
  `torch.manual_seed(0)`; `r_build = 5.0` ⇒ `n_bins == (9,9,9)`, `k_i = 2`, so the
  stencil is 125 of 729 bins — real pruning, no lattice symmetry to hide behind.
  The kernel is the oracle; no count golden.
- **(c) Triclinic.** Link 01's golden cell `[[10,0,0],[6,8,0],[0,0,10]]`
  (`w = (8, 8, 10) Å`, guard bound 4.000 Å), 40 atoms from
  `torch.manual_seed(1); frac = torch.rand(40,3,dtype=torch.float64)`,
  `pos = frac @ cell`; `cutoff=3.0`, `skin=0.5` ⇒ `r_build = 3.5`, strictly inside
  the guard. Golden `n_bins == (4,4,5)` is **discriminating**: sizing on the row
  norm `‖a_2‖ = 10` instead of the perpendicular width `w_2 = 8` would give
  `(4,5,5)`. This fixture is what pins the `|f_i| ≤ 1/2` lemma — the binned path's
  fractional rounding must agree with the kernel's sequential reduction.
- **(d) Skin at `r_build`.** Fixture (a) at `skin ∈ {0.0, 1.5}`: binned
  `num_edges` equals the kernel's in both (`384`, `1152`), proving the binned path
  builds at `r_build`, not `cutoff`.
- **(e) Unwrapped positions.** Fixture (a) with 8 atoms translated by whole cell
  vectors (`±12 Å`): binned == kernel on that system, and the multiset of
  `(min,max, rounded displacement ‖·‖)` is unchanged from (a) — pinning that the
  fractional wrap is an indexing device only and that shifts refer to the stored,
  unwrapped positions (the link-04 invariant this path must not break).
- **(f) Coincident atoms.** Fixture (a) with one atom duplicated at another's
  position and one atom translated by exactly `a_1`: both paths drop both
  zero-distance pairs (`r > 0` filter parity) and agree on everything else.

**Bin-size independence.** On fixture (b), `bin ∈ {0.0, 2.5, 5.0, 24.0}` all give
the identical canonical edge set and edge count — the bin size is a cost knob, never
a physics knob.

**Policy interplay.** With `bin=0.0` set, the link-04 `update()` decision sequence
(`ago`, `rebuild_count`, `ndanger`, and the returned booleans) over a scripted
displacement schedule is identical to `bin=None`, and the live edge set after each
triggered rebuild still matches the kernel path. `isinstance(nl, NeighborStrategy)`
still holds (the protocol is unchanged).

**Validation errors.** `bin=-1.0` raises `ValueError` naming `bin`; `bin=0.05` at
`r_build = 5.0` raises `ValueError` naming the implied stencil half-width (100),
the cap (8) and the automatic alternative. The completeness tripwire (validation 3)
is unreachable from user input by construction and is therefore documented rather
than exercised — the equivalence fixtures are what actually falsify completeness.

**Device / dtype.** `nl.to(torch.float32)` followed by
`rebuild(pos.to(torch.float32))` still reports `num_edges == 1152` on fixture (a)
(i.e. `_inv_cell` and `_stencil` were carried along), and `edge_index` stays
`torch.long`. A grep-level check that `_build_binned` and `_configure_bins` create
no tensor without an explicit `device=` keeps the path CUDA-usable without a GPU in
CI.

**Advisory (non-binding, no criterion).** The binned-vs-O(N²) build wall clock at
N ≈ 4000 is measured once and recorded in the docstring and the user guide with
device, torch version and date. It is machine-bound: **no timing threshold is
asserted in any test, regression, or acceptance criterion.**

**Regression example.** `regressions/md-neighborlist-skin-06-bins.py`, run as
`PYTHONPATH=src python regressions/md-neighborlist-skin-06-bins.py`, public API
only (`from molix.md import NeighborList`), prints `OK` and exits 0, exits 1 on
drift. The oracle is **in-repo** (the compiled kernel path) — no third party is
imported, subprocessed or downloaded — and all goldens are literals:

1. 64-atom 12 Å cubic lattice built in-script from `arange`/`meshgrid` (no RNG),
   `cutoff=3.5`, `skin=1.5`: `n_bins == (4,4,4)`, `num_edges == 1152` on **both**
   paths, canonical edge sets identical, no duplicate keys.
2. Same system at `skin=0.0`: `n_bins == (6,6,6)`, `num_edges == 384` on both paths,
   sets identical.
3. Explicit `bin=5.0` ⇒ `n_bins == (2,2,2)` and `bin=12.0` ⇒ `n_bins == (1,1,1)`:
   both reproduce scenario 1's edge set exactly.
4. Triclinic `[[10,0,0],[6,8,0],[0,0,10]]` with 12 **literal** fractional
   coordinates mapped by `frac @ cell`, `cutoff=3.0`, `skin=0.5`:
   `n_bins == (4,4,5)` and identical sets, against a golden edge-count literal
   captured once from the in-repo kernel oracle at implementation time.
5. Scenario 1 with 8 atoms translated by whole cell vectors: sets still identical
   between paths, and the displacement multiset matches scenario 1's.

The header records the capture command, commit sha, torch version, date and
device/precision per `regressions/README.md`. The script may **print** the two
build wall clocks for the 512-atom system; it asserts nothing about them.

## Out of scope

- **Modifying the compiled kernel** (`src/molix/op/`, `molix.F.locality`). The
  binned path is pure torch beside it; the kernel stays the default and the oracle.
- **A binned path for the data pipeline** (`molix.data.tasks.neighbor.NeighborList`
  / `NeighborListTask`), whose TODO at `:116` named this gap. It has its own
  `task_id` / cache-invalidation implications and belongs to a follow-up link; the
  TODO comment is left in place there rather than half-answered.
- **Lifting the single-image restriction.** `r_build ≤ min_i w_i / 2` (links 01/04)
  still holds and the binned path relies on it for the `|f_i| ≤ 1/2` lemma. Ghost
  atoms / multi-image expansion for `r_build > w/2` is a separate algorithm.
- **Compiling or CUDA-graph-capturing the build.** Like `_compute` and `update()`,
  `_build_binned` is eager, between force evaluations; the *force* path is what
  stays in the graph.
- **Any performance gate.** No timing assertion, no benchmark under `benchmarks/`,
  no `mol:bench` criterion — only the advisory recorded note.
- **Varying cells (NPT), non-periodic/open systems, half-pair-only output.** The
  grid is derived once from the fixed cell; the class is periodic-only; the output
  convention stays full bidirectional as `NeighborListTask(symmetry=True)`.
- **LAMMPS binning features with no consumer here**: `neigh_modify binsize/one/page`,
  `nbin_multi` / per-type cutoffs, hybrid and multi-cutoff lists, domain
  decomposition, and incremental (re-)binning between rebuilds.
- **Integrator / driver rewiring and `bin` defaults in `MD(...)`** — link 07. This
  link leaves `bin=None` as the constructor default so nothing changes for existing
  callers until someone opts in.
- **Promoting `molix.data.collate._gather_indices` to a shared device-aware helper**
  — considered and rejected above (one new caller, hot path in another subpackage);
  revisit at a third call site.
