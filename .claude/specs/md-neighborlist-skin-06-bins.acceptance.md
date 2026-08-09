---
slug: md-neighborlist-skin-06-bins
criteria:
  - id: ac-001
    summary: bin=None keeps the compiled O(N^2) path and the kernel untouched
    type: code
    evaluator_hint: "pytest tests/test_molix/test_md/test_neighbors.py"
    pass_when: |
      With bin omitted or None, NeighborList.n_bins is None and every
      pre-existing test in tests/test_molix/test_md/test_neighbors.py passes
      unchanged; the diff for this spec touches no file under src/molix/op/ and
      does not modify src/molix/data/tasks/neighbor.py.
    status: pending
  - id: ac-002
    summary: bin grid is derived from r_build and the perpendicular widths
    type: code
    evaluator_hint: "pytest tests/test_molix/test_md/test_neighbors.py::TestNeighborListBinned"
    pass_when: |
      At cutoff=3.5, skin=1.5 (r_build 5.0 A) with bin=0.0, n_bins == (4,4,4) on
      the 64-atom 12 A cube and (9,9,9) on the 512-atom 24 A cube; at skin=0.0
      (r_build 3.5 A) n_bins == (6,6,6) on the 12 A cube; explicit bin=5.0 gives
      (2,2,2) and bin=12.0 gives (1,1,1); on the triclinic cell
      [[10,0,0],[6,8,0],[0,0,10]] at r_build 3.5 A, n_bins == (4,4,5) — the value
      that row-norm sizing would report as (4,5,5).
    status: pending
  - id: ac-003
    summary: binned build equals the kernel build as an edge set, without duplicates
    type: code
    evaluator_hint: "pytest tests/test_molix/test_md/test_neighbors.py::TestNeighborListBinned"
    pass_when: |
      On each of the five fixtures — (a) 64-atom cubic lattice at r_build 5.0 A
      (stencil-aliasing regime), (b) 512-atom lattice jittered +/-0.4 A under
      manual_seed(0) (pruning regime), (c) 40-atom triclinic
      [[10,0,0],[6,8,0],[0,0,10]] at r_build 3.5 A, (e) fixture (a) with 8 atoms
      translated by whole cell vectors, (f) fixture (a) with coincident and
      exactly-lattice-translated atom pairs — the binned and kernel paths give
      set(canonical keys) equal, num_edges equal, and len(set(keys)) == num_edges
      (no duplicated edge), where the canonical key is
      (min(s,t), max(s,t), shift oriented low->high rounded to 6 decimals); each
      fixture also asserts that no pair lies within 1e-9 A of r_build.
    status: pending
  - id: ac-004
    summary: binned path builds at r_build and leaves the update() policy intact
    type: code
    evaluator_hint: "pytest tests/test_molix/test_md/test_neighbors.py::TestNeighborListBinned"
    pass_when: |
      On the 64-atom 12 A cube the binned path reports num_edges == 1152 at
      skin=1.5 and 384 at skin=0.0, equal to the kernel path in both cases; and
      over a scripted displacement schedule the update() return sequence, ago,
      rebuild_count and ndanger are identical with bin=0.0 and with bin=None,
      with the live edge set after each triggered rebuild still matching the
      kernel path. isinstance(nl, NeighborStrategy) still holds.
    status: pending
  - id: ac-005
    summary: bin size is a cost knob only — the edge set is independent of it
    type: code
    evaluator_hint: "pytest tests/test_molix/test_md/test_neighbors.py::TestNeighborListBinned"
    pass_when: |
      On the 512-atom jittered system at r_build 5.0 A, bin in {0.0, 2.5, 5.0,
      24.0} all produce the identical canonical edge set and the identical
      num_edges, equal to the kernel path's.
    status: pending
  - id: ac-006
    summary: bin validation refuses negative and absurdly small bin sizes
    type: code
    evaluator_hint: "pytest tests/test_molix/test_md/test_neighbors.py::TestNeighborListBinned"
    pass_when: |
      bin=-1.0 raises ValueError naming bin and stating that 0.0 selects the
      automatic r_build/2 size; bin=0.05 at r_build 5.0 A raises ValueError naming
      the effective bin thickness, the implied stencil half-width 100, the cap 8,
      and the automatic alternative; the completeness invariant
      k_i * b_i >= r_build is checked at construction and raises ValueError naming
      bin, n_i, b_i, k_i and r_build if it ever fails.
    status: pending
  - id: ac-007
    summary: binned state survives to() and creates no device-pinned tensors
    type: code
    evaluator_hint: "pytest tests/test_molix/test_md/test_neighbors.py::TestNeighborListBinned"
    pass_when: |
      After nl.to(torch.float32) on the 64-atom fixture, rebuild(pos.to(float32))
      still reports num_edges == 1152 and edge_index stays torch.long; and no
      tensor-creating call inside _configure_bins or _build_binned (torch.arange,
      torch.tensor, torch.zeros, torch.eye, ...) omits an explicit device=
      argument derived from the positions/cell tensors.
    status: pending
  - id: ac-008
    summary: docstring and user guide carry the lemmas, caveats and advisory timing
    type: docs
    pass_when: |
      The src/molix/md/neighbors.py module docstring states the stencil
      completeness relation (separation > (m-1)*b_i, hence k_i*b_i >= r_build),
      the |f_i| <= 1/2 minimum-image lemma that makes fractional rounding exact
      inside the r_build <= min_i w_i / 2 guard, the 0 < r <= r_build filter parity
      with the compiled kernel, the explicit statement that edge ORDER is not part
      of the contract (set equality is), and the graceful all-pairs degeneration on
      small cells; it cites Allen & Tildesley
      (10.1093/oso/9780198803195.001.0001), LAMMPS (10.1016/j.cpc.2021.108171 plus
      nbin_standard binsize_optimal = 0.5*cutneighmax) and Quentrec & Brot
      (10.1016/0021-9991(73)90046-6) with the paywall caveat. bin, n_bins,
      _configure_bins and _build_binned carry google-style docstrings with units
      (A / dimensionless). The bin docstring and docs/molix/user-guide/md.md record
      the measured N=4000 binned-vs-O(N^2) build wall clock with device, torch
      version and date, and state that no timing threshold is asserted anywhere.
    status: pending
  - id: ac-009
    summary: regression script reproduces the binned/kernel goldens in-repo
    type: runtime
    pass_when: |
      `PYTHONPATH=src python regressions/md-neighborlist-skin-06-bins.py` prints OK
      and exits 0, reproducing its hard-coded literals: 64-atom 12 A cubic lattice
      at cutoff=3.5/skin=1.5 gives n_bins == (4,4,4) and num_edges == 1152 on both
      paths with identical canonical edge sets and no duplicate keys; at skin=0.0,
      n_bins == (6,6,6) and 384 edges on both paths; explicit bin=5.0 (n_bins
      (2,2,2)) and bin=12.0 (n_bins (1,1,1)) reproduce the same edge set; the
      12-atom triclinic [[10,0,0],[6,8,0],[0,0,10]] case at cutoff=3.0/skin=0.5
      gives n_bins == (4,4,5) and matches its golden edge-count literal on both
      paths; the whole-cell-translated variant still matches. Timings, if printed,
      are not asserted. No third-party software is imported or subprocessed.
    status: pending
---

# Acceptance criteria

- **ac-001 — the default must not move.** The whole safety argument of this link
  is that the compiled O(N²) path stays exactly what it was, so it can serve as
  the oracle for the new one. A diff into `src/molix/op/` or the pipeline task
  invalidates the oracle and the criterion fails regardless of test results.
- **ac-002 — the grid is where triclinic correctness is decided.** `n_bins` is the
  only public window onto the derived grid, and the `(4,4,5)` triclinic golden is
  chosen precisely because the plausible wrong implementation (sizing on the row
  norm `‖a_2‖ = 10` instead of the perpendicular width `w_2 = 8`) reports
  `(4,5,5)`. A guard that only ever sees cubic cells proves nothing.
- **ac-003 — the primary falsification (invariant (b)).** Everything else can pass
  while the list is quietly incomplete or quietly doubled. Three clauses are needed
  together: set equality catches *missing* edges, `num_edges` equality plus the
  unique-key count catches *duplicated* edges (the wrapped-stencil failure mode
  that the `(4,4,4)` aliasing fixture is built to trigger), and the
  no-pair-near-`r_build` precondition keeps the comparison from being a float
  coin-flip. The triclinic fixture is what tests the `|f_i| ≤ 1/2` lemma; the
  translated fixture is what tests that the fractional wrap never leaks into the
  stored positions or the shifts; the coincident fixture is what tests `r > 0`
  parity with the kernel.
- **ac-004 — the two knobs are orthogonal.** `skin` decides the radius, `bin`
  decides the search strategy, `update()` decides the moment. If the binned path
  silently built at `cutoff` instead of `r_build`, the Verlet skin from link 04
  would be dead while every energy still looked plausible; the 1152/384 pair is the
  cheapest direct measurement of that.
- **ac-005 — bin size is not physics.** A bin size that changes the edge set is a
  broken stencil derivation, full stop. Sweeping from the auto size to a single-bin
  grid covers both the pruning and the full-degeneration regimes.
- **ac-006 — refuse at construction, name the numbers.** A negative bin is a typo;
  a 0.05 Å bin is a hang. Both are refused before any buffer is allocated, in the
  repo's measured-numbers-first message style. The third clause is a fail-loud
  tripwire on the completeness invariant: unreachable from user input today, kept
  because an incomplete stencil is a silently wrong energy and the iron law forbids
  discovering that from a drifting trajectory.
- **ac-007 — GPU-capability without a GPU in CI.** The device discipline check is
  what makes "GPU-capable" a verifiable claim on a CPU runner; `to()` carrying the
  stencil and inverse cell is the concrete way that claim usually breaks first.
- **ac-008 — the derivation lives with the code.** The two lemmas are the reason
  this path is allowed to disagree on edge *order* while being provably identical
  as a *set*; a reader who cannot find them will eventually "fix" the ordering. The
  timing note is deliberately advisory: it is recorded with its machine, and no
  criterion anywhere gates on it.
- **ac-009 — reproducibility with an in-repo oracle.** The regression is the
  standalone, public-API record that the two build paths agree on hard-coded
  systems, using this repo's own kernel as the oracle. It stays `runtime` (verified
  by `/mol:impl` at delivery) rather than `scientific`: it lives in `regressions/`,
  imports nothing outside molnex, and asserts literals, not measurements.
