"""Public-API scenario for the binned (cell-list) build backend of the MD neighbour list.

Spec: `md-neighborlist-skin-06-bins`.

`molix.md.NeighborList` gained a keyword-only `bin` argument selecting a second
build **backend**: `bin=None` (default) hands the system to the compiled O(N^2)
pair kernel, a float switches to the pure-torch periodic cell list.  `bin` is a
**cost** knob, never a physics knob — which is a claim about the *edge set*, and
that is what this file pins.

The oracle is **in repo**: the untouched `bin=None` kernel path.  Nothing here
imports, subprocesses or downloads a third party (torch is the repo's own core
dependency); no network, no filesystem, no RNG, no wall-clock *assertion*.  Every
system is built from integer arithmetic or written-out literals on this page.

What set equality alone cannot see
----------------------------------
A binned build searches a stencil of neighbouring bins.  When `2*k_i + 1 > n_i`
the raw stencil wraps onto the same bin twice, so a correct-looking *set* of
edges can hide every pair being emitted **twice**.  Comparison here is therefore
three-fold, on both paths:

1. `set(canonical_keys_binned) == set(canonical_keys_kernel)` — same pairs, same
   periodic images;
2. `len(set(directed_keys)) == num_edges` — no duplicate directed edge;
3. `2 * len(set(canonical_keys)) == num_edges` — the bidirectional list collapses
   exactly 2:1 onto orientation-free keys.

A directed key is `(source, target, round(shift, 6))`.  A canonical key is the
same triple re-oriented low index first, with the shift negated when `s > t`, so
it is orientation-free while still separating periodic images of the same pair.
Six decimals is far coarser than float64 noise on shifts that are integer
combinations of cell vectors (~1e-15 A here) and far finer than the smallest
distinct shift component (6.0 A), so the rounding can neither merge two images
nor split one.

Section 1 — cubic lattice, automatic bin, the aliasing regime
-------------------------------------------------------------
64 atoms on a 4x4x4 simple-cubic lattice, `a = 3.0 A`, in a 12.0 A cube (built
in-script from `arange`/`meshgrid`; no RNG).  `cutoff = 3.5 A`, `skin = 1.5 A`:

    r_build   = 3.5 + 1.5 = 5.0 A          (<= 12.0/2 = 6.0 A, guard clear)
    bin=0.0   => requested b = r_build/2 = 2.5 A   (LAMMPS `binsize_optimal`)
    n_i       = floor(12.0 / 2.5) = floor(4.8) = 4      => n_bins == (4, 4, 4)
    b_i       = 12.0 / 4 = 3.0 A
    k_i       = ceil(5.0 / 3.0) = 2  =>  2*k_i + 1 = 5 > 4 = n_i

so the raw stencil wraps onto itself: this is exactly the regime where the
unique-residue construction is load-bearing, and where a naive stencil would
double every edge while keeping the set right.

Simple cubic at `a = 3.0 A` has exact shells — 6 at 3.000 A, 12 at 4.243 A, 8 at
5.196 A — so `r_build = 5.0 A` admits 6 + 12 = 18 neighbours per atom:

    num_edges = 64 * 18 = 1152 directed edges (full bidirectional list)
    unordered pairs = 1152 / 2 = 576

Four sites per axis keeps the `+a` and `-a` neighbours distinct atoms, so the
multiplicities are not double-counted images.  Both shells sit far from the
5.0 A radius (4.243 and 5.196), so no pair can flip class on rounding.

Section 2 — the grid is derived from `r_build`, not `cutoff`
-------------------------------------------------------------
The same lattice at `skin = 0.0` (`r_build = 3.5 A`):

    requested b = 1.75 A;  n_i = floor(12.0/1.75) = floor(6.857) = 6
    n_bins == (6, 6, 6);  b_i = 2.0 A;  k_i = ceil(3.5/2.0) = 2
    2*k_i + 1 = 5 <= 6  =>  125 of 216 bins searched — real pruning
    num_edges = 64 * 6 = 384    (only the 3.0 A shell is inside 3.5 A)

`(6,6,6)` versus section 1's `(4,4,4)` on the *same cell* is what pins the grid
to `r_build`: a backend that binned on `cutoff` would report `(6,6,6)` in section
1 too, and one that binned on `r_build` while *filtering* on `cutoff` would
return 384 edges there instead of 1152.

Section 3 — the bin size is a cost knob
----------------------------------------
Section 1's system, built with explicit bin thicknesses:

    bin=5.0 A  =>  n_i = floor(12.0/5.0)  = 2  =>  n_bins == (2, 2, 2)
    bin=12.0 A =>  n_i = floor(12.0/12.0) = 1  =>  n_bins == (1, 1, 1)

Both must reproduce section 1's canonical edge set **exactly** — same 576
unordered pairs, same 1152 directed edges.  `(1,1,1)` is the documented graceful
degeneration: one bin holding all 64 atoms is an all-pairs search, correct and
merely not faster.  Three different grids over one configuration is the strongest
available statement that `bin` never moves an atom's neighbours.

Section 4 — triclinic, where the perpendicular width matters
-------------------------------------------------------------
Cell rows `a_1 = (10, 0, 0)`, `a_2 = (6, 8, 0)`, `a_3 = (0, 0, 10)` (Angstrom),
`V = 800 A^3`, perpendicular widths `w_i = V / ||a_j x a_k||`:

    w = (800/100, 800/100, 800/80) = (8, 8, 10) A     (guard: min w / 2 = 4.0 A)

12 atoms are written out as **literal fractional coordinates** and mapped by
`frac @ cell`; `cutoff = 3.0 A`, `skin = 0.5 A` => `r_build = 3.5 A`, strictly
inside the 4.0 A guard.  With `bin=0.0` (requested 1.75 A):

    n_bins == (floor(8/1.75), floor(8/1.75), floor(10/1.75)) = (4, 4, 5)

which is **discriminating**: sizing on the row norm `||a_2|| = 10` instead of the
perpendicular width `w_2 = 8` would give `(4, 5, 5)`.  This is also the fixture
that exercises the `|f_i| <= 1/2` lemma — the binned path's minimum image comes
from fractional rounding, the kernel's from a sequential diagonal reduction, and
on a sheared cell those two agree only because `r_build <= min_i w_i / 2`.

The edge count `24` here is **not** analytic: it was captured once from the
in-repo kernel oracle at implementation time (see Goldens).  The script also
recomputes, from the literals alone, the closest minimum-image pair distance and
the margin from `r_build` over all 66 pairs and all 27 images, and refuses to run
on a fixture whose margin has collapsed — a float tie at the radius would make
the comparison flaky rather than wrong.

Section 5 — unwrapped positions
--------------------------------
Section 1's lattice with 8 atoms translated by whole cell vectors (`+-12 A`,
including two diagonal images).  The binned path wraps fractional coordinates
into `[0, 1)` **for indexing only**: displacements are taken from the unwrapped
coordinates, and the shifts refer to the stored, unwrapped positions.  So both
paths must still agree edge for edge, and the multiset of
`(min(s,t), max(s,t), round(||pos[t] - pos[s] + shift||, 6))` must be *identical*
to section 1's — the physics is translation invariant even though the individual
shift labels are not.  A backend that differenced the wrapped copy would return
the same distances but wrong shifts; a backend that stored wrapped positions
would break the link-04 raw-displacement invariant.  Comparing distances *and*
edge sets catches both.

Advisory timing (asserted nowhere)
-----------------------------------
The two build wall clocks for a 512-atom / 24 A cube are printed for information.
**No timing threshold is asserted in this file, and none belongs here**: the
numbers are machine-, thread- and build-bound (the binned path loses to the
kernel at high `OMP_NUM_THREADS`, where per-op OpenMP region overhead dominates
its 125-offset loop). They are printed, never checked.

Drift policy
------------
A disagreement between this file and the runtime is a **DEFECT REPORT, never a
golden edit**.  Sections 1-3 and 5 are analytic — simple-cubic shell
multiplicities, `floor(w_i / b)` arithmetic and translation invariance, all
derived on this page — and section 4's single captured literal came from the
kernel oracle, which this file's whole point is to hold the binned path against.
If a count or a set comparison comes back wrong, open a defect against
`src/molix/md/neighbors.py`; do not retune a literal to whatever the run printed.
A duplicated edge doubles a pair's contribution to the energy and a missing one
deletes it, and both of those are silent at runtime — this file is where they are
supposed to become loud.

Goldens
-------
    capture command : PYTHONPATH=src python regressions/md-neighborlist-skin-06-bins.py
                      Sections 1-3 and 5 captured nothing: 64*18 = 1152,
                      64*6 = 384, floor(12/2.5) = 4, floor(12/1.75) = 6,
                      floor(12/5) = 2, floor(12/12) = 1, floor(8/1.75) = 4 and
                      floor(10/1.75) = 5 are arithmetic on this page.
                      Section 4's `TRICLINIC_EDGES = 24` was captured ONCE, from
                      the in-repo `bin=None` kernel path on the 12 literal
                      fractional coordinates below (which also reported closest
                      pair 2.012461 A and margin 0.145898 A from r_build).
    commit          : 639c9df (639c9df8818102a602b96fa326dbc4e7eab5b013), with
                      the `md-neighborlist-skin-06-bins` working tree on top (at
                      639c9df itself `NeighborList.__init__` takes no `bin`, so
                      every scenario here raises `TypeError` — that is the RED
                      this file was written against).
    torch           : 2.12.1+cpu   (python 3.14.5)
    date            : 2026-08-09
    device / dtype  : CPU, float64 — cells and positions are float64 literals, so
                      `shifts` is float64 too; `edge_index` is int64 regardless.
    oracle          : in repo. The compiled O(N^2) kernel path reached through
                      the public constructor as `bin=None` — the backend this
                      link does not touch. No third-party package, no network, no
                      subprocess, no downloaded reference, no RNG.
    tolerance       : exact integer equality on every edge count, bin-count tuple
                      and set cardinality — they are counters, and a counter has
                      no tolerance. Shifts and distances are compared as keys
                      rounded to 6 decimals (float64 "position" band is 1e-12;
                      the values here are integer combinations of cell vectors
                      carrying ~1e-15 A of noise, and the smallest distinct shift
                      component is 6.0 A, so the rounding is unambiguous).

Run:
    PYTHONPATH=src python regressions/md-neighborlist-skin-06-bins.py
"""

from __future__ import annotations

import itertools
import sys
import time

import torch

from molix.md import NeighborList

# ---------------------------------------------------------------------------
# The cubic system of sections 1, 2, 3 and 5.
# ---------------------------------------------------------------------------

#: Lattice spacing `a` in Angstrom.
SPACING = 3.0

#: Sites per axis; 4 keeps the `+a` and `-a` neighbours distinct atoms.
N_SIDE = 4

#: Cubic cell edge in Angstrom — four spacings, so the lattice tiles exactly.
BOX = SPACING * N_SIDE

#: Interaction cutoff in Angstrom: between the 3.0 A first shell (in) and the
#: 4.243 A second shell (out), so no pair can flip class on rounding.
CUTOFF = 3.5

#: Verlet skin in Angstrom. `r_build = 5.0 A` sits between the 4.243 A second
#: shell (in) and the 5.196 A third shell (out).
SKIN = 1.5

CUBIC_CELL = torch.tensor(
    [
        [BOX, 0.0, 0.0],
        [0.0, BOX, 0.0],
        [0.0, 0.0, BOX],
    ],
    dtype=torch.float64,
)

#: `bin=0.0` asks for the automatic `r_build / 2` thickness (LAMMPS
#: `nbin_standard`); every other float is an explicit requested thickness in A.
AUTO_BIN = 0.0

# ---------------------------------------------------------------------------
# Section 1 goldens — automatic bin at `r_build = 5.0 A`.
# ---------------------------------------------------------------------------

#: floor(12.0 / (5.0/2)) = floor(4.8) = 4 bins per axis; b_i = 3.0 A and
#: k_i = ceil(5.0/3.0) = 2, so 2*k_i + 1 = 5 > 4 — the wrapping stencil.
CUBIC_N_BINS_AT_R_BUILD = (4, 4, 4)

#: 64 sites x (6 at 3.0 A + 12 at 4.243 A), both directions kept.
CUBIC_EDGES_AT_R_BUILD = 1152

#: Directed edges collapse 2:1 onto orientation-free keys.
CUBIC_PAIRS_AT_R_BUILD = CUBIC_EDGES_AT_R_BUILD // 2

# ---------------------------------------------------------------------------
# Section 2 goldens — same cell, bare cutoff (`skin = 0.0`).
# ---------------------------------------------------------------------------

#: floor(12.0 / (3.5/2)) = floor(6.857) = 6; b_i = 2.0 A, k_i = 2, so the
#: stencil is 125 of 216 bins — pruning, not aliasing.
CUBIC_N_BINS_AT_CUTOFF = (6, 6, 6)

#: 64 sites x 6 at 3.0 A — only the first shell is inside 3.5 A.
CUBIC_EDGES_AT_CUTOFF = 384

# ---------------------------------------------------------------------------
# Section 3 goldens — explicit bin thicknesses on section 1's system.
# ---------------------------------------------------------------------------

#: (requested thickness in A, expected `n_bins`). 5.0 A gives two bins per axis
#: (b_i = 6.0 A, k_i = 1, residues {0,1} = the whole grid); 12.0 A is the whole
#: cell in one bin — the documented graceful degeneration to an all-pairs search.
EXPLICIT_BINS: tuple[tuple[float, tuple[int, int, int]], ...] = (
    (5.0, (2, 2, 2)),
    (12.0, (1, 1, 1)),
)

# ---------------------------------------------------------------------------
# Section 4 — the triclinic system.
# ---------------------------------------------------------------------------

TRICLINIC_CELL = torch.tensor(
    [
        [10.0, 0.0, 0.0],
        [6.0, 8.0, 0.0],
        [0.0, 0.0, 10.0],
    ],
    dtype=torch.float64,
)

#: 12 literal fractional coordinates, mapped to Angstrom by `frac @ cell`.
#: Chosen so the closest minimum-image pair is 2.012 A (well clear of the `r > 0`
#: filter) and no pair sits within 0.145 A of `r_build = 3.5 A` (well clear of a
#: float tie at the radius) — both re-verified at run time below.
TRICLINIC_FRACTIONAL = torch.tensor(
    [
        [0.00, 0.15, 0.20],
        [0.00, 0.45, 0.50],
        [0.30, 0.20, 0.90],
        [0.35, 0.25, 0.15],
        [0.35, 0.50, 0.25],
        [0.35, 0.70, 0.40],
        [0.50, 0.15, 0.80],
        [0.60, 0.40, 0.70],
        [0.75, 0.05, 0.75],
        [0.75, 0.60, 0.00],
        [0.90, 0.20, 0.80],
        [0.90, 0.25, 0.00],
    ],
    dtype=torch.float64,
)

#: Interaction cutoff and skin in Angstrom: `r_build = 3.5 A`, strictly inside
#: the guard `min_i w_i / 2 = 4.0 A` for this cell.
TRICLINIC_CUTOFF = 3.0
TRICLINIC_SKIN = 0.5

#: floor(w / 1.75) on the perpendicular widths w = (8, 8, 10) A. Sizing on the
#: row norm ||a_2|| = 10 A instead would give (4, 5, 5).
TRICLINIC_N_BINS = (4, 4, 5)

#: Captured ONCE from the in-repo `bin=None` kernel oracle on the literals above
#: (2026-08-09, torch 2.12.1+cpu, CPU float64): 12 unordered pairs, both
#: directions kept. A disagreement is a defect report, not a new capture.
TRICLINIC_EDGES = 24

#: Smallest allowed `| r - r_build |` over all pairs and images, in Angstrom.
#: Not a golden — a precondition floor, ~5 orders above float64 noise on these
#: distances and ~5 orders below the fixture's actual 0.145898 A margin. It fails
#: only if the literals above are edited into a tie at the build radius.
MARGIN_FLOOR = 1e-6

# ---------------------------------------------------------------------------
# Section 5 — whole-cell translations of section 1's lattice.
# ---------------------------------------------------------------------------

#: `(atom index, cell-vector image)` for the 8 translated atoms: three axes in
#: each direction plus two diagonals, so the fixture is not a single-axis
#: special case. Each atom moves by an exact lattice vector, so the periodic
#: system is unchanged and only the *labels* (shifts) may move.
TRANSLATED_ATOMS: tuple[tuple[int, tuple[int, int, int]], ...] = (
    (0, (1, 0, 0)),
    (5, (0, 1, 0)),
    (9, (0, 0, 1)),
    (17, (-1, 0, 0)),
    (23, (0, -1, 0)),
    (38, (0, 0, -1)),
    (47, (1, -1, 0)),
    (60, (-1, 1, 1)),
)

# ---------------------------------------------------------------------------
# Advisory timing system (printed, never asserted).
# ---------------------------------------------------------------------------

#: 8x8x8 sites at 3.0 A — 512 atoms in a 24 A cube, where `n_bins == (9,9,9)`
#: and the stencil prunes to 125 of 729 bins.
TIMING_N_SIDE = 8

# ---------------------------------------------------------------------------
# Key construction — the comparison currency.
# ---------------------------------------------------------------------------

#: Decimals kept when rounding a shift component or a distance into a key.
#: Coarser than float64 noise on integer combinations of cell vectors (~1e-15 A),
#: finer than the smallest distinct shift component (6.0 A).
KEY_DECIMALS = 6

Shift = tuple[float, ...]
EdgeKey = tuple[int, int, Shift]
PairDistance = tuple[int, int, float]


def cubic_lattice(n_side: int) -> torch.Tensor:
    """Build a simple-cubic lattice of `n_side**3` sites at `SPACING` Angstrom.

    Args:
        n_side: Sites per axis.

    Returns:
        Positions ``(n_side**3, 3)`` in Angstrom, float64, in lexicographic
        order. Built from ``arange``/``meshgrid``: no RNG, no data file.
    """
    axis = torch.arange(n_side, dtype=torch.float64) * SPACING
    grid_x, grid_y, grid_z = torch.meshgrid(axis, axis, axis, indexing="ij")
    return torch.stack((grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1)), dim=-1)


def translated_lattice() -> torch.Tensor:
    """Section 1's lattice with `TRANSLATED_ATOMS` moved by whole cell vectors.

    Returns:
        Positions ``(64, 3)`` in Angstrom, float64 — the same periodic system as
        :func:`cubic_lattice`, deliberately *not* wrapped back into the box.
    """
    positions = cubic_lattice(N_SIDE)
    for atom, image in TRANSLATED_ATOMS:
        positions[atom] += torch.tensor(image, dtype=torch.float64) @ CUBIC_CELL
    return positions


def directed_keys(neighbor_list: NeighborList) -> list[EdgeKey]:
    """Read the live edges as `(source, target, rounded shift)` triples.

    Args:
        neighbor_list: A built list; only its public buffers are read.

    Returns:
        One key per live directed edge, in buffer order (which is *not* part of
        the contract — every comparison below is a set or a sorted multiset).
    """
    edge_index = neighbor_list.edge_index[: neighbor_list.num_edges].tolist()
    shifts = neighbor_list.shifts[: neighbor_list.num_edges].tolist()
    return [
        (int(source), int(target), tuple(round(component, KEY_DECIMALS) for component in shift))
        for (source, target), shift in zip(edge_index, shifts, strict=True)
    ]


def canonical_keys(neighbor_list: NeighborList) -> list[EdgeKey]:
    """Re-orient :func:`directed_keys` low index first, negating the shift.

    The shift is the periodic remainder of ``pos[target] - pos[source]``, so it
    flips sign wholesale with the edge. Re-orienting makes the key blind to which
    way round the bidirectional list stored a pair while still separating
    periodic images of it.

    Args:
        neighbor_list: A built list.

    Returns:
        One orientation-free key per live directed edge; a duplicate-free
        bidirectional list yields each unordered pair exactly twice.
    """
    canonical: list[EdgeKey] = []
    for source, target, shift in directed_keys(neighbor_list):
        if source > target:
            source, target, shift = target, source, tuple(-component for component in shift)
        canonical.append((source, target, shift))
    return canonical


def pair_distances(neighbor_list: NeighborList, positions: torch.Tensor) -> list[PairDistance]:
    """Reconstruct each live edge's length from the stored positions and shifts.

    ``edge_diff = pos[target] - pos[source] + shift`` is the documented way a
    consumer recovers the displacement, and it is the step that fails if the
    shifts refer to a wrapped copy of the positions rather than the stored ones.

    Args:
        neighbor_list: A built list.
        positions: The positions it was built at ``(N, 3)`` in Angstrom.

    Returns:
        Sorted ``(min index, max index, rounded distance in Angstrom)`` triples —
        a multiset, so a duplicated edge is visible as a repeated entry.
    """
    edge_index = neighbor_list.edge_index[: neighbor_list.num_edges]
    shifts = neighbor_list.shifts[: neighbor_list.num_edges]
    displacement = positions[edge_index[:, 1]] - positions[edge_index[:, 0]] + shifts
    distance = torch.linalg.norm(displacement, dim=-1)
    return sorted(
        (min(int(source), int(target)), max(int(source), int(target)), round(length, KEY_DECIMALS))
        for (source, target), length in zip(edge_index.tolist(), distance.tolist(), strict=True)
    )


def minimum_image_extremes(positions: torch.Tensor, cell: torch.Tensor) -> tuple[float, float]:
    """Closest minimum-image pair distance, and the distance closest to `r_build`.

    Brute force over all unordered pairs and all 27 images — independent of the
    neighbour list under test, so it can serve as its precondition.

    Args:
        positions: Positions ``(N, 3)`` in Angstrom.
        cell: Cell vectors ``(3, 3)`` in Angstrom, one per row.

    Returns:
        ``(closest, nearest_to_boundary)`` in Angstrom: the smallest pair
        distance, and the pair distance minimising ``|r - r_build|`` (returned as
        the distance itself, so the caller reports the margin it cares about).
    """
    images = torch.tensor(list(itertools.product((-1, 0, 1), repeat=3)), dtype=torch.float64) @ cell
    n_atoms = int(positions.shape[0])
    closest = float("inf")
    boundary = float("inf")
    r_build = TRICLINIC_CUTOFF + TRICLINIC_SKIN
    for first, second in itertools.combinations(range(n_atoms), 2):
        separation = positions[second] - positions[first] + images
        distance = float(torch.linalg.norm(separation, dim=-1).min())
        closest = min(closest, distance)
        if abs(distance - r_build) < abs(boundary - r_build):
            boundary = distance
    return closest, boundary


# ---------------------------------------------------------------------------
# Checking
# ---------------------------------------------------------------------------


class Checker:
    """Collects every deviation so one run reports all failures, not the first."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def _row(self, name: str, got: object, ok: bool) -> None:
        print(f"  {name:<46} {got!s:<24} {'ok' if ok else 'FAILED'}")

    def exact(self, name: str, got: object, want: object) -> None:
        """Assert a counter, a bin-count tuple or a set cardinality."""
        ok = got == want
        if not ok:
            self.failures.append(f"{name}: got {got!r}, want {want!r}")
        self._row(name, got, ok)

    def same_edges(self, name: str, binned: list[EdgeKey], kernel: list[EdgeKey]) -> None:
        """Assert the two backends produced the same canonical edge set."""
        binned_set, kernel_set = set(binned), set(kernel)
        ok = binned_set == kernel_set
        if not ok:
            only_binned = sorted(binned_set - kernel_set)[:3]
            only_kernel = sorted(kernel_set - binned_set)[:3]
            self.failures.append(
                f"{name}: {len(binned_set - kernel_set)} keys only in the binned path "
                f"(e.g. {only_binned}), {len(kernel_set - binned_set)} only in the kernel "
                f"path (e.g. {only_kernel})"
            )
        self._row(name, f"{len(binned_set)} keys", ok)

    def same_distances(
        self, name: str, measured: list[PairDistance], reference: list[PairDistance]
    ) -> None:
        """Assert two builds carry the same multiset of `(pair, distance)` entries."""
        ok = measured == reference
        if not ok:
            self.failures.append(
                f"{name}: {len(measured)} entries against {len(reference)}; first difference "
                f"{next((pair for pair in zip(measured, reference) if pair[0] != pair[1]), None)}"
            )
        self._row(name, f"{len(measured)} entries", ok)

    def above(self, name: str, got: float, floor: float) -> None:
        """Assert a measured margin (Angstrom) clears a precondition floor."""
        ok = got > floor
        if not ok:
            self.failures.append(f"{name}: margin {got!r} A is not above {floor!r} A")
        self._row(name, f"{got:.6f} A", ok)


def check_duplicate_freedom(checker: Checker, label: str, neighbor_list: NeighborList) -> None:
    """Assert the list emits each directed edge once and collapses 2:1.

    Args:
        checker: Failure collector.
        label: Prefix for the reported rows.
        neighbor_list: A built list.
    """
    checker.exact(
        f"{label}.unique_directed_keys",
        len(set(directed_keys(neighbor_list))),
        neighbor_list.num_edges,
    )
    checker.exact(
        f"{label}.canonical_keys_x2",
        2 * len(set(canonical_keys(neighbor_list))),
        neighbor_list.num_edges,
    )


def check_cubic_auto_bin(checker: Checker) -> list[EdgeKey]:
    """Section 1 — the wrapping-stencil regime at `r_build = 5.0 A`.

    Args:
        checker: Failure collector.

    Returns:
        The binned path's canonical edge keys, reused as section 3's reference.
    """
    print(f"\nSection 1 — 64-atom cubic lattice, cutoff {CUTOFF} A, skin {SKIN} A, bin=0.0")

    positions = cubic_lattice(N_SIDE)
    kernel = NeighborList(cell=CUBIC_CELL, cutoff=CUTOFF, positions=positions, skin=SKIN)
    binned = NeighborList(
        cell=CUBIC_CELL, cutoff=CUTOFF, positions=positions, skin=SKIN, bin=AUTO_BIN
    )

    checker.exact("cubic.kernel_n_bins", kernel.n_bins, None)
    checker.exact("cubic.binned_n_bins", binned.n_bins, CUBIC_N_BINS_AT_R_BUILD)
    checker.exact("cubic.kernel_num_edges", kernel.num_edges, CUBIC_EDGES_AT_R_BUILD)
    checker.exact("cubic.binned_num_edges", binned.num_edges, CUBIC_EDGES_AT_R_BUILD)

    binned_keys = canonical_keys(binned)
    checker.same_edges("cubic.edge_sets_agree", binned_keys, canonical_keys(kernel))
    checker.exact("cubic.unordered_pairs", len(set(binned_keys)), CUBIC_PAIRS_AT_R_BUILD)
    check_duplicate_freedom(checker, "cubic", binned)
    return binned_keys


def check_bare_cutoff_grid(checker: Checker) -> None:
    """Section 2 — the same cell at `skin = 0.0` grids on `r_build`, not `cutoff`.

    Args:
        checker: Failure collector.
    """
    print(f"\nSection 2 — same lattice, cutoff {CUTOFF} A, skin 0.0 A, bin=0.0")

    positions = cubic_lattice(N_SIDE)
    kernel = NeighborList(cell=CUBIC_CELL, cutoff=CUTOFF, positions=positions, skin=0.0)
    binned = NeighborList(
        cell=CUBIC_CELL, cutoff=CUTOFF, positions=positions, skin=0.0, bin=AUTO_BIN
    )

    checker.exact("bare.binned_n_bins", binned.n_bins, CUBIC_N_BINS_AT_CUTOFF)
    checker.exact("bare.kernel_num_edges", kernel.num_edges, CUBIC_EDGES_AT_CUTOFF)
    checker.exact("bare.binned_num_edges", binned.num_edges, CUBIC_EDGES_AT_CUTOFF)
    checker.same_edges("bare.edge_sets_agree", canonical_keys(binned), canonical_keys(kernel))
    check_duplicate_freedom(checker, "bare", binned)


def check_explicit_bin_sizes(checker: Checker, reference: list[EdgeKey]) -> None:
    """Section 3 — explicit bin thicknesses reproduce section 1's edge set.

    Args:
        checker: Failure collector.
        reference: Section 1's canonical keys from the automatic grid.
    """
    print("\nSection 3 — same system at explicit bin thicknesses")

    positions = cubic_lattice(N_SIDE)
    for thickness, expected_bins in EXPLICIT_BINS:
        label = f"bin{thickness:g}"
        binned = NeighborList(
            cell=CUBIC_CELL, cutoff=CUTOFF, positions=positions, skin=SKIN, bin=thickness
        )
        checker.exact(f"{label}.n_bins", binned.n_bins, expected_bins)
        checker.exact(f"{label}.num_edges", binned.num_edges, CUBIC_EDGES_AT_R_BUILD)
        checker.same_edges(f"{label}.edge_set_matches_auto", canonical_keys(binned), reference)
        check_duplicate_freedom(checker, label, binned)


def check_triclinic(checker: Checker) -> None:
    """Section 4 — sheared cell, perpendicular-width sizing, captured golden count.

    Args:
        checker: Failure collector.
    """
    print(
        f"\nSection 4 — 12-atom triclinic cell, cutoff {TRICLINIC_CUTOFF} A, "
        f"skin {TRICLINIC_SKIN} A, bin=0.0"
    )

    positions = TRICLINIC_FRACTIONAL @ TRICLINIC_CELL
    closest, boundary = minimum_image_extremes(positions, TRICLINIC_CELL)
    r_build = TRICLINIC_CUTOFF + TRICLINIC_SKIN
    # Precondition, not a golden: a pair sitting exactly at r_build would make
    # the comparison a coin toss between two correct backends.
    checker.above("triclinic.closest_pair", closest, MARGIN_FLOOR)
    checker.above("triclinic.margin_from_r_build", abs(boundary - r_build), MARGIN_FLOOR)

    kernel = NeighborList(
        cell=TRICLINIC_CELL,
        cutoff=TRICLINIC_CUTOFF,
        positions=positions,
        skin=TRICLINIC_SKIN,
    )
    binned = NeighborList(
        cell=TRICLINIC_CELL,
        cutoff=TRICLINIC_CUTOFF,
        positions=positions,
        skin=TRICLINIC_SKIN,
        bin=AUTO_BIN,
    )

    checker.exact("triclinic.binned_n_bins", binned.n_bins, TRICLINIC_N_BINS)
    checker.exact("triclinic.kernel_num_edges", kernel.num_edges, TRICLINIC_EDGES)
    checker.exact("triclinic.binned_num_edges", binned.num_edges, TRICLINIC_EDGES)
    checker.same_edges("triclinic.edge_sets_agree", canonical_keys(binned), canonical_keys(kernel))
    check_duplicate_freedom(checker, "triclinic", binned)


def check_unwrapped(checker: Checker) -> None:
    """Section 5 — whole-cell translations move labels, never physics.

    Args:
        checker: Failure collector.
    """
    print(
        f"\nSection 5 — section 1's lattice with {len(TRANSLATED_ATOMS)} atoms "
        f"translated by +-{BOX:g} A"
    )

    reference_positions = cubic_lattice(N_SIDE)
    reference = NeighborList(
        cell=CUBIC_CELL, cutoff=CUTOFF, positions=reference_positions, skin=SKIN, bin=AUTO_BIN
    )

    positions = translated_lattice()
    kernel = NeighborList(cell=CUBIC_CELL, cutoff=CUTOFF, positions=positions, skin=SKIN)
    binned = NeighborList(
        cell=CUBIC_CELL, cutoff=CUTOFF, positions=positions, skin=SKIN, bin=AUTO_BIN
    )

    checker.exact("unwrapped.kernel_num_edges", kernel.num_edges, CUBIC_EDGES_AT_R_BUILD)
    checker.exact("unwrapped.binned_num_edges", binned.num_edges, CUBIC_EDGES_AT_R_BUILD)
    checker.same_edges("unwrapped.edge_sets_agree", canonical_keys(binned), canonical_keys(kernel))
    check_duplicate_freedom(checker, "unwrapped", binned)
    checker.same_distances(
        "unwrapped.distances_match_section_1",
        pair_distances(binned, positions),
        pair_distances(reference, reference_positions),
    )


def print_build_timings() -> None:
    """Print both backends' build wall clock at 512 atoms. Asserts nothing.

    The numbers are machine-, thread- and build-bound — at a high thread count
    the kernel's single fused op beats the binned path's 125-offset loop, and at
    a low one it loses badly. That is why no timing threshold is asserted in this
    file, in the unit tests, or in the spec's acceptance criteria.
    """
    positions = cubic_lattice(TIMING_N_SIDE)
    n_atoms = int(positions.shape[0])
    cell = CUBIC_CELL * (TIMING_N_SIDE / N_SIDE)

    kernel = NeighborList(cell=cell, cutoff=CUTOFF, positions=positions, skin=SKIN)
    binned = NeighborList(cell=cell, cutoff=CUTOFF, positions=positions, skin=SKIN, bin=AUTO_BIN)

    start = time.perf_counter()
    kernel.rebuild(positions)
    kernel_seconds = time.perf_counter() - start

    start = time.perf_counter()
    binned.rebuild(positions)
    binned_seconds = time.perf_counter() - start

    print(
        f"\nAdvisory (nothing below is asserted) — {n_atoms} atoms, "
        f"{float(cell[0, 0]):g} A cube, r_build {CUTOFF + SKIN:g} A, "
        f"n_bins {binned.n_bins}, {torch.get_num_threads()} torch threads"
    )
    print(f"  kernel (bin=None) build          {kernel_seconds:.4f} s  {kernel.num_edges} edges")
    print(f"  binned (bin=0.0)  build          {binned_seconds:.4f} s  {binned.num_edges} edges")


def main() -> int:
    """Pin the binned backend against the in-repo kernel oracle."""
    checker = Checker()

    cubic_reference = check_cubic_auto_bin(checker)
    check_bare_cutoff_grid(checker)
    check_explicit_bin_sizes(checker, cubic_reference)
    check_triclinic(checker)
    check_unwrapped(checker)
    print_build_timings()

    if checker.failures:
        print("\nFAILED — the binned build no longer agrees with the kernel path:")
        for failure in checker.failures:
            print(f"  {failure}")
        print(
            "\nEvery literal above is analytic (64*18 = 1152, 64*6 = 384, "
            "floor(12/2.5) = 4, floor(12/1.75) = 6, floor(8/1.75) = 4, "
            "floor(10/1.75) = 5) except the triclinic edge count, which was "
            "captured once from the in-repo bin=None kernel oracle. A "
            "disagreement is a DEFECT REPORT against src/molix/md/neighbors.py, "
            "never a reason to edit a literal: a duplicated edge doubles a "
            "pair's energy contribution and a missing one deletes it, and both "
            "are silent everywhere except here."
        )
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
