"""Public-API scenario for the `PeriodicNeighborList` cutoff bound.

Spec: `md-neighborlist-skin-01-guard`.

The claim this file pins, in one screenful, for the golden triclinic cell

    cell = [[10, 0, 0],
            [ 6, 8, 0],
            [ 0, 0, 10]]        V = |det| = 800 A^3

is that the admissible cutoff is bounded by **half the smallest perpendicular
cell width**, not by half the shortest row norm:

    ||a_2 x a_3|| = 100,  ||a_3 x a_1|| = 100,  ||a_1 x a_2|| = 80   (A^2)
    w = (V/100, V/100, V/80) = (8.000, 8.000, 10.000) A
    bound = min_i w_i / 2 = 4.000 A          <- what the constructor must enforce
    min_i ||a_i|| / 2    = 5.000 A           <- what the old row-norm guard allowed

Everything between 4.000 A and 5.000 A is the bug: the old guard admitted those
cutoffs, and the kernel's *sequential* minimum-image reduction (subtract
round(dz/c_zz)*a_3, then round(dy/b_yy)*a_2, then round(dx/a_xx)*a_1) then
returns a displacement longer than the true minimum image, so pairs that are
inside the cutoff are silently dropped — wrong energy, wrong forces, no error.

Part 1 — the bound.  `cutoff=5.0` (the exact value the old guard admitted) and
`cutoff=4.5` both raise `ValueError`, and the message names the measured bound
`4.000 A` and the minimum width `8.000 A`, so the *number* is pinned and not
merely the failure.  `cutoff=4.0` — exactly the bound — constructs, and
`cutoff=4.0001` does not; that brackets the bound to a ten-thousandth of an
Angstrom through the public error message alone.  `cutoff=3.9` constructs and
actually builds a non-empty edge set, so "accepted" means "built", not "did not
raise".

Part 2 — completeness at the admitted cutoff.  Twelve atoms, written below as
literal fractional coordinates and mapped to Cartesian by `frac @ cell`, are
handed to `PeriodicNeighborList(cutoff=3.9)`.  The live half-pair set it
produces is compared against a brute-force reference computed **in this file**:
for every i < j, minimise ||r_j - r_i + n . cell|| over all 27 shifts
n in {-1, 0, 1}^3.  Same pairs, same distances, nothing missed.  Since
3.9 A < w_min/2 = 4.0 A, at most one periodic image of a pair can lie inside
the cutoff, so that minimum *is* the complete answer — the brute force is an
analytic oracle, not a third-party one.  Eight of the fourteen reference pairs
are cross-boundary (non-zero shift n), which is what makes this a periodic test
rather than an open-boundary one.

The neighbour list is symmetry-expanded (`E = 2 * n_pairs`), so `edge_index`
rows are deduplicated to `(min, max)` half pairs here; that every pair appears
exactly twice is asserted, since the deduplication would otherwise hide a
missing reverse edge.  Distances are recovered from the public buffers as
`|| pos[target] - pos[source] + shifts ||`, the identity `shifts` is defined by.

That this layout is a real counterexample and not just a passing one was
checked out of band while writing the file: with the guard forced back to the
old row-norm bound, the same twelve atoms at `cutoff=5.0` give 41 reference
half pairs and only 39 reported ones — pairs `(1, 8)` at 4.769 A and `(3, 10)`
at 4.827 A are silently dropped.  That measurement is **not** asserted here: it
needs a private helper monkeypatched, and this file stays on the public API.
Part 1 refusing 5.0 A is the public-API form of the same claim.

What is deliberately **not** pinned: `capacity` and the dead-edge tail padding
(fixed-capacity buffering is a different concern from the min-image bound), and
anything timed.

Goldens
-------
    capture command : PYTHONPATH=src python regressions/md-neighborlist-skin-01-guard.py
                      The one measured literal is N_HALF_PAIRS = 14, captured
                      from the in-script brute-force reference (not from the
                      neighbour list) while writing this file.  Every other
                      golden is arithmetic over the cell above: V = 800, the
                      three face areas 100 / 100 / 80, w = (8, 8, 10), the
                      bound 4.000.
    commit          : 786d2b7 (786d2b74370371b039bae0fb6412a8f48466e28c), with
                      the `md-neighborlist-skin-01-guard` working tree on top
                      (the perpendicular-width guard in
                      `src/molix/md/neighbors.py` is new there).
    torch           : 2.12.1+cpu   (python 3.14.5)
    date            : 2026-08-09
    device / dtype  : CPU, float64 throughout — the cell and the positions are
                      float64 literals, so the neighbour list's buffers are
                      float64 too.  float64 is what makes the 1e-9 A distance
                      tolerance meaningful; the observed agreement is exact.
    oracle          : none.  No third-party package (torch is the repo's own
                      core dependency), no network, no subprocess, no RNG, no
                      wall-clock value, no filesystem access.
    tolerance       : 1e-9 A on distances (float64 "position" band, 1e-8, with a
                      decade of slack unused: the observed maximum deviation is
                      0.0, bit-for-bit).  Exact equality on pair sets and counts.

Run:
    PYTHONPATH=src python regressions/md-neighborlist-skin-01-guard.py
"""

from __future__ import annotations

import itertools
import sys

import torch

from molix.md import PeriodicNeighborList

# ---------------------------------------------------------------------------
# The golden triclinic cell.  Rows are the cell vectors a_1, a_2, a_3 (A).
#
# Lower triangular, which is the form the kernel's reduction assumes.  a_2 is
# sheared by 6 A along x, and that shear is the whole story: it costs the cell
# 2 A of perpendicular width along x (w_1 = 8, not 10) while leaving ||a_1|| =
# ||a_3|| = 10 and ||a_2|| = 10 — every row norm is 10, so the row-norm guard
# sees a 10 A cube that is not there.
# ---------------------------------------------------------------------------

CELL = torch.tensor(
    [
        [10.0, 0.0, 0.0],
        [6.0, 8.0, 0.0],
        [0.0, 0.0, 10.0],
    ],
    dtype=torch.float64,
)

#: `V / max_i ||a_j x a_k||` = 800 / 100.  Not read from the code under test —
#: the private helper is never imported here; this is the paper value.
MIN_WIDTH = 8.0

#: The contract: `cutoff <= min_i w_i / 2`.  Formatted as `4.000 A` by the
#: constructor's error message, which is the only public window onto it.
BOUND = 4.0

#: What the superseded row-norm guard admitted: `min_i ||a_i|| / 2` = 10 / 2.
#: Every cutoff in (4.000, 5.000] used to be accepted and is now refused.
OLD_ROW_NORM_BOUND = 5.0

#: Substrings the `ValueError` must carry, so the numbers survive a reword.
BOUND_TEXT = "4.000 A"
MIN_WIDTH_TEXT = "8.000 A"

# ---------------------------------------------------------------------------
# Twelve atoms as literal fractional coordinates (no RNG, no fixture).
#
# Cartesian is `frac @ cell`, i.e. r = (10*f1 + 6*f2, 8*f2, 10*f3).  The layout
# is three loose layers along b with a deliberate spread along c, chosen so that
#
#   * fourteen half pairs sit inside the 3.9 A cutoff and eight of them only
#     through a periodic image, and
#   * no pair separation lands near the cutoff — the closest miss is 4.139 A and
#     the closest hit 3.582 A, a 0.239 A margin either side of 3.9, so the pair
#     *set* cannot flip on an arithmetic reordering (the tolerance that matters
#     for set membership is ~1e-1, not ~1e-9).
#
# Closest approach overall is 3.036 A, so nothing is unphysically overlapped.
# ---------------------------------------------------------------------------

FRACTIONAL = torch.tensor(
    [
        [0.00, 0.06, 0.10],
        [0.30, 0.12, 0.15],
        [0.46, 0.07, 0.85],
        [0.76, 0.15, 0.55],
        [0.06, 0.40, 0.35],
        [0.36, 0.45, 0.60],
        [0.61, 0.38, 0.05],
        [0.85, 0.42, 0.81],
        [0.05, 0.70, 0.20],
        [0.33, 0.75, 0.91],
        [0.65, 0.68, 0.45],
        [0.82, 0.72, 0.70],
    ],
    dtype=torch.float64,
)

#: The admitted cutoff for part 2: below the 4.000 A bound, above the old
#: guard's threshold for nothing — it is simply a legal cutoff at which the
#: minimum-image reduction is guaranteed complete.
CUTOFF = 3.9

#: **Captured once** from the in-script brute-force reference below (not from
#: the neighbour list), at the commit in the header.  Fourteen of the 66 half
#: pairs are within 3.9 A; the neighbour list is symmetry-expanded, so this is
#: 28 live edges.
N_HALF_PAIRS = 14

#: Of those fourteen, the number reachable only across a periodic boundary
#: (minimising shift n != 0).  Same capture.  Pinned separately because a
#: neighbour list that silently lost periodicity would still find the other six.
N_CROSS_BOUNDARY_PAIRS = 8

#: Position-band tolerance in Angstrom for float64 (CLAUDE tester contract:
#: 1e-8 numerical, 1e-12 exact).  Observed deviation is 0.0.
DIST_TOL = 1e-9

#: All 27 periodic images n in {-1, 0, 1}^3.  Enough because the search radius
#: 3.9 A is below every half-width (4.0, 4.0, 5.0 A), so no second-shell image
#: can be the minimiser.
IMAGE_SHIFTS: tuple[tuple[int, ...], ...] = tuple(itertools.product((-1, 0, 1), repeat=3))


# ---------------------------------------------------------------------------
# Checking
# ---------------------------------------------------------------------------


class Checker:
    """Collects every deviation so one run reports all failures, not the first."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def _row(self, name: str, got: object, ok: bool) -> None:
        print(f"  {name:<40} {got!s:<32} {'ok' if ok else 'FAILED'}")

    def exact(self, name: str, got: object, want: object) -> None:
        """Assert a count, a pair set or a string — no tolerance applies."""
        ok = got == want
        if not ok:
            self.failures.append(f"{name}: got {got!r}, want {want!r}")
        self._row(name, got, ok)

    def within(self, name: str, got: float, tol: float) -> None:
        """Assert a measured deviation is at most *tol* (Angstrom)."""
        ok = got <= tol
        if not ok:
            self.failures.append(f"{name}: deviation {got!r} A exceeds {tol!r} A")
        self._row(name, f"{got:.3e} A", ok)

    def truth(self, name: str, holds: bool, message: str) -> None:
        """Assert a boolean contract (a cutoff was refused, a list was built, ...)."""
        if not holds:
            self.failures.append(f"{name}: {message}")
        self._row(name, holds, holds)


def positions() -> torch.Tensor:
    """Cartesian coordinates ``(12, 3)`` in Angstrom for the literal layout.

    Returns:
        ``FRACTIONAL @ CELL`` in float64 — fractional coordinates are the
        readable form, Cartesian is what the public API takes.
    """
    return FRACTIONAL @ CELL


def refuses(cutoff: float) -> str | None:
    """Construct at *cutoff* and report the refusal message, if any.

    Args:
        cutoff: Model cutoff ``r_cut`` in Angstrom.

    Returns:
        The ``ValueError`` message if the constructor refused, else ``None``.
    """
    try:
        PeriodicNeighborList(cell=CELL, cutoff=cutoff, positions=positions())
    except ValueError as error:
        return str(error)
    return None


def brute_force_pairs(pos: torch.Tensor) -> dict[tuple[int, int], float]:
    """Minimum-image half pairs within :data:`CUTOFF`, by exhaustive search.

    For every ``i < j`` the true minimum-image separation is
    ``min_n ||r_j - r_i + n . cell||`` over the 27 shifts ``n in {-1,0,1}^3``.
    No shortcut, no sequential reduction — this is the reference the kernel's
    reduction is supposed to reproduce.

    Args:
        pos: Cartesian positions ``(N, 3)`` in Angstrom, float64.

    Returns:
        ``{(i, j): distance}`` in Angstrom for every half pair whose minimum
        image lies within :data:`CUTOFF`, with ``i < j``.
    """
    images = torch.tensor(IMAGE_SHIFTS, dtype=pos.dtype) @ CELL  # (27, 3)
    inside: dict[tuple[int, int], float] = {}
    n_atoms = int(pos.shape[0])
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            distance = float(torch.linalg.norm(pos[j] - pos[i] + images, dim=-1).min())
            if distance <= CUTOFF:
                inside[(i, j)] = distance
    return inside


def brute_force_cross_boundary(pos: torch.Tensor) -> int:
    """How many reference pairs are reachable only through a periodic image.

    Args:
        pos: Cartesian positions ``(N, 3)`` in Angstrom, float64.

    Returns:
        The count of in-cutoff half pairs whose minimising shift is non-zero.
    """
    shifts = torch.tensor(IMAGE_SHIFTS, dtype=pos.dtype)
    images = shifts @ CELL
    count = 0
    n_atoms = int(pos.shape[0])
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            norms = torch.linalg.norm(pos[j] - pos[i] + images, dim=-1)
            best = int(norms.argmin())
            if float(norms[best]) <= CUTOFF and bool(shifts[best].abs().sum() > 0):
                count += 1
    return count


def live_half_pairs(
    neighbors: PeriodicNeighborList, pos: torch.Tensor
) -> dict[tuple[int, int], list[float]]:
    """Deduplicate the live, symmetry-expanded edge buffer into half pairs.

    The list is built with ``symmetry=True``, so each pair occupies two rows
    (``i -> j`` and ``j -> i``); both are collected under the ``(min, max)`` key
    so the caller can assert each pair appears exactly twice. Distances come
    from the public buffers via the identity ``shifts`` is defined by:
    ``edge_diff = pos[target] - pos[source] + shifts``.

    Args:
        neighbors: A built list; only its public ``edge_index`` / ``shifts`` /
            ``num_edges`` members are read.
        pos: The Cartesian positions the list was built at, ``(N, 3)``.

    Returns:
        ``{(i, j): [distance, ...]}`` in Angstrom over the live rows
        ``[0, num_edges)``.
    """
    edge_index = neighbors.edge_index[: neighbors.num_edges]
    source, target = edge_index[:, 0], edge_index[:, 1]
    edge_diff = pos[target] - pos[source] + neighbors.shifts[: neighbors.num_edges]
    edge_dist = torch.linalg.norm(edge_diff, dim=-1)
    pairs: dict[tuple[int, int], list[float]] = {}
    for row in range(int(edge_index.shape[0])):
        i, j = int(source[row]), int(target[row])
        pairs.setdefault((min(i, j), max(i, j)), []).append(float(edge_dist[row]))
    return pairs


def check_bound(checker: Checker) -> None:
    """Part 1 — the constructor bounds the cutoff at 4.000 A, not 5.000 A.

    Args:
        checker: Failure collector.
    """
    print("Bound (cell rows 10 / 6,8 / 10 A; V = 800 A^3; w = (8.000, 8.000, 10.000) A)")

    old_guard_message = refuses(OLD_ROW_NORM_BOUND)
    checker.truth(
        "bound.rejects_5.0",
        old_guard_message is not None,
        "cutoff=5.0 A was accepted — that is half the shortest row norm, not "
        "half the smallest perpendicular width; the min-image reduction drops "
        "pairs inside the cutoff there",
    )
    checker.truth(
        "bound.rejects_4.5",
        refuses(4.5) is not None,
        "cutoff=4.5 A was accepted — above the 4.000 A bound, below the old "
        "row-norm guard's 5.000 A, i.e. squarely in the silently-wrong window",
    )
    checker.truth(
        "bound.message_names_4.000_A",
        old_guard_message is not None and BOUND_TEXT in old_guard_message,
        f"the refusal does not contain {BOUND_TEXT!r}, so the measured bound is "
        f"not observable through the public API: {old_guard_message!r}",
    )
    checker.truth(
        "bound.message_names_8.000_A",
        old_guard_message is not None and MIN_WIDTH_TEXT in old_guard_message,
        f"the refusal does not contain {MIN_WIDTH_TEXT!r} (min_i w_i = "
        f"{MIN_WIDTH} A): {old_guard_message!r}",
    )
    # Brackets the bound: accepted at exactly min_i w_i / 2, refused a
    # ten-thousandth of an Angstrom above it.
    checker.truth(
        "bound.accepts_exactly_4.0",
        refuses(BOUND) is None,
        "cutoff=4.000 A was refused; the contract is `cutoff <= min_i w_i / 2`, "
        "so the bound itself is admissible",
    )
    checker.truth(
        "bound.rejects_4.0001",
        refuses(4.0001) is not None,
        "cutoff=4.0001 A was accepted; the bound is 4.000 A exactly",
    )

    accepted = PeriodicNeighborList(cell=CELL, cutoff=CUTOFF, positions=positions())
    checker.truth(
        "bound.accepts_3.9_and_builds",
        accepted.num_edges > 0,
        f"cutoff={CUTOFF} A constructed but produced {accepted.num_edges} edges — "
        "acceptance must mean 'actually built', not merely 'did not raise'",
    )


def check_completeness(checker: Checker) -> None:
    """Part 2 — at the admitted cutoff the list equals the 27-image reference.

    Args:
        checker: Failure collector.
    """
    print(f"\nCompleteness at cutoff = {CUTOFF} A (12 atoms, brute force over 27 images)")
    pos = positions()
    reference = brute_force_pairs(pos)
    checker.exact("reference.n_half_pairs", len(reference), N_HALF_PAIRS)
    checker.exact(
        "reference.n_cross_boundary",
        brute_force_cross_boundary(pos),
        N_CROSS_BOUNDARY_PAIRS,
    )

    neighbors = PeriodicNeighborList(cell=CELL, cutoff=CUTOFF, positions=pos)
    observed = live_half_pairs(neighbors, pos)
    checker.exact("list.num_edges", neighbors.num_edges, 2 * N_HALF_PAIRS)
    checker.exact("list.n_half_pairs", len(observed), N_HALF_PAIRS)
    checker.truth(
        "list.each_pair_bidirectional",
        all(len(distances) == 2 for distances in observed.values()),
        "a half pair does not occupy exactly two rows: "
        f"{sorted(pair for pair, d in observed.items() if len(d) != 2)}",
    )

    missed = sorted(set(reference) - set(observed))
    spurious = sorted(set(observed) - set(reference))
    checker.exact("completeness.missed_pairs", tuple(missed), ())
    checker.exact("completeness.spurious_pairs", tuple(spurious), ())
    if missed:
        print("    pairs inside the cutoff that the list never reported:")
        for pair in missed:
            print(f"      {pair}: true minimum image {reference[pair]:.9f} A")

    shared = set(reference) & set(observed)
    deviation = max(
        (abs(distance - reference[pair]) for pair in shared for distance in observed[pair]),
        default=0.0,
    )
    checker.within("completeness.max_distance_deviation", deviation, DIST_TOL)


def main() -> int:
    """Pin the cutoff bound and the minimum-image completeness beneath it."""
    checker = Checker()
    check_bound(checker)
    check_completeness(checker)

    if checker.failures:
        print("\nFAILED — PeriodicNeighborList no longer matches the golden bound / reference:")
        for failure in checker.failures:
            print(f"  {failure}")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
