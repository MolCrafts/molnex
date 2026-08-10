"""Public-API scenario for the `PeriodicNeighborList` -> `NeighborList` rename.

Spec: `md-neighborlist-skin-03-rename`.

The rename is declared **behaviour-neutral**: identifiers, docstrings, `__all__`
entries and prose move; no logic, signature, default or numerical result does.
A rename that quietly changed a number would be indistinguishable from a
successful one if the only evidence were "the suite is green after the suite was
edited" — every unit-test call site had its identifier substituted in the same
commit.  This file is the independent witness: it was written against the
*renamed* symbol only, it touches nothing private, and every literal it asserts
is arithmetic on the page rather than a value read off a run.

Section 1 — the surface.  `from molix.md import NeighborList` resolves, and
`molix.md.PeriodicNeighborList` does not exist.  Stage is `experimental` and the
spec forbids a back-compat alias, so the old name must be *gone*, not shimmed:
a `PeriodicNeighborList = NeighborList` line, a module `__getattr__` fallback or
a stale re-export would all be caught here.  `"PeriodicNeighborList" not in
molix.md.__all__` is checked separately from the attribute, because a name can
survive in `__all__` (breaking `from molix.md import *`) while `hasattr` is
already False, and vice versa.

Section 2 — the numbers the rename must not have moved.  A 3x3x3 simple-cubic
lattice, spacing 3.0 A, in a cubic 9.0 A cell, at `cutoff=3.5 A`:

    nearest-neighbour separation      3.000 A  <= 3.5   in   (6 per site)
    face-diagonal separation   sqrt2*3 = 4.243 A  >  3.5   out
    body-diagonal separation   sqrt3*3 = 5.196 A  >  3.5   out

so each of the 27 sites has exactly its 6 axis neighbours (+-x, +-y, +-z) inside
the cutoff — distinct sites, since 3 lattice points per axis means the +3.0 A and
-3.0 A neighbours are different atoms, not the same image twice.  Under the repo
convention (`symmetry=True`, full bidirectional list):

    num_edges = 27 * 6 = 162
    capacity  = ceil(1.35 * 162) = ceil(218.7) = 219      (default capacity_factor)
    edge_index (219, 2)      shifts (219, 3)

The 0.5 A margin on either side of the cutoff (3.0 in, 4.243 out) means no pair
can flip class on floating-point noise, which is what makes `162` a hard integer
golden and not a tolerance question.  `cutoff = 3.5 A <= 4.5 A = 9.0/2` also
clears the constructor's half-perpendicular-width bound, so the minimum-image
reduction is complete here.

Section 3 — the dead-edge tail.  Rows `[num_edges, capacity)` (57 of them) are
padding and must be *inert*: source and target both atom 0, displaced by
`DEAD_EDGE_CUTOFF_FACTOR (10.0) * cutoff (3.5) = 35.0 A`, far outside every
cutoff envelope, so the padding contributes exactly zero energy and — since
`pos[0] - pos[0]` cancels — exactly zero force.  35.0 A is asserted directly on
the norm of the padding rows.

Section 4 — rebuild under a rigid translation.  Shifting every atom by
`(1.234, 0, 0)` A preserves every pairwise displacement exactly, so the periodic
neighbour set is unchanged: `num_edges` back to 162, buffer shapes untouched
(the whole point of fixed capacity — the force path stays CUDA-graph
capturable), and `rebuild_count == 1` after exactly one `rebuild` call.

Drift policy
------------
A disagreement between this file and the runtime is a **DEFECT REPORT, never a
golden edit**.  Every literal here is derived analytically from the lattice above
and from the two documented constants (`capacity_factor=1.35`,
`DEAD_EDGE_CUTOFF_FACTOR=10.0`); none was captured from a run.  If `num_edges`
comes back as anything but 162, the neighbour path lost or gained pairs, and the
correct response is to open a defect against `src/molix/md/neighbors.py` — not to
adjust the number below to whatever the run printed.

Goldens
-------
    capture command : PYTHONPATH=src python regressions/md-neighborlist-skin-03-rename.py
                      Nothing was captured from a run.  Every literal is
                      arithmetic on the page: 27*6 = 162, ceil(1.35*162) = 219,
                      10.0*3.5 = 35.0, sqrt(2)*3 = 4.243 > 3.5.
    commit          : d7b0ea2 (d7b0ea2bceb2740fc14b21dd5901a72bbb2c0228), with
                      the `md-neighborlist-skin-03-rename` working tree on top
                      (at d7b0ea2 itself Section 1 fails by construction — the
                      class is still `PeriodicNeighborList` there; that is the
                      RED this file was written against).
    torch           : 2.12.1+cpu   (python 3.14.5)
    date            : 2026-08-09
    device / dtype  : CPU, float64 — the cell and the positions are float64
                      literals, so the neighbour list's `shifts` buffer is
                      float64 too.  `edge_index` is int64 regardless.
    oracle          : none.  No third-party package (torch is the repo's own
                      core dependency), no network, no subprocess, no RNG, no
                      wall-clock value, no filesystem access.
    tolerance       : 1e-12 A on the dead-edge shift norm (float64 "position"
                      exact band; 35.0 = 10.0 * 3.5 is exact in binary and the
                      observed deviation is 0.0).  Exact integer equality on
                      edge counts, capacity, shapes, rebuild count and every
                      surface assertion.

Run:
    PYTHONPATH=src python regressions/md-neighborlist-skin-03-rename.py
"""

from __future__ import annotations

import itertools
import sys

import torch

import molix.md
from molix.md import NeighborList

# ---------------------------------------------------------------------------
# Section 1 goldens — the renamed public surface.
# ---------------------------------------------------------------------------

#: The name that must have disappeared entirely: no attribute, no `__all__`
#: entry, no alias.  `stage: experimental` buys the hard rename.
RETIRED_NAME = "PeriodicNeighborList"

#: The name that must resolve on `molix.md` after the rename.
CURRENT_NAME = "NeighborList"

# ---------------------------------------------------------------------------
# Section 2 goldens — the 3x3x3 simple-cubic lattice.
#
# Three lattice points per axis at 0 / 3 / 6 A in a 9 A cube: the spacing is
# uniform under periodicity (6 -> 9 == 0), so every site is equivalent and the
# edge count is exactly 27 * (number of in-cutoff neighbours per site).
# ---------------------------------------------------------------------------

#: Lattice spacing in Angstrom.
SPACING = 3.0

#: Cubic cell edge in Angstrom — three spacings, so the lattice tiles exactly.
BOX = 9.0

#: Between the nearest-neighbour separation (3.0 A, in) and the face diagonal
#: (sqrt2 * 3.0 = 4.243 A, out), and at most half the 9.0 A perpendicular width
#: (4.5 A), so minimum image is complete.
CUTOFF = 3.5

CELL = torch.tensor(
    [
        [BOX, 0.0, 0.0],
        [0.0, BOX, 0.0],
        [0.0, 0.0, BOX],
    ],
    dtype=torch.float64,
)

#: 27 sites x 6 axis neighbours each, doubled-counted as directed edges by the
#: full-bidirectional convention (`symmetry=True`, the MD list's fixed setting).
EXPECTED_NUM_EDGES = 162

#: ceil(1.35 * 162) = ceil(218.7), with the constructor's default
#: `capacity_factor=1.35`.
EXPECTED_CAPACITY = 219

EXPECTED_EDGE_INDEX_SHAPE = (EXPECTED_CAPACITY, 2)
EXPECTED_SHIFTS_SHAPE = (EXPECTED_CAPACITY, 3)

# ---------------------------------------------------------------------------
# Section 3 goldens — the inert padding tail.
# ---------------------------------------------------------------------------

#: `DEAD_EDGE_CUTOFF_FACTOR (10.0) * CUTOFF (3.5)`, written out rather than
#: imported: this file pins the *number* a dead edge carries, so importing the
#: constant would make the assertion tautological.
EXPECTED_DEAD_SHIFT_NORM = 35.0

#: Both endpoints of a dead edge are atom 0, so `pos[0] - pos[0]` cancels and no
#: spurious force reaches it; the *set* of indices in the padding rows is
#: therefore the single value 0.
EXPECTED_DEAD_ENDPOINTS = (0,)

#: float64 "position" exact band; 35.0 is representable exactly.
NORM_TOL = 1e-12

# ---------------------------------------------------------------------------
# Section 4 goldens — rigid translation.
# ---------------------------------------------------------------------------

#: An arbitrary non-lattice offset: rigid, so every pairwise displacement (and
#: hence every minimum image) is preserved exactly, but not a symmetry of the
#: lattice, so a rebuild that silently reused stale state would still be caught.
TRANSLATION = torch.tensor([1.234, 0.0, 0.0], dtype=torch.float64)

#: One `rebuild` call; construction itself does not count as a rebuild.
EXPECTED_REBUILD_COUNT = 1


def simple_cubic_lattice() -> torch.Tensor:
    """Build the 3x3x3 simple-cubic lattice used by every section.

    Returns:
        Positions ``(27, 3)`` in Angstrom, float64: the Cartesian product of
        ``{0.0, 3.0, 6.0}`` with itself three times, in lexicographic order.
    """
    coordinates = (0.0 * SPACING, 1.0 * SPACING, 2.0 * SPACING)
    return torch.tensor(
        list(itertools.product(coordinates, repeat=3)),
        dtype=torch.float64,
    )


# ---------------------------------------------------------------------------
# Checking
# ---------------------------------------------------------------------------


class Checker:
    """Collects every deviation so one run reports all failures, not the first."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def _row(self, name: str, got: object, ok: bool) -> None:
        print(f"  {name:<42} {got!s:<24} {'ok' if ok else 'FAILED'}")

    def exact(self, name: str, got: object, want: object) -> None:
        """Assert an integer, a shape or a name — no tolerance applies."""
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
        """Assert a boolean contract (a name is gone, a shape held, ...)."""
        if not holds:
            self.failures.append(f"{name}: {message}")
        self._row(name, holds, holds)


def check_renamed_surface(checker: Checker) -> None:
    """Section 1 — the new name resolves and the old one is gone without a shim.

    Args:
        checker: Failure collector.
    """
    print("Renamed surface (molix.md.NeighborList)")

    checker.truth(
        "surface.neighborlist_attribute",
        hasattr(molix.md, CURRENT_NAME),
        "`molix.md.NeighborList` does not resolve; the MD buffer owner must be "
        "reachable under the bare name freed by link 02",
    )
    checker.truth(
        "surface.neighborlist_export",
        CURRENT_NAME in molix.md.__all__,
        "`NeighborList` is missing from `molix.md.__all__`; the attribute alone "
        "is not the declared surface",
    )
    checker.truth(
        "surface.no_periodic_attribute",
        not hasattr(molix.md, RETIRED_NAME),
        "`molix.md.PeriodicNeighborList` still resolves — a back-compat alias, "
        "a module `__getattr__` fallback or a stale re-export survived; the "
        "spec forbids all three at `stage: experimental`",
    )
    checker.truth(
        "surface.no_periodic_export",
        RETIRED_NAME not in molix.md.__all__,
        "`PeriodicNeighborList` is still in `molix.md.__all__`; "
        "`from molix.md import *` would resurrect the retired name",
    )
    checker.exact("surface.class_name", NeighborList.__name__, CURRENT_NAME)


def check_lattice_buffers(checker: Checker, neighbor_list: NeighborList) -> None:
    """Section 2 — edge count, capacity and buffer shapes on the lattice.

    Args:
        checker: Failure collector.
        neighbor_list: List built on the 3x3x3 lattice at ``cutoff=3.5 A``.
    """
    print(f"\nLattice buffers (3x3x3 sc, {SPACING} A spacing, {BOX} A cell, cutoff {CUTOFF} A)")

    checker.exact("lattice.num_edges", neighbor_list.num_edges, EXPECTED_NUM_EDGES)
    checker.exact("lattice.capacity", neighbor_list.capacity, EXPECTED_CAPACITY)
    checker.exact(
        "lattice.edge_index_shape",
        tuple(neighbor_list.edge_index.shape),
        EXPECTED_EDGE_INDEX_SHAPE,
    )
    checker.exact(
        "lattice.shifts_shape",
        tuple(neighbor_list.shifts.shape),
        EXPECTED_SHIFTS_SHAPE,
    )


def check_dead_edge_tail(checker: Checker, neighbor_list: NeighborList) -> None:
    """Section 3 — the padding rows carry an inert 35.0 A self-loop.

    Args:
        checker: Failure collector.
        neighbor_list: List built on the 3x3x3 lattice at ``cutoff=3.5 A``.
    """
    print(f"\nDead-edge tail (rows [{EXPECTED_NUM_EDGES}, {EXPECTED_CAPACITY}))")

    dead_shifts = neighbor_list.shifts[neighbor_list.num_edges :]
    dead_edges = neighbor_list.edge_index[neighbor_list.num_edges :]

    checker.exact(
        "dead.row_count",
        int(dead_shifts.shape[0]),
        EXPECTED_CAPACITY - EXPECTED_NUM_EDGES,
    )

    norms = torch.linalg.norm(dead_shifts, dim=-1)
    deviation = (
        float((norms - EXPECTED_DEAD_SHIFT_NORM).abs().max()) if norms.numel() else float("inf")
    )
    checker.within("dead.shift_norm_deviation", deviation, NORM_TOL)
    checker.truth(
        "dead.shift_norm_is_35A",
        deviation <= NORM_TOL,
        f"dead-edge shift norms {norms.unique().tolist()} A are not "
        f"{EXPECTED_DEAD_SHIFT_NORM} A (= 10.0 x {CUTOFF} A); padding closer than "
        "the cutoff would leak a spurious pair contribution into the energy",
    )

    endpoints = tuple(int(value) for value in dead_edges.unique().tolist())
    checker.truth(
        "dead.endpoints_are_atom_zero",
        endpoints == EXPECTED_DEAD_ENDPOINTS,
        f"dead-edge endpoints {endpoints} are not all atom 0; a dead edge must "
        "be a self-loop so `pos[0] - pos[0]` cancels and no force reaches it",
    )


def check_rigid_translation(checker: Checker, neighbor_list: NeighborList) -> None:
    """Section 4 — a rigid shift leaves the neighbour set and the shapes alone.

    Args:
        checker: Failure collector.
        neighbor_list: List built on the 3x3x3 lattice; rebuilt in place here.
    """
    print(f"\nRigid translation ({TRANSLATION.tolist()} A)")

    neighbor_list.rebuild(simple_cubic_lattice() + TRANSLATION)

    checker.exact("translated.num_edges", neighbor_list.num_edges, EXPECTED_NUM_EDGES)
    checker.exact("translated.capacity", neighbor_list.capacity, EXPECTED_CAPACITY)
    checker.exact(
        "translated.edge_index_shape",
        tuple(neighbor_list.edge_index.shape),
        EXPECTED_EDGE_INDEX_SHAPE,
    )
    checker.exact(
        "translated.shifts_shape",
        tuple(neighbor_list.shifts.shape),
        EXPECTED_SHIFTS_SHAPE,
    )
    checker.exact("translated.rebuild_count", neighbor_list.rebuild_count, EXPECTED_REBUILD_COUNT)


def main() -> int:
    """Pin the renamed surface and the buffer numbers the rename must not move."""
    checker = Checker()

    check_renamed_surface(checker)

    neighbor_list = NeighborList(
        cell=CELL,
        cutoff=CUTOFF,
        positions=simple_cubic_lattice(),
    )
    check_lattice_buffers(checker, neighbor_list)
    check_dead_edge_tail(checker, neighbor_list)
    check_rigid_translation(checker, neighbor_list)

    if checker.failures:
        print("\nFAILED — the rename was not behaviour-neutral, or the old name is back:")
        for failure in checker.failures:
            print(f"  {failure}")
        print(
            "\nEvery golden above is analytic (27*6 = 162, ceil(1.35*162) = 219, "
            "10.0*3.5 = 35.0). A disagreement is a DEFECT REPORT against the "
            "neighbour path, never a reason to edit the literal."
        )
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
