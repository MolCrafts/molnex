"""Public-API scenario for the neighbour-graph homonym prune.

Spec: `md-neighborlist-skin-02-prune`.

Two dead neighbour-graph implementations are removed from the tree so that the
name `NeighborList` is unambiguous:

    molix.nn.locality.NeighborList   an nn.Module wrapper over
                                     molix.F.locality.get_neighbor_pairs;
                                     zero call sites
    molpot.graph.radius_graph        the same kernel plus a cross-molecule
                                     mask, with the pre-Edge-Convention sign
                                     (pos_j - pos_i, unnegated); zero call
                                     sites, no __init__.py, never a declared
                                     surface

Stage is `experimental`, so this is a **hard removal** — no alias, no
DeprecationWarning, no shim.  This file is the public-API form of the two
claims that makes such a removal safe.

Section 1 — the deleted surface stays deleted.  `import molix.nn` and
`import molpot` still succeed (the prune is import-time-only: nothing in the
training loop, the MD driver or any encoder resolved either symbol), while
`molix.nn.NeighborList`, the `molix.nn.locality` module, the `molpot.graph`
package, the `molpot.graph.radius` module and a `molpot.radius_graph`
attribute are all absent.  `find_spec("molpot.graph")` is checked rather than
an import: `src/molpot/graph/` has no `__init__.py`, so a *surviving directory*
— a leftover `__pycache__`, say — is still a live namespace package and makes
the spec non-None even though nothing imports.  That leftover is precisely the
drift being caught.

Section 2 — the capability was never lost, only the duplicate wrapper.  The
surviving `molix.data.tasks.NeighborList` (module home
`molix.data.tasks.neighbor`) is run on a hard-coded 3-atom chain and must still
produce the documented bidirectional edge list.

The goldens are arithmetic on the page, not a captured measurement.  For atoms
at x = 0.0, 1.0, 2.0 A on a line and `cutoff=1.5 A`:

    d(0,1) = 1.0 A  <= 1.5   in
    d(1,2) = 1.0 A  <= 1.5   in
    d(0,2) = 2.0 A  >  1.5   out

so two half pairs survive, and `symmetry=True` (the default, the full
bidirectional list every aggregating model assumes) doubles them:

    E = 2 x 2 = 4  ->  edge_index (4, 2), edge_dist == 1.0 in all four entries

Only the shape and the distances are pinned.  Row order and the source/target
orientation within a row are **not** asserted: both are kernel-internal and the
Edge Convention (`edge_index[:,0]` = source, `edge_diff = pos[target] -
pos[source]`) is already pinned by the unit suite; re-asserting an ordering
here would make this file fail on a legal kernel change.

Goldens
-------
    capture command : PYTHONPATH=src python regressions/md-neighborlist-skin-02-prune.py
                      Nothing was captured from a run.  Every literal below is
                      derived on the page from the chain geometry above: the
                      four names of `molix.nn.__all__`, E = 4, d = 1.0 A.
    commit          : 0111076 (0111076aba240e3e3b7b326f33341bf601fcd6ed), with
                      the `md-neighborlist-skin-02-prune` working tree on top
                      (at 0111076 itself Section 1 fails by construction — the
                      two modules are still present; that is the RED this file
                      was written against).
    torch           : 2.12.1+cpu   (python 3.14.5)
    date            : 2026-08-09
    device / dtype  : CPU, float64 — the positions are float64 literals, so the
                      neighbour list's distances come back float64 too.
    oracle          : none.  No third-party package (torch is the repo's own
                      core dependency), no network, no subprocess, no RNG, no
                      wall-clock value, no filesystem access.
    tolerance       : 1e-6 A on the distances (spec-mandated; the float64
                      "position" band is 1e-8 and the observed deviation is
                      0.0, so six decades of slack are unused).  Exact equality
                      on every import-surface assertion and on the edge shape.

Run:
    PYTHONPATH=src python regressions/md-neighborlist-skin-02-prune.py
"""

from __future__ import annotations

import importlib
import importlib.util
import sys

import torch

import molix.nn
import molpot
from molix.data.tasks import NeighborList

# ---------------------------------------------------------------------------
# Section 1 goldens — the surviving public surface of `molix.nn`.
#
# Hard-coded as a list, not a set: the order pins the alphabetization rule
# (`.claude/notes/notes.md:243`) that the deletion's rewrite has to honour.
# ---------------------------------------------------------------------------

SURVIVING_NN_EXPORTS = ["BatchAggregation", "KeyedMLP", "KeyedMLPSpec", "ScatterSum"]

#: Fully-qualified names that must no longer resolve to anything importable.
DELETED_MODULES = ("molix.nn.locality", "molpot.graph")

# ---------------------------------------------------------------------------
# Section 2 goldens — the 3-atom chain.
#
# Three atoms on the x axis one Angstrom apart; y = z = 0 throughout, so every
# separation is a difference of the x column and can be read off by eye.
# ---------------------------------------------------------------------------

POSITIONS = torch.tensor(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ],
    dtype=torch.float64,
)

#: Between the nearest-neighbour separation (1.0 A, in) and the end-to-end one
#: (2.0 A, out), with a 0.5 A margin either side — the pair *set* cannot flip
#: on arithmetic noise.
CUTOFF = 1.5

#: Two half pairs — (0,1) and (1,2) — each expanded to a forward and a reverse
#: edge by `symmetry=True`.
EXPECTED_EDGE_SHAPE = (4, 2)

#: Every surviving pair is a nearest-neighbour pair, so all four rows carry the
#: same distance.
EXPECTED_EDGE_DIST = 1.0

#: Spec-mandated distance tolerance in Angstrom.
DIST_TOL = 1e-6


# ---------------------------------------------------------------------------
# Checking
# ---------------------------------------------------------------------------


class Checker:
    """Collects every deviation so one run reports all failures, not the first."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def _row(self, name: str, got: object, ok: bool) -> None:
        print(f"  {name:<42} {got!s:<34} {'ok' if ok else 'FAILED'}")

    def exact(self, name: str, got: object, want: object) -> None:
        """Assert a name list, a shape or a flag — no tolerance applies."""
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
        """Assert a boolean contract (a module is gone, an import raised, ...)."""
        if not holds:
            self.failures.append(f"{name}: {message}")
        self._row(name, holds, holds)


def import_error_of(module: str) -> str | None:
    """Import *module* and report the ``ModuleNotFoundError``, if any.

    Args:
        module: Fully-qualified module name, e.g. ``"molpot.graph.radius"``.

    Returns:
        The exception message if the import raised :class:`ModuleNotFoundError`,
        else ``None`` — the module is still importable, which is the drift.
    """
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as error:
        return str(error)
    return None


def check_deleted_surface(checker: Checker) -> None:
    """Section 1 — neither dead neighbour-graph symbol resolves any more.

    Args:
        checker: Failure collector.
    """
    print("Deleted surface (molix.nn.locality.NeighborList, molpot.graph.radius_graph)")

    # The prune is import-time-only; the two packages must still come up.
    checker.truth(
        "surface.molix_nn_imports",
        molix.nn is not None,
        "`import molix.nn` did not yield a module — the deletion was supposed "
        "to drop one re-export line, not break the package",
    )
    checker.truth(
        "surface.molpot_imports",
        molpot is not None,
        "`import molpot` did not yield a module — `molpot.graph` was never a "
        "declared surface, so removing it must be invisible here",
    )

    checker.exact("surface.molix_nn_all", list(molix.nn.__all__), SURVIVING_NN_EXPORTS)
    checker.truth(
        "surface.no_neighborlist_attribute",
        not hasattr(molix.nn, "NeighborList"),
        "`molix.nn.NeighborList` still resolves; the name must stay free for "
        "molix.data.tasks.neighbor.NeighborList / molix.md.NeighborList",
    )
    checker.truth(
        "surface.no_neighborlist_export",
        "NeighborList" not in molix.nn.__all__,
        "`NeighborList` is back in `molix.nn.__all__`",
    )

    for module in DELETED_MODULES:
        # `find_spec`, not `import`: `src/molpot/graph/` has no `__init__.py`,
        # so a surviving directory (e.g. a stale `__pycache__`) is a live
        # namespace package with a non-None spec and no import error.
        spec = importlib.util.find_spec(module)
        checker.truth(
            f"surface.{module}_has_no_spec",
            spec is None,
            f"`{module}` still resolves to {spec!r} — the module file or its "
            "directory (stale `__pycache__` included) survived the deletion",
        )

    checker.truth(
        "surface.molpot_graph_radius_unimportable",
        import_error_of("molpot.graph.radius") is not None,
        "`import molpot.graph.radius` succeeded; the pre-Edge-Convention "
        "radius_graph builder is still in the tree",
    )
    checker.truth(
        "surface.molpot_has_no_radius_graph",
        not hasattr(molpot, "radius_graph"),
        "`molpot.radius_graph` resolves; the deleted free function was never "
        "exported at package level and must not appear now",
    )


def check_surviving_neighbor_list(checker: Checker) -> None:
    """Section 2 — the live pipeline neighbour list still builds the chain graph.

    Args:
        checker: Failure collector.
    """
    print(f"\nSurviving neighbour list (3-atom chain at x = 0, 1, 2 A; cutoff = {CUTOFF} A)")

    # `molix.data.tasks.NeighborList` is the package re-export of
    # `molix.data.tasks.neighbor.NeighborList` — the same class object.
    task = NeighborList(cutoff=CUTOFF, symmetry=True)
    sample = task.execute({"pos": POSITIONS})

    edge_index = sample["edge_index"]
    edge_dist = sample["edge_dist"]

    checker.exact("chain.edge_index_shape", tuple(edge_index.shape), EXPECTED_EDGE_SHAPE)
    checker.truth(
        "chain.edge_dist_all_one",
        bool(
            torch.allclose(
                edge_dist,
                torch.full_like(edge_dist, EXPECTED_EDGE_DIST),
                atol=DIST_TOL,
                rtol=0.0,
            )
        ),
        f"edge distances {edge_dist.tolist()} are not all {EXPECTED_EDGE_DIST} A — "
        "within a 1.5 A cutoff only the two nearest-neighbour pairs survive, and "
        "both are exactly 1.0 A",
    )
    deviation = (
        float((edge_dist - EXPECTED_EDGE_DIST).abs().max()) if edge_dist.numel() else float("inf")
    )
    checker.within("chain.max_distance_deviation", deviation, DIST_TOL)


def main() -> int:
    """Pin the two deletions and the surviving neighbour list's behaviour."""
    checker = Checker()
    check_deleted_surface(checker)
    check_surviving_neighbor_list(checker)

    if checker.failures:
        print("\nFAILED — the pruned neighbour-graph surface is back, or the survivor drifted:")
        for failure in checker.failures:
            print(f"  {failure}")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
