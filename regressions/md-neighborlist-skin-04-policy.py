"""Public-API scenario for the Verlet skin + `neigh_modify` rebuild policy.

Spec: `md-neighborlist-skin-04-policy`.

`molix.md.NeighborList` now builds at `r_build = cutoff + skin` and decides once
per force evaluation, in `update(positions)`, whether the frozen list is still
valid — under the LAMMPS `neigh_modify every/delay/check` gate.  The whole value
of that machinery is *when it rebuilds*: too eager and the skin bought nothing,
too lazy and pairs are silently missed (an `O(1)`, one-signed energy injection
per missed event, not an `O(dt^2)` discretisation artefact).  A unit test can
pin one gate arm at a time; this file pins the **decision sequence** a user
actually observes, end to end, over three scripted schedules whose every literal
is arithmetic on this page.

The shared configuration is a 64-atom simple-cubic lattice (4x4x4 sites, 3.0 A
spacing) in a 12.0 A cube, CPU float64.  Nothing moves by itself: displacements
are applied by hand, so no integrator, no thermostat and no RNG stand between
the schedule and the counters.

Section 1 — the crystallographic goldens (what the skin costs)
--------------------------------------------------------------
Simple cubic at `a = 3.0 A` has exact neighbour shells:

    6 neighbours at        a = 3.000 A
    12 neighbours at  sqrt2*a = 4.243 A
    8 neighbours at   sqrt3*a = 5.196 A
    6 neighbours at       2*a = 6.000 A

With `cutoff = 3.5 A, skin = 1.5 A`:

    r_build = 3.5 + 1.5 = 5.0 A
    in-build shells: 6 + 12 = 18   (5.196 A is out; 4.243 A is in)
    num_edges = 64 * 18 = 1152     (full bidirectional list, `symmetry=True`)
    capacity  = ceil(1.35 * 1152) = ceil(1555.2) = 1556

versus the bare cutoff `skin = 0.0`:

    r_build = 3.5 A;   only the 6-shell is in
    num_edges = 64 * 6 = 384
    capacity  = ceil(1.35 * 384) = ceil(518.4) = 519

1152/384 = 3.0 is the `(1 + skin/r_cut)^3 = (1 + 1.5/3.5)^3 = 2.91` growth law
measured directly (the lattice is discrete, so the ratio lands on the next shell
rather than exactly on the continuum estimate).  Four sites per axis means the
`+a` and `-a` neighbours are distinct atoms, so the shell multiplicities above
are not double-counted images.  The `2*a = 6.0 A` shell is exactly the
minimum-image half-width and would be ambiguous — it sits well outside
`r_build = 5.0 A`, so it never enters.  `r_build = 5.0 A <= 6.0 A = 12.0/2`
also clears the constructor's half-perpendicular-width guard.

Section 2 — gating arithmetic (`every=2, delay=4, check=True`)
--------------------------------------------------------------
Atom 0 is translated by `+0.1 A` along x per update; every other atom is frozen,
so the largest displacement since the last build is exactly `0.1 * ago` A.
Half-skin is `skin/2 = 0.75 A`.  The gate is conjunctive — a rebuild is
*permitted* only at `ago >= delay` **and** `ago % every == 0`:

    ago:          1  2  3  4  5  6  7  8
    >= delay 4:   .  .  .  x  x  x  x  x
    % every 2:    .  x  .  x  .  x  .  x
    permitted:    .  .  .  x  .  x  .  x
    displacement:          0.4   0.6   0.8   A
    vs 0.75 A:             no    no    YES

so the first rebuild lands at the 8th update since the build, not the 4th
(displacement too small) and not the 7th (`ago` odd).  `rebuild` resets `ago` to
0, so the pattern repeats verbatim: over 40 updates `update()` returns True at
exactly `{8, 16, 24, 32, 40}` — `rebuild_count == 5`.  Every one of those fired
at `ago == 8`, never at the first permitted opportunity `ago == max(every,
delay) == 4`, so `ndanger == 0`: no rebuild was ever overdue.

This one schedule falsifies all three ways the gate can be wrong.  A disjunctive
gate (`or` instead of `and`) would fire at update 6.  A gate that ignored the
displacement check would fire at update 4.  A `>=` instead of the strict `>` on
the half-skin would not change these numbers (0.4/0.6/0.8 never equal 0.75) —
which is why the strict comparison is pinned in the unit tests instead, and why
this file pins the *schedule*.

Section 3 — the `ndanger` alarm (`every=1, delay=0, check=True`)
----------------------------------------------------------------
At the default gate `max(every, delay) == 1`, so *every* rebuild lands on the
first permitted opportunity and every rebuild is counted dangerous.  Atom 0 is
jumped `1.0 A` along x and back, alternately, so the displacement since the last
build is `1.0 A > 0.75 A` on every single update: 5 updates give
`rebuild_count == 5` and `ndanger == 5`.

That is the alarm behaving as designed, not a bug: `ndanger` says "a rebuild
fired at the earliest moment the gate allowed, so it may already have been
overdue on a step the gate skipped".  With `every=1` no step is skipped, so the
counter carries no information there — pinned here precisely so the documented
degenerate reading stays visible and nobody "fixes" it into silence.

Section 4 — cadence only (`every=5, delay=0, check=False`)
----------------------------------------------------------
Nothing moves at all.  With `check=False` the displacement criterion — and with
it the unwrapped-positions guard — is skipped entirely, so the list rebuilds on
raw cadence: updates `{5, 10, 15, 20}` out of 20, `rebuild_count == 4`, and
`ndanger == 0` because `ndanger` is only ever incremented inside the
displacement branch.  Rebuilding four times on a configuration that never
changed is the visible price of `check=False`; under `check=True` the same run
would rebuild zero times.

Drift policy
------------
A disagreement between this file and the runtime is a **DEFECT REPORT, never a
golden edit**.  Every literal below is derived on this page from the lattice
geometry, the two documented constants (`capacity_factor = 1.35`, shell
multiplicities of the simple-cubic lattice) and the LAMMPS gate as specified;
none was read off a run and then written down.  If the fired-update sequence
comes back as anything but `{8, 16, 24, 32, 40}`, the gate changed semantics —
open a defect against `src/molix/md/neighbors.py` and fix the gate, do not
retune the tuple below to whatever the run printed.

Goldens
-------
    capture command : PYTHONPATH=src python regressions/md-neighborlist-skin-04-policy.py
                      Nothing was captured from a run.  Every literal is
                      arithmetic on the page: 64*18 = 1152, 64*6 = 384,
                      ceil(1.35*1152) = 1556, ceil(1.35*384) = 519,
                      3.5 + 1.5 = 5.0, and the three gate tables above.
    commit          : 58fb180 (58fb180faab256bd03083782fcda8e3af9c4a883), with
                      the `md-neighborlist-skin-04-policy` working tree on top
                      (at 58fb180 itself `NeighborList` has no `skin` / `update`
                      at all, so construction raises `TypeError` — that is the
                      RED this file was written against).
    torch           : 2.12.1+cpu   (python 3.14.5)
    date            : 2026-08-09
    device / dtype  : CPU, float64 — the cell and the positions are float64
                      literals, so `shifts` and `_x_hold` are float64 too.
                      `edge_index` is int64 regardless.
    oracle          : none.  No third-party package (torch is the repo's own
                      core dependency), no network, no subprocess, no RNG, no
                      wall-clock value, no filesystem access.  LAMMPS is the
                      *specified* behaviour, not a runtime dependency: its gate
                      is transcribed into the tables above, never executed.
    tolerance       : 1e-12 A on `r_build` (float64 "position" exact band; both
                      3.5 and 1.5 are exactly representable, so 5.0 is exact and
                      the observed deviation is 0.0).  Exact integer equality on
                      every edge count, capacity, rebuild count, `ago`,
                      `ndanger` and fired-update sequence — they are counters,
                      and a counter has no tolerance.

Run:
    PYTHONPATH=src python regressions/md-neighborlist-skin-04-policy.py
"""

from __future__ import annotations

import itertools
import sys

import torch

from molix.md import NeighborList

# ---------------------------------------------------------------------------
# The shared lattice.
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
#: shell (in) and the 5.196 A third shell (out) — again a hard integer boundary,
#: not a tolerance question — and half-skin is a round 0.75 A.
SKIN = 1.5

CELL = torch.tensor(
    [
        [BOX, 0.0, 0.0],
        [0.0, BOX, 0.0],
        [0.0, 0.0, BOX],
    ],
    dtype=torch.float64,
)

# ---------------------------------------------------------------------------
# Section 1 goldens — what the skin costs.
# ---------------------------------------------------------------------------

#: `CUTOFF + SKIN`, written out rather than recomputed from the two constants:
#: this file pins the derived radius, so deriving it here would be tautological.
EXPECTED_R_BUILD = 5.0

#: 64 sites x (6 at 3.0 A + 12 at 4.243 A) directed edges, both directions kept
#: (`symmetry=True`, the MD list's fixed setting).
EXPECTED_EDGES_AT_R_BUILD = 1152

#: 64 sites x 6 at 3.0 A — the same lattice with `skin = 0.0`.
EXPECTED_EDGES_AT_CUTOFF = 384

#: ceil(1.35 * 1152) = ceil(1555.2), with the default `capacity_factor = 1.35`.
EXPECTED_CAPACITY_AT_R_BUILD = 1556

#: ceil(1.35 * 384) = ceil(518.4).
EXPECTED_CAPACITY_AT_CUTOFF = 519

#: float64 "position" exact band; 3.5 + 1.5 = 5.0 is exact in binary.
LENGTH_TOL = 1e-12

# ---------------------------------------------------------------------------
# Section 2 goldens — `every=2, delay=4, check=True`.
# ---------------------------------------------------------------------------

#: Per-update translation of atom 0 along x, in Angstrom.
STEP_DISPLACEMENT = 0.1

#: Updates driven in scenario 1.
SCENARIO_1_UPDATES = 40

SCENARIO_1_EVERY = 2
SCENARIO_1_DELAY = 4

#: Permitted `ago` values are {4, 6, 8, ...}; the displacement `0.1 * ago` first
#: exceeds half-skin 0.75 A at `ago == 8`, and `rebuild` re-phases `ago` to 0.
SCENARIO_1_FIRED_UPDATES = (8, 16, 24, 32, 40)

SCENARIO_1_REBUILD_COUNT = 5

#: Every rebuild fired at `ago == 8`, never at the first permitted opportunity
#: `max(every, delay) == 4`, so no rebuild was ever overdue.
SCENARIO_1_NDANGER = 0

#: The 40th update *is* a rebuild, so the clock is back at 0 when the loop ends.
SCENARIO_1_FINAL_AGO = 0

# ---------------------------------------------------------------------------
# Section 3 goldens — `every=1, delay=0, check=True`.
# ---------------------------------------------------------------------------

#: Jump of atom 0 along x, in Angstrom; alternated with 0.0 so the displacement
#: *since the last build* is 1.0 A on every update, not a growing drift.
JUMP_DISPLACEMENT = 1.0

SCENARIO_2_UPDATES = 5

#: 1.0 A > half-skin 0.75 A every time, and `every=1` permits every step.
SCENARIO_2_FIRED_UPDATES = (1, 2, 3, 4, 5)

SCENARIO_2_REBUILD_COUNT = 5

#: `max(every, delay) == max(1, 0) == 1 == ago` at every rebuild: the documented
#: degenerate reading of the alarm, where it merely counts rebuilds.
SCENARIO_2_NDANGER = 5

# ---------------------------------------------------------------------------
# Section 4 goldens — `every=5, delay=0, check=False`, nothing moves.
# ---------------------------------------------------------------------------

SCENARIO_3_UPDATES = 20
SCENARIO_3_EVERY = 5

#: Pure cadence: `ago % 5 == 0` and `check=False` short-circuits before the
#: displacement branch, so a motionless system still rebuilds four times.
SCENARIO_3_FIRED_UPDATES = (5, 10, 15, 20)

SCENARIO_3_REBUILD_COUNT = 4

#: `ndanger` is only ever incremented inside the displacement branch, which
#: `check=False` skips.
SCENARIO_3_NDANGER = 0


def simple_cubic_lattice() -> torch.Tensor:
    """Build the 4x4x4 simple-cubic lattice every section runs on.

    Returns:
        Positions ``(64, 3)`` in Angstrom, float64: the Cartesian product of
        ``{0.0, 3.0, 6.0, 9.0}`` with itself three times, lexicographic order.
    """
    coordinates = tuple(index * SPACING for index in range(N_SIDE))
    return torch.tensor(
        list(itertools.product(coordinates, repeat=3)),
        dtype=torch.float64,
    )


def displaced(base: torch.Tensor, offset: float) -> torch.Tensor:
    """Return *base* with atom 0 translated by *offset* Angstrom along x.

    Args:
        base: Reference positions ``(N, 3)`` in Angstrom.
        offset: Translation of atom 0 along x, in Angstrom.

    Returns:
        A fresh ``(N, 3)`` tensor; *base* is never mutated, so each schedule is
        an absolute displacement from the lattice rather than an accumulation
        of rounding.
    """
    positions = base.clone()
    positions[0, 0] += offset
    return positions


# ---------------------------------------------------------------------------
# Checking
# ---------------------------------------------------------------------------


class Checker:
    """Collects every deviation so one run reports all failures, not the first."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def _row(self, name: str, got: object, ok: bool) -> None:
        print(f"  {name:<44} {got!s:<26} {'ok' if ok else 'FAILED'}")

    def exact(self, name: str, got: object, want: object) -> None:
        """Assert a counter, a shape or an update sequence — no tolerance applies."""
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


def check_skin_geometry(checker: Checker) -> None:
    """Section 1 — `r_build`, the edge counts it implies, and the fresh counters.

    Args:
        checker: Failure collector.
    """
    print(f"Skin geometry ({N_SIDE}x{N_SIDE}x{N_SIDE} sc, {SPACING} A spacing, {BOX} A cell)")

    lattice = simple_cubic_lattice()
    skinned = NeighborList(cell=CELL, cutoff=CUTOFF, positions=lattice, skin=SKIN)
    bare = NeighborList(cell=CELL, cutoff=CUTOFF, positions=lattice, skin=0.0)

    checker.within("skin.r_build_deviation", abs(skinned.r_build - EXPECTED_R_BUILD), LENGTH_TOL)
    checker.exact("skin.cutoff_unchanged", skinned.cutoff, CUTOFF)
    checker.exact("skin.skin", skinned.skin, SKIN)
    checker.exact("skin.num_edges_at_r_build", skinned.num_edges, EXPECTED_EDGES_AT_R_BUILD)
    checker.exact("skin.capacity_at_r_build", skinned.capacity, EXPECTED_CAPACITY_AT_R_BUILD)

    checker.exact("bare.num_edges_at_cutoff", bare.num_edges, EXPECTED_EDGES_AT_CUTOFF)
    checker.exact("bare.capacity_at_cutoff", bare.capacity, EXPECTED_CAPACITY_AT_CUTOFF)

    checker.exact("fresh.ago", skinned.ago, 0)
    checker.exact("fresh.rebuild_count", skinned.rebuild_count, 0)
    checker.exact("fresh.ndanger", skinned.ndanger, 0)


def check_gate_schedule(checker: Checker) -> None:
    """Section 2 — the `every=2, delay=4, check=True` decision sequence.

    Args:
        checker: Failure collector.
    """
    print(
        f"\nGate schedule (every={SCENARIO_1_EVERY}, delay={SCENARIO_1_DELAY}, check=True, "
        f"{STEP_DISPLACEMENT} A/update on atom 0)"
    )

    lattice = simple_cubic_lattice()
    neighbor_list = NeighborList(
        cell=CELL,
        cutoff=CUTOFF,
        positions=lattice,
        skin=SKIN,
        every=SCENARIO_1_EVERY,
        delay=SCENARIO_1_DELAY,
        check=True,
    )

    fired: list[int] = []
    for update in range(1, SCENARIO_1_UPDATES + 1):
        if neighbor_list.update(displaced(lattice, STEP_DISPLACEMENT * update)):
            fired.append(update)

    checker.exact("gate.fired_updates", tuple(fired), SCENARIO_1_FIRED_UPDATES)
    checker.exact("gate.rebuild_count", neighbor_list.rebuild_count, SCENARIO_1_REBUILD_COUNT)
    checker.exact("gate.ndanger", neighbor_list.ndanger, SCENARIO_1_NDANGER)
    checker.exact("gate.final_ago", neighbor_list.ago, SCENARIO_1_FINAL_AGO)


def check_danger_alarm(checker: Checker) -> None:
    """Section 3 — `ndanger` at the default gate, where every rebuild is dangerous.

    Args:
        checker: Failure collector.
    """
    print(f"\nDanger alarm (every=1, delay=0, check=True, {JUMP_DISPLACEMENT} A jump per update)")

    lattice = simple_cubic_lattice()
    neighbor_list = NeighborList(
        cell=CELL,
        cutoff=CUTOFF,
        positions=lattice,
        skin=SKIN,
        every=1,
        delay=0,
        check=True,
    )

    fired: list[int] = []
    for update in range(1, SCENARIO_2_UPDATES + 1):
        # Alternate there-and-back so the displacement *since the last build* is
        # 1.0 A every time, rather than a drift that would also pass a broken
        # gate reading absolute position.
        offset = JUMP_DISPLACEMENT if update % 2 else 0.0
        if neighbor_list.update(displaced(lattice, offset)):
            fired.append(update)

    checker.exact("danger.fired_updates", tuple(fired), SCENARIO_2_FIRED_UPDATES)
    checker.exact("danger.rebuild_count", neighbor_list.rebuild_count, SCENARIO_2_REBUILD_COUNT)
    checker.exact("danger.ndanger", neighbor_list.ndanger, SCENARIO_2_NDANGER)


def check_cadence_only(checker: Checker) -> None:
    """Section 4 — `check=False` rebuilds a motionless system on cadence alone.

    Args:
        checker: Failure collector.
    """
    print(f"\nCadence only (every={SCENARIO_3_EVERY}, delay=0, check=False, nothing moves)")

    lattice = simple_cubic_lattice()
    neighbor_list = NeighborList(
        cell=CELL,
        cutoff=CUTOFF,
        positions=lattice,
        skin=SKIN,
        every=SCENARIO_3_EVERY,
        delay=0,
        check=False,
    )

    fired: list[int] = []
    for update in range(1, SCENARIO_3_UPDATES + 1):
        if neighbor_list.update(lattice):
            fired.append(update)

    checker.exact("cadence.fired_updates", tuple(fired), SCENARIO_3_FIRED_UPDATES)
    checker.exact("cadence.rebuild_count", neighbor_list.rebuild_count, SCENARIO_3_REBUILD_COUNT)
    checker.exact("cadence.ndanger", neighbor_list.ndanger, SCENARIO_3_NDANGER)
    checker.exact("cadence.num_edges_unchanged", neighbor_list.num_edges, EXPECTED_EDGES_AT_R_BUILD)


def main() -> int:
    """Pin the rebuild-decision sequence the skin policy produces."""
    checker = Checker()

    check_skin_geometry(checker)
    check_gate_schedule(checker)
    check_danger_alarm(checker)
    check_cadence_only(checker)

    if checker.failures:
        print("\nFAILED — the rebuild policy no longer matches the LAMMPS gate:")
        for failure in checker.failures:
            print(f"  {failure}")
        print(
            "\nEvery golden above is analytic (64*18 = 1152, 64*6 = 384, "
            "ceil(1.35*1152) = 1556, and the three gate tables in the module "
            "docstring). A disagreement is a DEFECT REPORT against "
            "src/molix/md/neighbors.py, never a reason to edit the literal: a "
            "gate that rebuilds late misses pairs and leaks NVE energy, and a "
            "golden retuned to match it would hide exactly that."
        )
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
