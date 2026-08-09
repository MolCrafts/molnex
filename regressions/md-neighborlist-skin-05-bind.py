"""Public-API scenario for the TensorDict bind surface of `molix.md.NeighborList`.

Spec: `md-neighborlist-skin-05-bind`.

`nl.build(batch)` rebuilds the list at `batch["atoms", "pos"]` and writes
`batch["edges"]` as a `TensorDict` holding the live `edge_index` / `shifts`
buffers **by reference**; `nl.update(batch)` then drives the LAMMPS
`every/delay/check` policy from the same batch, dispatching on
`TensorDictBase` so the raw `(N, 3)` MD hot path and the batch path share one
door.  The load-bearing property is the *tie*: because every rebuild is in
place, a consumer holding that batch sees the current neighbour set with no
re-binding and no shape change — which is exactly what keeps a compiled or
CUDA-graph-captured force path valid across a rebuild.

A unit test can pin the identity `batch["edges", "edge_index"] is nl.edge_index`
at bind time.  What it cannot show is that the tie *stays alive across a driven
run* and that what a consumer reads through the batch is still physically
right afterwards.  That is this file: bind once, drive twenty steps through the
batch, then reconstruct the interatomic distances **out of the batch** and check
them against crystallography.  A binding that ever copied on assignment would
sail through the first three sections and fail the fourth with a frozen
neighbour set — which is the failure mode worth a regression file.

The configuration is the same 64-atom simple-cubic lattice link 04 uses (4x4x4
sites, 3.0 A spacing) in a 12.0 A cube, CPU float64,
`NeighborList(cutoff=3.5, skin=1.5, every=1, delay=0, check=True)`.  Nothing
moves by itself: positions are written by hand, so no integrator, no thermostat
and no RNG stand between the schedule and the counters.

Section 1 — the lattice goldens (inherited, re-pinned at the bind)
------------------------------------------------------------------
Simple cubic at `a = 3.0 A` has exact neighbour shells:

    6 neighbours at        a = 3.000 A
    12 neighbours at  sqrt2*a = 4.243 A
    8 neighbours at   sqrt3*a = 5.196 A

With `cutoff = 3.5 A, skin = 1.5 A`:

    r_build   = 3.5 + 1.5 = 5.0 A     (the 5.196 A shell is out, 4.243 A is in)
    num_edges = 64 * (6 + 12) = 1152  (full bidirectional list, `symmetry=True`)
    capacity  = ceil(1.35 * 1152) = ceil(1555.2) = 1556

`r_build = 5.0 A <= 6.0 A = 12.0/2` clears the constructor's
half-perpendicular-width guard.  Four sites per axis keeps the `+a` and `-a`
neighbours distinct atoms, so the multiplicities are not double-counted images.

Section 2 — what `build` binds
-------------------------------
`build` returns the *same* batch object (so `potential(nl.build(batch))`
composes with the repo's `forward(td) -> td` convention), and the two leaves it
writes are the list's own buffers, not copies:

    nl.build(batch) is batch
    batch["edges", "edge_index"] is nl.edge_index
    batch["edges", "shifts"]     is nl.shifts
    set(batch["edges"].keys())   == {"edge_index", "shifts"}
    batch["edges"].batch_size    == [1556]        (capacity, not num_edges)

`build` is a *binding* operation, not physics: `rebuild_count` stays 0 across it
(it also runs on every `.to()` re-sync, and counting it would make a dtype cast
look like a rebuild), while `ago` restarts at 0 because the buffers are fresh.

Section 3 — driving the policy through the batch
-------------------------------------------------
Every update translates **all 64 atoms** rigidly by `+0.2 A` along x, written in
place into the batch's `("atoms", "pos")` leaf — the shape an integrator
produces.  Half-skin is `skin/2 = 0.75 A`, and the criterion is strict `>`:

    ago:            1     2     3     4
    displacement:  0.2   0.4   0.6   0.8   A
    vs 0.75 A:      no    no    no   YES

`every=1, delay=0` permits every step, so the first rebuild lands at `ago == 4`;
`rebuild` re-phases `ago` to 0, so the pattern repeats verbatim.  Over 20
updates `update(batch)` returns True at exactly `{4, 8, 12, 16, 20}` —
`rebuild_count == 5`, and the 20th update *is* a rebuild so the run ends at
`ago == 0`.

`ndanger == 0`: at this gate `max(every, delay) == 1`, but every rebuild fired at
`ago == 4`, never at that first permitted opportunity.  This is the *informative*
corner of the alarm, and the complement of link 04's Section 3 — there
`every=1` with a 1.0 A jump made every rebuild fire at `ago == 1` and
`ndanger` merely counted rebuilds.  Here the skin does its job and the counter
correctly reports "no rebuild was ever overdue".

Section 4 — reading the physics back **out of the batch**
----------------------------------------------------------
A rigid translation preserves every minimum-image displacement exactly, so the
neighbour set is invariant: `num_edges == 1152` still, after five rebuilds and a
total 4.0 A drift (the lattice is deliberately left *unwrapped* — x runs to
13.0 A in a 12.0 A cell — which the raw-displacement invariant requires).

The check reconstructs distances the way a potential does, through the batch and
not through `nl`:

    r = || pos[target] - pos[source] + shift ||   over edge_index[:num_edges]

with `pos`, `edge_index` and `shifts` all read from `batch`.  Against the shell
table of Section 1 that must give, exactly:

    min r                    = 3.000 A          (first shell)
    max r          = sqrt2*a = 4.2426406871 A   (second shell)
    count(r <= 3.5 A)        = 64 *  6 =  384   (interaction cutoff)
    count(3.5 < r <= 5.0 A)  = 64 * 12 =  768   (the skin band)
    count(r > 5.0 A)         = 0               (nothing beyond r_build)

Section 5 — dispatch equivalence at the public-API level
---------------------------------------------------------
A second, identically constructed list is driven over the *same* schedule with
raw `update(pos)` tensors instead of `update(batch)`.  Both end at
`rebuild_count == 5`, `ndanger == 0`, `num_edges == 1152`, and `torch.equal` on
the full `edge_index` / `shifts` buffers — dead padding tail included.  One
policy, one state machine, two front doors.

Drift policy
------------
A disagreement between this file and the runtime is a **DEFECT REPORT, never a
golden edit**.  Every literal below is derived on this page from the lattice
geometry, the documented `capacity_factor = 1.35`, and the LAMMPS gate as
specified; none was read off a run and then written down.  If `max r` comes back
as anything but `sqrt2 * 3.0 A`, or `num_edges` drifts off 1152 after the run,
the bind went stale — open a defect against `src/molix/md/neighbors.py`, do not
retune the literal.  A batch that silently *copied* on assignment would show up
here as exactly that kind of drift, and a golden edited to match it would hide a
frozen potential-energy surface.

Known spec-text correction (recorded, not tuned)
-------------------------------------------------
The spec's "Regression example" paragraph asks for `max r == 5.0 A`.  That is a
slip in the prose, contradicted by the spec's own shell table three lines
earlier: `5.0 A` is `r_build`, an upper **bound** on realised pair distances, not
a realised one.  Simple-cubic distances are `3 * sqrt(i^2+j^2+k^2)` A, i.e.
`{3.000, 4.243, 5.196, 6.000, ...}` — `5.0` is not in the set, so no pair can
ever sit there.  This file therefore pins the crystallographic value
`max r = sqrt2 * a = 4.2426406871 A` **and** the bound the prose was reaching
for (`count(r > r_build) == 0`, `max r < r_build`).  The golden comes from the
lattice, not from a run; the spec sentence should be corrected to match.

Goldens
-------
    capture command : PYTHONPATH=src python regressions/md-neighborlist-skin-05-bind.py
                      Nothing was captured from a run.  Every literal is
                      arithmetic on the page: 64*18 = 1152, 64*6 = 384,
                      64*12 = 768, ceil(1.35*1152) = 1556, 3.5 + 1.5 = 5.0,
                      3.0*sqrt(2) = 4.2426406871192851, and the gate table of
                      Section 3.
    commit          : d59a38f (d59a38fc77b700daec57c5a90676d42d32cf01be), with
                      the `md-neighborlist-skin-05-bind` working tree on top
                      (at d59a38f itself `NeighborList` has no `build` at all,
                      so `nl.build(batch)` raises `AttributeError` — that is the
                      RED this file was written against).
    torch           : 2.12.1+cpu   (python 3.14.5)
    date            : 2026-08-09
    device / dtype  : CPU, float64 — the cell and the positions are float64
                      literals, so `shifts` and `_x_hold` are float64 too.
                      `edge_index` is int64 regardless.
    oracle          : none.  No third-party package (torch and tensordict are
                      the repo's own core dependencies), no network, no
                      subprocess, no RNG, no wall-clock value, no filesystem
                      access.  LAMMPS is the *specified* behaviour, not a
                      runtime dependency: its gate is transcribed into the table
                      in Section 3, never executed.
    tolerance       : 1e-12 A on `r_build` (float64 "position" exact band; 3.5
                      and 1.5 are exactly representable, so 5.0 is exact).
                      1e-9 A on the reconstructed pair distances, which pass
                      through a subtraction, an addition and a norm in float64
                      (observed deviation ~1e-15 A).  Exact integer equality on
                      every edge count, capacity, rebuild count, `ago`,
                      `ndanger`, pair tally and fired-update sequence — they are
                      counters, and a counter has no tolerance.  Exact identity
                      (`is`) on the bound leaves: the whole point.

Run:
    PYTHONPATH=src python regressions/md-neighborlist-skin-05-bind.py
"""

from __future__ import annotations

import itertools
import math
import sys

import torch
from tensordict import TensorDict

from molix.md import NeighborList

# ---------------------------------------------------------------------------
# The lattice every section runs on.
# ---------------------------------------------------------------------------

#: Lattice spacing `a` in Angstrom.
SPACING = 3.0

#: Sites per axis; 4 keeps the `+a` and `-a` neighbours distinct atoms.
N_SIDE = 4

#: Cubic cell edge in Angstrom — four spacings, so the lattice tiles exactly.
BOX = SPACING * N_SIDE

#: Atomic number written into the batch. Argon: chemically inert here, since no
#: potential is evaluated — the key exists because a batch carries it.
ATOMIC_NUMBER = 18

#: Interaction cutoff in Angstrom: between the 3.0 A first shell (in) and the
#: 4.243 A second shell (out), so no pair can flip class on rounding.
CUTOFF = 3.5

#: Verlet skin in Angstrom. `r_build = 5.0 A` sits between the 4.243 A second
#: shell (in) and the 5.196 A third shell (out) — a hard integer boundary, not a
#: tolerance question — and half-skin is a round 0.75 A.
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
# Section 1 goldens — the lattice at `r_build`.
# ---------------------------------------------------------------------------

#: `CUTOFF + SKIN`, written out rather than recomputed from the two constants:
#: this file pins the derived radius, so deriving it here would be tautological.
EXPECTED_R_BUILD = 5.0

#: 64 sites x (6 at 3.0 A + 12 at 4.243 A) directed edges, both directions kept
#: (`symmetry=True`, the MD list's fixed setting).
EXPECTED_NUM_EDGES = 1152

#: ceil(1.35 * 1152) = ceil(1555.2), with the default `capacity_factor = 1.35`.
EXPECTED_CAPACITY = 1556

#: float64 "position" exact band; 3.5 + 1.5 = 5.0 is exact in binary.
LENGTH_TOL = 1e-12

# ---------------------------------------------------------------------------
# Section 2 goldens — the bind.
# ---------------------------------------------------------------------------

#: `build` binds exactly the two live buffers, replacing the namespace wholesale.
EXPECTED_EDGE_KEYS = ("edge_index", "shifts")

# ---------------------------------------------------------------------------
# Section 3 goldens — `every=1, delay=0, check=True` under a rigid drift.
# ---------------------------------------------------------------------------

#: Rigid translation of *all* atoms along x per update, in Angstrom.
STEP_DISPLACEMENT = 0.2

#: Updates driven through the batch.
N_UPDATES = 20

#: Displacement since the last build is `0.2 * ago` A; it first exceeds
#: half-skin 0.75 A at `ago == 4` (0.8 A), and `rebuild` re-phases `ago` to 0.
EXPECTED_FIRED_UPDATES = (4, 8, 12, 16, 20)

EXPECTED_REBUILD_COUNT = 5

#: Every rebuild fired at `ago == 4`, never at the first permitted opportunity
#: `max(every, delay) == 1`, so no rebuild was ever overdue.
EXPECTED_NDANGER = 0

#: The 20th update *is* a rebuild, so the clock is back at 0 when the loop ends.
EXPECTED_FINAL_AGO = 0

# ---------------------------------------------------------------------------
# Section 4 goldens — the geometry read back out of the batch.
# ---------------------------------------------------------------------------

#: First shell: the lattice spacing itself, exactly.
EXPECTED_MIN_PAIR_DISTANCE = SPACING

#: Second shell — the face diagonal `sqrt2 * a = 4.2426406871192851 A`, the
#: largest distance that fits inside `r_build = 5.0 A`. Crystallography, not a
#: captured run value: the next shell is `sqrt3 * a = 5.196 A`, outside.
EXPECTED_MAX_PAIR_DISTANCE = SPACING * math.sqrt(2.0)

#: 64 sites x 6 first-shell neighbours, inside the *interaction* cutoff 3.5 A.
EXPECTED_PAIRS_WITHIN_CUTOFF = 384

#: 64 sites x 12 second-shell neighbours — the skin band `(3.5, 5.0] A`, edges
#: the list carries and the model's own envelope masks to zero.
EXPECTED_PAIRS_IN_SKIN_BAND = 768

#: `r_build` is a hard horizon: the kernel builds at it, so nothing is beyond.
EXPECTED_PAIRS_BEYOND_R_BUILD = 0

#: float64 through a subtract, an add and a 3-vector norm; observed ~1e-15 A.
DISTANCE_TOL = 1e-9


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


def md_batch(positions: torch.Tensor) -> TensorDict:
    """Wrap *positions* in the batch shape an MD driver hands the list.

    Args:
        positions: Positions ``(N, 3)`` in Angstrom, float64.

    Returns:
        A ``TensorDict`` with root ``batch_size=[]`` carrying ``("atoms",
        "pos")`` / ``"Z"`` / ``"batch"`` at ``batch_size=[N]`` — the two-tier
        contract's post-collate tier, single system, no ``"edges"`` yet. The
        positions leaf is stored by reference, so writing into it in place is
        what an integrator does.
    """
    n_atoms = int(positions.shape[0])
    return TensorDict(
        {
            "atoms": TensorDict(
                {
                    "pos": positions,
                    "Z": torch.full((n_atoms,), ATOMIC_NUMBER, dtype=torch.long),
                    "batch": torch.zeros(n_atoms, dtype=torch.long),
                },
                batch_size=[n_atoms],
            ),
        },
        batch_size=[],
    )


def rigidly_translated(base: torch.Tensor, offset: float) -> torch.Tensor:
    """Return *base* with **every** atom translated by *offset* along x.

    A rigid translation leaves every interatomic displacement — and so every
    minimum image — exactly unchanged, which is what lets Section 4 assert the
    neighbour set is bit-identical after a 4.0 A drift.

    Args:
        base: Reference positions ``(N, 3)`` in Angstrom.
        offset: Translation along x, in Angstrom, applied to all atoms.

    Returns:
        A fresh ``(N, 3)`` tensor; *base* is never mutated, so each step is an
        absolute displacement from the lattice rather than an accumulation of
        rounding.
    """
    shift = torch.tensor([offset, 0.0, 0.0], dtype=base.dtype, device=base.device)
    return base + shift


def pair_distances(batch: TensorDict, num_edges: int) -> torch.Tensor:
    """Reconstruct live pair distances **through the batch**, as a potential does.

    Reads positions, edge indices and periodic shifts from *batch* only — never
    from the list — so a binding that copied instead of aliasing shows up as a
    stale neighbour set here.

    Args:
        batch: Batch carrying ``("atoms", "pos")`` and the bound
            ``("edges", "edge_index")`` / ``("edges", "shifts")``.
        num_edges: Live edge count; rows ``[num_edges, capacity)`` are dead
            padding and are excluded.

    Returns:
        Minimum-image distances ``(num_edges,)`` in Angstrom, computed as
        ``|| pos[target] - pos[source] + shift ||`` per the repo edge
        convention (``edge_index[:, 0]`` source, ``[:, 1]`` target).
    """
    edge_index = batch["edges", "edge_index"][:num_edges]
    shifts = batch["edges", "shifts"][:num_edges]
    positions = batch["atoms", "pos"]
    source, target = edge_index[:, 0], edge_index[:, 1]
    return torch.linalg.norm(positions[target] - positions[source] + shifts, dim=-1)


def fresh_list(positions: torch.Tensor) -> NeighborList:
    """Construct the one configuration every section shares.

    Args:
        positions: Initial positions ``(N, 3)`` in Angstrom the list builds at.

    Returns:
        A ``NeighborList`` at ``cutoff=3.5 A``, ``skin=1.5 A``, default gate
        ``every=1, delay=0, check=True``.
    """
    return NeighborList(
        cell=CELL,
        cutoff=CUTOFF,
        positions=positions,
        skin=SKIN,
        every=1,
        delay=0,
        check=True,
    )


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
        """Assert a counter, a shape, an identity or a sequence — no tolerance."""
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


def check_bind(checker: Checker) -> tuple[NeighborList, TensorDict]:
    """Sections 1-2 — the lattice at `r_build`, and what `build` writes.

    Args:
        checker: Failure collector.

    Returns:
        The bound list and its batch, for the driven schedule to reuse.
    """
    print(f"Bind ({N_SIDE}x{N_SIDE}x{N_SIDE} sc, {SPACING} A spacing, {BOX} A cell, float64 CPU)")

    lattice = simple_cubic_lattice()
    neighbor_list = fresh_list(lattice.clone())
    batch = md_batch(lattice.clone())

    checker.within(
        "bind.r_build_deviation",
        abs(neighbor_list.r_build - EXPECTED_R_BUILD),
        LENGTH_TOL,
    )
    checker.exact("bind.num_edges", neighbor_list.num_edges, EXPECTED_NUM_EDGES)
    checker.exact("bind.capacity", neighbor_list.capacity, EXPECTED_CAPACITY)

    bound = neighbor_list.build(batch)

    # The returned object *is* the argument, so `potential(nl.build(batch))`
    # composes with the repo's forward(td) -> td convention.
    checker.exact("bind.returns_same_batch", bound is batch, True)
    # Identity, not equality: a TensorDict that copied on assignment would give
    # a frozen neighbour set that still compares equal at bind time.
    checker.exact(
        "bind.edge_index_is_buffer",
        batch["edges", "edge_index"] is neighbor_list.edge_index,
        True,
    )
    checker.exact(
        "bind.shifts_is_buffer",
        batch["edges", "shifts"] is neighbor_list.shifts,
        True,
    )
    checker.exact("bind.edge_keys", tuple(sorted(batch["edges"].keys())), EXPECTED_EDGE_KEYS)
    # Capacity, not num_edges: the buffers are fixed-shape, tail padded dead.
    checker.exact(
        "bind.edges_batch_size",
        tuple(batch["edges"].batch_size),
        (EXPECTED_CAPACITY,),
    )
    # A bind is not physics: it also runs on every .to() re-sync.
    checker.exact("bind.rebuild_count", neighbor_list.rebuild_count, 0)
    checker.exact("bind.ago", neighbor_list.ago, 0)
    checker.exact("bind.ndanger", neighbor_list.ndanger, 0)

    return neighbor_list, batch


def check_driven_schedule(checker: Checker, neighbor_list: NeighborList, batch: TensorDict) -> None:
    """Section 3 — twenty `update(batch)` calls under a rigid +0.2 A/step drift.

    Args:
        checker: Failure collector.
        neighbor_list: The list bound in Section 2.
        batch: Its bound batch; positions are written in place, as an integrator
            would, so the bound leaves are never rebound.
    """
    print(
        f"\nDriven schedule (every=1, delay=0, check=True, rigid {STEP_DISPLACEMENT} A/update "
        f"on all {N_SIDE**3} atoms)"
    )

    lattice = simple_cubic_lattice()
    positions = batch["atoms", "pos"]

    fired: list[int] = []
    for update in range(1, N_UPDATES + 1):
        positions.copy_(rigidly_translated(lattice, STEP_DISPLACEMENT * update))
        if neighbor_list.update(batch):
            fired.append(update)

    checker.exact("driven.fired_updates", tuple(fired), EXPECTED_FIRED_UPDATES)
    checker.exact("driven.rebuild_count", neighbor_list.rebuild_count, EXPECTED_REBUILD_COUNT)
    checker.exact("driven.ndanger", neighbor_list.ndanger, EXPECTED_NDANGER)
    checker.exact("driven.final_ago", neighbor_list.ago, EXPECTED_FINAL_AGO)
    # The tie survived five in-place rebuilds: still the same tensors.
    checker.exact("driven.pos_leaf_not_rebound", batch["atoms", "pos"] is positions, True)
    checker.exact(
        "driven.edge_index_still_bound",
        batch["edges", "edge_index"] is neighbor_list.edge_index,
        True,
    )
    checker.exact(
        "driven.shifts_still_bound",
        batch["edges", "shifts"] is neighbor_list.shifts,
        True,
    )


def check_geometry_through_batch(
    checker: Checker, neighbor_list: NeighborList, batch: TensorDict
) -> None:
    """Section 4 — reconstruct the shells from the batch after the driven run.

    Args:
        checker: Failure collector.
        neighbor_list: The driven list (read only for ``num_edges``).
        batch: Its bound batch — the sole source of positions, indices, shifts.
    """
    print("\nGeometry read back through the batch (after 4.0 A of rigid, unwrapped drift)")

    # A rigid translation preserves every minimum image, so the neighbour set is
    # invariant under the whole schedule.
    checker.exact("geometry.num_edges_unchanged", neighbor_list.num_edges, EXPECTED_NUM_EDGES)

    distances = pair_distances(batch, neighbor_list.num_edges)

    checker.exact("geometry.distances_counted", int(distances.numel()), EXPECTED_NUM_EDGES)
    checker.within(
        "geometry.min_distance_deviation",
        abs(float(distances.min()) - EXPECTED_MIN_PAIR_DISTANCE),
        DISTANCE_TOL,
    )
    checker.within(
        "geometry.max_distance_deviation",
        abs(float(distances.max()) - EXPECTED_MAX_PAIR_DISTANCE),
        DISTANCE_TOL,
    )
    checker.exact(
        "geometry.pairs_within_cutoff",
        int((distances <= CUTOFF).sum()),
        EXPECTED_PAIRS_WITHIN_CUTOFF,
    )
    checker.exact(
        "geometry.pairs_in_skin_band",
        int(((distances > CUTOFF) & (distances <= EXPECTED_R_BUILD)).sum()),
        EXPECTED_PAIRS_IN_SKIN_BAND,
    )
    checker.exact(
        "geometry.pairs_beyond_r_build",
        int((distances > EXPECTED_R_BUILD).sum()),
        EXPECTED_PAIRS_BEYOND_R_BUILD,
    )


def check_dispatch_equivalence(checker: Checker, driven: NeighborList) -> None:
    """Section 5 — the raw-tensor front door reaches the same state.

    Args:
        checker: Failure collector.
        driven: The batch-driven list to compare buffers against.
    """
    print("\nDispatch equivalence (same schedule through raw update(pos))")

    lattice = simple_cubic_lattice()
    neighbor_list = fresh_list(lattice.clone())

    fired: list[int] = []
    for update in range(1, N_UPDATES + 1):
        positions = rigidly_translated(lattice, STEP_DISPLACEMENT * update)
        if neighbor_list.update(positions):
            fired.append(update)

    checker.exact("raw.fired_updates", tuple(fired), EXPECTED_FIRED_UPDATES)
    checker.exact("raw.rebuild_count", neighbor_list.rebuild_count, EXPECTED_REBUILD_COUNT)
    checker.exact("raw.ndanger", neighbor_list.ndanger, EXPECTED_NDANGER)
    checker.exact("raw.num_edges", neighbor_list.num_edges, driven.num_edges)
    # Whole buffers, dead padding tail included — not just the live prefix.
    checker.exact(
        "raw.edge_index_equals_batch_path",
        bool(torch.equal(neighbor_list.edge_index, driven.edge_index)),
        True,
    )
    checker.exact(
        "raw.shifts_equals_batch_path",
        bool(torch.equal(neighbor_list.shifts, driven.shifts)),
        True,
    )


def main() -> int:
    """Pin the bind surface: build, drive through the batch, read the physics back."""
    checker = Checker()

    neighbor_list, batch = check_bind(checker)
    check_driven_schedule(checker, neighbor_list, batch)
    check_geometry_through_batch(checker, neighbor_list, batch)
    check_dispatch_equivalence(checker, neighbor_list)

    if checker.failures:
        print("\nFAILED — the TensorDict bind surface no longer holds:")
        for failure in checker.failures:
            print(f"  {failure}")
        print(
            "\nEvery golden above is analytic (64*18 = 1152, 64*6 = 384, "
            "64*12 = 768, ceil(1.35*1152) = 1556, 3.0*sqrt(2) = 4.2426406871, "
            "and the gate table in the module docstring). A disagreement is a "
            "DEFECT REPORT against src/molix/md/neighbors.py, never a reason to "
            "edit the literal. In particular a broken identity assertion means "
            "the batch holds a *copy* of the buffers: the neighbour set a "
            "potential reads would freeze at bind time while the list keeps "
            "rebuilding, and a golden retuned to match would hide a frozen "
            "potential-energy surface."
        )
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
