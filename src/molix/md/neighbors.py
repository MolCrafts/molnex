"""Rebuilding, fixed-capacity neighbour list for production MD.

A frozen neighbour list is only valid while no atom moves far enough to change
its neighbour set — tens of steps for liquid water, not the millions a
production trajectory needs. This module rebuilds it under the LAMMPS
``neigh_modify every/delay/check`` policy (see below) while keeping **every
tensor shape constant**, which is what lets the force evaluation stay inside a
CUDA graph across the whole run.

The trick is the *capacity*: edge tensors are allocated once at
``capacity = ceil(factor * E_initial)`` and only their contents change. Unused
rows carry a **dead edge** — source and target both atom 0, displaced by a shift
longer than the cutoff. Such an edge contributes exactly zero:

* ``r = |shift| > r_cut`` so the polynomial cutoff envelope is 0, and the radial
  features it multiplies are 0, so the message and the learned density are 0;
* pair-repulsion envelopes cut off at the pair's covalent radii, well inside
  ``r_cut``, so they are 0 too;
* ``pos[0] - pos[0]`` cancels exactly, so no spurious force reaches atom 0.

Periodicity is handled by the minimum-image convention of the compiled
neighbour kernel, whose sequential reduction is complete only up to half the
smallest **perpendicular width** of the cell: ``w_i = V / ||a_j x a_k||`` in
Angstrom, for cell vectors ``a_1, a_2, a_3`` (the rows of ``cell``) and volume
``V = |det(cell)|``. So the *build* radius must not exceed ``min_i w_i / 2`` —
:class:`NeighborList` refuses to construct otherwise rather than
silently dropping pairs that are inside the cutoff. For an orthorhombic cell
``w_i = ||a_i||``, i.e. the familiar half-shortest-cell-vector bound.

The Verlet skin
---------------

``cutoff`` is the **interaction** cutoff ``r_cut`` every consumer means. The
list is built at the enlarged radius

``r_build = cutoff + skin``            (Angstrom, :attr:`NeighborList.r_build`)

so that pairs which walk *into* ``r_cut`` between rebuilds are already in the
buffers. The model's own cutoff envelope masks the skin-region pairs to zero
(``LennardJonesCutForceField`` compares against ``cutoff_sq``), so the skin
costs edges — the live count grows as ``(1 + skin/r_cut)^3`` — but changes no
energy. Both the capacity and the kernel's ``max_num_pairs`` are therefore
sized from ``r_build``, never from ``cutoff``.

**Half-skin completeness criterion.** For atoms *i*, *j* let
``d_i = ||x_i(t) - x_i(t0)||`` be the displacement since the last build at
``t0``. The triangle inequality gives ``r_ij(t) >= r_ij(t0) - d_i - d_j``, so
any pair inside ``r_cut`` at time *t* was inside ``r_cut + d_i + d_j`` at
``t0``. A list built at ``r_build = r_cut + s`` is therefore still **complete**
while ``d_(1) + d_(2) <= s`` for the two largest displacements. Which two atoms
those are is unknown, so the conservative sufficient condition — and the one
LAMMPS tests — is the **half**-skin bound

``max_i d_i <= s / 2``     (worst case ``d_(1) = d_(2) = s/2``; hence *half*)

with a **strict** ``>`` triggering the rebuild: exactly ``s/2`` still satisfies
the proof, so rebuilding there would be wasted work.

**Raw displacements, unwrapped positions (load-bearing invariant).** The
displacement test uses the *raw* difference ``x - x_hold``, never a minimum
image — min-imaging it would clamp a genuine ``> L/2`` excursion and so
*suppress* the very rebuild it is meant to force. That is correct only while
positions drift unwrapped, which is what this repo's MD does (nothing in
``molix.md.integrators`` / ``molix.md.runner`` wraps; LAMMPS wraps only on
reneighbour steps, before ``xhold`` is stored). :meth:`NeighborList.update`
promotes it to a checked invariant: a displacement at or beyond
``min_i w_i / 2`` raises :class:`RuntimeError`, which catches mid-run wrapping,
a changed cell and a blown-up trajectory alike. Do not wrap positions mid-run.

**``ndanger`` — the correctness alarm.** A rebuild that fires at the *first*
permitted opportunity, ``ago == max(every, delay)``, may already have been
overdue on an earlier, non-permitted step, so pairs may have been missed;
:attr:`NeighborList.ndanger` counts those (LAMMPS "Dangerous builds"). A run
that reports ``ndanger > 0`` should be rerun with a larger ``skin`` or a
tighter gate. **Caveat at the default gate** ``every=1, delay=0``:
``max(every, delay) == 1``, so *every* rebuild lands on the first permitted
opportunity and ``ndanger`` simply counts rebuilds. There the counter carries
no information — read it only for a coarser gate or a nonzero skin.

**What ``check=False`` and a coarse ``delay`` cost.** A pair inside ``r_cut``
but absent from the list contributes exactly zero, and the next build inserts
it discontinuously: the total energy takes an ``O(1)``, one-signed injection
per missed event — not the ``O(dt^2)`` error of a discretisation artefact.
Repeated events integrate into a systematic NVE energy leak. ``check=False``
(and any ``delay`` long enough to skip a needed rebuild) buys speed by
accepting that leak; it is never a free optimisation. ``check=False``
additionally disables the unwrapped-positions guard, which lives inside the
displacement branch.

**LAMMPS parity.** The gate, the strict comparison, the raw difference and the
``ndanger`` threshold are ``Neighbor::decide`` / ``Neighbor::check_distance``
verbatim, and the constructor mirrors ``Neighbor::init`` in rejecting a
``delay`` that is not a multiple of ``every`` (which would make the danger
threshold an ``ago`` the gate never permits, silently killing the alarm).

The binned build
----------------

``bin=None`` (the default) hands the whole system to the compiled O(N^2) pair
kernel, which enumerates ``N(N-1)/2`` candidates per rebuild. Passing a float
switches to a pure-torch **cell list**: atoms are sorted into a periodic grid
of bins derived once from the cell and ``r_build``, and only a fixed stencil of
neighbouring bins is searched, so the build is linear in ``N`` at fixed density
with no per-atom Python loop and no device-specific code
(:meth:`NeighborList._configure_bins` derives the grid,
:meth:`NeighborList._build_binned` runs the search). Both backends return the
same triple into the same ``_write``, so the capacity, the dead-edge padding
and every consumer are unaffected by which one ran. ``bin`` is a **cost** knob:
the edge set is identical, and the kernel path is the equivalence oracle the
binned one is tested against.

**Stencil completeness — why a bounded search is exact.** Bins are cubes in
*fractional* space: atom A sits in bin ``p_i = floor(s_i n_i)`` along axis
``i``. If two atoms are ``m`` bins apart along axis ``i`` (minimal modular
difference), then ``s_B >= (p + m)/n_i`` while ``s_A < (p + 1)/n_i``, so
``|Ds_i| > (m - 1)/n_i`` **strictly**. The displacement's component along the
axis normal is ``|Ds_i| w_i``, and ``||d|| >= |d . n_i| = |Ds_i| w_i``, so with
the *effective* perpendicular bin thickness ``b_i = w_i / n_i`` (Angstrom)

``||d|| > (m - 1) * b_i``

A stencil half-width ``k_i`` with ``k_i * b_i >= r_build`` is therefore
**complete**: any pair more than ``k_i`` bins apart on some axis has
``||d|| > k_i b_i >= r_build`` and is excluded by the ``r <= r_build`` filter
anyway. The strict inequality is what makes the textbook statement exact at the
boundary — ``b_i = r_build`` gives ``k_i = 1``, i.e. the 27-cell 3x3x3 stencil,
with no epsilon fudge. ``_configure_bins`` re-checks the relation per axis
instead of trusting the arithmetic that produced it: an incomplete stencil is a
silently short neighbour list, which is a wrong energy that never raises.

**Bin size.** The candidate volume for ``b = r_build / k`` is
``(2 + 1/k)^3 r_build^3`` — ``27 r^3`` at ``k = 1``, ``15.6 r^3`` at ``k = 2``,
against the ``4.19 r^3`` sphere actually needed — decreasing in ``k`` while the
sorting and gather cost grows with the bin count. LAMMPS settles this at
``k = 2`` (``src/nbin_standard.cpp``, ``binsize_optimal = 0.5 * cutneighmax``),
which is what ``bin=0.0`` requests here.

**Minimum image by fractional rounding is exact inside the half-width guard.**
Write a periodic displacement as ``d = sum_i f_i a_i``. Its component along the
axis-``i`` normal is ``|f_i| w_i = |d . n_i| <= ||d||``. So whenever
``||d|| <= r_build <= min_i w_i / 2 <= w_i / 2`` — exactly the guard above —
``|f_i| <= 1/2`` on every axis: **any** in-range image is already the one
``f <- f - round(f)`` selects. Fractional rounding and the kernel's sequential
diagonal reduction therefore return the same, unique minimum image everywhere
the guard admits, which is what makes edge-set equality between the two
backends a theorem rather than a coincidence. (Ties at ``|f_i| = 1/2`` are
reachable only when ``r_build = min_i w_i / 2`` exactly *and* a pair sits
exactly on the boundary, where ``torch.round``'s half-to-even and C's
half-away-from-zero can differ: measure zero, and outside every fixture.)

**Filter parity.** The binned path accepts a pair iff ``0 < r <= r_build``,
which is the compiled backends' filter verbatim (``(distances <= cutoff) &
(distances > 0)`` in the C++ kernel; ``distance2 > cutoff2 || distance2 == 0``
dropped in the CUDA one) — the same *closed* upper bound and the same ``r > 0``
rejection, so coincident atoms and pairs separated by exactly one lattice
vector (whose minimum image is the zero vector) are dropped identically.

**Edge order is not part of the contract.** The binned path emits edges in
bin-sorted order, the kernel in upper-triangle index order. What binds is the
*set* of ``(source, target, shift)`` triples plus the edge count: consumers
reduce with order-independent scatter / ``index_add_`` and read the shift
buffer positionally alongside ``edge_index``, never by index-order assumption.

**Small cells degenerate gracefully.** When ``2 k_i + 1 >= n_i`` on every axis
the stencil's residue sets cover the whole grid and the search enumerates all
pairs — correct, just not faster (and a single bin per axis, the ``bin`` wider
than the cell case, is exactly that). The O(N) win appears once
``n_i > 2 k_i + 1``, i.e. cells wider than about ``5 r_build / 2`` per axis at
the automatic bin size, so a small-cell timing is not a regression.

References:
    LAMMPS ``neigh_modify`` documentation —
    https://docs.lammps.org/neigh_modify.html — and ``lammps/lammps`` develop
    ``src/neighbor.cpp`` (``Neighbor::decide``, ``Neighbor::check_distance``,
    ``Neighbor::init``) / ``src/verlet.cpp``. The binning policy behind
    ``bin=0.0`` is ``src/nbin_standard.cpp``
    (``binsize_optimal = 0.5 * cutneighmax``) and
    https://docs.lammps.org/neighbor.html; the paper of record is
    A. P. Thompson et al., *Comput. Phys. Commun.* **271**, 108171 (2022),
    https://doi.org/10.1016/j.cpc.2021.108171.

    K. Nordlund, *Introduction to molecular dynamics simulations*, lecture 3,
    https://www.mv.helsinki.fi/home/knordlun/moldyn/lecture03.pdf — the open,
    directly re-verified source for the two-atom criterion above.

    M. P. Allen & D. J. Tildesley, *Computer Simulation of Liquids*, 2nd ed.,
    Oxford University Press (2017),
    https://doi.org/10.1093/oso/9780198803195.001.0001 — cell (link-cell)
    lists, the 27-cell stencil and minimum-image validity.

    L. Verlet, *Phys. Rev.* **159**, 98 (1967),
    https://doi.org/10.1103/PhysRev.159.98 — the original neighbour list.

    B. Quentrec & C. Brot, *J. Comput. Phys.* **13**, 430 (1973),
    https://doi.org/10.1016/0021-9991(73)90046-6 — the original cell method,
    and the skin refinement.

    Caveat: the Verlet 1967 and Quentrec & Brot 1973 texts are paywalled and
    were **not** re-verified here; they are cited for attribution only. Every
    equation above is verified against the Nordlund notes and the LAMMPS
    source, and the stencil-completeness and ``|f_i| <= 1/2`` relations are
    derived in line above rather than taken from a text.

Owning ``edges``: the bind surface
----------------------------------

Two entry points, two audiences, one owner. The MD hot path drives the list
with raw ``(N, 3)`` position tensors in Angstrom (:meth:`NeighborList.rebuild`
forces a build, :meth:`NeighborList.update` applies the policy); a TensorDict
caller drives it with the batch itself (:meth:`NeighborList.build` builds *and*
binds, :meth:`NeighborList.update` again — one method with an ``isinstance``
dispatch at the top, never an ``update_td`` / ``update_pos`` pair of twins).

**The list owns ``edges`` once bound.** :meth:`NeighborList.build` writes
``batch["edges"]`` as a ``TensorDict`` holding the live ``edge_index`` /
``shifts`` buffers **by reference**, replacing whatever was there — including a
precomputed ``edge_diff`` / ``edge_dist`` pair, which a potential would
otherwise consume straight through as a value and so freeze the PES. Because
every rebuild is in place, that tie survives all later ``rebuild`` / ``update``
calls with no re-binding: the potential simply sees the current neighbour set.
It does **not** survive :meth:`NeighborList.to`, which rebinds the buffers to
new tensors — the owner re-binds (see ``PeriodicPotentialForceField._apply``).

Two ``NeighborList`` classes, deliberately
-----------------------------------------

The repository holds two classes named ``NeighborList``, in different layers,
and the collision is intentional rather than an accident awaiting cleanup:

* :class:`molix.md.neighbors.NeighborList` (this one) — the **MD engine**
  symbol: a *stateful, fixed-capacity buffer owner* holding ``edge_index
  (capacity, 2)``, ``shifts (capacity, 3)``, ``num_edges`` and
  ``rebuild_count``, rebuilt in place so shapes never change and the force path
  stays CUDA-graph capturable. One instance per run, held by a
  :class:`~molix.md.forcefield.ForceField`.
* :class:`molix.data.tasks.neighbor.NeighborList` — the **data-pipeline**
  symbol: a *stateless* ``SampleTask`` mapping one flat sample dict to
  ``edge_index`` / ``edge_diff`` / ``edge_dist`` and contributing to ``task_id``
  for cache keying. Constructed once per pipeline definition, no per-call state.

They are not two variants of one concept — a per-run mutable buffer with an
overflow policy versus a pure pipeline transform — and each is the shortest
natural name in its own layer, so neither gives up the bare name. Because this
module *consumes* the pipeline task (see the import below), and a bare
``from ... import NeighborList`` here would be rebound by the class definition
further down — making the constructor call itself recursively — the import is
aliased to ``NeighborListTask``.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import torch
from tensordict import TensorDict, TensorDictBase

# The one owner of kernel-output normalisation (pbc handling, NaN-padding
# strip, symmetry expansion, edge-sign convention); reimplementing that here
# against the raw ``molix.F.locality`` kernel would fork it. Aliased because
# the MD buffer owner defined below is *also* named ``NeighborList`` and would
# otherwise shadow its own dependency — the class would then call itself.
from molix.data.tasks.neighbor import NeighborList as NeighborListTask
from molix.units import DEAD_EDGE_CUTOFF_FACTOR

#: Largest stencil half-width ``k_i`` (in bins) the binned build accepts. At the
#: cap the Python loop runs over ``17^3 = 4913`` bin offsets; beyond it an
#: absurdly small explicit ``bin`` stops being a fine grid and becomes a hang.
_MAX_STENCIL_HALF_WIDTH = 8


def _perpendicular_widths(cell: torch.Tensor) -> torch.Tensor:
    """Distances between the three pairs of opposite faces of a periodic cell.

    For cell vectors ``a_1, a_2, a_3`` — the **rows** of ``cell`` — the width
    perpendicular to the face spanned by ``a_j`` and ``a_k`` is
    ``w_i = V / ||a_j x a_k||`` with ``V = |det(cell)|``. For an orthorhombic
    cell ``||a_j x a_k|| = ||a_j||*||a_k||`` and ``V = ||a_1||*||a_2||*||a_3||``,
    hence ``w_i = ||a_i||``.

    Two callers need different reductions of the same three numbers: the
    minimum-image guard bounds ``r_build`` by ``min_i w_i / 2``, while the
    binned build sizes its grid **per axis** so that a requested bin thickness
    means the same thing on a sheared cell as on a cube.

    Evaluated in ``float64`` so a ``float32`` cell cannot jitter an
    accept/reject decision taken right at the bound.

    Args:
        cell: Cell vectors ``(3, 3)`` in Angstrom, one vector per row.

    Returns:
        The perpendicular widths ``(w_1, w_2, w_3)`` as a ``(3,)`` ``float64``
        tensor in Angstrom, in cell-row order.

    Raises:
        ValueError: If ``cell`` is not ``(3, 3)``, or is degenerate (volume
            zero or non-finite, or two rows collinear). A degenerate cell has
            no finite width, and returning ``nan`` would make every ``>``
            comparison against the bound silently succeed.
    """
    if tuple(cell.shape) != (3, 3):
        raise ValueError(f"cell must have shape (3, 3), got {tuple(cell.shape)}")
    vectors = cell.detach().to(torch.float64)
    volume = float(torch.linalg.det(vectors).abs())
    areas = torch.linalg.norm(
        torch.stack(
            (
                torch.linalg.cross(vectors[1], vectors[2]),
                torch.linalg.cross(vectors[2], vectors[0]),
                torch.linalg.cross(vectors[0], vectors[1]),
            )
        ),
        dim=-1,
    )
    max_area = float(areas.max())
    if not math.isfinite(volume) or volume <= 0.0 or not math.isfinite(max_area) or max_area <= 0.0:
        raise ValueError(
            f"degenerate cell: volume {volume} A^3, largest face area {max_area} A^2; "
            "a cell with no interior has no perpendicular width to bound the cutoff by."
        )
    # A positive volume forces every face area positive (a zero area means two
    # rows are collinear, which collapses the determinant), so this cannot divide
    # by zero once the guard above has passed.
    return volume / areas


@runtime_checkable
class NeighborStrategy(Protocol):
    """Contract a force field expects from a rebuildable neighbour list.

    Any strategy (minimum-image, Verlet-skin, cell list, …) is usable by
    :class:`~molix.md.forcefield.PeriodicPotentialForceField` /
    :class:`~molix.md.forcefield.CallableForceField` as long as it exposes
    these members with fixed-shape, in-place-rebuilt buffers.

    Attributes:
        edge_index: Edge buffer ``(capacity, 2)`` — ``[:, 0]`` source,
            ``[:, 1]`` target, per the repo edge convention.
        shifts: Periodic shift vectors ``(capacity, 3)``.
        num_edges: Live edges occupy ``[0, num_edges)``; the rest are dead.
        capacity: Fixed buffer length.
        cutoff: Interaction cutoff ``r_cut`` in Angstrom — the radius the list
            guarantees *complete* between rebuilds, and the horizon a force
            field's own cutoff must not exceed. A strategy may build at a
            larger radius (see ``skin``); that is its business, not the
            consumer's.
        skin: Verlet skin in Angstrom, ``0.0`` for a strategy that rebuilds at
            the bare cutoff.
    """

    edge_index: torch.Tensor
    shifts: torch.Tensor
    num_edges: int
    capacity: int
    cutoff: float
    skin: float

    def rebuild(self, positions: torch.Tensor) -> None:
        """Recompute the neighbour list at ``positions``, in place."""
        ...

    def build(self, batch: TensorDict) -> TensorDict:
        """Rebuild at ``batch["atoms", "pos"]`` and bind the buffers into ``batch``.

        Returns the same batch object, with ``batch["edges"]`` holding the
        strategy's live buffers by reference — the strategy owns that namespace
        from here on. ``PeriodicPotentialForceField`` calls this through this
        annotation, at construction and after every ``.to()``.
        """
        ...

    def update(self, positions: TensorDict | torch.Tensor) -> bool:
        """Rebuild at ``positions`` if the strategy's policy says so.

        Called once per force evaluation, with either the raw ``(N, 3)``
        positions of the MD hot path or the batch carrying them. Returns
        whether a rebuild happened.
        """
        ...

    def to(
        self,
        device: torch.device | str | torch.dtype | None = None,
        dtype: torch.dtype | None = None,
    ) -> "NeighborStrategy":
        """Move / cast the buffers, mirroring ``Tensor.to`` semantics."""
        ...


class NeighborList:
    """Minimum-image Verlet-skin neighbour list in fixed-capacity buffers.

    Not an ``nn.Module``: it owns plain buffers and a rebuild policy, and is held
    by a :class:`~molix.md.forcefield.ForceField`. Keeping it out of the module
    tree also keeps it out of ``state_dict``, where a per-run neighbour list has
    no business — the owning force field forwards device/dtype changes through
    :meth:`to` instead (see ``PeriodicPotentialForceField._apply``).

    Two entry points drive it: :meth:`rebuild` forces a build unconditionally,
    :meth:`update` applies the ``every`` / ``delay`` / ``check`` policy. A
    TensorDict caller uses :meth:`build` (force a build *and* bind the buffers
    into the batch) and the same :meth:`update`, which takes either input type.
    See the module docstring for the half-skin criterion, the
    unwrapped-positions invariant and what ``check=False`` costs.

    Args:
        cell: Cell vectors ``(3, 3)`` in Angstrom, one vector per row.
        cutoff: **Interaction** cutoff ``r_cut`` in Angstrom — what every
            consumer means by "cutoff", and the radius the list stays complete
            to between rebuilds. The build radius is :attr:`r_build`.
        positions: Initial positions ``(N, 3)`` in Angstrom; the constructor
            builds the list at them (not counted in :attr:`rebuild_count`) and
            sizes the capacity from the result.
        skin: Verlet skin ``s`` in Angstrom. The list is built at
            ``r_build = cutoff + skin`` and is provably complete out to
            ``cutoff`` while no atom has moved more than ``s/2`` since the last
            build. ``0.0`` reproduces the pre-skin rebuild-on-any-motion
            behaviour.
        every: Attempt a rebuild only when the number of steps since the last
            build is a multiple of this (steps).
        delay: Attempt no rebuild until at least this many steps have passed
            since the last build (steps). Must be a multiple of ``every``
            (LAMMPS ``Neighbor::init`` parity); ``0`` is always legal.
        check: Rebuild only when the maximum displacement since the last build
            exceeds ``skin/2``. ``False`` rebuilds on cadence alone — cheaper,
            but it drops both the completeness criterion and the
            unwrapped-positions guard (module docstring).
        capacity_factor: Buffer capacity as a multiple of the initial edge
            count at ``r_build``. Density fluctuations grow the edge count
            during a run; overflow raises rather than truncating. **Caveat for
            crystalline starts:** a lattice initial configuration systematically
            *under*-estimates the live-edge count a warm run reaches, because a
            coordination shell sitting just outside ``r_build`` at ``t = 0`` is
            pulled in by thermal motion (the LJ-lattice test harness needed
            ``2.5`` where the default would have allocated 519 rows against 600
            live edges). Raise it when starting from a lattice.
        bin: Selects the **build backend** (never the physics — see the module
            docstring). ``None`` (default) keeps the compiled O(N^2) pair
            kernel. A float switches to the pure-torch binned (cell-list)
            build and is the *requested* perpendicular bin thickness in
            Angstrom: ``0.0`` asks for the automatic ``r_build / 2`` (LAMMPS
            ``nbin_standard`` ``binsize_optimal``), a positive value asks for
            that thickness. The effective thickness is ``w_i / n_i`` with
            ``n_i = max(1, floor(w_i / bin))``, so it is never *smaller* than
            requested and is reported per axis through :attr:`n_bins`. The
            binned path wins on cells wider than about ``5 * r_build / 2`` per
            axis and merely degenerates to an all-pairs search below that.
            Measured 2026-08-09 (torch 2.12.1+cpu, x86_64, N=4096, 48 A cube,
            r_build 5.0, n_bins (19,19,19)): binned 0.037 s vs kernel 0.450 s
            per rebuild at ``OMP_NUM_THREADS=4`` — but 7.2 s vs 1.0 s at the
            node default of 48 threads, where per-op OpenMP region overhead
            dominates the 125-offset loop. The thread count is part of any
            such number; no timing threshold is asserted anywhere.
        device: Device for the buffers; defaults to ``positions``'.

    Raises:
        ValueError: If ``cell`` is not ``(3, 3)`` or is degenerate; if ``skin``,
            ``every`` or ``delay`` is out of domain or ``delay`` is not a
            multiple of ``every``; if ``r_build`` exceeds half the smallest
            perpendicular cell width ``w_i = V / ||a_j x a_k||`` (Angstrom;
            ``w_i = ||a_i||`` for an orthorhombic cell), beyond which the
            minimum-image reduction silently drops pairs that lie inside the
            cutoff; if ``skin`` reaches the dead-edge padding radius
            ``(DEAD_EDGE_CUTOFF_FACTOR - 1) * cutoff``; or if ``bin`` is
            negative or so small that the stencil half-width exceeds
            ``_MAX_STENCIL_HALF_WIDTH`` bins.
    """

    def __init__(
        self,
        *,
        cell: torch.Tensor,
        cutoff: float,
        positions: torch.Tensor,
        skin: float = 0.0,
        every: int = 1,
        delay: int = 0,
        check: bool = True,
        capacity_factor: float = 1.35,
        bin: float | None = None,
        device: torch.device | None = None,
    ) -> None:
        cutoff_f, skin_f = float(cutoff), float(skin)
        if skin_f < 0.0:
            raise ValueError(f"skin must be >= 0 A, got {skin_f} A")
        every_i, delay_i = int(every), int(delay)
        if every_i != every or every_i < 1:
            raise ValueError(f"every must be an integer >= 1 step, got {every!r}")
        if delay_i != delay or delay_i < 0:
            raise ValueError(f"delay must be an integer >= 0 steps, got {delay!r}")
        if delay_i % every_i:
            raise ValueError(
                f"delay {delay_i} steps must be a multiple of every {every_i} steps "
                f"({delay_i} % {every_i} = {delay_i % every_i}); LAMMPS Neighbor::init "
                f"rejects the same pair, because the danger threshold max(every, delay) = "
                f"{max(every_i, delay_i)} would be an ago the gate never permits, leaving "
                "ndanger silently dead."
            )

        min_width = float(_perpendicular_widths(cell).min())
        half_width = 0.5 * min_width
        r_build = cutoff_f + skin_f
        if r_build > half_width:
            raise ValueError(
                f"r_build {r_build} A (cutoff {cutoff_f} A + skin {skin_f} A) exceeds half the "
                f"minimum perpendicular cell width "
                f"({half_width:.3f} A; widths from V/||a_j x a_k||, minimum {min_width:.3f} A); "
                "the kernel's sequential minimum-image reduction would silently drop pairs "
                "inside the cutoff. Use a larger cell, a shorter cutoff, or a thinner skin."
            )
        dead_edge_headroom = (DEAD_EDGE_CUTOFF_FACTOR - 1.0) * cutoff_f
        if skin_f >= dead_edge_headroom:
            raise ValueError(
                f"skin {skin_f} A reaches the dead padding edges: they sit at "
                f"DEAD_EDGE_CUTOFF_FACTOR * cutoff = {DEAD_EDGE_CUTOFF_FACTOR} * {cutoff_f} = "
                f"{DEAD_EDGE_CUTOFF_FACTOR * cutoff_f} A, so the skin must stay below "
                f"(DEAD_EDGE_CUTOFF_FACTOR - 1) * cutoff = {dead_edge_headroom} A or a dead "
                "edge falls inside r_build and is built as a real pair."
            )

        self.cutoff = cutoff_f
        self.skin = skin_f
        self.every = every_i
        self.delay = delay_i
        self.check = bool(check)
        self._r_build = r_build
        self.capacity_factor = float(capacity_factor)
        self._device = device if device is not None else positions.device
        self._dtype = positions.dtype
        self.cell = cell.to(device=self._device, dtype=self._dtype)
        n_atoms = int(positions.shape[0])
        # Built at r_build, not cutoff — and with an explicit half-pair bound, so
        # the task's stale 512 default can never truncate the enlarged radius.
        self._nl = NeighborListTask(
            cutoff=self._r_build,
            max_num_pairs=max(1, n_atoms * (n_atoms - 1) // 2),
            pbc=True,
            symmetry=True,
        )
        #: Requested perpendicular bin thickness in Angstrom, or ``None`` for
        #: the compiled O(N^2) backend. A cost knob, never a physics knob.
        self.bin = None if bin is None else float(bin)
        #: Bins per cell axis ``(n_1, n_2, n_3)``, ``None`` when ``bin is None``
        #: and no grid was derived. The only public window onto the stencil.
        self.n_bins: tuple[int, int, int] | None = None
        if self.bin is not None:
            self._configure_bins(self.bin)

        # Through the dispatch, so the capacity is sized by the backend that
        # will keep refilling the buffers (the two agree, by the equivalence).
        source, target, shifts = self._build_pairs(positions)
        self.capacity = max(1, int(math.ceil(self.capacity_factor * source.numel())))
        self.edge_index = torch.zeros(self.capacity, 2, dtype=torch.long, device=self._device)
        self.shifts = torch.zeros(self.capacity, 3, dtype=self._dtype, device=self._device)
        #: Edges live in ``[0, num_edges)``; the rest are dead. Kept for
        #: diagnostics — the model needs no mask, dead edges self-annihilate.
        self.num_edges = 0
        #: Rebuilds since construction; the initial build is not one of them.
        self.rebuild_count = 0
        #: Steps since the last build (LAMMPS ``ago``); 0 on a fresh list.
        self.ago = 0
        #: Rebuilds that fired at the first permitted opportunity and may
        #: therefore have come too late (LAMMPS "Dangerous builds").
        self.ndanger = 0
        self._half_skin_sq = (0.5 * skin_f) ** 2
        self._wrap_guard_sq = half_width**2
        self._danger_ago = max(every_i, delay_i)  # LAMMPS neighbor.cpp:2488, verbatim
        # Reference configuration of the last build, differenced raw (never
        # minimum-imaged) by ``update``. In place from here on: no per-rebuild
        # allocation, no shape churn.
        self._x_hold = torch.zeros(n_atoms, 3, dtype=self._dtype, device=self._device)
        self._write(source, target, shifts)
        self._hold(positions)

    @property
    def r_build(self) -> float:
        """Build radius ``cutoff + skin`` in Angstrom (derived, never settable).

        The capacity, the kernel's radius and the half-width guard were all
        sized from this at construction, so a writable ``r_build`` could only
        drift out of step with them.
        """
        return self._r_build

    def _compute(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the compiled kernel; returns ``(source, target, shifts)``."""
        pos = positions.detach()
        graph = self._nl.execute({"pos": pos, "cell": self.cell.to(pos.dtype)})
        edge_index = graph["edge_index"]  # (E, 2) — canonical layout throughout
        source, target = edge_index[:, 0], edge_index[:, 1]
        # The kernel returns minimum-image displacements; the model recomputes
        # pos[target]-pos[source] itself, so hand it the periodic remainder.
        shifts = graph["edge_diff"] - (pos[target] - pos[source])
        return source, target, shifts

    def _configure_bins(self, requested: float) -> None:
        """Derive the bin grid and the search stencil from the cell and ``r_build``.

        Called once from ``__init__`` when ``bin`` is not ``None``. The grid
        depends only on the (fixed) cell and the (derived) build radius, so
        nothing here runs again per rebuild; :meth:`to` only carries the two
        tensors it produces to their new device / dtype.

        Per axis ``i``, with the perpendicular widths ``w_i`` in Angstrom
        (:func:`_perpendicular_widths`) and all counts dimensionless:

        1. requested thickness ``b = requested`` in Angstrom, or ``r_build / 2``
           when ``requested == 0.0`` (LAMMPS ``nbin_standard``
           ``binsize_optimal``);
        2. ``n_i = max(1, floor(w_i / b))`` — sized on the **perpendicular**
           width, not the row norm, so a requested thickness means the same
           thing on a sheared cell as on a cube;
        3. effective thickness ``b_i = w_i / n_i`` in Angstrom, never below the
           request except where the ``max(1, ...)`` clamp caught a cell thinner
           than one requested bin;
        4. half-width ``k_i = ceil(r_build / b_i)`` bins, bumped while
           ``k_i * b_i < r_build`` — a float-exactness guard that fires at most
           once, and the completeness relation the whole search rests on (module
           docstring);
        5. offsets ``o_i = unique(arange(-k_i, k_i + 1) mod n_i)``. The
           **distinct residues** are load-bearing: as soon as ``2 k_i + 1 > n_i``
           the raw stencil wraps onto the same bin twice, and every candidate
           pair in it would be emitted twice.

        Sets :attr:`n_bins` (the public diagnostic), ``self._stencil`` — the
        ``(S, 3)`` Cartesian product of the three residue sets, ``S <= (2 *
        _MAX_STENCIL_HALF_WIDTH + 1) ** 3`` — and ``self._inv_cell``, the cached
        ``cell^-1`` the build maps positions to fractional coordinates with.

        Args:
            requested: Requested perpendicular bin thickness in Angstrom;
                ``0.0`` asks for the automatic ``r_build / 2``.

        Raises:
            ValueError: If ``requested`` is negative (``0.0`` is how one asks
                for the automatic size); if any ``k_i`` exceeds
                ``_MAX_STENCIL_HALF_WIDTH``, i.e. the bin is so far below
                ``r_build`` that the stencil loop stops being a search and
                becomes a hang; or — a fail-loud tripwire on step 4 rather than
                a user knob — if the completeness relation ``k_i * b_i >=
                r_build`` fails on any axis, which would be a silently short
                neighbour list.
        """
        if requested < 0.0:
            raise ValueError(
                f"bin must be >= 0 A, got {requested} A: a negative bin thickness has no "
                f"meaning. bin=0.0 selects the automatic r_build / 2 = {0.5 * self._r_build} A "
                "size (LAMMPS nbin_standard), any positive value is an explicit requested "
                "perpendicular bin thickness, and bin=None keeps the compiled O(N^2) backend."
            )
        thickness = requested if requested > 0.0 else 0.5 * self._r_build
        widths = [float(width) for width in _perpendicular_widths(self.cell)]
        counts = [max(1, int(math.floor(width / thickness))) for width in widths]
        effective = [width / count for width, count in zip(widths, counts, strict=True)]
        halves: list[int] = []
        for size in effective:
            half = int(math.ceil(self._r_build / size))
            while half * size < self._r_build:  # float-exactness bump; fires at most once
                half += 1
            halves.append(half)

        if max(halves) > _MAX_STENCIL_HALF_WIDTH:
            raise ValueError(
                f"bin {requested} A gives effective bin thicknesses "
                f"{[round(size, 6) for size in effective]} A, so the stencil half-widths are "
                f"{halves} bins — above the cap of {_MAX_STENCIL_HALF_WIDTH}. The build would "
                f"loop over {math.prod(2 * half + 1 for half in halves)} bin offsets per "
                f"rebuild, against the {(2 * _MAX_STENCIL_HALF_WIDTH + 1) ** 3} the cap allows. "
                f"Pass a larger bin, or bin=0.0 for the automatic "
                f"r_build / 2 = {0.5 * self._r_build} A size."
            )
        for count, size, half in zip(counts, effective, halves, strict=True):
            if half * size < self._r_build:
                raise ValueError(
                    f"incomplete stencil from bin {requested} A: an axis with n_i = {count} "
                    f"bins of b_i = {size} A reaches only k_i * b_i = {half * size} A at "
                    f"half-width k_i = {half}, short of r_build {self._r_build} A. Pairs "
                    "further apart than the stencil are dropped without being measured, so "
                    "this is a tripwire on the derivation above, not a user knob."
                )

        axes = [
            torch.unique(
                torch.arange(-half, half + 1, dtype=torch.long, device=self._device) % count
            )
            for count, half in zip(counts, halves, strict=True)
        ]
        self._stencil = torch.stack(
            [axis.reshape(-1) for axis in torch.meshgrid(*axes, indexing="ij")], dim=-1
        )
        # Inverted in float64 and cast down: a float32 cell inverted in float32
        # loses digits the bin index is then floored from.
        self._inv_cell = torch.linalg.inv(self.cell.to(torch.float64)).to(
            device=self._device, dtype=self.cell.dtype
        )
        self.n_bins = (counts[0], counts[1], counts[2])

    def _build_binned(
        self, positions: torch.Tensor, n_bins: tuple[int, int, int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the pair list from the bin grid; returns ``(source, target, shifts)``.

        The pure-torch O(N) backend behind ``bin=``, with the same return
        contract as :meth:`_compute` so :meth:`_write` — capacity, overflow and
        dead-edge padding — is shared verbatim. Atoms are sorted into the
        :meth:`_configure_bins` grid once, then each of the ``S`` stencil offsets
        gathers one neighbour bin per atom; the Python loop runs over those
        offsets (``S <= 4913``, independent of ``N``) and everything inside it is
        vectorised over all atoms at once. Every tensor is created with an
        explicit ``device=`` / ``dtype=`` taken from the positions, so the path
        runs on CUDA unchanged. Eager, like every other build here.

        Fractional coordinates are wrapped into ``[0, 1)`` **for indexing only**:
        displacements are taken from the unwrapped ``frac`` and the stored
        ``positions``, so a trajectory that has drifted out of the box keeps
        both its coordinates and its shifts (module docstring, "Raw
        displacements, unwrapped positions").

        Args:
            positions: Positions ``(N, 3)`` in Angstrom, wrapped or not. Cast to
                the cell's dtype — the precision the buffers already hold.
            n_bins: Bins per axis, i.e. :attr:`n_bins` narrowed to non-``None``
                by :meth:`_build_pairs`, which is the only caller.

        Returns:
            ``(source, target, shifts)``: atom indices ``(E,)`` per the repo
            edge convention and periodic remainders ``(E, 3)`` in Angstrom, as a
            **full bidirectional** list — each pair appears as ``(s, t, D)`` and
            ``(t, s, -D)``. Edge *order* is not part of the contract (module
            docstring).
        """
        cell = self.cell
        pos = positions.detach().to(cell.dtype)
        device, dtype = pos.device, pos.dtype
        n_atoms = int(pos.shape[0])
        counts_per_axis = torch.tensor(n_bins, dtype=torch.long, device=device)
        strides = torch.tensor(
            [n_bins[1] * n_bins[2], n_bins[2], 1], dtype=torch.long, device=device
        )

        frac = pos @ self._inv_cell
        # Indexing device only — never differenced, never stored.
        wrapped = frac - frac.floor()
        bin_ijk = (wrapped * counts_per_axis.to(dtype)).floor().long()
        # clamp: frac_w can round to exactly 1.0, and a negative index would wrap
        # silently through Python's semantics instead of landing in bin 0.
        bin_ijk = bin_ijk.clamp_(min=0).minimum(counts_per_axis - 1)
        bin_id = (bin_ijk * strides).sum(-1)

        order = torch.argsort(bin_id, stable=True)
        occupancy = torch.bincount(bin_id, minlength=n_bins[0] * n_bins[1] * n_bins[2])
        starts = torch.cumsum(occupancy, 0) - occupancy
        atoms = torch.arange(n_atoms, dtype=torch.long, device=device)
        r_build_sq = self._r_build**2

        sources: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        remainders: list[torch.Tensor] = []
        for offset in self._stencil:
            neighbour_bin = (((bin_ijk + offset) % counts_per_axis) * strides).sum(-1)
            occupied = occupancy[neighbour_bin]
            total = int(occupied.sum())
            if total == 0:
                continue
            # Ragged gather, the molix.data.collate._gather_indices idiom (that
            # one is CPU-pinned for DataLoader workers, so it is followed, not
            # imported): counts -> segment ids -> exclusive cumsum -> row index.
            segment = torch.repeat_interleave(atoms, occupied)
            exclusive = torch.cumsum(occupied, 0) - occupied
            candidate = order[
                starts[neighbour_bin][segment]
                + (torch.arange(total, dtype=torch.long, device=device) - exclusive[segment])
            ]

            half_pair = segment < candidate  # each unordered pair once, no self pairs
            source, target = segment[half_pair], candidate[half_pair]
            if source.numel() == 0:
                continue
            # Minimum image by fractional rounding — exact here because
            # r_build <= min_i w_i / 2 bounds every in-range image by |f_i| <= 1/2.
            fractional = frac[target] - frac[source]
            displacement = (fractional - fractional.round()) @ cell
            distance_sq = (displacement * displacement).sum(-1)
            # 0 < r <= r_build, the compiled kernels' filter verbatim: coincident
            # atoms and pairs separated by exactly one lattice vector are dropped.
            keep = (distance_sq <= r_build_sq) & (distance_sq > 0)
            source, target, displacement = source[keep], target[keep], displacement[keep]
            sources.append(source)
            targets.append(target)
            # Same definition as _compute's edge_diff - (pos[target] - pos[source]),
            # against the stored (unwrapped) positions.
            remainders.append(displacement - (pos[target] - pos[source]))

        if sources:
            half_source, half_target = torch.cat(sources), torch.cat(targets)
            half_shifts = torch.cat(remainders)
        else:  # an isolated system at this radius: no half pairs to expand
            half_source = torch.zeros(0, dtype=torch.long, device=device)
            half_target = torch.zeros(0, dtype=torch.long, device=device)
            half_shifts = torch.zeros(0, 3, dtype=dtype, device=device)
        # Symmetry expansion to the full bidirectional list: the shift flips
        # sign wholesale with the displacement it is the remainder of.
        return (
            torch.cat((half_source, half_target)),
            torch.cat((half_target, half_source)),
            torch.cat((half_shifts, -half_shifts)),
        )

    def _build_pairs(
        self, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route the build to the backend ``bin`` selected; ``(source, target, shifts)``.

        The single seam between the two backends: the constructor's initial
        (capacity-sizing) build and every :meth:`_build_at` go through it, so a
        binned list is binned from the first edge on and its capacity is sized
        by the backend that will keep refilling it.

        Args:
            positions: Positions ``(N, 3)`` in Angstrom to build at.
        """
        grid = self.n_bins
        if grid is None:
            return self._compute(positions)
        return self._build_binned(positions, grid)

    def _dead_shift(self) -> torch.Tensor:
        """A displacement long enough that every cutoff envelope evaluates to 0."""
        return torch.tensor(
            [DEAD_EDGE_CUTOFF_FACTOR * self.cutoff, 0.0, 0.0],
            dtype=self._dtype,
            device=self._device,
        )

    def _write(self, source: torch.Tensor, target: torch.Tensor, shifts: torch.Tensor) -> None:
        """Fill the buffers in place; pad the tail with dead edges."""
        n = int(source.numel())
        if n > self.capacity:
            raise RuntimeError(
                f"neighbour-list overflow: {n} edges > capacity {self.capacity}. Raise "
                f"capacity_factor (currently {self.capacity_factor})."
            )
        # In-place so the compiled graph keeps seeing the same tensors.
        self.edge_index[:n, 0] = source
        self.edge_index[:n, 1] = target
        self.shifts[:n] = shifts.to(self._dtype)
        self.edge_index[n:] = 0
        self.shifts[n:] = self._dead_shift()
        self.num_edges = n

    def _hold(self, positions: torch.Tensor) -> None:
        """Adopt ``positions`` as the reference of the current build.

        Restarts the ``ago`` clock and refreshes ``_x_hold`` in place (cast to
        the buffer dtype, so a list that was moved with :meth:`to` keeps
        differencing in one precision).
        """
        self.ago = 0
        self._x_hold.copy_(positions.detach().to(self._x_hold.dtype))

    def _positions_from(self, batch: TensorDictBase) -> torch.Tensor:
        """Validate that ``batch`` describes *this* system; return its positions.

        Shared by :meth:`build` and the batch arm of :meth:`update`, so the
        per-step path cannot drift from the bind-time path. The checks are
        metadata compares — key presence, shape, ``device``, ``dtype`` — and so
        cost the hot loop nothing measurable; the single value read is the
        optional cell comparison, and an MD working batch carries no cell.

        Args:
            batch: Batch ``TensorDict`` with positions at ``("atoms", "pos")``
                in Angstrom, optionally carrying ``("graphs", "cell")``.

        Returns:
            The batch's positions ``(N, 3)`` in Angstrom — the *same* tensor,
            never a cast copy.

        Raises:
            ValueError: If ``("atoms", "pos")`` is missing; if its shape is not
                ``(N, 3)`` for the ``N`` this list was constructed with; if its
                device or dtype differ from the list's buffers (the owner
                casts, via :meth:`to` — no silent cast here); or if
                ``("graphs", "cell")`` is present and is neither ``(3, 3)`` nor
                ``(1, 3, 3)`` equal to the constructor cell within ``1e-8`` A
                (``rtol=0``) — loose enough to survive a float32 template
                round-trip, far below any physically meaningful difference. The
                constructor cell stays the **owner**: the batch's copy is
                checked, never adopted, because every frozen shift in the
                buffers is a lattice vector of *that* cell.
        """
        if ("atoms", "pos") not in batch.keys(include_nested=True):
            present = sorted(
                "/".join(key) if isinstance(key, tuple) else str(key)
                for key in batch.keys(include_nested=True)
            )
            raise ValueError(
                "the batch carries no ('atoms', 'pos'): a neighbour list builds at atom "
                f"positions, and this batch holds {present}. Post-collate batches nest "
                "positions under the 'atoms' namespace (two-tier data contract); a flat "
                "sample dict is the other tier and is not what this binds into."
            )
        pos = batch["atoms", "pos"]
        if not isinstance(pos, torch.Tensor):
            raise ValueError(
                "('atoms', 'pos') must be a positions tensor (N, 3) in Angstrom, but the "
                f"batch holds a {type(pos).__name__} there — a nested namespace, not the "
                "leaf this list builds at."
            )
        n_atoms = int(self._x_hold.shape[0])
        if tuple(pos.shape) != (n_atoms, 3):
            raise ValueError(
                f"positions must have shape ({n_atoms}, 3) in Angstrom — this list was "
                f"constructed for {n_atoms} atoms — but the batch's ('atoms', 'pos') has "
                f"shape {tuple(pos.shape)}. A different atom count is a different system: "
                "the capacity was sized for the original and the displacement reference "
                "_x_hold has its shape."
            )
        # _x_hold, not _device/_dtype: it is the tensor these positions are
        # actually differenced against, and it tracks every to() exactly.
        if pos.device != self._x_hold.device or pos.dtype != self._x_hold.dtype:
            raise ValueError(
                f"positions are on {pos.device} in {pos.dtype}, but this list's buffers are "
                f"on {self._x_hold.device} in {self._x_hold.dtype}. Nothing is cast here: a "
                "silently promoted difference against _x_hold is a mixed-precision "
                "comparison nobody asked for, and a cross-device index is worse. The owner "
                "casts — call NeighborList.to(device, dtype) first."
            )
        if ("graphs", "cell") not in batch.keys(include_nested=True):
            return pos
        cell = batch["graphs", "cell"]
        if not isinstance(cell, torch.Tensor):
            raise ValueError(
                "('graphs', 'cell') must be a cell tensor (3, 3) — or (1, 3, 3) for a "
                f"single-system batch — in Angstrom, but the batch holds a "
                f"{type(cell).__name__} there."
            )
        vectors = cell
        if vectors.dim() == 3:
            if vectors.shape[0] != 1:
                raise ValueError(
                    f"('graphs', 'cell') has a leading batch dimension of {vectors.shape[0]} "
                    f"(shape {tuple(vectors.shape)}), but this neighbour list is "
                    "single-system: one cell, one displacement reference, one capacity. "
                    "Multi-system batched MD is not supported here rather than silently "
                    "reduced to the first cell."
                )
            vectors = vectors[0]
        if tuple(vectors.shape) != (3, 3):
            raise ValueError(
                "('graphs', 'cell') must be (3, 3), or (1, 3, 3) for a single-system batch, "
                f"in Angstrom; got shape {tuple(cell.shape)}."
            )
        reference = self.cell.detach().to(torch.float64)
        candidate = vectors.detach().to(device=reference.device, dtype=torch.float64)
        if not torch.allclose(candidate, reference, rtol=0.0, atol=1e-8):
            raise ValueError(
                f"the batch's ('graphs', 'cell')\n{candidate.tolist()}\ndisagrees with the "
                f"cell this list was built against\n{reference.tolist()}\n(Angstrom, tolerance "
                "atol=1e-8, rtol=0). The stored shifts are lattice vectors of the "
                "constructor's cell, so adopting a different one would leave every periodic "
                "remainder silently wrong; construct a new list instead."
            )
        return pos

    def _build_at(self, positions: torch.Tensor) -> None:
        """Recompute and rewrite the buffers at ``positions``, restarting ``ago``.

        The shared body of :meth:`rebuild` (which adds the counter increment)
        and :meth:`build` (which adds the batch validation and the bind).
        Deliberately does *not* touch :attr:`rebuild_count`: only the caller
        knows whether this build is a rebuild driven by the run.

        Args:
            positions: Positions ``(N, 3)`` in Angstrom to build at.
        """
        source, target, shifts = self._build_pairs(positions)
        self._write(source, target, shifts)
        self._hold(positions)

    def rebuild(self, positions: torch.Tensor) -> None:
        """Recompute the neighbour list at ``positions`` (eager, outside any graph).

        The unconditional primitive: it builds whatever the policy would have
        said. ``ago`` restarts here, so a forced rebuild also re-phases the
        ``every`` / ``delay`` schedule :meth:`update` runs on.

        Args:
            positions: Positions ``(N, 3)`` in Angstrom to build at.
        """
        self._build_at(positions)
        self.rebuild_count += 1

    def build(self, batch: TensorDict) -> TensorDict:
        """Rebuild at the batch's positions and bind the live buffers into it.

        The TensorDict-side forced build: it validates that ``batch`` describes
        the system this list was constructed for, rebuilds at
        ``batch["atoms", "pos"]`` (Angstrom), then writes ``batch["edges"]`` as
        a ``TensorDict`` of :attr:`edge_index` ``(capacity, 2)`` and
        :attr:`shifts` ``(capacity, 3)`` held **by reference**, at
        ``batch_size=[capacity]``. Every later in-place :meth:`rebuild` /
        :meth:`update` is therefore visible to whatever reads that batch, with
        no shape change and no re-binding — which is what keeps a compiled or
        graph-captured force path valid across a rebuild.

        **The list owns ``edges`` once bound.** The namespace is replaced
        wholesale, not merged: a pre-existing ``edge_diff`` / ``edge_dist``
        pair would be consumed straight through as a *value* by a potential and
        freeze the PES, and a surviving shorter ``edge_index`` would disagree
        with the capacity.

        This is *not* counted as a :attr:`rebuild_count` rebuild — it is a
        binding operation, and it runs again on every ``.to()`` re-sync, so
        counting it would make a dtype cast look like physics. It does restart
        the policy clock (:attr:`ago` back to 0, ``_x_hold`` refreshed): the
        buffers are fresh here.

        Warning:
            ``build`` returns the batch so it composes with the repo's
            ``forward(td) -> td`` convention (``potential(nl.build(batch))``) —
            **not** so the policy call can be chained.
            ``nl.build(batch).update(batch)`` parses, but that ``.update`` is
            ``TensorDict.update``: it merges the batch into itself and never
            touches this list. The idiom is two statements::

                nl.build(batch)      # once, and after every .to()
                ...
                nl.update(batch)     # per step — NeighborList.update

        Args:
            batch: Batch ``TensorDict`` carrying ``("atoms", "pos")``
                ``(N, 3)`` in Angstrom and optionally ``("graphs", "cell")``
                ``(3, 3)`` or ``(1, 3, 3)`` in Angstrom, which is validated
                against the constructor cell and never adopted.

        Returns:
            The **same** ``batch`` object, with ``batch["edges"]`` bound.

        Raises:
            ValueError: Per :meth:`_positions_from` — missing positions, a
                different atom count, a device/dtype the owner has not cast, or
                a cell that is not this list's.
            RuntimeError: If the rebuilt edge count overflows the capacity.
        """
        positions = self._positions_from(batch)
        self._build_at(positions)
        batch["edges"] = TensorDict(
            {"edge_index": self.edge_index, "shifts": self.shifts},
            batch_size=[self.capacity],
        )
        return batch

    def update(self, positions: TensorDict | torch.Tensor) -> bool:
        """Rebuild at ``positions`` if the ``every``/``delay``/``check`` gate says so.

        Call once per force evaluation, at the positions being evaluated. The
        gate is ``Neighbor::decide`` verbatim: ``ago`` is incremented, a rebuild
        is *permitted* only when ``ago >= delay`` **and** ``ago % every == 0``
        (conjunctive), and — with ``check`` — happens only when the largest raw
        displacement since the last build exceeds ``skin/2``. Crossing that
        bound at the first permitted opportunity ``ago == max(every, delay)``
        increments :attr:`ndanger`.

        **One method, two input types.** A raw ``(N, 3)`` tensor is the MD hot
        path; a batch ``TensorDict`` is the bound path, dispatched by an
        ``isinstance`` test against ``TensorDictBase`` (so lazy / stacked
        batches dispatch too) and reduced to its positions by the same
        validation :meth:`build` runs — a batch whose ``pos`` was silently
        re-cast therefore fails loud here instead of promoting against
        ``_x_hold`` for the rest of the run. Everything after the dispatch is
        identical: same decisions, same :attr:`ago` / :attr:`rebuild_count` /
        :attr:`ndanger` bookkeeping, same buffers. There is no ``update_td`` /
        ``update_pos`` pair — the policy state lives behind one door.

        Note:
            ``nl.update(batch)`` is :class:`NeighborList`'s ``update``;
            ``batch.update(...)`` is ``TensorDict.update``, a merge that never
            touches this list. Keep the bind and the step as two statements
            (see :meth:`build`).

        Args:
            positions: Positions ``(N, 3)`` in Angstrom, **unwrapped** (see
                Raises and the module docstring), or the batch
                ``TensorDict`` carrying them at ``("atoms", "pos")``.

        Returns:
            ``True`` if the list was rebuilt, ``False`` if the frozen list is
            still valid (or the gate simply did not permit a build this step).

        Raises:
            ValueError: If a batch was passed and it does not describe this
                system — see :meth:`_positions_from`.
            RuntimeError: If the largest displacement since the last build
                reaches half the smallest perpendicular cell width — mid-run
                wrapping, a changed cell, or a blown-up trajectory. Not raised
                under ``check=False``, which skips the displacement branch
                entirely.
        """
        if isinstance(positions, TensorDictBase):
            positions = self._positions_from(positions)
        self.ago += 1
        if self.ago < self.delay or self.ago % self.every:
            return False  # not a permitted opportunity
        if not self.check:
            self.rebuild(positions)
            return True
        # Raw difference, never a minimum image: min-imaging would clamp a
        # genuine > L/2 excursion and suppress the rebuild it should force.
        max_d2 = float(((positions.detach() - self._x_hold) ** 2).sum(-1).max())
        if max_d2 >= self._wrap_guard_sq:
            raise RuntimeError(
                f"positions are no longer unwrapped: the largest displacement since the last "
                f"build is {math.sqrt(max_d2):.3f} A, at or beyond half the minimum "
                f"perpendicular cell width ({math.sqrt(self._wrap_guard_sq):.3f} A). The frozen "
                "periodic shifts and this raw displacement test hold only for continuously "
                "drifting coordinates, so a jump this large means the positions were wrapped "
                "mid-run, the cell changed, or the trajectory blew up. All three are fatal; "
                "none is recoverable by rebuilding."
            )
        if max_d2 > self._half_skin_sq:  # strict: exactly skin/2 is still complete
            if self.ago == self._danger_ago:
                self.ndanger += 1
            self.rebuild(positions)
            return True
        return False

    def to(
        self,
        device: torch.device | str | torch.dtype | None = None,
        dtype: torch.dtype | None = None,
    ) -> "NeighborList":
        """Move / cast the buffers, mirroring ``Tensor.to`` semantics.

        Accepts ``nl.to("cuda")``, ``nl.to(torch.float64)`` and
        ``nl.to(device, dtype)`` alike. ``_x_hold`` travels with the rest: a
        displacement reference left behind in the old dtype would silently
        promote the next :meth:`update` comparison instead of failing.

        Warning:
            **This severs any :meth:`build` binding.** The move rebinds
            :attr:`edge_index` / :attr:`shifts` to *new* tensors, so a batch
            bound beforehand keeps pointing at the old ones and would freeze at
            the pre-cast neighbour set. Re-binding from inside ``to`` is
            deliberately not done — it would make the list hold a reference to
            a batch it does not own — so the **owner** re-binds: call
            ``nl.build(batch)`` after the cast (this is exactly what
            ``PeriodicPotentialForceField._apply`` does).
        """
        if isinstance(device, torch.dtype):
            if dtype is not None:
                raise TypeError("dtype given twice")
            device, dtype = None, device
        if device is not None:
            self._device = torch.device(device)
            self.edge_index = self.edge_index.to(self._device)
            self.shifts = self.shifts.to(self._device)
            self.cell = self.cell.to(self._device)
            self._x_hold = self._x_hold.to(self._device)
        if dtype is not None:
            self._dtype = dtype
            self.shifts = self.shifts.to(dtype)
            self.cell = self.cell.to(dtype)
            self._x_hold = self._x_hold.to(dtype)
        if self.n_bins is not None:
            # The grid itself is a property of the cell and r_build, so only its
            # two tensors move: the stencil is integer offsets, and the inverse
            # cell is re-derived (in float64, then cast) rather than converted,
            # so a float32 hop does not compound its own rounding.
            self._stencil = self._stencil.to(self._device)
            self._inv_cell = torch.linalg.inv(self.cell.to(torch.float64)).to(
                device=self._device, dtype=self.cell.dtype
            )
        return self
