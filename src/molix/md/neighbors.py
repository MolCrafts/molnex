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

References:
    LAMMPS ``neigh_modify`` documentation —
    https://docs.lammps.org/neigh_modify.html — and ``lammps/lammps`` develop
    ``src/neighbor.cpp`` (``Neighbor::decide``, ``Neighbor::check_distance``,
    ``Neighbor::init``) / ``src/verlet.cpp``.

    K. Nordlund, *Introduction to molecular dynamics simulations*, lecture 3,
    https://www.mv.helsinki.fi/home/knordlun/moldyn/lecture03.pdf — the open,
    directly re-verified source for the two-atom criterion above.

    L. Verlet, *Phys. Rev.* **159**, 98 (1967),
    https://doi.org/10.1103/PhysRev.159.98 — the original neighbour list.

    B. Quentrec & C. Brot, *J. Comput. Phys.* **13**, 430 (1973),
    https://doi.org/10.1016/0021-9991(73)90046-6 — cell/skin refinement.

    Caveat: the Verlet 1967 and Quentrec & Brot 1973 texts are paywalled and
    were **not** re-verified here; they are cited for attribution only. Every
    equation above is verified against the Nordlund notes and the LAMMPS source.

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

# The one owner of kernel-output normalisation (pbc handling, NaN-padding
# strip, symmetry expansion, edge-sign convention); reimplementing that here
# against the raw ``molix.F.locality`` kernel would fork it. Aliased because
# the MD buffer owner defined below is *also* named ``NeighborList`` and would
# otherwise shadow its own dependency — the class would then call itself.
from molix.data.tasks.neighbor import NeighborList as NeighborListTask
from molix.units import DEAD_EDGE_CUTOFF_FACTOR


def _min_perpendicular_width(cell: torch.Tensor) -> float:
    """Smallest distance between two opposite faces of a periodic cell.

    For cell vectors ``a_1, a_2, a_3`` — the **rows** of ``cell`` — the width
    perpendicular to the face spanned by ``a_j`` and ``a_k`` is
    ``w_i = V / ||a_j x a_k||`` with ``V = |det(cell)|``, so the minimum is
    ``V / max_i ||a_j x a_k||``: one division instead of three. For an
    orthorhombic cell ``||a_j x a_k|| = ||a_j||*||a_k||`` and
    ``V = ||a_1||*||a_2||*||a_3||``, hence ``w_i = ||a_i||``.

    Evaluated in ``float64`` so a ``float32`` cell cannot jitter an
    accept/reject decision taken right at the bound.

    Args:
        cell: Cell vectors ``(3, 3)`` in Angstrom, one vector per row.

    Returns:
        The smallest perpendicular width ``min_i w_i`` in Angstrom.

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
    return volume / max_area


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

    def update(self, positions: torch.Tensor) -> bool:
        """Rebuild at ``positions`` if the strategy's policy says so.

        Called once per force evaluation. Returns whether a rebuild happened.
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
    :meth:`update` applies the ``every`` / ``delay`` / ``check`` policy. See the
    module docstring for the half-skin criterion, the unwrapped-positions
    invariant and what ``check=False`` costs.

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
        device: Device for the buffers; defaults to ``positions``'.

    Raises:
        ValueError: If ``cell`` is not ``(3, 3)`` or is degenerate; if ``skin``,
            ``every`` or ``delay`` is out of domain or ``delay`` is not a
            multiple of ``every``; if ``r_build`` exceeds half the smallest
            perpendicular cell width ``w_i = V / ||a_j x a_k||`` (Angstrom;
            ``w_i = ||a_i||`` for an orthorhombic cell), beyond which the
            minimum-image reduction silently drops pairs that lie inside the
            cutoff; or if ``skin`` reaches the dead-edge padding radius
            ``(DEAD_EDGE_CUTOFF_FACTOR - 1) * cutoff``.
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

        min_width = _min_perpendicular_width(cell)
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

        source, target, shifts = self._compute(positions)
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

    def _compute(self, positions: torch.Tensor):
        """Run the compiled kernel; returns ``(source, target, shifts)``."""
        pos = positions.detach()
        graph = self._nl.execute({"pos": pos, "cell": self.cell.to(pos.dtype)})
        edge_index = graph["edge_index"]  # (E, 2) — canonical layout throughout
        source, target = edge_index[:, 0], edge_index[:, 1]
        # The kernel returns minimum-image displacements; the model recomputes
        # pos[target]-pos[source] itself, so hand it the periodic remainder.
        shifts = graph["edge_diff"] - (pos[target] - pos[source])
        return source, target, shifts

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

    def rebuild(self, positions: torch.Tensor) -> None:
        """Recompute the neighbour list at ``positions`` (eager, outside any graph).

        The unconditional primitive: it builds whatever the policy would have
        said. ``ago`` restarts here, so a forced rebuild also re-phases the
        ``every`` / ``delay`` schedule :meth:`update` runs on.

        Args:
            positions: Positions ``(N, 3)`` in Angstrom to build at.
        """
        source, target, shifts = self._compute(positions)
        self._write(source, target, shifts)
        self.rebuild_count += 1
        self._hold(positions)

    def update(self, positions: torch.Tensor) -> bool:
        """Rebuild at ``positions`` if the ``every``/``delay``/``check`` gate says so.

        Call once per force evaluation, at the positions being evaluated. The
        gate is ``Neighbor::decide`` verbatim: ``ago`` is incremented, a rebuild
        is *permitted* only when ``ago >= delay`` **and** ``ago % every == 0``
        (conjunctive), and — with ``check`` — happens only when the largest raw
        displacement since the last build exceeds ``skin/2``. Crossing that
        bound at the first permitted opportunity ``ago == max(every, delay)``
        increments :attr:`ndanger`.

        Args:
            positions: Positions ``(N, 3)`` in Angstrom, **unwrapped** (see
                Raises and the module docstring).

        Returns:
            ``True`` if the list was rebuilt, ``False`` if the frozen list is
            still valid (or the gate simply did not permit a build this step).

        Raises:
            RuntimeError: If the largest displacement since the last build
                reaches half the smallest perpendicular cell width — mid-run
                wrapping, a changed cell, or a blown-up trajectory. Not raised
                under ``check=False``, which skips the displacement branch
                entirely.
        """
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
        return self
