"""Rebuilding, fixed-capacity neighbour list for production MD.

A frozen neighbour list is only valid while no atom moves far enough to change
its neighbour set — tens of steps for liquid water, not the millions a
production trajectory needs. This module rebuilds it on a fixed step cadence
while keeping **every tensor shape constant**, which is what lets the force
evaluation stay inside a CUDA graph across the whole run.

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
``V = |det(cell)|``. So ``r_cut`` must not exceed ``min_i w_i / 2`` —
:class:`NeighborList` refuses to construct otherwise rather than
silently dropping pairs that are inside the cutoff. For an orthorhombic cell
``w_i = ||a_i||``, i.e. the familiar half-shortest-cell-vector bound.

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
    """

    edge_index: torch.Tensor
    shifts: torch.Tensor
    num_edges: int
    capacity: int

    def rebuild(self, positions: torch.Tensor) -> None:
        """Recompute the neighbour list at ``positions``, in place."""
        ...

    def to(
        self,
        device: torch.device | str | torch.dtype | None = None,
        dtype: torch.dtype | None = None,
    ) -> "NeighborStrategy":
        """Move / cast the buffers, mirroring ``Tensor.to`` semantics."""
        ...


class NeighborList:
    """Minimum-image neighbour list rebuilt into fixed-capacity buffers.

    Not an ``nn.Module``: it owns plain buffers and a rebuild policy, and is held
    by a :class:`~molix.md.forcefield.ForceField`. Keeping it out of the module
    tree also keeps it out of ``state_dict``, where a per-run neighbour list has
    no business — the owning force field forwards device/dtype changes through
    :meth:`to` instead (see ``PeriodicPotentialForceField._apply``).

    Args:
        cell: Cell vectors ``(3, 3)``.
        cutoff: Model cutoff ``r_cut`` in Angstrom.
        positions: Initial positions ``(N, 3)``, used to size the capacity.
        capacity_factor: Buffer capacity as a multiple of the initial edge
            count. Density fluctuations grow the edge count during a run;
            overflow raises rather than truncating.
        device: Device for the buffers; defaults to ``positions``'.

    Raises:
        ValueError: If ``cell`` is not ``(3, 3)`` or is degenerate, or if
            ``cutoff`` exceeds half the smallest perpendicular cell width
            ``w_i = V / ||a_j x a_k||`` (Angstrom; ``w_i = ||a_i||`` for an
            orthorhombic cell), beyond which the minimum-image reduction
            silently drops pairs that lie inside the cutoff.
    """

    def __init__(
        self,
        *,
        cell: torch.Tensor,
        cutoff: float,
        positions: torch.Tensor,
        capacity_factor: float = 1.35,
        device: torch.device | None = None,
    ) -> None:
        min_width = _min_perpendicular_width(cell)
        half_width = 0.5 * min_width
        if cutoff > half_width:
            raise ValueError(
                f"cutoff {cutoff} A exceeds half the minimum perpendicular cell width "
                f"({half_width:.3f} A; widths from V/||a_j x a_k||, minimum {min_width:.3f} A); "
                "the kernel's sequential minimum-image reduction would silently drop pairs "
                "inside the cutoff. Use a larger cell or a shorter cutoff."
            )
        self.cutoff = float(cutoff)
        self.capacity_factor = float(capacity_factor)
        self._device = device if device is not None else positions.device
        self._dtype = positions.dtype
        self.cell = cell.to(device=self._device, dtype=self._dtype)
        self._nl = NeighborListTask(cutoff=self.cutoff, pbc=True, symmetry=True)

        source, target, shifts = self._compute(positions)
        self.capacity = max(1, int(math.ceil(self.capacity_factor * source.numel())))
        self.edge_index = torch.zeros(self.capacity, 2, dtype=torch.long, device=self._device)
        self.shifts = torch.zeros(self.capacity, 3, dtype=self._dtype, device=self._device)
        #: Edges live in ``[0, num_edges)``; the rest are dead. Kept for
        #: diagnostics — the model needs no mask, dead edges self-annihilate.
        self.num_edges = 0
        self.rebuild_count = 0
        self._write(source, target, shifts)

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

    def rebuild(self, positions: torch.Tensor) -> None:
        """Recompute the neighbour list at ``positions`` (eager, outside any graph)."""
        source, target, shifts = self._compute(positions)
        self._write(source, target, shifts)
        self.rebuild_count += 1

    def to(
        self,
        device: torch.device | str | torch.dtype | None = None,
        dtype: torch.dtype | None = None,
    ) -> "NeighborList":
        """Move / cast the buffers, mirroring ``Tensor.to`` semantics.

        Accepts ``nl.to("cuda")``, ``nl.to(torch.float64)`` and
        ``nl.to(device, dtype)`` alike.
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
        if dtype is not None:
            self._dtype = dtype
            self.shifts = self.shifts.to(dtype)
            self.cell = self.cell.to(dtype)
        return self
