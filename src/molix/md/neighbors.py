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
neighbour kernel, so ``r_cut`` must not exceed half the shortest cell vector —
:class:`PeriodicNeighborList` refuses to construct otherwise rather than
silently missing periodic images.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import torch

# The one owner of kernel-output normalisation (pbc handling, NaN-padding
# strip, symmetry expansion, edge-sign convention); reimplementing that here
# against the raw ``molix.F.locality`` kernel would fork it.
from molix.data.tasks.neighbor import NeighborList
from molix.units import DEAD_EDGE_CUTOFF_FACTOR


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


class PeriodicNeighborList:
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
        ValueError: If ``cutoff`` exceeds half the shortest cell vector, which
            would make the minimum-image convention miss periodic images.
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
        half_box = 0.5 * float(torch.linalg.norm(cell, dim=-1).min())
        if cutoff > half_box:
            raise ValueError(
                f"cutoff {cutoff} A exceeds half the shortest cell vector ({half_box:.3f} A); "
                "the minimum-image neighbour list would miss periodic images. Use a larger "
                "cell or a shorter cutoff."
            )
        self.cutoff = float(cutoff)
        self.capacity_factor = float(capacity_factor)
        self._device = device if device is not None else positions.device
        self._dtype = positions.dtype
        self.cell = cell.to(device=self._device, dtype=self._dtype)
        self._nl = NeighborList(cutoff=self.cutoff, pbc=True, symmetry=True)

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
    ) -> "PeriodicNeighborList":
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
