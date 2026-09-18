"""Mesh interpolation for particle-mesh methods (Lagrange and P3M)."""

from typing import NamedTuple, Optional

import torch

from molpot.potentials.elec.lib.kvectors import MeshShape


class MeshStencil(NamedTuple):
    """Where each particle lands on the mesh, and with what weight.

    The value :meth:`MeshInterpolator.compute_weights` returns and both
    transfer methods — :meth:`MeshInterpolator.points_to_mesh` and
    :meth:`MeshInterpolator.mesh_to_points` — take back: one stencil of
    ``interpolation_nodes**3`` mesh points per particle.

    A value rather than state on the interpolator, because every field is a
    function of the positions and the cell of **one** call: dynamo treats
    leaf-derived tensors stored on a module as leaked graph outputs, and under
    a derivative (forces, virial) they require grad, so every such store broke
    the traced graph. Passed along, the stencil lives and dies inside the one
    graph that made it.

    Attributes:
        weights: Assignment weight of every stencil point
            ``(interpolation_nodes**3, n_atoms)`` — the product of the three
            1-D weights, summing to one per particle.
        x_indices: Mesh index along ``x`` of every stencil point
            ``(interpolation_nodes**3, n_atoms)``, wrapped into the mesh.
        y_indices: The same along ``y``.
        z_indices: The same along ``z``.
    """

    weights: torch.Tensor
    x_indices: torch.Tensor
    y_indices: torch.Tensor
    z_indices: torch.Tensor


class MeshInterpolator(torch.nn.Module):
    """Interpolation between particles and a regular 3D mesh.

    Handles both forward interpolation (particles → mesh) and backward
    interpolation (mesh → particles) using either Lagrange or P3M
    cardinal B-spline assignment functions.

    Nothing derived from a cell or from positions is kept on the instance:
    :meth:`compute_weights` returns a :class:`MeshStencil` that the two
    transfer methods take back, and :meth:`get_mesh_xyz` takes the cell it
    lays the grid on. What the instance holds is metadata (mesh shape, dtype,
    device) and the leaf-independent stencil offsets.

    The mesh shape arrives in one of two modes, and they trace differently:

    * **frozen** — ``ns_mesh`` given as ``tuple[int, int, int]``, the shape a
      caller fixed ahead of time
      (:meth:`molpot.potentials.elec.calculators.pme.PMECalculator.bind`).
      The grid counts are host-side ints, so :meth:`get_mesh_xyz` and
      :meth:`points_to_mesh` size their tensors without reading anything back
      from the device and the interpolation traces as a single graph.
    * **live** — ``ns_mesh`` given as a ``(3,)`` tensor, sized off the cell.
      The same two methods then read the counts back to the host, which is a
      synchronisation and breaks a compiled graph. This mode is for eager
      callers that resize their mesh per cell.

    :attr:`ns_mesh` is the ``(3,)`` tensor form in both modes.

    Args:
        cell: Unit cell matrix ``(3, 3)`` in Å, rows being the lattice vectors
            — read here for its dtype and device only. The lattice itself is
            an argument of :meth:`compute_weights` and :meth:`get_mesh_xyz`
            and is never stored.
        ns_mesh: Mesh dimensions ``(3,)`` as a tensor, or three Python ints —
            see the two modes above.
        interpolation_nodes: Number of mesh points per axis each particle is
            spread over (3-7 for Lagrange, 1-5 for P3M). More nodes mean a
            smoother, more accurate spread at a higher cost.
        method: ``"Lagrange"`` or ``"P3M"`` — which family of assignment
            functions weights those nodes.
    """

    def __init__(
        self,
        cell: torch.Tensor,
        ns_mesh: torch.Tensor | tuple[int, int, int],
        interpolation_nodes: int,
        method: str,
    ):
        super().__init__()

        if method == "Lagrange":
            if interpolation_nodes not in [3, 4, 5, 6, 7]:
                raise ValueError(
                    f"`interpolation_nodes` is {interpolation_nodes} but only values "
                    f"from 3 to 7 for method 'Lagrange' are allowed"
                )
        elif method == "P3M":
            if interpolation_nodes not in [1, 2, 3, 4, 5]:
                raise ValueError(
                    f"`interpolation_nodes` is {interpolation_nodes} but only values "
                    "from 1 to 5 for method 'P3M' are allowed"
                )
        else:
            raise ValueError(f"method '{method}' is not supported. Choose from 'Lagrange' or 'P3M'")

        self.method: str = method
        self.interpolation_nodes: int = interpolation_nodes

        self._ns_shape: MeshShape = MeshShape(ns_mesh, cell.device)
        self._dtype: torch.dtype = cell.dtype
        self._device: torch.device = cell.device
        # Stencil offsets ``{0..nodes-1}^3``, flattened: which of the
        # ``nodes**3`` mesh points around a particle each stencil entry is.
        # Leaf-independent constants — built once, moved only by ``update``
        # when the device changes, never per call.
        self.x_shifts, self.y_shifts, self.z_shifts = (
            axis.flatten()
            for axis in torch.meshgrid(
                *(torch.arange(interpolation_nodes, device=cell.device),) * 3, indexing="ij"
            )
        )
        self.update(cell, ns_mesh)

    @property
    def ns_mesh(self) -> torch.Tensor:
        """Mesh dimensions ``(3,)`` as a long tensor, in either mode.

        In frozen mode this is built once from the ints and reused; the ints
        themselves are what the grid-sizing code reads.
        """
        return self._ns_shape.tensor

    def update(
        self,
        cell: Optional[torch.Tensor] = None,
        ns_mesh: Optional[torch.Tensor | tuple[int, int, int]] = None,
    ) -> None:
        """Adopt the dtype and device of ``cell`` and/or a new mesh resolution.

        Only Python-level metadata changes hands here: the cell's dtype and
        device, and the mesh shape. The cell itself is not kept — it is a
        function of the strain leaf under a virial, and a leaf-derived tensor
        stored on a module is a leaked graph output to dynamo. It is an
        argument of every per-call method instead.

        Args:
            cell: Unit cell matrix ``(3, 3)`` in Å, rows being the lattice
                vectors — read for its dtype and device.
            ns_mesh: Mesh dimensions ``(3,)`` as a tensor (live mode), or three
                Python ints (frozen mode) — see the class docstring. The mode
                is whichever this call was given; a frozen shape repeated
                unchanged reuses the tensor form it already built.

        Raises:
            ValueError: ``cell`` is not ``(3, 3)``, or ``cell`` and the mesh
                shape live on different devices.
        """
        if cell is not None:
            if cell.shape != (3, 3):
                raise ValueError(f"cell of shape {list(cell.shape)} should be of shape (3, 3)")
            if cell.device != self._device:
                self.x_shifts = self.x_shifts.to(cell.device)
                self.y_shifts = self.y_shifts.to(cell.device)
                self.z_shifts = self.z_shifts.to(cell.device)
            self._dtype = cell.dtype
            self._device = cell.device

        if ns_mesh is not None:
            self._ns_shape.update(ns_mesh, self._device)

        if self._device != self.ns_mesh.device:
            raise ValueError(
                "`cell` and `ns_mesh` are on different devices, got "
                f"{self._device} and {self.ns_mesh.device}"
            )

    def get_mesh_xyz(self, cell: torch.Tensor) -> torch.Tensor:
        """Return Cartesian positions of all mesh points.

        Args:
            cell: Unit cell matrix ``(3, 3)`` in Å, rows being the lattice
                vectors, that the grid is laid on.

        Returns:
            Mesh positions ``(nx, ny, nz, 3)`` in Å.
        """
        nx, ny, nz = self._ns_shape.counts

        grid_scaled = torch.stack(
            torch.meshgrid(
                torch.arange(nx, dtype=cell.dtype, device=cell.device) / nx,
                torch.arange(ny, dtype=cell.dtype, device=cell.device) / ny,
                torch.arange(nz, dtype=cell.dtype, device=cell.device) / nz,
                indexing="ij",
            ),
            dim=-1,
        )
        return torch.matmul(grid_scaled, cell)

    def _compute_1d_weights(self, x: torch.Tensor) -> torch.Tensor:
        if self.method == "Lagrange":
            return self._compute_1d_weights_Lagrange(x)
        if self.method == "P3M":
            return self._compute_1d_weights_P3M(x)
        raise ValueError("Only `method` `Lagrange` and `P3M` are allowed")

    def _compute_1d_weights_P3M(self, x: torch.Tensor) -> torch.Tensor:
        """P3M cardinal B-spline weights.

        Args:
            x: Relative positions ``(interpolation_nodes, n_atoms)``.

        Returns:
            Weights ``(interpolation_nodes, n_atoms)``.
        """
        if self.interpolation_nodes == 1:
            return torch.ones((1, x.shape[0], x.shape[1]), dtype=self._dtype, device=self._device)
        if self.interpolation_nodes == 2:
            return torch.stack([0.5 * (1 - 2 * x), 0.5 * (1 + 2 * x)])

        x2 = x * x
        if self.interpolation_nodes == 3:
            return torch.stack(
                [
                    1 / 8 * (1 - 4 * x + 4 * x2),
                    1 / 4 * (3 - 4 * x2),
                    1 / 8 * (1 + 4 * x + 4 * x2),
                ]
            )

        x3 = x * x2
        if self.interpolation_nodes == 4:
            return torch.stack(
                [
                    1 / 48 * (1 - 6 * x + 12 * x2 - 8 * x3),
                    1 / 48 * (23 - 30 * x - 12 * x2 + 24 * x3),
                    1 / 48 * (23 + 30 * x - 12 * x2 - 24 * x3),
                    1 / 48 * (1 + 6 * x + 12 * x2 + 8 * x3),
                ]
            )

        x4 = x * x3
        if self.interpolation_nodes == 5:
            return torch.stack(
                [
                    1 / 384 * (1 - 8 * x + 24 * x2 - 32 * x3 + 16 * x4),
                    1 / 96 * (19 - 44 * x + 24 * x2 + 16 * x3 - 16 * x4),
                    1 / 192 * (115 - 120 * x2 + 48 * x4),
                    1 / 96 * (19 + 44 * x + 24 * x2 - 16 * x3 - 16 * x4),
                    1 / 384 * (1 + 8 * x + 24 * x2 + 32 * x3 + 16 * x4),
                ]
            )
        raise ValueError("Only `interpolation_nodes` from 1 to 5 are allowed")

    def _compute_1d_weights_Lagrange(self, x: torch.Tensor) -> torch.Tensor:
        """Lagrange interpolation weights.

        Args:
            x: Relative positions ``(interpolation_nodes, n_atoms)``.

        Returns:
            Weights ``(interpolation_nodes, n_atoms)``.
        """
        x2 = x * x
        if self.interpolation_nodes == 3:
            return torch.stack(
                [
                    1 / 2 * (-x + x2),
                    1 / 2 * (2 - 2 * x2),
                    1 / 2 * (x + x2),
                ]
            )

        x3 = x * x2
        if self.interpolation_nodes == 4:
            return torch.stack(
                [
                    1 / 48 * (-3 + 2 * x + 12 * x2 - 8 * x3),
                    1 / 48 * (27 - 54 * x - 12 * x2 + 24 * x3),
                    1 / 48 * (27 + 54 * x - 12 * x2 - 24 * x3),
                    1 / 48 * (-3 - 2 * x + 12 * x2 + 8 * x3),
                ]
            )

        x4 = x * x3
        if self.interpolation_nodes == 5:
            return torch.stack(
                [
                    1 / 24 * (2 * x - x2 - 2 * x3 + x4),
                    1 / 24 * (-16 * x + 16 * x2 + 4 * x3 - 4 * x4),
                    1 / 24 * (24 - 30 * x2 + 6 * x4),
                    1 / 24 * (16 * x + 16 * x2 - 4 * x3 - 4 * x4),
                    1 / 24 * (-2 * x - x2 + 2 * x3 + x4),
                ]
            )

        x5 = x * x4
        if self.interpolation_nodes == 6:
            return torch.stack(
                [
                    1 / 3840 * (45 - 18 * x - 200 * x2 + 80 * x3 + 80 * x4 - 32 * x5),
                    1 / 3840 * (-375 + 250 * x + 1560 * x2 - 1040 * x3 - 240 * x4 + 160 * x5),
                    1 / 3840 * (2250 - 4500 * x - 1360 * x2 + 2720 * x3 + 160 * x4 - 320 * x5),
                    1 / 3840 * (2250 + 4500 * x - 1360 * x2 - 2720 * x3 + 160 * x4 + 320 * x5),
                    1 / 3840 * (-375 - 250 * x + 1560 * x2 + 1040 * x3 - 240 * x4 - 160 * x5),
                    1 / 3840 * (45 + 18 * x - 200 * x2 - 80 * x3 + 80 * x4 + 32 * x5),
                ]
            )
        x6 = x * x5
        if self.interpolation_nodes == 7:
            return torch.stack(
                [
                    1 / 720 * (-12 * x + 4 * x2 + 15 * x3 - 5 * x4 - 3 * x5 + x6),
                    1 / 720 * (108 * x - 54 * x2 - 120 * x3 + 60 * x4 + 12 * x5 - 6 * x6),
                    1 / 720 * (-540 * x + 540 * x2 + 195 * x3 - 195 * x4 - 15 * x5 + 15 * x6),
                    1 / 720 * (720 - 980 * x2 + 280 * x4 - 20 * x6),
                    1 / 720 * (540 * x + 540 * x2 - 195 * x3 - 195 * x4 + 15 * x5 + 15 * x6),
                    1 / 720 * (-108 * x - 54 * x2 + 120 * x3 + 60 * x4 - 12 * x5 - 6 * x6),
                    1 / 720 * (12 * x + 4 * x2 - 15 * x3 - 5 * x4 + 3 * x5 + x6),
                ]
            )
        raise ValueError("Only `interpolation_nodes` from 3 to 7 are allowed")

    def compute_weights(self, positions: torch.Tensor, cell: torch.Tensor) -> MeshStencil:
        """Compute the mesh stencil — indices and weights — of every atom.

        The result is what :meth:`points_to_mesh` and :meth:`mesh_to_points`
        take: a value, not state on the instance (see :class:`MeshStencil`
        for why). The inverse cell is computed here from the argument, per
        call, and not stored either.

        Args:
            positions: Cartesian coordinates ``(n_atoms, 3)`` in Å.
            cell: Unit cell matrix ``(3, 3)`` in Å, rows being the lattice
                vectors, that the positions are reduced against.

        Returns:
            The :class:`MeshStencil` of these positions on this cell.

        Raises:
            ValueError: ``positions`` is on another device than this
                interpolator, or is not ``(n_atoms, 3)``.
        """
        if positions.device != self._device:
            raise ValueError(
                f"`positions` device {positions.device} is not the same as instance "
                f"device {self._device}"
            )

        n_positions = len(positions)
        if positions.shape != (n_positions, 3):
            raise ValueError(f"shape {list(positions.shape)} of `positions` has to be (N, 3)")

        # ``inv_ex`` on CUDA: ``inv`` checks its info tensor on the host, a
        # sync on the force path — a singular cell is not a lattice either way.
        inverse_cell = torch.linalg.inv_ex(cell)[0] if cell.is_cuda else torch.linalg.inv(cell)
        positions_rel = self.ns_mesh * torch.matmul(positions, inverse_cell)

        even = self.interpolation_nodes % 2 == 0
        if even:
            positions_rel_idx = torch.floor(positions_rel).long()
            offsets = positions_rel - (positions_rel_idx + 1 / 2)
        else:
            positions_rel_idx = torch.round(positions_rel).long()
            offsets = positions_rel - positions_rel_idx

        # 1-D weights ``(interpolation_nodes, n_atoms, 3)``, one per axis.
        weights_1d = self._compute_1d_weights(offsets)

        indices_to_interpolate = torch.stack(
            [
                (positions_rel_idx + i) % self.ns_mesh
                for i in range(
                    1 - (self.interpolation_nodes + 1) // 2,
                    1 + self.interpolation_nodes // 2,
                )
            ],
            dim=0,
        )

        xs, ys, zs = self.x_shifts, self.y_shifts, self.z_shifts
        return MeshStencil(
            # Spatial weights are channel-independent — the product once, here.
            weights=weights_1d[xs, :, 0] * weights_1d[ys, :, 1] * weights_1d[zs, :, 2],
            x_indices=indices_to_interpolate[xs, :, 0],
            y_indices=indices_to_interpolate[ys, :, 1],
            z_indices=indices_to_interpolate[zs, :, 2],
        )

    def points_to_mesh(self, particle_weights: torch.Tensor, stencil: MeshStencil) -> torch.Tensor:
        """Interpolate particle weights onto the mesh.

        Args:
            particle_weights: Per-particle weights ``(n_atoms, n_channels)`` —
                the partial charges in ``e`` on the electrostatic path.
            stencil: The particles' :class:`MeshStencil`, from
                :meth:`compute_weights` on the same positions.

        Returns:
            Mesh density ``(n_channels, nx, ny, nz)`` in the weights' own unit
            accumulated *per mesh point*, not per volume: the division by the
            cell volume happens later, in the calculator
            (:meth:`~molpot.potentials.elec.calculators.pme.PMECalculator._compute_kspace`).
        """
        if particle_weights.device != self._device:
            raise ValueError(
                f"`particle_weights` device {particle_weights.device} is not the same "
                f"as instance device {self._device}"
            )

        if particle_weights.dim() != 2:
            raise ValueError(
                f"`particle_weights` of dimension {particle_weights.dim()} has to be of dimension 2"
            )

        n_channels = particle_weights.shape[1]
        nx, ny, nz = self._ns_shape.counts
        rho_mesh = torch.zeros((n_channels, nx, ny, nz), dtype=self._dtype, device=self._device)
        for a in range(n_channels):
            rho_mesh[a].index_put_(
                (stencil.x_indices, stencil.y_indices, stencil.z_indices),
                particle_weights[:, a] * stencil.weights,
                accumulate=True,
            )

        return rho_mesh

    def mesh_to_points(self, mesh_vals: torch.Tensor, stencil: MeshStencil) -> torch.Tensor:
        """Interpolate mesh values back to particle positions.

        Args:
            mesh_vals: Mesh values ``(n_channels, nx, ny, nz)``.
            stencil: The particles' :class:`MeshStencil`, from
                :meth:`compute_weights` on the positions to read at.

        Returns:
            Interpolated values ``(n_atoms, n_channels)``.
        """
        if mesh_vals.dim() != 4:
            raise ValueError(f"`mesh_vals` of dimension {mesh_vals.dim()} has to be of dimension 4")

        return (
            (
                mesh_vals[:, stencil.x_indices, stencil.y_indices, stencil.z_indices]
                * stencil.weights
            )
            .sum(dim=1)
            .T
        )
