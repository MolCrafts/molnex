"""Mesh interpolation for particle-mesh methods (Lagrange and P3M)."""

from typing import Optional

import torch


class MeshInterpolator(torch.nn.Module):
    """Interpolation between particles and a regular 3D mesh.

    Handles both forward interpolation (particles → mesh) and backward
    interpolation (mesh → particles) using either Lagrange or P3M
    cardinal B-spline assignment functions.

    Args:
        cell: Unit cell matrix ``(3, 3)``.
        ns_mesh: Mesh dimensions ``(3,)``.
        interpolation_nodes: Number of nodes per axis (3-7 for Lagrange, 1-5 for P3M).
        method: ``"Lagrange"`` or ``"P3M"``.
    """

    def __init__(
        self,
        cell: torch.Tensor,
        ns_mesh: torch.Tensor,
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

        self.update(cell, ns_mesh)

        self.interpolation_weights: torch.Tensor = torch.zeros(
            1, device=self._device, dtype=self._dtype
        )
        self.x_shifts: torch.Tensor = torch.zeros(1, device=self._device)
        self.y_shifts: torch.Tensor = torch.zeros(1, device=self._device)
        self.z_shifts: torch.Tensor = torch.zeros(1, device=self._device)
        self.x_indices: torch.Tensor = torch.zeros(1, device=self._device)
        self.y_indices: torch.Tensor = torch.zeros(1, device=self._device)
        self.z_indices: torch.Tensor = torch.zeros(1, device=self._device)

    def update(
        self,
        cell: Optional[torch.Tensor] = None,
        ns_mesh: Optional[torch.Tensor] = None,
    ) -> None:
        """Update buffers when cell or mesh resolution changes.

        Args:
            cell: Unit cell matrix ``(3, 3)``.
            ns_mesh: Mesh dimensions ``(3,)``.
        """
        if cell is not None:
            if cell.shape != (3, 3):
                raise ValueError(f"cell of shape {list(cell.shape)} should be of shape (3, 3)")
            self.cell = cell
            self.inverse_cell = cell.clone()
            self._dtype = cell.dtype
            self._device = cell.device

            if self.cell.is_cuda:
                self.inverse_cell = torch.linalg.inv_ex(cell)[0]
            else:
                self.inverse_cell = torch.linalg.inv(cell)

        if ns_mesh is not None:
            if ns_mesh.shape != (3,):
                raise ValueError(f"shape {list(ns_mesh.shape)} of `ns_mesh` has to be (3,)")
            self.ns_mesh = ns_mesh

        if self.cell.device != self.ns_mesh.device:
            raise ValueError(
                "`cell` and `ns_mesh` are on different devices, got "
                f"{self.cell.device} and {self.ns_mesh.device}"
            )

    def get_mesh_xyz(self) -> torch.Tensor:
        """Return Cartesian positions of all mesh points.

        Returns:
            Mesh positions ``(nx, ny, nz, 3)``.
        """
        nx = int(self.ns_mesh[0])
        ny = int(self.ns_mesh[1])
        nz = int(self.ns_mesh[2])

        grid_scaled = torch.stack(
            torch.meshgrid(
                torch.arange(nx, dtype=self._dtype, device=self._device) / nx,
                torch.arange(ny, dtype=self._dtype, device=self._device) / ny,
                torch.arange(nz, dtype=self._dtype, device=self._device) / nz,
                indexing="ij",
            ),
            dim=-1,
        )
        return torch.matmul(grid_scaled, self.cell)

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

    def compute_weights(self, positions: torch.Tensor):
        """Compute interpolation weights for all atoms.

        Must be called before :meth:`points_to_mesh` or :meth:`mesh_to_points`.

        Args:
            positions: Cartesian coordinates ``(n_atoms, 3)``.
        """
        if positions.device != self._device:
            raise ValueError(
                f"`positions` device {positions.device} is not the same as instance "
                f"device {self._device}"
            )

        n_positions = len(positions)
        if positions.shape != (n_positions, 3):
            raise ValueError(f"shape {list(positions.shape)} of `positions` has to be (N, 3)")

        positions_rel = self.ns_mesh * torch.matmul(positions, self.inverse_cell)

        even = self.interpolation_nodes % 2 == 0
        if even:
            positions_rel_idx = torch.floor(positions_rel).long()
            offsets = positions_rel - (positions_rel_idx + 1 / 2)
        else:
            positions_rel_idx = torch.round(positions_rel).long()
            offsets = positions_rel - positions_rel_idx

        self.interpolation_weights = self._compute_1d_weights(offsets)

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

        x_shifts, y_shifts, z_shifts = torch.meshgrid(
            torch.arange(self.interpolation_nodes, device=self._device),
            torch.arange(self.interpolation_nodes, device=self._device),
            torch.arange(self.interpolation_nodes, device=self._device),
            indexing="ij",
        )
        self.x_shifts = x_shifts.flatten()
        self.y_shifts = y_shifts.flatten()
        self.z_shifts = z_shifts.flatten()

        self.x_indices = indices_to_interpolate[self.x_shifts, :, 0]
        self.y_indices = indices_to_interpolate[self.y_shifts, :, 1]
        self.z_indices = indices_to_interpolate[self.z_shifts, :, 2]

    def points_to_mesh(self, particle_weights: torch.Tensor) -> torch.Tensor:
        """Interpolate particle weights onto the mesh.

        Args:
            particle_weights: Per-particle weights ``(n_atoms, n_channels)``.

        Returns:
            Mesh density ``(n_channels, nx, ny, nz)``.
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
        nx = int(self.ns_mesh[0])
        ny = int(self.ns_mesh[1])
        nz = int(self.ns_mesh[2])
        rho_mesh = torch.zeros((n_channels, nx, ny, nz), dtype=self._dtype, device=self._device)
        for a in range(n_channels):
            rho_mesh[a].index_put_(
                (self.x_indices, self.y_indices, self.z_indices),
                (
                    particle_weights[:, a]
                    * self.interpolation_weights[self.x_shifts, :, 0]
                    * self.interpolation_weights[self.y_shifts, :, 1]
                    * self.interpolation_weights[self.z_shifts, :, 2]
                ),
                accumulate=True,
            )

        return rho_mesh

    def mesh_to_points(self, mesh_vals: torch.Tensor) -> torch.Tensor:
        """Interpolate mesh values back to particle positions.

        Args:
            mesh_vals: Mesh values ``(n_channels, nx, ny, nz)``.

        Returns:
            Interpolated values ``(n_atoms, n_channels)``.
        """
        if mesh_vals.dim() != 4:
            raise ValueError(f"`mesh_vals` of dimension {mesh_vals.dim()} has to be of dimension 4")

        return (
            (
                mesh_vals[:, self.x_indices, self.y_indices, self.z_indices]
                * self.interpolation_weights[self.x_shifts, :, 0]
                * self.interpolation_weights[self.y_shifts, :, 1]
                * self.interpolation_weights[self.z_shifts, :, 2]
            )
            .sum(dim=1)
            .T
        )
