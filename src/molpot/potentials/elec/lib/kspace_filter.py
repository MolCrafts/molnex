"""K-space convolution filter for reciprocal-space potential calculations."""

from typing import Optional

import torch

from molpot.potentials.elec.lib.kvectors import generate_kvectors_for_mesh


class KSpaceKernel(torch.nn.Module):
    """Base class for a reciprocal-space convolution kernel.

    Provides the interface for computing :math:`\\phi(|\\mathbf{k}|^2)`,
    the filter applied in Fourier space.

    Subclasses or :class:`Potential` instances implementing
    ``kernel_from_k_sq`` can be used as kernels.
    """

    def __init__(self):
        super().__init__()

    def kernel_from_k_sq(self, kvectors: torch.Tensor) -> torch.Tensor:
        """Compute the kernel on a k-vector grid.

        Args:
            kvectors: K-vectors ``(nx, ny, nz, 3)``.

        Returns:
            Kernel values at each k-point.
        """
        raise NotImplementedError(
            f"kernel_from_k_sq is not implemented for '{self.__class__.__name__}'"
        )


class KSpaceFilter(torch.nn.Module):
    """Apply a reciprocal-space filter to a real-space mesh.

    Computes :math:`f \\to \\hat{f} \\to \\hat{f} \\cdot \\phi \\to \\tilde{f}`
    via FFT → multiply kernel → IFFT.

    Args:
        cell: Unit cell matrix ``(3, 3)``.
        ns_mesh: Mesh dimensions ``(3,)``.
        kernel: :class:`KSpaceKernel` providing ``kernel_from_k_sq``.
        fft_norm: Normalization for forward FFT (``"forward"``, ``"backward"``,
            ``"ortho"``).
        ifft_norm: Normalization for inverse FFT (``"forward"``, ``"backward"``,
            ``"ortho"``).
    """

    def __init__(
        self,
        cell: torch.Tensor,
        ns_mesh: torch.Tensor,
        kernel: KSpaceKernel,
        fft_norm: str = "ortho",
        ifft_norm: str = "ortho",
    ):
        super().__init__()

        self._fft_norm = fft_norm
        self._ifft_norm = ifft_norm
        if fft_norm not in ["ortho", "forward", "backward"]:
            raise ValueError(f"Invalid option '{fft_norm}' for the `fft_norm` parameter.")
        if ifft_norm not in ["ortho", "forward", "backward"]:
            raise ValueError(f"Invalid option '{ifft_norm}' for the `ifft_norm` parameter.")

        self.kernel = kernel
        self.update(cell, ns_mesh)

    def update(
        self,
        cell: Optional[torch.Tensor] = None,
        ns_mesh: Optional[torch.Tensor] = None,
    ) -> None:
        """Update buffers when cell, mesh, or kernel parameters change.

        Args:
            cell: Unit cell matrix ``(3, 3)``.
            ns_mesh: Mesh dimensions ``(3,)``.
        """
        self._prep_kvectors(cell, ns_mesh)
        self._kfilter = self.kernel.kernel_from_k_sq(self._k_sq)

    def forward(self, mesh_values: torch.Tensor) -> torch.Tensor:
        """Apply the k-space filter to a real-space mesh.

        Args:
            mesh_values: Mesh values ``(n_channels, nx, ny, nz)``.

        Returns:
            Filtered mesh ``(n_channels, nx, ny, nz)``.
        """
        if mesh_values.dim() != 4:
            raise ValueError(
                f"`mesh_values` needs to be a 4 dimensional tensor, got {mesh_values.dim()}"
            )

        if mesh_values.device != self._kfilter.device:
            raise ValueError(
                "`mesh_values` and the k-space filter are on different devices, got "
                f"{mesh_values.device} and {self._kfilter.device}"
            )

        dims = (1, 2, 3)
        mesh_hat = torch.fft.rfftn(mesh_values, norm=self._fft_norm, dim=dims)

        if mesh_hat.shape[-3:] != self._kfilter.shape[-3:]:
            raise ValueError("The real-space mesh is inconsistent with the k-space grid.")

        filter_hat = mesh_hat * self._kfilter

        result = torch.fft.irfftn(
            filter_hat,
            norm=self._ifft_norm,
            dim=dims,
            s=mesh_values.shape[-3:],
        )

        if torch.isnan(result).any():
            raise ValueError(
                "NaNs detected in the k-space filter result. This are probably caused "
                "by an unsuitable `mesh_spacing`, resulting in a problematic grid of "
                f"shape: {list(mesh_values.shape)}. Try adjsuting the grid by using a "
                "different `mesh_spacing` value."
            )

        return result

    def _prep_kvectors(self, cell: Optional[torch.Tensor], ns_mesh: Optional[torch.Tensor]):
        if cell is not None:
            if cell.shape != (3, 3):
                raise ValueError(f"cell of shape {list(cell.shape)} should be of shape (3, 3)")
            self.cell = cell

        if ns_mesh is not None:
            if ns_mesh.shape != (3,):
                raise ValueError(f"shape {list(ns_mesh.shape)} of `ns_mesh` has to be (3,)")
            self.ns_mesh = ns_mesh

        if self.cell.device != self.ns_mesh.device:
            raise ValueError(
                "`cell` and `ns_mesh` are on different devices, got "
                f"{self.cell.device} and {self.ns_mesh.device}"
            )

        if cell is not None or ns_mesh is not None:
            self._kvectors = generate_kvectors_for_mesh(ns=self.ns_mesh, cell=self.cell)
            self._k_sq = torch.linalg.norm(self._kvectors, dim=3) ** 2


class P3MKSpaceFilter(KSpaceFilter):
    """Specialized k-space filter for the P3M method.

    Uses a cell-dependent Green's function with optimized influence function
    and differential operator discretization.

    Args:
        cell: Unit cell matrix ``(3, 3)``.
        ns_mesh: Mesh dimensions ``(3,)``.
        interpolation_nodes: Order of charge assignment (1-5).
        kernel: :class:`KSpaceKernel` for the base potential.
        fft_norm: Forward FFT normalization.
        ifft_norm: Inverse FFT normalization.
        mode: 0 for potential, 1 for energy, 2 for dipolar torque, 3 for dipolar force.
        differential_order: Order of the difference operator (1-6).

    Reference:
        Deserno, M. & Holm, C. J. Chem. Phys. 109, 7678–7693 (1998)
    """

    def __init__(
        self,
        cell: torch.Tensor,
        ns_mesh: torch.Tensor,
        interpolation_nodes: int,
        kernel: KSpaceKernel,
        fft_norm: str = "ortho",
        ifft_norm: str = "ortho",
        mode: int = 0,
        differential_order: int = 2,
    ):
        self.interpolation_nodes = interpolation_nodes
        if mode not in [0, 1, 2, 3]:
            raise ValueError(f"`mode` should be one of [0, 1, 2, 3], but got {mode}")
        self.mode = mode
        if differential_order not in [1, 2, 3, 4, 5, 6]:
            raise ValueError(
                f"`differential_order` should be one between 1 and 6, but got {differential_order}"
            )
        self.differential_order = differential_order

        super().__init__(cell, ns_mesh, kernel, fft_norm, ifft_norm)
        self.register_buffer(
            "_diff_coeff",
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [4 / 3, -1 / 3, 0.0, 0.0, 0.0, 0.0],
                    [3 / 2, -3 / 5, 1 / 10, 0.0, 0.0, 0.0],
                    [8 / 5, -4 / 5, 8 / 35, -1 / 35, 0.0, 0.0],
                    [5 / 3, -20 / 21, 5 / 14, -5 / 63, 1 / 126, 0.0],
                    [12 / 7, -15 / 14, 10 / 21, -1 / 7, 2 / 77, -1 / 465],
                ]
            ),
        )

    def update(
        self,
        cell: Optional[torch.Tensor] = None,
        ns_mesh: Optional[torch.Tensor] = None,
    ) -> None:
        """Update buffers with P3M-specific influence function.

        Args:
            cell: Unit cell matrix ``(3, 3)``.
            ns_mesh: Mesh dimensions ``(3,)``.
        """
        self._prep_kvectors(cell, ns_mesh)
        self._kfilter = self._compute_influence(self._kvectors) * self.kernel.kernel_from_k_sq(
            self._k_sq
        )

    def _compute_influence(self, kvectors: torch.Tensor) -> torch.Tensor:
        cell_dimensions = torch.linalg.norm(self.cell, dim=1)
        actual_mesh_spacing = (cell_dimensions / self.ns_mesh).reshape(1, 1, 1, 3)

        kh = kvectors * actual_mesh_spacing
        U2 = self._charge_assignment(kh)
        if self.mode == 0:
            masked = torch.where(U2 == 0, 1.0, U2)
            return torch.where(U2 == 0, 0.0, torch.reciprocal(masked))

        D = self._differential_operator(kh, actual_mesh_spacing)
        D_to_4mode = torch.linalg.norm(D, dim=-1) ** (4 * self.mode)

        numerator = torch.sum(kvectors * D, dim=-1) ** self.mode
        denominator = U2 * D_to_4mode

        masked = torch.where(denominator == 0, 1.0, denominator)
        return torch.where(denominator == 0, 0.0, numerator / masked)

    def _differential_operator(
        self, kh: torch.Tensor, actual_mesh_spacing: torch.Tensor
    ) -> torch.Tensor:
        """Approximate the differential operator :math:`i\\mathbf{k}`.

        Args:
            kh: Scaled k-vectors ``(nx, ny, nz, 3)``.
            actual_mesh_spacing: Mesh spacing per dimension ``(1, 1, 1, 3)``.

        Returns:
            Difference operator values ``(nx, ny, nz, 3)``.
        """
        temp = torch.zeros(kh.shape, dtype=kh.dtype, device=kh.device)
        for i, coef in enumerate(
            self._diff_coeff[self.differential_order - 1][: self.differential_order]
        ):
            temp += (coef / (i + 1)) * torch.sin(kh * (i + 1))
        return temp / actual_mesh_spacing

    def _charge_assignment(self, kh: torch.Tensor) -> torch.Tensor:
        """Fourier-transformed charge assignment function (squared).

        Args:
            kh: Scaled k-vectors ``(nx, ny, nz, 3)``.

        Returns:
            Assignment values ``(nx, ny, nz)``.
        """
        return torch.prod(
            torch.sinc(kh / (2 * torch.pi)),
            dim=-1,
        ) ** (self.interpolation_nodes * 2)
