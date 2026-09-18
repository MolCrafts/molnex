"""K-space convolution filter for reciprocal-space potential calculations."""

import os

import torch

from molpot.potentials.elec.lib.kvectors import MeshShape, generate_kvectors_for_mesh

#: Default for :attr:`KSpaceFilter.check_nan`, read once at import.
#: The guard is a full-mesh reduction — a device sync barrier on the PME hot
#: path — so it is off unless ``MOLNEX_PME_CHECK_NAN=1`` is exported. Set
#: ``filter.check_nan = True`` (or pass ``check_nan=True``) to enable it for a
#: single instance without touching the environment.
CHECK_NAN_DEFAULT: bool = os.environ.get("MOLNEX_PME_CHECK_NAN", "") == "1"


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
        """Compute the kernel on a grid of **squared** k-vector norms.

        Despite the parameter name, every caller passes ``|k|²`` — one scalar
        per k-point, built by :meth:`KSpaceFilter.kfilter` as the squared norm
        of its k-vector grid — never the ``(nx, ny, nz, 3)`` vectors
        themselves. The implementation that actually runs in this repo,
        :meth:`~molpot.potentials.elec.potentials.Potential.kernel_from_k_sq`,
        names the same argument ``k_sq``.

        Args:
            kvectors: Squared k-vector norms ``|k|²`` ``(nx, ny, nz)`` in Å⁻².

        Returns:
            Kernel values ``(nx, ny, nz)``, one per k-point — for the smeared
            Coulomb kernel ``4π e^{−σ²k²/2} / k²`` times the electrostatics
            prefactor, i.e. kcal Å³ mol⁻¹ e⁻².
        """
        raise NotImplementedError(
            f"kernel_from_k_sq is not implemented for '{self.__class__.__name__}'"
        )


class KSpaceFilter(torch.nn.Module):
    """Apply a reciprocal-space filter to a real-space mesh.

    Computes :math:`f \\to \\hat{f} \\to \\hat{f} \\cdot \\phi \\to \\tilde{f}`
    via FFT → multiply kernel → IFFT.

    The k-vector grid and the filter on it are built from the cell
    :meth:`forward` is handed, on every call (:meth:`kfilter`), and never
    stored: they are functions of the strain leaf under a virial, and dynamo
    treats a leaf-derived tensor stored on a module as a leaked graph output.
    Under ``torch.compile`` the build is traced once; eagerly it is a few
    small factory ops per step — and the cell changes every NPT step anyway.
    What the instance holds is the mesh shape and the kernel.

    The mesh shape arrives in one of two modes, and they trace differently:

    * **frozen** — ``ns_mesh`` given as ``tuple[int, int, int]``, the shape a
      caller fixed ahead of time
      (:meth:`molpot.potentials.elec.calculators.pme.PMECalculator.bind`). The
      k-vector grid is then built from host-side ints and :meth:`forward`
      traces as part of a single graph.
    * **live** — ``ns_mesh`` given as a ``(3,)`` tensor sized off the cell.
      Building the grid reads those counts back to the host, a
      synchronisation that breaks a compiled graph; the mode is for eager
      callers that resize their mesh per cell.

    :attr:`ns_mesh` is the ``(3,)`` tensor form in both modes.

    Args:
        cell: Unit cell matrix ``(3, 3)`` in Å, rows being the lattice vectors
            — read here for its device only, the one a frozen mesh shape's
            tensor form is built on. The lattice itself is an argument of
            :meth:`forward` and is never stored.
        ns_mesh: Mesh dimensions ``(3,)`` as a tensor, or three Python ints —
            see the two modes above.
        kernel: :class:`KSpaceKernel` providing ``kernel_from_k_sq``.
        fft_norm: Normalization for forward FFT (``"forward"``, ``"backward"``,
            ``"ortho"``).
        ifft_norm: Normalization for inverse FFT (``"forward"``, ``"backward"``,
            ``"ortho"``).
        check_nan: Raise :class:`ValueError` when :meth:`forward` produces NaNs
            (usually an unsuitable ``mesh_spacing``). The scan is a full-mesh
            reduction and therefore a device sync barrier on the PME hot path,
            so it defaults to :data:`CHECK_NAN_DEFAULT` (off unless
            ``MOLNEX_PME_CHECK_NAN=1``). Enable it per instance when debugging
            a grid; the attribute stays writable after construction.
    """

    def __init__(
        self,
        cell: torch.Tensor,
        ns_mesh: torch.Tensor | tuple[int, int, int],
        kernel: KSpaceKernel,
        fft_norm: str = "ortho",
        ifft_norm: str = "ortho",
        check_nan: bool | None = None,
    ):
        super().__init__()

        self._fft_norm = fft_norm
        self._ifft_norm = ifft_norm
        self.check_nan = CHECK_NAN_DEFAULT if check_nan is None else bool(check_nan)
        if fft_norm not in ["ortho", "forward", "backward"]:
            raise ValueError(f"Invalid option '{fft_norm}' for the `fft_norm` parameter.")
        if ifft_norm not in ["ortho", "forward", "backward"]:
            raise ValueError(f"Invalid option '{ifft_norm}' for the `ifft_norm` parameter.")

        self.kernel = kernel
        self._ns_shape: MeshShape = MeshShape(ns_mesh, cell.device)

    @property
    def ns_mesh(self) -> torch.Tensor:
        """Mesh dimensions ``(3,)`` as a long tensor, in either mode."""
        return self._ns_shape.tensor

    def update(self, ns_mesh: torch.Tensor | tuple[int, int, int]) -> None:
        """Adopt a new mesh resolution.

        The shape is the only state this filter keeps; the k-grid and the
        filter values follow the cell :meth:`forward` is handed, per call.

        Args:
            ns_mesh: Mesh dimensions ``(3,)`` as a tensor (live mode), or three
                Python ints (frozen mode) — see the class docstring. A frozen
                shape's tensor form is built on the device the current one
                lives on; a frozen shape repeated unchanged reuses it.

        Raises:
            ValueError: A tensor ``ns_mesh`` is not ``(3,)``.
        """
        self._ns_shape.update(ns_mesh, self.ns_mesh.device)

    def kfilter(self, cell: torch.Tensor) -> torch.Tensor:
        """The filter :math:`\\phi(|\\mathbf{k}|^2)` on this mesh's k-grid for ``cell``.

        Built from the argument on every call — nothing is cached, so a
        kernel parameter or the lattice changing between calls is seen at
        once, and the strain derivative through the k-grid exists.

        Args:
            cell: Unit cell matrix ``(3, 3)`` in Å, rows being the lattice
                vectors.

        Returns:
            Filter values ``(nx, ny, nz // 2 + 1)`` on the ``rfftn`` half-grid
            of the mesh, in the kernel's unit.

        Raises:
            ValueError: ``cell`` is not ``(3, 3)``, or ``cell`` and the mesh
                shape live on different devices.
        """
        if cell.shape != (3, 3):
            raise ValueError(f"cell of shape {list(cell.shape)} should be of shape (3, 3)")
        if cell.device != self.ns_mesh.device:
            raise ValueError(
                "`cell` and `ns_mesh` are on different devices, got "
                f"{cell.device} and {self.ns_mesh.device}"
            )
        # Frozen counts go down as ints; only the live mode hands over a
        # tensor the generator has to read back.
        frozen = self._ns_shape.frozen
        ns = self._ns_shape.tensor if frozen is None else frozen
        return self._filter_from_kvectors(generate_kvectors_for_mesh(ns=ns, cell=cell), cell)

    def _filter_from_kvectors(self, kvectors: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        """The kernel on ``|k|²`` of a k-vector grid; P3M multiplies in its influence function.

        Args:
            kvectors: K-vectors ``(nx, ny, nz // 2 + 1, 3)`` in Å⁻¹.
            cell: The unit cell ``(3, 3)`` the grid was built from.

        Returns:
            Filter values ``(nx, ny, nz // 2 + 1)``.
        """
        k_sq = torch.linalg.norm(kvectors, dim=3) ** 2
        return self.kernel.kernel_from_k_sq(k_sq)

    def forward(self, mesh_values: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        """Apply the k-space filter for ``cell`` to a real-space mesh.

        Args:
            mesh_values: Mesh values ``(n_channels, nx, ny, nz)``.
            cell: Unit cell matrix ``(3, 3)`` in Å, rows being the lattice
                vectors, the mesh is laid on — see :meth:`kfilter`.

        Returns:
            Filtered mesh ``(n_channels, nx, ny, nz)``.
        """
        if mesh_values.dim() != 4:
            raise ValueError(
                f"`mesh_values` needs to be a 4 dimensional tensor, got {mesh_values.dim()}"
            )

        kfilter = self.kfilter(cell)
        if mesh_values.device != kfilter.device:
            raise ValueError(
                "`mesh_values` and the k-space filter are on different devices, got "
                f"{mesh_values.device} and {kfilter.device}"
            )

        dims = (1, 2, 3)
        mesh_hat = torch.fft.rfftn(mesh_values, norm=self._fft_norm, dim=dims)

        if mesh_hat.shape[-3:] != kfilter.shape[-3:]:
            raise ValueError("The real-space mesh is inconsistent with the k-space grid.")

        filter_hat = mesh_hat * kfilter

        result = torch.fft.irfftn(
            filter_hat,
            norm=self._ifft_norm,
            dim=dims,
            s=mesh_values.shape[-3:],
        )

        # Full-mesh NaN scan is a device sync barrier on the PME hot path —
        # opt in per instance (see ``check_nan`` in the class docstring).
        if self.check_nan and torch.isnan(result).any():
            raise ValueError(
                "NaNs detected in the k-space filter result. This are probably caused "
                "by an unsuitable `mesh_spacing`, resulting in a problematic grid of "
                f"shape: {list(mesh_values.shape)}. Try adjusting the grid by using a "
                "different `mesh_spacing` value."
            )

        return result


class P3MKSpaceFilter(KSpaceFilter):
    """Specialized k-space filter for the P3M method.

    Uses a cell-dependent Green's function with optimized influence function
    and differential operator discretization — both built from the cell
    :meth:`~KSpaceFilter.forward` is handed, per call, like the base filter.

    Args:
        cell: Unit cell matrix ``(3, 3)`` in Å, rows being the lattice vectors
            — read for its device only, see :class:`KSpaceFilter`.
        ns_mesh: Mesh dimensions ``(3,)`` as a tensor, or three Python ints —
            see :class:`KSpaceFilter` for the two modes.
        interpolation_nodes: Order of charge assignment (1-5).
        kernel: :class:`KSpaceKernel` for the base potential.
        fft_norm: Forward FFT normalization.
        ifft_norm: Inverse FFT normalization.
        mode: 0 for potential, 1 for energy, 2 for dipolar torque, 3 for dipolar force.
        differential_order: Order of the difference operator (1-6).
        check_nan: See :class:`KSpaceFilter`.

    Reference:
        Deserno, M. & Holm, C. J. Chem. Phys. 109, 7678–7693 (1998)
    """

    def __init__(
        self,
        cell: torch.Tensor,
        ns_mesh: torch.Tensor | tuple[int, int, int],
        interpolation_nodes: int,
        kernel: KSpaceKernel,
        fft_norm: str = "ortho",
        ifft_norm: str = "ortho",
        mode: int = 0,
        differential_order: int = 2,
        check_nan: bool | None = None,
    ):
        super().__init__(cell, ns_mesh, kernel, fft_norm, ifft_norm, check_nan)
        self.interpolation_nodes = interpolation_nodes
        if mode not in [0, 1, 2, 3]:
            raise ValueError(f"`mode` should be one of [0, 1, 2, 3], but got {mode}")
        self.mode = mode
        if differential_order not in [1, 2, 3, 4, 5, 6]:
            raise ValueError(
                f"`differential_order` should be one between 1 and 6, but got {differential_order}"
            )
        self.differential_order = differential_order
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

    def _filter_from_kvectors(self, kvectors: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        """The base kernel times the P3M influence function for ``cell``.

        Args:
            kvectors: K-vectors ``(nx, ny, nz // 2 + 1, 3)`` in Å⁻¹.
            cell: The unit cell ``(3, 3)`` the grid was built from.

        Returns:
            Filter values ``(nx, ny, nz // 2 + 1)``.
        """
        return self._compute_influence(kvectors, cell) * super()._filter_from_kvectors(
            kvectors, cell
        )

    def _compute_influence(self, kvectors: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        cell_dimensions = torch.linalg.norm(cell, dim=1)
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
