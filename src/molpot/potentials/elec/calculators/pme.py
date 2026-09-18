"""PME calculator — O(N log N) Particle-Mesh Ewald with FFT acceleration."""

from typing import Optional

import torch

from molpot.potentials.elec.calculators.calculator import Calculator
from molpot.potentials.elec.lib.kspace_filter import KSpaceFilter
from molpot.potentials.elec.lib.kvectors import get_ns_mesh
from molpot.potentials.elec.lib.mesh_interpolator import MeshInterpolator
from molpot.potentials.elec.potentials import Potential


class PMECalculator(Calculator):
    """Particle-Mesh Ewald (PME) calculator with FFT-based k-space evaluation.

    Particle-Mesh Ewald spreads the atomic charges onto a regular grid, does the
    reciprocal-space part of the Ewald sum as one fast Fourier transform (FFT)
    on that grid, and interpolates the result back to the atoms. Scaling:
    :math:`\\mathcal{O}(N \\log N)` with particle count :math:`N`, against the
    :math:`\\mathcal{O}(N^2)` of summing every k-vector explicitly.

    Args:
        potential: :class:`Potential` with ``smearing`` set — the Gaussian width
            (Å) of the real/reciprocal split, which this calculator evaluates the
            reciprocal half of.
        mesh_spacing: Target real-space mesh spacing in Å; the FFT grid size is
            derived from it by
            :func:`~molpot.potentials.elec.lib.kvectors.get_ns_mesh`. Ignored
            once :meth:`bind` freezes a shape.
        interpolation_nodes: Number of mesh points per axis each charge is spread
            over (3-7).
        full_neighbor_list: ``True`` when the neighbour list handed to
            :meth:`Calculator.forward` holds both ``(i, j)`` and ``(j, i)``.

    Raises:
        ValueError: ``potential.smearing`` is unset or not positive — without a
            split width there is no reciprocal part to evaluate.

    Reference:
        Darden, T. et al. J. Chem. Phys. 98, 10089–10092 (1993)
    """

    def __init__(
        self,
        potential: Potential,
        mesh_spacing: float,
        interpolation_nodes: int = 4,
        full_neighbor_list: bool = False,
    ):
        super().__init__(potential=potential, full_neighbor_list=full_neighbor_list)

        if potential.smearing is None:
            raise ValueError("Must specify smearing to use a potential with PMECalculator")
        if potential.smearing <= 0:
            raise ValueError(f"`smearing` is {potential.smearing} but must be positive")

        self.mesh_spacing: float = mesh_spacing

        cell = torch.eye(
            3,
            device=self.potential.smearing.device,
            dtype=self.potential.smearing.dtype,
        )
        ns_mesh = torch.ones(3, dtype=torch.long, device=cell.device)

        self.kspace_filter: KSpaceFilter = KSpaceFilter(
            cell=cell,
            ns_mesh=ns_mesh,
            kernel=self.potential,
            fft_norm="backward",
            ifft_norm="forward",
        )

        self.mesh_interpolator: MeshInterpolator = MeshInterpolator(
            cell=cell,
            ns_mesh=ns_mesh,
            interpolation_nodes=interpolation_nodes,
            method="Lagrange",
        )
        self.interpolation_nodes: int = interpolation_nodes
        # Frozen mesh shape, set by :meth:`bind`; ``None`` means unbound. Three
        # Python ints, never a tensor: the mesh libraries size their grids off
        # these, and an int read off a device tensor is the host sync that
        # breaks the compiled graph.
        self._bound_ns_mesh: tuple[int, int, int] | None = None

    def bind(self, ns_mesh: tuple[int, int, int]) -> None:
        """Freeze the FFT mesh shape, so later calls size nothing off the cell.

        A bound calculator sizes nothing off the cell: the interpolator and
        the k-space filter keep this shape, and build their stencil and k-grid
        from the **live** cell on every :meth:`_compute_kspace` call — which is
        what a barostat needs, where the filter has to follow the lattice for
        the strain derivative to exist.

        Idempotent: binding the same shape twice replaces one constant with an
        equal one. The owner that sized the shape is responsible for refusing
        cells the mesh cannot resolve — see
        :meth:`molpot.forcefield.pair_terms.CoulombPeriodicPotential.bind_cell`.

        The shape travels on as three Python ints — the interpolator and the
        k-space filter take their grid counts from them directly, so nothing
        on the bound path reads a mesh dimension back off the device (see
        :class:`~molpot.potentials.elec.lib.mesh_interpolator.MeshInterpolator`
        for the two modes).

        Args:
            ns_mesh: Mesh points per axis, three Python ints (powers of two for
                a fast FFT).
        """
        self._bound_ns_mesh = (int(ns_mesh[0]), int(ns_mesh[1]), int(ns_mesh[2]))

    def _compute_kspace(
        self,
        charges: torch.Tensor,
        cell: torch.Tensor,
        positions: torch.Tensor,
        periodic: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
        kvectors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the PME k-space contribution via FFT.

        Bound (:meth:`bind`) the mesh keeps the frozen shape; unbound it is
        sized off the cell, a host read for eager callers only. Either way the
        stencil and the k-space filter are built from the live cell and
        positions on every call and passed along, never stored: the
        interpolator and the filter hold metadata only, because a leaf-derived
        tensor kept on a module is a leaked graph output to dynamo.

        Args:
            charges: Atomic charges ``(n_atoms, n_channels)`` in ``e``.
            cell: Unit cell matrix ``(3, 3)`` in Å, rows being the lattice
                vectors. Read live: the stencil and the k-space filter are
                built from it on every call.
            positions: Cartesian coordinates ``(n_atoms, 3)`` in Å.
            periodic: Boolean mask ``(3,)`` of which lattice directions are
                periodic. It selects the slab correction the potential adds
                (:meth:`~molpot.potentials.elec.potentials.Potential.pbc_correction`):
                a mask with exactly two ``True`` entries is the 2D-periodic
                case, and ``None`` means all three directions are periodic.
            node_mask: Must be ``None``. A per-atom mask belongs to the batched
                path, which the mesh calculators do not have.
            kvectors: Must be ``None``. The mesh path builds its own k-vector
                grid from ``cell`` at the current mesh shape; precomputed
                vectors are the explicit-Ewald route's argument.

        Returns:
            K-space potential per atom ``(n_atoms, n_channels)``, in the
            potential's own unit per unit charge (kcal mol⁻¹ e⁻¹ with the
            ``real`` prefactor). Multiplying by the charges and summing gives
            the reciprocal-space energy.

        Raises:
            NotImplementedError: ``node_mask`` or ``kvectors`` is not ``None``.
        """
        if node_mask is not None or kvectors is not None:
            raise NotImplementedError("Batching not implemented for mesh-based calculators")
        bound_ns: Optional[tuple[int, int, int]] = self._bound_ns_mesh
        ns = get_ns_mesh(cell, self.mesh_spacing) if bound_ns is None else bound_ns
        # Metadata only — dtype, device, mesh shape. The cell and positions go
        # to each call below as arguments; the stencil and the filter they
        # yield are values that live inside this one call, never state on the
        # interpolator or the filter (a leaf-derived tensor parked on a module
        # is a leaked graph output to dynamo).
        self.mesh_interpolator.update(cell, ns)
        self.kspace_filter.update(ns)

        stencil = self.mesh_interpolator.compute_weights(positions, cell)
        rho_mesh = self.mesh_interpolator.points_to_mesh(charges, stencil)

        potential_mesh = self.kspace_filter.forward(rho_mesh, cell)

        ivolume = torch.abs(cell.det()).pow(-1)
        interpolated_potential = (
            self.mesh_interpolator.mesh_to_points(potential_mesh, stencil) * ivolume
        )

        interpolated_potential -= charges * self.potential.self_contribution()

        charge_tot = torch.sum(charges, dim=0)
        prefac = self.potential.background_correction()
        interpolated_potential -= 2 * prefac * charge_tot * ivolume

        interpolated_potential += self.potential.pbc_correction(periodic, positions, cell, charges)

        return interpolated_potential / 2
