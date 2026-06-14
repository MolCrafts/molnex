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

    Scaling: :math:`\\mathcal{O}(N \\log N)` with particle count.

    Args:
        potential: :class:`Potential` with smearing set.
        mesh_spacing: Target real-space mesh spacing (determines FFT grid size).
        interpolation_nodes: Number of interpolation nodes per axis (3-7).
        full_neighbor_list: If True, full neighbor list expected.

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

        Args:
            charges: Atomic charges ``(n_atoms, n_channels)``.
            cell: Unit cell matrix ``(3, 3)``.
            positions: Cartesian coordinates ``(n_atoms, 3)``.
            periodic: Boolean mask ``(3,)`` for periodic directions (unused).
            node_mask: Optional boolean mask ``(n_atoms,)`` (unused).
            kvectors: Optional precomputed k-vectors (unused).

        Returns:
            K-space potential ``(n_atoms, n_channels)``.
        """
        if node_mask is not None or kvectors is not None:
            raise NotImplementedError("Batching not implemented for mesh-based calculators")
        ns = get_ns_mesh(cell, self.mesh_spacing)

        self.mesh_interpolator.update(cell, ns)
        self.kspace_filter.update(cell, ns)

        self.mesh_interpolator.compute_weights(positions)
        rho_mesh = self.mesh_interpolator.points_to_mesh(particle_weights=charges)

        potential_mesh = self.kspace_filter.forward(rho_mesh)

        ivolume = torch.abs(cell.det()).pow(-1)
        interpolated_potential = self.mesh_interpolator.mesh_to_points(potential_mesh) * ivolume

        interpolated_potential -= charges * self.potential.self_contribution()

        charge_tot = torch.sum(charges, dim=0)
        prefac = self.potential.background_correction()
        interpolated_potential -= 2 * prefac * charge_tot * ivolume

        interpolated_potential += self.potential.pbc_correction(periodic, positions, cell, charges)

        return interpolated_potential / 2
