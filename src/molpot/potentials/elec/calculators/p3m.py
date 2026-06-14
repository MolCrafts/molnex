"""P3M calculator — Particle-Particle Particle-Mesh with optimized Green's function."""

import torch

from molpot.potentials.elec.calculators.pme import PMECalculator
from molpot.potentials.elec.lib.kspace_filter import P3MKSpaceFilter
from molpot.potentials.elec.lib.mesh_interpolator import MeshInterpolator
from molpot.potentials.elec.potentials import Potential


class P3MCalculator(PMECalculator):
    """Particle-Particle Particle-Mesh (P3M) calculator.

    Uses an optimized influence function and P3M-specific charge assignment
    for improved accuracy over standard PME at the same grid resolution.

    Args:
        potential: :class:`Potential` with smearing set.
        mesh_spacing: Target real-space mesh spacing.
        interpolation_nodes: Number of P3M interpolation nodes per axis (1-5).
        full_neighbor_list: If True, full neighbor list expected.

    Reference:
        Deserno, M. & Holm, C. J. Chem. Phys. 109, 7678–7693 (1998)
    """

    def __init__(
        self,
        potential: Potential,
        mesh_spacing: float,
        interpolation_nodes: int = 4,
        full_neighbor_list: bool = False,
    ):
        super().__init__(
            potential=potential,
            mesh_spacing=mesh_spacing,
            full_neighbor_list=full_neighbor_list,
        )

        cell = torch.eye(
            3,
            device=self.potential.smearing.device,
            dtype=self.potential.smearing.dtype,
        )
        ns_mesh = torch.ones(3, dtype=torch.long, device=cell.device)

        self.kspace_filter: P3MKSpaceFilter = P3MKSpaceFilter(
            cell=cell,
            ns_mesh=ns_mesh,
            interpolation_nodes=interpolation_nodes,
            kernel=self.potential,
            mode=0,
            differential_order=2,
            fft_norm="backward",
            ifft_norm="forward",
        )

        self.mesh_interpolator: MeshInterpolator = MeshInterpolator(
            cell=cell,
            ns_mesh=ns_mesh,
            interpolation_nodes=interpolation_nodes,
            method="P3M",
        )
        self.interpolation_nodes: int = interpolation_nodes
