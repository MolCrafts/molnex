from typing import Any

import torch

from molpot.potentials.base import BasePotential
from molpot.potentials.geom import dihedral_cos_sin


class DihedralHarmonic(BasePotential):
    """Harmonic dihedral / improper-style form ``E = ½ k (φ − φ₀)²``.

    **Not** a Class-I proper torsion. For AMBER / GAFF / SMIRNOFF propers use
    :class:`~molpot.potentials.dihedrals.periodic.ProperTorsionPeriodic`.

    Energy formula::

        E = 0.5 * k * (phi - phi0)^2

    Parameters are stored as type-indexed vectors:

    - ``k[dihedral_type]``: Force constant
    - ``phi0[dihedral_type]``: Equilibrium dihedral angle (radians)

    Attributes:
        k: Force constants ``[num_dihedral_types]``.
        phi0: Equilibrium dihedral angles in radians ``[num_dihedral_types]``.
    """

    name = "dihedral_harmonic_torch"
    type = "dihedral"

    k: torch.Tensor
    phi0: torch.Tensor

    def __init__(
        self,
        k: torch.Tensor,
        phi0: torch.Tensor,
    ):
        """Initialize DihedralHarmonic potential.

        Args:
            k: Force constant vector [num_dihedral_types]
            phi0: Equilibrium dihedral angle vector in radians [num_dihedral_types]
        """
        super().__init__()

        # Validate shapes
        if k.shape != phi0.shape:
            raise ValueError(
                f"k and phi0 must have same shape, got k: {k.shape}, phi0: {phi0.shape}"
            )

        if k.ndim != 1:
            raise ValueError(f"k must be 1D vector [num_dihedral_types], got shape {k.shape}")

        # Register as buffers
        self.register_buffer("k", k)
        self.register_buffer("phi0", phi0)

    def forward(self, data: dict[str, Any] | None = None, **kwargs: Any) -> torch.Tensor:
        """Compute harmonic dihedral energy.

        Args:
            data: Optional dictionary with molecular fields
            **kwargs: Alternate way to pass explicit tensors, including:
                - pos: Positions [N, 3]
                - dihedral_index: Dihedral indices [4, num_dihedrals] (i-j-k-l)
                - dihedral_types: Dihedral types [num_dihedrals]

        Returns:
            Total dihedral energy (scalar)
        """
        # Extract data
        pos = kwargs.get("pos")
        dihedral_index = kwargs.get("dihedral_index")
        dihedral_types = kwargs.get("dihedral_types")

        if pos is None and data is not None:
            if isinstance(data, dict):
                pos = data.get("pos")
                dihedral_index = (
                    data.get("dihedral_index") if dihedral_index is None else dihedral_index
                )
                dihedral_types = (
                    data.get("dihedral_types") if dihedral_types is None else dihedral_types
                )
                if pos is None and "atoms" in data:
                    pos = data["atoms"]["x"]

        if pos is None or dihedral_index is None or dihedral_types is None:
            raise ValueError("DihedralHarmonic requires pos, dihedral_index, and dihedral_types.")

        # Convert numpy to torch if needed
        if not isinstance(pos, torch.Tensor):
            pos = torch.from_numpy(pos).float()
            dihedral_index = torch.from_numpy(dihedral_index).long()
            dihedral_types = torch.from_numpy(dihedral_types).long()

        # Handle empty dihedrals
        if dihedral_index.size(1) == 0:
            return torch.tensor(0.0, device=pos.device, dtype=pos.dtype)

        b1 = pos[dihedral_index[1]] - pos[dihedral_index[0]]
        b2 = pos[dihedral_index[2]] - pos[dihedral_index[1]]
        b3 = pos[dihedral_index[3]] - pos[dihedral_index[2]]
        cos_phi, sin_phi = dihedral_cos_sin(b1, b2, b3)
        phi = torch.atan2(sin_phi, cos_phi)

        # Look up parameters for each dihedral
        k_dihedrals = self.k[dihedral_types]  # [num_dihedrals]
        phi0_dihedrals = self.phi0[dihedral_types]  # [num_dihedrals]

        # Compute harmonic energy: E = 0.5 * k * (phi - phi0)^2
        displacement = phi - phi0_dihedrals
        energy_per_dihedral = 0.5 * k_dihedrals * displacement**2

        # Sum over all dihedrals
        total_energy = energy_per_dihedral.sum()

        return total_energy

    def __repr__(self) -> str:
        num_types = len(self.k)
        return f"DihedralHarmonic(num_dihedral_types={num_types})"
