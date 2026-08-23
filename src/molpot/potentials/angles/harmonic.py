from typing import Any

import torch

from molpot.potentials.base import BasePotential
from molpot.potentials.bonded_forces import angle_theta_grad
from molpot.potentials.geom import bond_cos_sin


class AngleHarmonic(BasePotential):
    """Harmonic angle bending potential.

    Energy formula (AMBER / OpenMM table convention)::

        E = 0.5 * k * (theta - theta0)^2

    LAMMPS ``angle_style harmonic`` writes the same polynomial as
    ``E = K (θ − θ0)²`` and states that the usual 1/2 is included in ``K``,
    so ``K_LAMMPS = k / 2``. ``θ = atan2(sin, cos)`` with the C^infty
    ``(cos, sin)`` of :func:`~molpot.potentials.geom.bond_cos_sin`.

    Parameters are stored as type-indexed vectors:
        k[angle_type]: Force constant
        theta0[angle_type]: Equilibrium angle (in radians)

    Attributes:
        k: Force constants [num_angle_types]
        theta0: Equilibrium angles in radians [num_angle_types]
    """

    name = "angle_harmonic_torch"
    type = "angle"

    k: torch.Tensor
    theta0: torch.Tensor

    def __init__(
        self,
        k: torch.Tensor,
        theta0: torch.Tensor,
    ):
        """Initialize AngleHarmonic potential.

        Args:
            k: Force constant vector [num_angle_types]
            theta0: Equilibrium angle vector in radians [num_angle_types]
        """
        super().__init__()

        # Validate shapes
        if k.shape != theta0.shape:
            raise ValueError(
                f"k and theta0 must have same shape, got k: {k.shape}, theta0: {theta0.shape}"
            )

        if k.ndim != 1:
            raise ValueError(f"k must be 1D vector [num_angle_types], got shape {k.shape}")

        # Register as buffers
        self.register_buffer("k", k)
        self.register_buffer("theta0", theta0)

    def forward(self, data: dict[str, Any] | None = None, **kwargs: Any) -> torch.Tensor:
        """Compute harmonic angle energy.

        Args:
            data: Optional dictionary with molecular fields
            **kwargs: Alternate way to pass explicit tensors, including:
                - pos: Positions [N, 3]
                - angle_index: Angle indices [3, num_angles] (i-j-k triplets)
                - angle_types: Angle types [num_angles]

        Returns:
            Total angle energy (scalar)
        """
        # Extract data
        pos = kwargs.get("pos")
        angle_index = kwargs.get("angle_index")
        angle_types = kwargs.get("angle_types")
        term_batch = kwargs.get("term_batch")
        num_graphs = kwargs.get("num_graphs")

        if pos is None and data is not None:
            if isinstance(data, dict):
                pos = data.get("pos")
                angle_index = data.get("angle_index") if angle_index is None else angle_index
                angle_types = data.get("angle_types") if angle_types is None else angle_types
                if pos is None and "atoms" in data:
                    pos = data["atoms"]["x"]

        if pos is None or angle_index is None or angle_types is None:
            raise ValueError("AngleHarmonic requires pos, angle_index, and angle_types.")

        # Convert numpy to torch if needed
        if not isinstance(pos, torch.Tensor):
            pos = torch.from_numpy(pos).float()
            angle_index = torch.from_numpy(angle_index).long()
            angle_types = torch.from_numpy(angle_types).long()

        # Handle empty angles
        if angle_index.size(1) == 0:
            return torch.tensor(0.0, device=pos.device, dtype=pos.dtype)

        # Get positions for each angle (i-j-k where j is the central atom)
        pos_i = pos[angle_index[0]]  # [num_angles, 3]
        pos_j = pos[angle_index[1]]  # [num_angles, 3] (central atom)
        pos_k = pos[angle_index[2]]  # [num_angles, 3]

        # Compute vectors from central atom
        vec_ji = pos_i - pos_j  # [num_angles, 3]
        vec_jk = pos_k - pos_j  # [num_angles, 3]
        cos_theta, sin_theta = bond_cos_sin(vec_ji, vec_jk)
        theta = torch.atan2(sin_theta, cos_theta)

        # Look up parameters for each angle
        k_angles = self.k[angle_types]  # [num_angles]
        theta0_angles = self.theta0[angle_types]  # [num_angles]

        # Compute harmonic energy: E = 0.5 * k * (theta - theta0)^2
        displacement = theta - theta0_angles
        energy_per_angle = 0.5 * k_angles * displacement**2

        # Sum over all angles
        if term_batch is not None:
            total_energy = energy_per_angle.new_zeros(int(num_graphs))
            total_energy.index_add_(0, term_batch.long(), energy_per_angle)
        else:
            total_energy = energy_per_angle.sum()

        return total_energy

    def energy_forces(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Energy and analytic forces ``F = -∂E/∂pos`` in one pass.

        ``E = ½k(θ − θ0)²`` with the C^∞ angle of
        :func:`~molpot.potentials.geom.bond_cos_sin`; forces use the exact
        gradient of that angle (:func:`angle_theta_grad`) with
        ``F_i = −g ∂θ/∂u``, ``F_k = −g ∂θ/∂v``, ``F_j = −(F_i + F_k)`` where
        ``g = k(θ − θ0)``.

        Args:
            **kwargs: ``pos`` ``[N, 3]``, ``angle_index`` ``[3, M]``,
                ``angle_types`` ``[M]``, and optional ``term_batch`` /
                ``num_graphs``.

        Returns:
            ``(energy, forces)`` — energy ``(B,)`` (or scalar) and forces
            ``[N, 3]``.
        """
        pos = kwargs.get("pos")
        angle_index = kwargs.get("angle_index")
        angle_types = kwargs.get("angle_types")
        term_batch = kwargs.get("term_batch")
        num_graphs = kwargs.get("num_graphs")
        if pos is None or angle_index is None or angle_types is None:
            raise ValueError("AngleHarmonic.energy_forces requires pos, angle_index, angle_types.")

        if angle_index.size(1) == 0:
            energy = torch.zeros(
                (int(num_graphs),) if term_batch is not None else (),
                device=pos.device,
                dtype=pos.dtype,
            )
            return energy, pos.new_zeros((pos.shape[0], 3))

        pos_i = pos[angle_index[0]]
        pos_j = pos[angle_index[1]]
        pos_k = pos[angle_index[2]]
        u = pos_i - pos_j
        v = pos_k - pos_j
        cos_theta, sin_theta = bond_cos_sin(u, v)
        theta = torch.atan2(sin_theta, cos_theta)
        k_angles = self.k[angle_types]
        theta0_angles = self.theta0[angle_types]
        displacement = theta - theta0_angles
        energy_per_angle = 0.5 * k_angles * displacement**2

        du, dv = angle_theta_grad(u, v)
        g = k_angles * displacement
        force_on_i = -(g.unsqueeze(-1) * du)
        force_on_k = -(g.unsqueeze(-1) * dv)
        force_on_j = -(force_on_i + force_on_k)
        forces = pos.new_zeros((pos.shape[0], 3))
        forces.index_add_(0, angle_index[0], force_on_i)
        forces.index_add_(0, angle_index[1], force_on_j)
        forces.index_add_(0, angle_index[2], force_on_k)

        if term_batch is not None:
            energy = energy_per_angle.new_zeros(int(num_graphs))
            energy.index_add_(0, term_batch.long(), energy_per_angle)
            return energy, forces
        return energy_per_angle.sum(), forces

    def __repr__(self) -> str:
        num_types = len(self.k)
        return f"AngleHarmonic(num_angle_types={num_types})"
