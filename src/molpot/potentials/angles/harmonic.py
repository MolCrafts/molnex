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
    category = "angle"

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

    def forward(
        self,
        *,
        pos: torch.Tensor,
        angle_index: torch.Tensor,
        angle_types: torch.Tensor | None = None,
        term_batch: torch.Tensor | None = None,
        num_graphs: int,
        k: torch.Tensor | None = None,
        theta0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute harmonic angle energy.

        Args:
            pos: Positions ``[N, 3]``.
            angle_index: Angle indices ``[3, num_angles]`` (i-j-k triplets,
                ``j`` central).
            angle_types: Angle types ``[num_angles]``, or ``None`` when the
                tables already carry one row per angle.
            term_batch: Graph index per angle ``[num_angles]``; ``None`` sums
                to a scalar.
            num_graphs: Graph count ``B`` — the length of the per-graph energy.
            k: Per-call force-constant table; ``None`` prices the registered one.
            theta0: Per-call equilibrium-angle table; ``None`` prices the
                registered one.

        Returns:
            Per-graph energy ``(B,)`` when ``term_batch`` is given, else a scalar.
        """
        tables = {"k": k, "theta0": theta0}

        # Vectors from the central atom j
        vec_ji = pos[angle_index[0]] - pos[angle_index[1]]  # [num_angles, 3]
        vec_jk = pos[angle_index[2]] - pos[angle_index[1]]  # [num_angles, 3]
        cos_theta, sin_theta = bond_cos_sin(vec_ji, vec_jk)
        theta = torch.atan2(sin_theta, cos_theta)

        k_angles = self.rows("k", tables, angle_types)  # [num_angles]
        theta0_angles = self.rows("theta0", tables, angle_types)  # [num_angles]

        displacement = theta - theta0_angles
        energy_per_angle = 0.5 * k_angles * displacement**2
        return self.per_graph(energy_per_angle, term_batch, num_graphs)

    def energy_forces(
        self,
        *,
        pos: torch.Tensor,
        angle_index: torch.Tensor,
        angle_types: torch.Tensor | None = None,
        term_batch: torch.Tensor | None = None,
        num_graphs: int,
        k: torch.Tensor | None = None,
        theta0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Energy and analytic forces ``F = -∂E/∂pos`` in one pass.

        ``E = ½k(θ − θ0)²`` with the C^∞ angle of
        :func:`~molpot.potentials.geom.bond_cos_sin`; forces use the exact
        gradient of that angle (:func:`angle_theta_grad`) with
        ``F_i = −g ∂θ/∂u``, ``F_k = −g ∂θ/∂v``, ``F_j = −(F_i + F_k)`` where
        ``g = k(θ − θ0)``.

        Args:
            pos: Positions ``[N, 3]``.
            angle_index: Angle indices ``[3, num_angles]``.
            angle_types: Angle types ``[num_angles]``, or ``None`` for
                per-interaction tables.
            term_batch: Graph index per angle; ``None`` sums to a scalar.
            num_graphs: Graph count ``B``.
            k: Per-call force-constant table; ``None`` prices the registered one.
            theta0: Per-call equilibrium-angle table; ``None`` prices the
                registered one.

        Returns:
            ``(energy, forces)`` — energy ``(B,)`` (or scalar) and forces
            ``[N, 3]``.
        """
        tables = {"k": k, "theta0": theta0}

        u = pos[angle_index[0]] - pos[angle_index[1]]
        v = pos[angle_index[2]] - pos[angle_index[1]]
        cos_theta, sin_theta = bond_cos_sin(u, v)
        theta = torch.atan2(sin_theta, cos_theta)
        k_angles = self.rows("k", tables, angle_types)
        theta0_angles = self.rows("theta0", tables, angle_types)
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
        return self.per_graph(energy_per_angle, term_batch, num_graphs), forces

    def __repr__(self) -> str:
        num_types = len(self.k)
        return f"AngleHarmonic(num_angle_types={num_types})"
