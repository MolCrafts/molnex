import torch

from molpot.potentials.base import BasePotential
from molpot.potentials.dihedrals.periodic import _dihedral_phi, _require_index_4_by_n


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

    def forward(
        self,
        *,
        pos: torch.Tensor,
        dihedral_index: torch.Tensor,
        dihedral_types: torch.Tensor | None = None,
        term_batch: torch.Tensor | None = None,
        num_graphs: int,
        k: torch.Tensor | None = None,
        phi0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute harmonic dihedral energy.

        Args:
            pos: Positions ``[N, 3]``.
            dihedral_index: Dihedral indices ``[4, num_dihedrals]`` (i-j-k-l).
            dihedral_types: Dihedral types ``[num_dihedrals]``, or ``None``
                when the tables already carry one row per dihedral.
            term_batch: Graph index per dihedral ``[num_dihedrals]``; ``None``
                sums to a scalar.
            num_graphs: Graph count ``B`` — the length of the per-graph energy.
            k: Per-call force-constant table; ``None`` prices the registered one.
            phi0: Per-call equilibrium-angle table; ``None`` prices the
                registered one.

        Returns:
            Per-graph energy ``(B,)`` when ``term_batch`` is given, else a scalar.

        Raises:
            ValueError: ``dihedral_index`` is not ``[4, num_dihedrals]``.
        """
        _require_index_4_by_n(dihedral_index, "dihedral_index")
        tables = {"k": k, "phi0": phi0}

        phi = _dihedral_phi(pos, dihedral_index)
        k_dihedrals = self.rows("k", tables, dihedral_types)  # [num_dihedrals]
        phi0_dihedrals = self.rows("phi0", tables, dihedral_types)  # [num_dihedrals]

        displacement = phi - phi0_dihedrals
        energy_per_dihedral = 0.5 * k_dihedrals * displacement**2
        return self.per_graph(energy_per_dihedral, term_batch, num_graphs)

    def __repr__(self) -> str:
        num_types = len(self.k)
        return f"DihedralHarmonic(num_dihedral_types={num_types})"
