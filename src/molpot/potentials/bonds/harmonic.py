import torch

from molpot.potentials.base import BasePotential


def _require_bond_index(bond_index: torch.Tensor) -> None:
    """Refuse anything but COO ``[2, num_bonds]``.

    Deliberately a different shape from the geometric neighbour graph
    ``pair_index`` ``[E, 2]``, so a cutoff pair can never be silently
    mis-indexed as a bond (CLAUDE.md, bond_index vs pair_index).
    """
    if bond_index.ndim != 2 or bond_index.shape[0] != 2:
        raise ValueError(
            f"bond_index must be COO [2, num_bonds] covalent topology; got shape "
            f"{tuple(bond_index.shape)}. A neighbour-graph pair_index [E, 2] is not "
            "a bond list — see the bond_index vs pair_index distinction in CLAUDE.md."
        )


class BondHarmonic(BasePotential):
    """Harmonic bond stretching potential.

    Energy formula:
        E = 0.5 * k * (r - r0)^2

    Parameters are stored as type-indexed vectors:
        k[bond_type]: Force constant
        r0[bond_type]: Equilibrium bond length

    Attributes:
        k: Force constants [num_bond_types]
        r0: Equilibrium bond lengths [num_bond_types]
    """

    name = "bond_harmonic_torch"
    type = "bond"

    k: torch.Tensor
    r0: torch.Tensor

    def __init__(
        self,
        k: torch.Tensor,
        r0: torch.Tensor,
    ):
        """Initialize BondHarmonic potential.

        Args:
            k: Force constant vector [num_bond_types]
            r0: Equilibrium bond length vector [num_bond_types]
        """
        super().__init__()

        # Validate shapes
        if k.shape != r0.shape:
            raise ValueError(f"k and r0 must have same shape, got k: {k.shape}, r0: {r0.shape}")

        if k.ndim != 1:
            raise ValueError(f"k must be 1D vector [num_bond_types], got shape {k.shape}")

        # Register as buffers
        self.register_buffer("k", k)
        self.register_buffer("r0", r0)

    def forward(
        self,
        *,
        pos: torch.Tensor,
        bond_index: torch.Tensor,
        bond_types: torch.Tensor | None = None,
        term_batch: torch.Tensor | None = None,
        num_graphs: int,
        k: torch.Tensor | None = None,
        r0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute harmonic bond energy.

        Args:
            pos: Positions ``[N, 3]``.
            bond_index: COO bond indices ``[2, num_bonds]`` (NOT ``[E, 2]``).
                There is no ``pair_index`` route — a geometric cutoff edge is
                not a bond.
            bond_types: Bond types ``[num_bonds]``, or ``None`` when the tables
                already carry one row per bond.
            term_batch: Graph index per bond ``[num_bonds]``; ``None`` sums to
                a scalar.
            num_graphs: Graph count ``B`` — the length of the per-graph energy.
            k: Per-call force-constant table; ``None`` prices the registered one.
            r0: Per-call equilibrium-length table; ``None`` prices the
                registered one.

        Returns:
            Per-graph energy ``(B,)`` when ``term_batch`` is given, else a scalar.

        Raises:
            ValueError: ``bond_index`` is not COO ``[2, num_bonds]`` (e.g. a
                geometric ``pair_index`` ``[E, 2]`` was passed).
        """
        _require_bond_index(bond_index)
        tables = {"k": k, "r0": r0}

        bond_vec = pos[bond_index[1]] - pos[bond_index[0]]  # [num_bonds, 3]
        bond_lengths = torch.norm(bond_vec, dim=-1)  # [num_bonds]

        k_bonds = self.rows("k", tables, bond_types)  # [num_bonds]
        r0_bonds = self.rows("r0", tables, bond_types)  # [num_bonds]

        displacement = bond_lengths - r0_bonds
        energy_per_bond = 0.5 * k_bonds * displacement**2
        return self.per_graph(energy_per_bond, term_batch, num_graphs)

    def energy_forces(
        self,
        *,
        pos: torch.Tensor,
        bond_index: torch.Tensor,
        bond_types: torch.Tensor | None = None,
        term_batch: torch.Tensor | None = None,
        num_graphs: int,
        k: torch.Tensor | None = None,
        r0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Energy and analytic forces ``F = -∂E/∂pos`` in one pass.

        Closed form ``F_i = k(r − r0) r̂``, ``F_j = −F_i`` (``r̂ = (pos_j −
        pos_i)/r``). No position leaf is built, so no double backward is paid;
        the force still carries the ordinary gradient w.r.t. ``k`` and ``r0``.

        Args:
            pos: Positions ``[N, 3]``.
            bond_index: COO bond indices ``[2, num_bonds]``.
            bond_types: Bond types ``[num_bonds]``, or ``None`` for
                per-interaction tables.
            term_batch: Graph index per bond; ``None`` sums to a scalar.
            num_graphs: Graph count ``B``.
            k: Per-call force-constant table; ``None`` prices the registered one.
            r0: Per-call equilibrium-length table; ``None`` prices the
                registered one.

        Returns:
            ``(energy, forces)`` — energy ``(B,)`` (or scalar) and forces
            ``[N, 3]``.
        """
        _require_bond_index(bond_index)
        tables = {"k": k, "r0": r0}

        bond_vec = pos[bond_index[1]] - pos[bond_index[0]]
        bond_lengths = torch.norm(bond_vec, dim=-1)
        k_bonds = self.rows("k", tables, bond_types)
        r0_bonds = self.rows("r0", tables, bond_types)
        displacement = bond_lengths - r0_bonds
        energy_per_bond = 0.5 * k_bonds * displacement**2

        rhat = bond_vec / bond_lengths.unsqueeze(-1)
        force_on_i = (k_bonds * displacement).unsqueeze(-1) * rhat
        forces = pos.new_zeros((pos.shape[0], 3))
        forces.index_add_(0, bond_index[0], force_on_i)
        forces.index_add_(0, bond_index[1], -force_on_i)
        return self.per_graph(energy_per_bond, term_batch, num_graphs), forces

    def __repr__(self) -> str:
        num_types = len(self.k)
        return f"BondHarmonic(num_bond_types={num_types})"
