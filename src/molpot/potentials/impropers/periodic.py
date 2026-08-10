"""Class-I multi-term cosine improper torsion.

Same cosine energy form as :class:`~molpot.potentials.dihedrals.periodic.ProperTorsionPeriodic`::

    E = Σ_impropers Σ_m (k[t,m] / s[t]) * [1 + cos(n[m] * φ − γ[t,m])]

**Index layout (molrs Topology source of truth):** ``improper_index`` is
``[4, N]`` with rows ``[center, i, j, k]`` — **center at row 0**, matching
molrs ``Topology`` impropers and molpy force-field topology. Internally the
dihedral angle is evaluated on the ordered quartet ``(i, center, j, k)``
(rows 1, 0, 2, 3). OpenFF trefoil reordering is an export adapter, not a
second in-kernel layout.

References:
    OpenMM User Guide §19.5
    SMIRNOFF specification (OpenFF) — impropers
    molrs ``Topology`` impropers ``[center, i, j, k]``
"""

from __future__ import annotations

from typing import Any

import torch

from molpot.potentials.base import BasePotential
from molpot.potentials.dihedrals.periodic import _dihedral_phi, _require_index_4_by_n


def _improper_to_dihedral_index(improper_index: torch.Tensor) -> torch.Tensor:
    """Map molrs ``[center, i, j, k]`` → dihedral order ``[i, center, j, k]``."""
    return torch.stack(
        (
            improper_index[1],
            improper_index[0],
            improper_index[2],
            improper_index[3],
        ),
        dim=0,
    )


class ImproperPeriodic(BasePotential):
    """Class-I multi-term cosine improper torsion.

    Energy::

        E = Σ_n (k_n / s) * [1 + cos(n * φ − γ_n)]

    Central atom of each improper is at **row index 0** of ``improper_index``
    (molrs ``[center, i, j, k]``).

    Attributes:
        k: Barrier heights ``[n_types, n_terms]``.
        periodicity: Periodicities ``[n_terms]``.
        phase: Phase offsets ``[n_types, n_terms]``.
        idivf: Scale factors ``[n_types]``.
    """

    name = "improper_periodic_torch"
    type = "improper"

    k: torch.Tensor
    periodicity: torch.Tensor
    phase: torch.Tensor
    idivf: torch.Tensor

    def __init__(
        self,
        k: torch.Tensor,
        periodicity: torch.Tensor,
        phase: torch.Tensor,
        idivf: torch.Tensor,
    ) -> None:
        """Initialize ImproperPeriodic.

        Args:
            k: Barrier heights ``[n_types, n_terms]``.
            periodicity: Integer periods ``[n_terms]`` (must be > 0).
            phase: Phase offsets γ in radians ``[n_types, n_terms]``.
            idivf: Scale factors ``s`` per type ``[n_types]``.
        """
        super().__init__()

        if k.ndim != 2:
            raise ValueError(f"k must be [n_types, n_terms], got shape {tuple(k.shape)}")
        if phase.shape != k.shape:
            raise ValueError(f"phase must match k shape {tuple(k.shape)}, got {tuple(phase.shape)}")
        n_types, n_terms = k.shape
        if periodicity.ndim != 1 or periodicity.shape[0] != n_terms:
            raise ValueError(
                f"periodicity must be [n_terms]={n_terms}, got shape {tuple(periodicity.shape)}"
            )
        if idivf.ndim != 1 or idivf.shape[0] != n_types:
            raise ValueError(f"idivf must be [n_types]={n_types}, got shape {tuple(idivf.shape)}")
        if not bool((periodicity > 0).all()):
            raise ValueError(f"periodicity entries must be > 0, got {periodicity.tolist()}")

        self.register_buffer("k", k)
        self.register_buffer("periodicity", periodicity)
        self.register_buffer("phase", phase)
        self.register_buffer("idivf", idivf)

    def forward(self, data: dict[str, Any] | None = None, **kwargs: Any) -> torch.Tensor:
        """Compute Class-I multi-term improper torsion energy.

        Args:
            data: Optional dictionary with molecular fields.
            **kwargs: Explicit tensors:
                - pos: Positions ``[n_atoms, 3]``.
                - improper_index: COO improper indices ``[4, N]`` molrs layout
                  ``[center, i, j, k]`` (center at **row 0**).
                - improper_types: Improper types ``[N]``.

        Returns:
            Total improper torsion energy (scalar).

        Raises:
            ValueError: Missing inputs or ``improper_index`` not ``[4, N]``.
        """
        pos = kwargs.get("pos")
        improper_index = kwargs.get("improper_index")
        improper_types = kwargs.get("improper_types")

        if data is not None and isinstance(data, dict):
            if pos is None:
                pos = data.get("pos")
                if pos is None and isinstance(data.get("atoms"), dict):
                    pos = data["atoms"].get("pos")
            if improper_index is None:
                improper_index = data.get("improper_index")
            if improper_types is None:
                improper_types = data.get("improper_types")

        if pos is None or improper_index is None or improper_types is None:
            raise ValueError("ImproperPeriodic requires pos, improper_index, and improper_types.")

        if not isinstance(pos, torch.Tensor):
            pos = torch.from_numpy(pos).float()
            improper_index = torch.from_numpy(improper_index).long()
            improper_types = torch.from_numpy(improper_types).long()

        _require_index_4_by_n(improper_index, "improper_index")

        if improper_index.size(1) == 0:
            return torch.tensor(0.0, device=pos.device, dtype=pos.dtype)

        # molrs [center,i,j,k] → dihedral (i,center,j,k) for atan2 path.
        phi = _dihedral_phi(pos, _improper_to_dihedral_index(improper_index))

        k_t = self.k[improper_types]
        phase_t = self.phase[improper_types]
        s_t = self.idivf[improper_types]
        n = self.periodicity.to(dtype=pos.dtype)

        arg = n.unsqueeze(0) * phi.unsqueeze(1) - phase_t
        term = (k_t / s_t.unsqueeze(1)) * (1.0 + torch.cos(arg))
        return term.sum()

    def __repr__(self) -> str:
        n_types, n_terms = self.k.shape
        return f"ImproperPeriodic(n_types={n_types}, n_terms={n_terms})"
