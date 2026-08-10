"""Class-I multi-term cosine proper torsion.

Energy formula (OpenMM §19.4 / SMIRNOFF / AMBER)::

    E = Σ_torsions Σ_m (k[t,m] / s[t]) * [1 + cos(n[m] * φ − γ[t,m])]

where ``t = proper_types[i]``, ``s = idivf``, ``n = periodicity``,
``γ = phase``, and ``φ`` is the i-j-k-l dihedral angle from the standard
atan2(n1, n2) construction (same geometry as :class:`DihedralHarmonic`).

References:
    OpenMM User Guide §19.4
    SMIRNOFF specification (OpenFF) — proper torsions
    Cornell et al., JACS 1995 DOI 10.1021/ja00124a002 (AMBER)
"""

from __future__ import annotations

from typing import Any

import torch

from molpot.potentials.base import BasePotential


def _dihedral_phi(
    pos: torch.Tensor,
    index: torch.Tensor,
) -> torch.Tensor:
    """Compute dihedral angles for COO ``index`` of shape ``[4, N]``.

    Args:
        pos: Positions ``[n_atoms, 3]``.
        index: Torsion indices ``[4, N]`` (i-j-k-l).

    Returns:
        Dihedral angles ``φ`` of shape ``[N]`` in ``[-π, π]``.
    """
    pos_i = pos[index[0]]
    pos_j = pos[index[1]]
    pos_k = pos[index[2]]
    pos_l = pos[index[3]]

    b1 = pos_j - pos_i
    b2 = pos_k - pos_j
    b3 = pos_l - pos_k

    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)

    n1_norm = n1 / (torch.norm(n1, dim=-1, keepdim=True) + 1e-8)
    n2_norm = n2 / (torch.norm(n2, dim=-1, keepdim=True) + 1e-8)

    cos_phi = torch.sum(n1_norm * n2_norm, dim=-1)
    cos_phi = torch.clamp(cos_phi, -1.0, 1.0)

    b2_norm = b2 / (torch.norm(b2, dim=-1, keepdim=True) + 1e-8)
    cross_n1_n2 = torch.cross(n1_norm, n2_norm, dim=-1)
    sin_phi = torch.sum(cross_n1_n2 * b2_norm, dim=-1)

    return torch.atan2(sin_phi, cos_phi)


def _require_index_4_by_n(index: torch.Tensor, name: str) -> None:
    if index.ndim != 2 or index.shape[0] != 4:
        raise ValueError(
            f"{name} must be COO [4, N] (i-j-k-l); got shape {tuple(index.shape)}. "
            "A geometric edge_index [E, 2] or row-major [N, 4] is not a torsion list."
        )


class ProperTorsionPeriodic(BasePotential):
    """Class-I multi-term cosine proper torsion.

    Energy::

        E = Σ_n (k_n / s) * [1 + cos(n * φ − γ_n)]

    Parameters are type-indexed tables:

    - ``k``: ``[n_types, n_terms]``
    - ``periodicity``: ``[n_terms]`` (shared integer periods)
    - ``phase``: ``[n_types, n_terms]`` (radians)
    - ``idivf``: ``[n_types]`` AMBER scale factor ``s``

    Attributes:
        k: Barrier heights ``[n_types, n_terms]``.
        periodicity: Periodicities ``[n_terms]``.
        phase: Phase offsets ``[n_types, n_terms]``.
        idivf: Identity / scale divisors ``[n_types]``.
    """

    name = "proper_torsion_periodic_torch"
    type = "dihedral"

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
        """Initialize ProperTorsionPeriodic.

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
        """Compute Class-I multi-term proper torsion energy.

        Args:
            data: Optional dictionary with molecular fields.
            **kwargs: Explicit tensors:
                - pos: Positions ``[n_atoms, 3]``.
                - proper_index: COO proper indices ``[4, N]`` (i-j-k-l).
                - proper_types: Proper types ``[N]``.

        Returns:
            Total proper torsion energy (scalar).

        Raises:
            ValueError: Missing inputs or ``proper_index`` not ``[4, N]``.
        """
        pos = kwargs.get("pos")
        proper_index = kwargs.get("proper_index")
        proper_types = kwargs.get("proper_types")

        if data is not None and isinstance(data, dict):
            if pos is None:
                pos = data.get("pos")
                if pos is None and isinstance(data.get("atoms"), dict):
                    pos = data["atoms"].get("pos")
            if proper_index is None:
                proper_index = data.get("proper_index")
            if proper_types is None:
                proper_types = data.get("proper_types")

        if pos is None or proper_index is None or proper_types is None:
            raise ValueError("ProperTorsionPeriodic requires pos, proper_index, and proper_types.")

        if not isinstance(pos, torch.Tensor):
            pos = torch.from_numpy(pos).float()
            proper_index = torch.from_numpy(proper_index).long()
            proper_types = torch.from_numpy(proper_types).long()

        _require_index_4_by_n(proper_index, "proper_index")

        if proper_index.size(1) == 0:
            return torch.tensor(0.0, device=pos.device, dtype=pos.dtype)

        phi = _dihedral_phi(pos, proper_index)  # [N]

        # Look up per-torsion parameter rows: [N, n_terms]
        k_t = self.k[proper_types]
        phase_t = self.phase[proper_types]
        s_t = self.idivf[proper_types]  # [N]
        n = self.periodicity.to(dtype=pos.dtype)  # [n_terms]

        # Broadcast: phi [N, 1], n [1, n_terms] → [N, n_terms]
        arg = n.unsqueeze(0) * phi.unsqueeze(1) - phase_t
        term = (k_t / s_t.unsqueeze(1)) * (1.0 + torch.cos(arg))
        return term.sum()

    def __repr__(self) -> str:
        n_types, n_terms = self.k.shape
        return f"ProperTorsionPeriodic(n_types={n_types}, n_terms={n_terms})"
