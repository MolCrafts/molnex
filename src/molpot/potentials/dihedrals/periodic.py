"""Class-I multi-term cosine proper torsion.

Energy formula (OpenMM §19.4 / SMIRNOFF / AMBER)::

    E = Σ_torsions Σ_m (k[t,m] / s[t]) * [1 + cos(n[m] * φ − γ[t,m])]

where ``t = proper_types[i]``, ``s = idivf``, ``n = periodicity``,
``γ = phase``. ``cos(nφ − γ)`` is the Chebyshev recurrence in
``(cos φ, sin φ)`` used by LAMMPS ``dihedral_style opls`` / CHARMM; ``φ``
is never formed. ``(cos φ, sin φ)`` come from
:func:`~molpot.potentials.geom.dihedral_cos_sin`.

References:
    OpenMM User Guide §19.4
    SMIRNOFF specification (OpenFF) — proper torsions
    Cornell et al., JACS 1995 DOI 10.1021/ja00124a002 (AMBER)
    LAMMPS ``dihedral_opls``
"""

from __future__ import annotations

from typing import Any

import torch

from molpot.potentials.base import BasePotential
from molpot.potentials.bonded_forces import dihedral_phi_grad
from molpot.potentials.geom import dihedral_cos_sin, fourier_cos, fourier_cos_sin


def _dihedral_bond_vectors(
    pos: torch.Tensor,
    index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``b1 = j−i``, ``b2 = k−j``, ``b3 = l−k`` for COO ``index`` ``[4, N]``."""
    return (
        pos[index[1]] - pos[index[0]],
        pos[index[2]] - pos[index[1]],
        pos[index[3]] - pos[index[2]],
    )


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
    cos_phi, sin_phi = dihedral_cos_sin(*_dihedral_bond_vectors(pos, index))
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
        term_batch = kwargs.get("term_batch")
        num_graphs = kwargs.get("num_graphs")

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

        cos_phi, sin_phi = dihedral_cos_sin(*_dihedral_bond_vectors(pos, proper_index))
        k_t = self.k[proper_types]
        phase_t = self.phase[proper_types]
        s_t = self.idivf[proper_types]
        cos_arg = fourier_cos(cos_phi, sin_phi, self.periodicity, phase_t)
        energy_per_proper = ((k_t / s_t.unsqueeze(1)) * (1.0 + cos_arg)).sum(dim=1)
        if term_batch is not None:
            energy = energy_per_proper.new_zeros(int(num_graphs))
            energy.index_add_(0, term_batch.long(), energy_per_proper)
            return energy
        return energy_per_proper.sum()

    def energy_forces(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Energy and analytic forces ``F = -∂E/∂pos`` in one pass.

        With ``g = Σ_m (k[t,m]/s) · n[m] · sin(n[m]φ − γ[t,m]) = −∂E/∂φ``, the
        force is ``g · ∂φ/∂pos`` using the Blondel–Karplus gradient
        (:func:`dihedral_phi_grad`). ``φ`` is never formed; the sine comes from
        the same Chebyshev recurrence as the energy.

        Args:
            **kwargs: ``pos`` ``[N, 3]``, ``proper_index`` ``[4, M]``,
                ``proper_types`` ``[M]``, and optional ``term_batch`` /
                ``num_graphs``.

        Returns:
            ``(energy, forces)`` — energy ``(B,)`` (or scalar) and forces
            ``[N, 3]``.
        """
        pos = kwargs.get("pos")
        proper_index = kwargs.get("proper_index")
        proper_types = kwargs.get("proper_types")
        term_batch = kwargs.get("term_batch")
        num_graphs = kwargs.get("num_graphs")
        if pos is None or proper_index is None or proper_types is None:
            raise ValueError(
                "ProperTorsionPeriodic.energy_forces requires pos, proper_index, proper_types."
            )
        _require_index_4_by_n(proper_index, "proper_index")

        if proper_index.size(1) == 0:
            energy = torch.zeros(
                (int(num_graphs),) if term_batch is not None else (),
                device=pos.device,
                dtype=pos.dtype,
            )
            return energy, pos.new_zeros((pos.shape[0], 3))

        b1, b2, b3 = _dihedral_bond_vectors(pos, proper_index)
        cos_phi, sin_phi = dihedral_cos_sin(b1, b2, b3)
        k_t = self.k[proper_types]
        phase_t = self.phase[proper_types]
        s_t = self.idivf[proper_types]
        cos_arg, sin_arg = fourier_cos_sin(cos_phi, sin_phi, self.periodicity, phase_t)
        scale = k_t / s_t.unsqueeze(1)
        energy_per = (scale * (1.0 + cos_arg)).sum(dim=1)

        n = self.periodicity.to(dtype=k_t.dtype)
        g = (scale * n.unsqueeze(0) * sin_arg).sum(dim=1)
        fi, fj, fk, fl = dihedral_phi_grad(b1, b2, b3)
        g = g.unsqueeze(-1)
        forces = pos.new_zeros((pos.shape[0], 3))
        forces.index_add_(0, proper_index[0], g * fi)
        forces.index_add_(0, proper_index[1], g * fj)
        forces.index_add_(0, proper_index[2], g * fk)
        forces.index_add_(0, proper_index[3], g * fl)

        if term_batch is not None:
            energy = energy_per.new_zeros(int(num_graphs))
            energy.index_add_(0, term_batch.long(), energy_per)
            return energy, forces
        return energy_per.sum(), forces

    def __repr__(self) -> str:
        n_types, n_terms = self.k.shape
        return f"ProperTorsionPeriodic(n_types={n_types}, n_terms={n_terms})"
