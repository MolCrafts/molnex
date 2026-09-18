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
from molpot.potentials.bonded_forces import dihedral_phi_grad
from molpot.potentials.dihedrals.periodic import (
    _dihedral_bond_vectors,
    _require_index_4_by_n,
)
from molpot.potentials.geom import dihedral_cos_sin, fourier_cos, fourier_cos_sin


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
        periodicity: Periodicities ``[n_types, n_terms]``.
        phase: Phase offsets ``[n_types, n_terms]``.
    """

    name = "improper_periodic_torch"
    type = "improper"

    k: torch.Tensor
    periodicity: torch.Tensor
    phase: torch.Tensor

    def __init__(
        self,
        k: torch.Tensor,
        periodicity: torch.Tensor,
        phase: torch.Tensor,
    ) -> None:
        """Initialize ImproperPeriodic.

        Args:
            k: Barrier heights ``[n_types, n_terms]``.
            periodicity: Integer periods ``[n_types, n_terms]`` (must be > 0).
            phase: Phase offsets γ in radians ``[n_types, n_terms]``.
        """
        super().__init__()

        if k.ndim != 2:
            raise ValueError(f"k must be [n_types, n_terms], got shape {tuple(k.shape)}")
        if phase.shape != k.shape:
            raise ValueError(f"phase must match k shape {tuple(k.shape)}, got {tuple(phase.shape)}")
        n_types, n_terms = k.shape
        if periodicity.ndim == 1 and periodicity.shape[0] == k.shape[1]:
            # A multiplicity set shared by every type — the OPLS/CHARMM case of
            # the general per-type table.
            periodicity = periodicity.expand(k.shape[0], -1).contiguous()
        if periodicity.shape != k.shape:
            raise ValueError(
                f"periodicity must match k shape {tuple(k.shape)}, got {tuple(periodicity.shape)}"
            )
        if not bool((periodicity > 0).all()):
            raise ValueError(f"periodicity entries must be > 0, got {periodicity.tolist()}")

        self.register_buffer("k", k)
        self.register_buffer("periodicity", periodicity)
        self.register_buffer("phase", phase)
        # Python int, fixed at construction: a trip count read off a
        # tensor is a data-dependent loop bound and a graph break.
        #
        # An empty table is a real, ordinary case: a learned field declares its
        # styles with no rows and the head fills them per batch, so whichever
        # batch assembles this container may carry no term of this kind at all
        # (a small molecule has no improper). ``max()`` over zero elements
        # raises, and guessing 1 is worse than raising — the next batch that
        # *does* bring rows gathers past the ladder and dies in a CUDA
        # index assert, far from here. With no rows the only capacity the
        # table still states is its own width, which for the canonical
        # ``n = 1…T`` tables these heads emit is exactly the depth needed.
        self.n_max = (
            int(periodicity.max()) if periodicity.numel() else max(int(periodicity.shape[-1]), 1)
        )

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
        term_batch = kwargs.get("term_batch")
        num_graphs = kwargs.get("num_graphs")

        if data is not None and isinstance(data, dict):
            if pos is None:
                pos = data.get("pos")
                if pos is None and isinstance(data.get("atoms"), dict):
                    pos = data["atoms"].get("pos")
            if improper_index is None:
                improper_index = data.get("improper_index")
            if improper_types is None:
                improper_types = data.get("improper_types")

        if pos is None or improper_index is None:
            raise ValueError("ImproperPeriodic requires pos, improper_index, and improper_types.")

        if not isinstance(pos, torch.Tensor):
            pos = torch.from_numpy(pos).float()
            improper_index = torch.from_numpy(improper_index).long()
            improper_types = (
                None if improper_types is None else torch.from_numpy(improper_types).long()
            )

        _require_index_4_by_n(improper_index, "improper_index")

        if improper_index.size(1) == 0:
            return torch.tensor(0.0, device=pos.device, dtype=pos.dtype)

        # molrs [center,i,j,k] → dihedral (i, center, j, k); Chebyshev, no φ.
        cos_phi, sin_phi = dihedral_cos_sin(
            *_dihedral_bond_vectors(pos, _improper_to_dihedral_index(improper_index))
        )
        k_t = self.rows("k", kwargs, improper_types)
        phase_t = self.rows("phase", kwargs, improper_types)
        n_t = self.rows("periodicity", kwargs, improper_types)
        cos_arg = fourier_cos(cos_phi, sin_phi, n_t, phase_t, n_max=self.n_max)
        energy_per = (k_t * (1.0 + cos_arg)).sum(dim=1)
        if term_batch is not None:
            energy = energy_per.new_zeros(int(num_graphs))
            energy.index_add_(0, term_batch.long(), energy_per)
            return energy
        return energy_per.sum()

    def energy_forces(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Energy and analytic forces ``F = -∂E/∂pos`` in one pass.

        Evaluated on the dihedral quartet ``(i, center, j, k)`` (rows 1, 0, 2, 3
        of ``improper_index``); the Blondel–Karplus gradient is scattered back
        onto the original ``[center, i, j, k]`` atoms. ``g = −∂E/∂φ`` is
        ``Σ_m (k[t,m]/s) · n[m] · sin(n[m]φ − γ[t,m])``.

        Args:
            **kwargs: ``pos`` ``[N, 3]``, ``improper_index`` ``[4, M]`` (center
                first), ``improper_types`` ``[M]``, and optional ``term_batch``
                / ``num_graphs``.

        Returns:
            ``(energy, forces)`` — energy ``(B,)`` (or scalar) and forces
            ``[N, 3]``.
        """
        pos = kwargs.get("pos")
        improper_index = kwargs.get("improper_index")
        improper_types = kwargs.get("improper_types")
        term_batch = kwargs.get("term_batch")
        num_graphs = kwargs.get("num_graphs")
        if pos is None or improper_index is None:
            raise ValueError(
                "ImproperPeriodic.energy_forces requires pos, improper_index, improper_types."
            )
        _require_index_4_by_n(improper_index, "improper_index")

        if improper_index.size(1) == 0:
            energy = torch.zeros(
                (int(num_graphs),) if term_batch is not None else (),
                device=pos.device,
                dtype=pos.dtype,
            )
            return energy, pos.new_zeros((pos.shape[0], 3))

        dihedral_index = _improper_to_dihedral_index(improper_index)
        b1, b2, b3 = _dihedral_bond_vectors(pos, dihedral_index)
        cos_phi, sin_phi = dihedral_cos_sin(b1, b2, b3)
        k_t = self.rows("k", kwargs, improper_types)
        phase_t = self.rows("phase", kwargs, improper_types)
        n_t = self.rows("periodicity", kwargs, improper_types)
        cos_arg, sin_arg = fourier_cos_sin(cos_phi, sin_phi, n_t, phase_t, n_max=self.n_max)
        scale = k_t
        energy_per = (scale * (1.0 + cos_arg)).sum(dim=1)

        n = n_t.to(dtype=k_t.dtype)
        g = (scale * n * sin_arg).sum(dim=1)
        fi, fj, fk, fl = dihedral_phi_grad(b1, b2, b3)
        g = g.unsqueeze(-1)
        forces = pos.new_zeros((pos.shape[0], 3))
        forces.index_add_(0, improper_index[1], g * fi)  # dihedral i
        forces.index_add_(0, improper_index[0], g * fj)  # dihedral j (center)
        forces.index_add_(0, improper_index[2], g * fk)  # dihedral k
        forces.index_add_(0, improper_index[3], g * fl)  # dihedral l

        if term_batch is not None:
            energy = energy_per.new_zeros(int(num_graphs))
            energy.index_add_(0, term_batch.long(), energy_per)
            return energy, forces
        return energy_per.sum(), forces

    def __repr__(self) -> str:
        n_types, n_terms = self.k.shape
        return f"ImproperPeriodic(n_types={n_types}, n_terms={n_terms})"
