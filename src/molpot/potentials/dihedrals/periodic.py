"""Class-I multi-term cosine proper torsion.

Energy formula (OpenMM §19.4 / SMIRNOFF / AMBER)::

    E = Σ_torsions Σ_m k[t,m] * [1 + cos(n[m] * φ − γ[t,m])]

where ``t = proper_types[i]``, ``n = periodicity``,
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
    - ``periodicity``: ``[n_types, n_terms]`` (per type, per term)
    - ``phase``: ``[n_types, n_terms]`` (radians)

    Attributes:
        k: Barrier heights ``[n_types, n_terms]``.
        periodicity: Periodicities ``[n_types, n_terms]``.
        phase: Phase offsets ``[n_types, n_terms]``.
    """

    name = "proper_torsion_periodic_torch"
    category = "dihedral"

    k: torch.Tensor
    periodicity: torch.Tensor
    phase: torch.Tensor

    def __init__(
        self,
        k: torch.Tensor,
        periodicity: torch.Tensor,
        phase: torch.Tensor,
    ) -> None:
        """Initialize ProperTorsionPeriodic.

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

    def forward(
        self,
        *,
        pos: torch.Tensor,
        proper_index: torch.Tensor,
        proper_types: torch.Tensor | None = None,
        term_batch: torch.Tensor | None = None,
        num_graphs: int,
        k: torch.Tensor | None = None,
        periodicity: torch.Tensor | None = None,
        phase: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute Class-I multi-term proper torsion energy.

        Args:
            pos: Positions ``[n_atoms, 3]``.
            proper_index: COO proper indices ``[4, N]`` (i-j-k-l).
            proper_types: Proper types ``[N]``, or ``None`` when the tables
                already carry one row per torsion.
            term_batch: Graph index per torsion ``[N]``; ``None`` sums to a
                scalar.
            num_graphs: Graph count ``B`` — the length of the per-graph energy.
            k: Per-call barrier table; ``None`` prices the registered one.
            periodicity: Per-call period table; ``None`` prices the registered
                one.
            phase: Per-call phase table; ``None`` prices the registered one.

        Returns:
            Per-graph energy ``(B,)`` when ``term_batch`` is given, else a scalar.

        Raises:
            ValueError: ``proper_index`` is not ``[4, N]``.
        """
        _require_index_4_by_n(proper_index, "proper_index")
        tables = {"k": k, "periodicity": periodicity, "phase": phase}

        cos_phi, sin_phi = dihedral_cos_sin(*_dihedral_bond_vectors(pos, proper_index))
        k_t = self.rows("k", tables, proper_types)
        phase_t = self.rows("phase", tables, proper_types)
        n_t = self.rows("periodicity", tables, proper_types)
        cos_arg = fourier_cos(cos_phi, sin_phi, n_t, phase_t, n_max=self.n_max)
        energy_per_proper = (k_t * (1.0 + cos_arg)).sum(dim=1)
        return self.per_graph(energy_per_proper, term_batch, num_graphs)

    def energy_forces(
        self,
        *,
        pos: torch.Tensor,
        proper_index: torch.Tensor,
        proper_types: torch.Tensor | None = None,
        term_batch: torch.Tensor | None = None,
        num_graphs: int,
        k: torch.Tensor | None = None,
        periodicity: torch.Tensor | None = None,
        phase: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Energy and analytic forces ``F = -∂E/∂pos`` in one pass.

        With ``g = Σ_m (k[t,m]/s) · n[m] · sin(n[m]φ − γ[t,m]) = −∂E/∂φ``, the
        force is ``g · ∂φ/∂pos`` using the Blondel–Karplus gradient
        (:func:`dihedral_phi_grad`). ``φ`` is never formed; the sine comes from
        the same Chebyshev recurrence as the energy.

        Args:
            pos: Positions ``[N, 3]``.
            proper_index: COO proper indices ``[4, M]``.
            proper_types: Proper types ``[M]``, or ``None`` for
                per-interaction tables.
            term_batch: Graph index per torsion; ``None`` sums to a scalar.
            num_graphs: Graph count ``B``.
            k: Per-call barrier table; ``None`` prices the registered one.
            periodicity: Per-call period table; ``None`` prices the registered
                one.
            phase: Per-call phase table; ``None`` prices the registered one.

        Returns:
            ``(energy, forces)`` — energy ``(B,)`` (or scalar) and forces
            ``[N, 3]``.
        """
        _require_index_4_by_n(proper_index, "proper_index")
        tables = {"k": k, "periodicity": periodicity, "phase": phase}

        b1, b2, b3 = _dihedral_bond_vectors(pos, proper_index)
        cos_phi, sin_phi = dihedral_cos_sin(b1, b2, b3)
        k_t = self.rows("k", tables, proper_types)
        phase_t = self.rows("phase", tables, proper_types)
        n_t = self.rows("periodicity", tables, proper_types)
        cos_arg, sin_arg = fourier_cos_sin(cos_phi, sin_phi, n_t, phase_t, n_max=self.n_max)
        energy_per = (k_t * (1.0 + cos_arg)).sum(dim=1)

        n = n_t.to(dtype=k_t.dtype)
        g = (k_t * n * sin_arg).sum(dim=1)
        fi, fj, fk, fl = dihedral_phi_grad(b1, b2, b3)
        g = g.unsqueeze(-1)
        forces = pos.new_zeros((pos.shape[0], 3))
        forces.index_add_(0, proper_index[0], g * fi)
        forces.index_add_(0, proper_index[1], g * fj)
        forces.index_add_(0, proper_index[2], g * fk)
        forces.index_add_(0, proper_index[3], g * fl)
        return self.per_graph(energy_per, term_batch, num_graphs), forces

    def __repr__(self) -> str:
        n_types, n_terms = self.k.shape
        return f"ProperTorsionPeriodic(n_types={n_types}, n_terms={n_terms})"
