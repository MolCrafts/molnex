"""C^infty bond-angle and dihedral trigonometry.

``‖x‖ = √(x·x)`` is not differentiable at the origin, so a unit-vector
``acos`` / ``atan2`` path is undefined for a linear bend and a colinear
torsion. Softening the squared length, ``√(x·x + ε)``, makes ``cos`` and
``sin`` of every angle and dihedral a single expression, defined for every
finite geometry.

``ε = 1e-12`` (Å⁴ on a product of two squared lengths) leaves well-bent
residuals inside float64 golden tolerance and shifts a 180° harmonic by
``< 1e-4`` kcal/mol at ``k ~ 100``.

LAMMPS ``angle_style harmonic`` is the same energy ``K (θ − θ0)²`` (their
``K`` already includes 1/2). LAMMPS ``dihedral_style opls`` / CHARMM never
form ``φ``: they evaluate ``cos(nφ)`` from ``cos φ`` by Chebyshev
recurrence and set ``1 / (|n1| |n2|) = 0`` when the plane-norm product
vanishes. The ``ε`` here is that regularization written without a branch.

References:
    LAMMPS ``angle_harmonic`` / ``dihedral_opls``
    Blondel, Karplus, J. Comput. Chem. 17, 1132 (1996)
"""

from __future__ import annotations

import torch

GEOM_EPS = 1e-12


def bond_cos_sin(u: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(cos θ, sin θ)`` of the angle between vectors ``u`` and ``v``.

    Args:
        u: ``(..., 3)`` Å.
        v: ``(..., 3)`` Å.

    Returns:
        Cosine and sine, each ``(...)``. Sine is non-negative.
    """
    uu = (u * u).sum(dim=-1)
    vv = (v * v).sum(dim=-1)
    uv = (u * v).sum(dim=-1)
    cr = torch.linalg.cross(u, v)
    cc = (cr * cr).sum(dim=-1)
    denom = (uu * vv + GEOM_EPS).sqrt()
    return uv / denom, (cc + GEOM_EPS).sqrt() / denom


def dihedral_cos_sin(
    b1: torch.Tensor,
    b2: torch.Tensor,
    b3: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(cos φ, sin φ)`` of the i-j-k-l dihedral from bond vectors.

    ``b1 = j−i``, ``b2 = k−j``, ``b3 = l−k``. Same sign convention as
    LAMMPS / OpenMM (``sin φ = (n1 × n2) · b2``).

    Args:
        b1: ``(..., 3)`` Å.
        b2: ``(..., 3)`` Å.
        b3: ``(..., 3)`` Å.

    Returns:
        Cosine and sine, each ``(...)``.
    """
    n1 = torch.linalg.cross(b1, b2)
    n2 = torch.linalg.cross(b2, b3)
    n1sq = (n1 * n1).sum(dim=-1)
    n2sq = (n2 * n2).sum(dim=-1)
    b2sq = (b2 * b2).sum(dim=-1)
    denom = (n1sq * n2sq + GEOM_EPS).sqrt()
    cos_phi = (n1 * n2).sum(dim=-1) / denom
    sin_phi = (torch.linalg.cross(n1, n2) * b2).sum(dim=-1) / (denom * (b2sq + GEOM_EPS).sqrt())
    return cos_phi, sin_phi


def _fourier_basis(
    cos_phi: torch.Tensor,
    sin_phi: torch.Tensor,
    periodicity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(cos nφ, sin nφ)`` per period via Chebyshev recurrence. Never forms ``φ``."""
    n_max = int(periodicity.max().item())
    cos_k = [torch.ones_like(cos_phi), cos_phi]
    sin_k = [torch.zeros_like(sin_phi), sin_phi]
    for _ in range(2, n_max + 1):
        cos_prev, sin_prev = cos_k[-1], sin_k[-1]
        cos_k.append(cos_phi * cos_prev - sin_phi * sin_prev)
        sin_k.append(sin_phi * cos_prev + cos_phi * sin_prev)
    n_idx = periodicity.to(dtype=torch.long)
    cos_n = torch.stack(cos_k, dim=-1)[:, n_idx]
    sin_n = torch.stack(sin_k, dim=-1)[:, n_idx]
    return cos_n, sin_n


def fourier_cos(
    cos_phi: torch.Tensor,
    sin_phi: torch.Tensor,
    periodicity: torch.Tensor,
    phase: torch.Tensor,
) -> torch.Tensor:
    """``cos(n φ − γ)`` by Chebyshev recurrence. Never forms ``φ``.

    Args:
        cos_phi: ``cos φ``, shape ``[N]``.
        sin_phi: ``sin φ``, shape ``[N]``.
        periodicity: Integer periods ``n``, shape ``[M]``.
        phase: Phase offsets ``γ`` in radians, shape ``[N, M]``.

    Returns:
        ``cos(n φ − γ)`` of shape ``[N, M]``.
    """
    cos_n, sin_n = _fourier_basis(cos_phi, sin_phi, periodicity)
    return cos_n * torch.cos(phase) + sin_n * torch.sin(phase)


def fourier_cos_sin(
    cos_phi: torch.Tensor,
    sin_phi: torch.Tensor,
    periodicity: torch.Tensor,
    phase: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(cos(n φ − γ), sin(n φ − γ))`` by Chebyshev recurrence.

    Args:
        cos_phi: ``cos φ``, shape ``[N]``.
        sin_phi: ``sin φ``, shape ``[N]``.
        periodicity: Integer periods ``n``, shape ``[M]``.
        phase: Phase offsets ``γ`` in radians, shape ``[N, M]``.

    Returns:
        ``cos(n φ − γ)`` and ``sin(n φ − γ)``, each ``[N, M]``.
    """
    cos_n, sin_n = _fourier_basis(cos_phi, sin_phi, periodicity)
    cos_phase = torch.cos(phase)
    sin_phase = torch.sin(phase)
    return (
        cos_n * cos_phase + sin_n * sin_phase,
        sin_n * cos_phase - cos_n * sin_phase,
    )
