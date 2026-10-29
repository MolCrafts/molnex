"""Proper-dihedral styles, as molrs prices them.

References:
    LAMMPS ``dihedral_style fourier``, https://docs.lammps.org/dihedral_fourier.html
    molrs ``molrs-python/docs/guides/forcefield-conventions.md`` §Dihedrals.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

from molpot.potentials.geom import dihedral_cos_sin
from molpot.potentials.potential import BondedPotential
from molpot.potentials.registry import register

__all__ = ["DihedralOpls", "DihedralPeriodic", "periodic_series"]


def periodic_series(
    phi: torch.Tensor, k: torch.Tensor, periodicity: torch.Tensor, phase: torch.Tensor
) -> torch.Tensor:
    """``Σₘ kₘ [1 + cos(nₘ φ − γₘ)]`` per row, ``γ`` in degrees.

    Args:
        phi: ``(M,)`` rad.
        k, periodicity, phase: ``(M, T)``, or ``(M,)`` for a one-term series.

    Returns:
        ``(M,)``.
    """
    if k.dim() == 1:
        k, periodicity, phase = k[:, None], periodicity[:, None], phase[:, None]
    arg = periodicity * phi[:, None] - phase * (math.pi / 180.0)
    return (k * (1.0 + torch.cos(arg))).sum(-1)


def dihedral_angle(x: torch.Tensor) -> torch.Tensor:
    """The signed dihedral of each ``(M, 4, 3)`` row in its stored order, rad."""
    cos_phi, sin_phi = dihedral_cos_sin(x[:, 1] - x[:, 0], x[:, 2] - x[:, 1], x[:, 3] - x[:, 2])
    return torch.atan2(sin_phi, cos_phi)


@register
class DihedralPeriodic(BondedPotential):
    """``dihedral periodic`` (LAMMPS ``fourier``): ``Σₘ kₘ [1 + cos(nₘ φ − γₘ)]``.

    Parameters ``k`` (energy), ``periodicity`` and ``phase`` (degrees), each
    ``(M, T)``: row ``i`` is the ``T``-term series of dihedral ``i``. A term
    with ``k = 0`` contributes nothing, so series of different lengths share
    one ``T``.
    """

    style = "dihedral:periodic"
    width = 4
    params = ("k", "periodicity", "phase")

    @classmethod
    def table_from_types(cls, types: Sequence[Any]) -> dict[str, torch.Tensor]:
        """molrs's per-term keys ``k{m}`` / ``periodicity{m}`` / ``phase{m}``.

        A type with the unindexed ``k`` / ``periodicity`` / ``phase`` is the
        one-term case. Shorter series are padded with ``k = 0`` terms.
        """
        rows = [cls._series(t) for t in types]
        width = max(len(row) for row in rows)
        padded = [row + [(0.0, 1.0, 0.0)] * (width - len(row)) for row in rows]
        table = torch.tensor(padded, dtype=torch.float64)
        return {"k": table[..., 0], "periodicity": table[..., 1], "phase": table[..., 2]}

    @staticmethod
    def _series(t: Any) -> list[tuple[float, float, float]]:
        params = t.params
        keys = set(params)
        terms: list[tuple[float, float, float]] = []
        m = 1
        while f"k{m}" in keys:
            terms.append(
                (
                    float(params[f"k{m}"]),
                    float(params[f"periodicity{m}"]),
                    float(params.get(f"phase{m}", 0.0)),
                )
            )
            m += 1
        if not terms and "k" in keys:
            terms.append(
                (float(params["k"]), float(params["periodicity"]), float(params.get("phase", 0.0)))
            )
        if not terms:
            raise ValueError(f"dihedral:periodic type {t.name!r} has no terms")
        return terms

    def term_energy(
        self,
        x: torch.Tensor,
        k: torch.Tensor,
        periodicity: torch.Tensor,
        phase: torch.Tensor,
    ) -> torch.Tensor:
        return periodic_series(dihedral_angle(x), k, periodicity, phase)


@register
class DihedralOpls(BondedPotential):
    """``dihedral opls``: ``½K₁(1+cos φ) + ½K₂(1−cos 2φ) + ½K₃(1+cos 3φ) + ½K₄(1−cos 4φ)``.

    Parameters ``k1`` … ``k4`` (energy), one row per dihedral. The series is the
    ``periodic`` one at ``n = 1..4`` with ``k = Kₙ/2`` and phases ``0, 180, 0,
    180°``, which is how LAMMPS defines the style.

    References:
        LAMMPS ``dihedral_style opls``, https://docs.lammps.org/dihedral_opls.html
    """

    style = "dihedral:opls"
    width = 4
    params = ("k1", "k2", "k3", "k4")

    def term_energy(
        self,
        x: torch.Tensor,
        k1: torch.Tensor,
        k2: torch.Tensor,
        k3: torch.Tensor,
        k4: torch.Tensor,
    ) -> torch.Tensor:
        phi = dihedral_angle(x)
        return 0.5 * (
            k1 * (1.0 + torch.cos(phi))
            + k2 * (1.0 - torch.cos(2.0 * phi))
            + k3 * (1.0 + torch.cos(3.0 * phi))
            + k4 * (1.0 - torch.cos(4.0 * phi))
        )
