"""Improper styles, as molrs prices them.

Both price the dihedral of the four atoms **in their stored order**, as every
LAMMPS improper style that is a dihedral does. ``improper cvff`` is stored
centre first (the batch's own convention); ``improper periodic`` is stored in
AMBER's order, the centre third — no centre-first order has the AMBER
dihedral, so a frame carrying one must list it that way.

References:
    LAMMPS ``improper_style cvff``, https://docs.lammps.org/improper_cvff.html
    molrs ``molrs-python/docs/guides/forcefield-conventions.md`` §Impropers.
"""

from __future__ import annotations

import torch

from molpot.potentials.dihedral import dihedral_angle
from molpot.potentials.potential import BondedPotential
from molpot.potentials.registry import register

__all__ = ["ImproperCvff"]


@register
class ImproperCvff(BondedPotential):
    """``improper cvff``: ``k [1 + sign · cos(n φ)]``, the centre first.

    Parameters ``k`` (energy), ``sign`` (±1) and ``periodicity``.
    """

    style = "improper:cvff"
    width = 4
    params = ("k", "sign", "periodicity")

    def term_energy(
        self, x: torch.Tensor, k: torch.Tensor, sign: torch.Tensor, periodicity: torch.Tensor
    ) -> torch.Tensor:
        return k * (1.0 + sign * torch.cos(periodicity * dihedral_angle(x)))
