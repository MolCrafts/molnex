"""Angle styles, as molrs prices them.

References:
    LAMMPS ``angle_style harmonic``, https://docs.lammps.org/angle_harmonic.html
    molrs ``molrs-python/docs/guides/forcefield-conventions.md`` §Angles.
"""

from __future__ import annotations

import math

import torch

from molpot.potentials.geom import bond_cos_sin
from molpot.potentials.potential import BondedPotential
from molpot.potentials.registry import register

__all__ = ["AngleHarmonic"]


@register
class AngleHarmonic(BondedPotential):
    """``angle harmonic``: ``k (θ − theta0)²`` per angle, ``θ`` at the middle atom.

    Parameters ``k`` (energy/rad², LAMMPS's ``K``, no ½) and ``theta0`` in
    **degrees**, as molrs stores it.
    """

    style = "angle:harmonic"
    width = 3
    params = ("k", "theta0")

    def term_energy(self, x: torch.Tensor, k: torch.Tensor, theta0: torch.Tensor) -> torch.Tensor:
        cos_t, sin_t = bond_cos_sin(x[:, 0] - x[:, 1], x[:, 2] - x[:, 1])
        theta = torch.atan2(sin_t, cos_t)
        return k * (theta - theta0 * (math.pi / 180.0)).square()
