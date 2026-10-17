"""Bond styles, as molrs prices them.

References:
    LAMMPS ``bond_style harmonic``, https://docs.lammps.org/bond_harmonic.html
    molrs ``molrs-python/docs/guides/forcefield-conventions.md`` §Bonds.
"""

from __future__ import annotations

import torch

from molpot.potentials.potential import BondedPotential
from molpot.potentials.registry import register

__all__ = ["BondHarmonic"]


@register
class BondHarmonic(BondedPotential):
    """``bond harmonic``: ``k (r − r0)²`` per bond — LAMMPS's ``K``, no ½.

    Parameters ``k`` (energy/Å²) and ``r0`` (Å).
    """

    style = "bond:harmonic"
    width = 2
    params = ("k", "r0")

    def term_energy(self, x: torch.Tensor, k: torch.Tensor, r0: torch.Tensor) -> torch.Tensor:
        d = x[:, 1] - x[:, 0]
        r = (d * d).sum(-1).clamp_min(1e-24).sqrt()
        return k * (r - r0).square()
