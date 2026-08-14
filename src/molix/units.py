"""Shared :class:`molpy.UnitSystem` — pint-style quantities.

Usage::

    from molix.units import units

    (1.0 * units.k_B).to("eV / kelvin")
    (1.0 * units.eV).to("amu * angstrom ** 2 / femtosecond ** 2")
    (1.0 * units.kilocalorie_per_mole).to(units.kilojoule_per_mole)
"""

from __future__ import annotations

from molpy import UnitSystem

#: Process-wide registry. All conversions go through pint-style ``Quantity.to``.
units = UnitSystem()

#: Padded ("dead") neighbour edges are displaced to ``factor · r_cut`` so every
#: cutoff envelope evaluates to exactly zero. Shared by
#: :class:`molix.md.neighbors.NeighborList` and
#: :class:`molix.engine.static.StaticForward`.
DEAD_EDGE_CUTOFF_FACTOR: float = 10.0
