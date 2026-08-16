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

#: Integrator energy unit: ``(amu, Å, fs)`` so ``v += Δt·F/m`` is consistent.
MD_ENERGY = "amu * angstrom ** 2 / femtosecond ** 2"


def preset_energy_to_md(style: str) -> float:
    """Magnitude of one energy unit of LAMMPS ``style`` in amu·Å²/fs².

    Resolves the style through :meth:`molpy.UnitSystem.preset` (``metal`` →
    ``electron_volt``, ``real`` → ``kilocalorie_per_mole``) and converts with
    pint ``.to``.

    Args:
        style: LAMMPS unit-style name stored on ``ForceField.units``.

    Returns:
        Factor from the preset energy unit to :data:`MD_ENERGY`.
    """
    energy = UnitSystem.preset(style).base_units["energy"]
    return float((1.0 * energy).to(MD_ENERGY).magnitude)


#: Padded ("dead") neighbour edges are displaced to ``factor · r_cut`` so every
#: cutoff envelope evaluates to exactly zero. Shared by
#: :class:`molix.md.neighbors.NeighborList` and
#: :class:`molix.engine.static.StaticForward`.
DEAD_EDGE_CUTOFF_FACTOR: float = 10.0
