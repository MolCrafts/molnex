"""Shared :class:`molpy.UnitSystem` — pint-style quantities.

pint (https://pint.readthedocs.io) is the Python physical-units library molpy
builds on. A pint *quantity* is a magnitude paired with a unit, and
``quantity.to(other_unit)`` re-expresses it in ``other_unit``, refusing
dimensionally impossible conversions. Every conversion in molnex goes through
it, so a conversion factor is never written out as a float in Python.

Two vocabularies meet here. molnex's own molecular-dynamics (MD) integrator
works in ``(amu, Å, fs)`` — see :data:`MD_ENERGY` — while LAMMPS, the external
MD engine molnex exports models to, names whole systems of units *unit styles*
(``real`` = kcal/mol, Å, fs; ``metal`` = eV, Å, ps). The two ``preset_energy_*``
functions below translate between those namings.

Usage::

    from molix.units import units

    (1.0 * units.k_B).to("eV / kelvin")
    (1.0 * units.eV).to("amu * angstrom ** 2 / femtosecond ** 2")
    (1.0 * units.kilocalorie_per_mole).to(units.kilojoule_per_mole)
"""

from __future__ import annotations

from molpy import Unit, UnitSystem

#: Process-wide registry. All conversions go through pint-style ``Quantity.to``.
units = UnitSystem()

#: Integrator energy unit: ``(amu, Å, fs)`` so ``v += Δt·F/m`` is consistent.
MD_ENERGY = "amu * angstrom ** 2 / femtosecond ** 2"


def _preset_energy(style: str) -> Unit:
    """Energy unit of LAMMPS unit style ``style`` (``real`` → kcal/mol, ``metal`` → eV).

    Shared by :func:`preset_energy_to_md` and :func:`preset_energy_ratio` so both
    resolve a style the same way. The lookup is molpy's preset table, which raises
    ``ValueError`` for a name it does not carry — LAMMPS ``lj``, for instance,
    whose energies are reduced (dimensionless) and have no absolute unit.
    """
    return UnitSystem.preset(style).base_units["energy"]


def preset_energy_to_md(style: str) -> float:
    """Magnitude of one energy unit of LAMMPS ``style`` in amu·Å²/fs².

    Resolves the style through :meth:`molpy.UnitSystem.preset` (``metal`` →
    ``electron_volt``, ``real`` → ``kilocalorie_per_mole``) and converts with
    pint ``.to``.

    Args:
        style: LAMMPS unit-style name stored on ``ForceField.units``.

    Returns:
        Dimensionless factor from the preset energy unit to :data:`MD_ENERGY`
        (amu·Å²/fs²): multiply an energy counted in ``style`` by it to get the
        same energy in integrator units.

    Raises:
        ValueError: If ``style`` is not a molpy unit preset — the lookup is
            :func:`_preset_energy`.
    """
    return float((1.0 * _preset_energy(style)).to(MD_ENERGY).magnitude)


def preset_energy_ratio(source_style: str, target_style: str) -> float:
    """Magnitude of one energy unit of ``source_style`` in the unit of ``target_style``.

    Both styles resolve through :meth:`molpy.UnitSystem.preset` (``real`` →
    kcal/mol, ``metal`` → eV) and the conversion is evaluated by pint ``.to``,
    so no conversion factor is ever spelled out in Python. Stamped into the
    LAMMPS ``meta.json`` ``energy_conv`` table, from which ``pair_style molnex``
    picks the row matching the running simulation's ``units`` command.

    Args:
        source_style: LAMMPS unit-style name the energy is expressed in.
        target_style: LAMMPS unit-style name to express it in.

    Returns:
        Dimensionless factor from the ``source_style`` energy unit to the
        ``target_style`` one: multiply an energy counted in ``source_style`` by
        it to get the same energy counted in ``target_style``.

    Raises:
        ValueError: If either style is not a molpy unit preset (e.g. LAMMPS
            ``lj``, which has no absolute energy unit).

    Examples:
        ``preset_energy_ratio("real", "metal")`` is one kcal/mol expressed in eV
        (≈ 0.0433641), ``preset_energy_ratio("metal", "real")`` is its reciprocal
        (≈ 23.06055), and a style measured against itself is exactly ``1.0``.
    """
    return float((1.0 * _preset_energy(source_style)).to(_preset_energy(target_style)).magnitude)


#: Padded ("dead") neighbour edges are displaced to ``factor · r_cut`` so every
#: cutoff envelope evaluates to exactly zero. Shared by
#: :class:`molix.md.neighbors.NeighborList` and
#: :class:`molix.engine.static.StaticForward`.
DEAD_EDGE_CUTOFF_FACTOR: float = 10.0
