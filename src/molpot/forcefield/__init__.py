"""ForceField + Potentials for fixed (tabulated) Class-I force fields.

::

    from molpot.forcefield import OPLSAAForceField

    ff = OPLSAAForceField()
    pots = ff.to_potentials()  # overrides molpy native → torch
    E = pots.calc_energy(batch)  # kcal/mol

A *learned* force field does not go through this container: it prices the
molpy ``ForceField`` IR through :meth:`molpot.potentials.collection.Potentials.from_forcefield`
(see :class:`molzoo.rondo.Rondo`).

Geometry helpers of the same shape live beside the container:
:class:`~molpot.forcefield.wilson.WilsonBMatrix` builds the Wilson
``B = ∂q_int/∂x`` matrix (rows = internal coordinates ``q_int`` — bond stretches in
Å and bend angles in rad — columns = the ``3N`` Cartesian components in Å) that a
projected-Hessian loss consumes. Wilson's ``q_int`` is written ``charge`` in the
vibrational-spectroscopy literature; it is *not* the partial charge ``charge`` above.
"""

from molpot.forcefield.container import (
    ClassIForceField,
    ForceField,
    OPLSAAForceField,
    TypeAssignment,
)
from molpot.forcefield.exclusions import ExclusionTable
from molpot.forcefield.potentials import Potentials
from molpot.forcefield.schema import (
    ForceFieldSchemaView,
    ParamSpec,
    PotentialMeta,
    StyleSchema,
)
from molpot.forcefield.wilson import WilsonBMatrix

__all__ = [
    "ExclusionTable",
    "ForceField",
    "ForceFieldSchemaView",
    "ClassIForceField",
    "OPLSAAForceField",
    "ParamSpec",
    "PotentialMeta",
    "Potentials",
    "StyleSchema",
    "TypeAssignment",
    "WilsonBMatrix",
]
