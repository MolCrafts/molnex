"""ForceField + Potentials + ClassicalIMM (OOP, no glue façade).

::

    ff = ClassIForceField()
    model = ClassicalIMM(encoder, forcefield=ff, bond_head=..., ...)
    ff_t = model(batch)
    pots = ff_t.to_potentials()               # overrides molpy native → torch
    E = pots.calc_energy(batch)
    ff_opls = model.project(OPLSForceField())
    # ff_t is a molpy.ForceField — molpy.io writers take it after realize()

:class:`~molpot.forcefield.predictions.ClassIPredictions` is the one hop from
that model path to the per-task prediction mapping a loss consumes
(``energy`` in kcal/mol, ``forces`` in kcal mol⁻¹ Å⁻¹, ``hessian`` in
kcal mol⁻¹ Å⁻², ``q`` = per-atom partial charges in ``e``), with the keys declared
at construction.

Geometry helpers of the same shape live beside the container:
:class:`~molpot.forcefield.wilson.WilsonBMatrix` builds the Wilson
``B = ∂q_int/∂x`` matrix (rows = internal coordinates ``q_int`` — bond stretches in
Å and bend angles in rad — columns = the ``3N`` Cartesian components in Å) that a
projected-Hessian loss consumes. Wilson's ``q_int`` is written ``q`` in the
vibrational-spectroscopy literature; it is *not* the partial charge ``q`` above.
"""

from molpot.forcefield.classical_imm import ClassicalIMM
from molpot.forcefield.container import ClassIForceField, ForceField, OPLSForceField
from molpot.forcefield.potentials import Potentials
from molpot.forcefield.predictions import ClassIPredictions
from molpot.forcefield.schema import (
    ForceFieldSchemaView,
    ParamSpec,
    PotentialMeta,
    StyleSchema,
)
from molpot.forcefield.wilson import WilsonBMatrix

__all__ = [
    "ClassIForceField",
    "ClassIPredictions",
    "ClassicalIMM",
    "ForceField",
    "ForceFieldSchemaView",
    "OPLSForceField",
    "ParamSpec",
    "PotentialMeta",
    "Potentials",
    "StyleSchema",
    "WilsonBMatrix",
]
