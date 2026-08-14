"""ForceField + Potentials + ClassicalIMM (OOP, no glue façade).

::

    ff = ClassIForceField()
    model = ClassicalIMM(encoder, forcefield=ff, bond_head=..., ...)
    ff_t = model(batch)
    pots = ff_t.to_potentials()               # overrides molpy native → torch
    E = pots.calc_energy(batch)
    ff_opls = model.project(OPLSForceField())
    # ff_t is a molpy.ForceField — molpy.io writers take it after realize()

Geometry helpers of the same shape live beside the container:
:class:`~molpot.forcefield.wilson.WilsonBMatrix` builds the Wilson
``B = ∂q/∂x`` matrix (rows = bond stretches and bend angles, columns = the
``3N`` Cartesian components) that a projected-Hessian loss consumes.
"""

from molpot.forcefield.classical_imm import ClassicalIMM
from molpot.forcefield.container import ClassIForceField, ForceField, OPLSForceField
from molpot.forcefield.potentials import Potentials
from molpot.forcefield.schema import (
    ForceFieldSchemaView,
    ParamSpec,
    PotentialMeta,
    StyleSchema,
)
from molpot.forcefield.wilson import WilsonBMatrix

__all__ = [
    "ClassIForceField",
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
