"""ForceField + Potentials + ClassicalIMM (OOP, no glue façade).

::

    ff = ClassIForceField()
    model = ClassicalIMM(encoder, forcefield=ff, bond_head=..., ...)
    ff_t = model(batch)
    pots = ff_t.to_potentials()               # overrides molpy native → torch
    E = pots.calc_energy(batch)
    ff_opls = model.project(OPLSForceField())
    # ff_t is a molpy.ForceField — molpy.io writers take it after realize()
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

__all__ = [
    "ClassIForceField",
    "ClassicalIMM",
    "ForceField",
    "OPLSForceField",
    "Potentials",
    "ForceFieldSchemaView",
    "ParamSpec",
    "StyleSchema",
    "PotentialMeta",
]
