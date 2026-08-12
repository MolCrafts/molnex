"""ForceField + Potentials + ClassicalIMM (OOP, no glue façade).

::

    ff = ForceField.class_i(dihedral="fourier")
    model = ClassicalIMM(encoder, forcefield=ff, bond_head=..., ...)
    ff_t = model(batch)
    E = ff_t.to_potentials().calc_energy(batch)
    ff_opls = model.project(opls_ff)
"""

from molpot.forcefield.classical_imm import ClassicalIMM
from molpot.forcefield.container import ForceField
from molpot.forcefield.potentials import Potentials
from molpot.forcefield.schema import (
    ForceFieldSchemaView,
    ParamSpec,
    PotentialMeta,
    StyleSchema,
)

__all__ = [
    "ForceField",
    "Potentials",
    "ClassicalIMM",
    "ForceFieldSchemaView",
    "ParamSpec",
    "StyleSchema",
    "PotentialMeta",
]
