"""ForceField + Potentials + Parameterizer (OOP, no glue façade).

::

    from molpot.forcefield import OPLSAAForceField, Parameterizer

    model = Parameterizer(
        encoder,
        OPLSAAForceField(),
        bond_head=...,
        angle_head=...,
        proper_head=...,  # general Fourier (a, b, periodicity)
        improper_head=...,
        atom_head=...,
        charge_model=...,
    )
    ff = model(batch)
    pots = ff.to_potentials()  # overrides molpy native → torch
    E = pots.calc_energy(batch)  # kcal/mol

:class:`~molpot.forcefield.predictions.ForceFieldPredictions` is the one hop
from that model path to the per-task prediction mapping a loss consumes
(``energy`` in kcal/mol, ``forces`` in kcal mol⁻¹ Å⁻¹, ``hessian`` in
kcal mol⁻¹ Å⁻², ``charge`` = per-atom partial charges in ``e``), with the keys
declared at construction.

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
from molpot.forcefield.parameterizer import Parameterizer
from molpot.forcefield.potentials import Potentials
from molpot.forcefield.predictions import ForceFieldPredictions
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
    "ForceFieldPredictions",
    "ForceFieldSchemaView",
    "ClassIForceField",
    "OPLSAAForceField",
    "ParamSpec",
    "Parameterizer",
    "PotentialMeta",
    "Potentials",
    "StyleSchema",
    "TypeAssignment",
    "WilsonBMatrix",
]
