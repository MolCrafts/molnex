"""Prediction heads.

All heads are plain PyTorch modules.
"""

from molpot.heads.charge_bond import BondChargeHead
from molpot.heads.charge_response import ChargeResponseHead
from molpot.heads.classical import (
    AngleHead,
    BondHead,
    CovalentRadiusSum,
    CvffHead,
    ExpAround,
    LinearAround,
    LJHead,
    PeriodicHead,
)
from molpot.heads.dipole import DipoleHead
from molpot.heads.edge import EdgeEnergyHead
from molpot.heads.electrostatics import (
    HardnessHead,
    HardnessHeadSpec,
    PolarizabilityHead,
    PolarizabilityHeadSpec,
)
from molpot.heads.element import (
    ElementAlphaTable,
    ElementChargeTable,
)
from molpot.heads.energy import AtomicEnergyMLP, AtomicReferenceEnergy, EnergyHead
from molpot.heads.multipole import (
    PermMultipoleHead,
    PermMultipoleHeadSpec,
)
from molpot.heads.provenance import (
    CoverageRegime,
    ParameterProvenance,
    SupportClassifier,
    attach_provenance,
)
from molpot.heads.rescale import GlobalRescale, PerSpeciesScaleShift
from molpot.heads.type import TypeHead

__all__ = [
    "AngleHead",
    "AtomicEnergyMLP",
    "BondHead",
    "CovalentRadiusSum",
    "CvffHead",
    "ExpAround",
    "LJHead",
    "LinearAround",
    "PeriodicHead",
    "AtomicReferenceEnergy",
    "BondChargeHead",
    "ChargeResponseHead",
    "CoverageRegime",
    "DipoleHead",
    "EdgeEnergyHead",
    "ElementAlphaTable",
    "ElementChargeTable",
    "EnergyHead",
    "GlobalRescale",
    "HardnessHead",
    "HardnessHeadSpec",
    "ParameterProvenance",
    "PermMultipoleHead",
    "PermMultipoleHeadSpec",
    "PolarizabilityHead",
    "PolarizabilityHeadSpec",
    "PerSpeciesScaleShift",
    "SupportClassifier",
    "TypeHead",
    "attach_provenance",
]
