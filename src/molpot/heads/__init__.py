"""Prediction heads.

All heads are plain PyTorch modules.
"""

from molpot.heads.charge_bond import BondChargeHead
from molpot.heads.charge_response import ChargeResponseHead
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
from molpot.heads.energy import AtomicEnergyMLP, EnergyHead
from molpot.heads.multipole import (
    PermMultipoleHead,
    PermMultipoleHeadSpec,
)
from molpot.heads.rescale import GlobalRescale, PerSpeciesScaleShift
from molpot.heads.type import TypeHead

__all__ = [
    "AtomicEnergyMLP",
    "BondChargeHead",
    "ChargeResponseHead",
    "DipoleHead",
    "EdgeEnergyHead",
    "ElementAlphaTable",
    "ElementChargeTable",
    "EnergyHead",
    "GlobalRescale",
    "HardnessHead",
    "HardnessHeadSpec",
    "PermMultipoleHead",
    "PermMultipoleHeadSpec",
    "PolarizabilityHead",
    "PolarizabilityHeadSpec",
    "PerSpeciesScaleShift",
    "TypeHead",
]
