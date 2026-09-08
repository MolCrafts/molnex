"""Potential Intermediate Representation (Class-I molecular mechanics).

Names interaction bags, canonical units, nonbonded scaling and the nonbonded
*method* (cutoffs and Coulomb solver) for the learnable classical force-field
stack. Bags are parameter containers only — evaluation lives in
:mod:`molpot.potentials`.
"""

from molpot.ir.bags import (
    AngleBag,
    BondBag,
    ChargeBag,
    ImproperHarmonicBag,
    ImproperPeriodicBag,
    LJBag,
    ProperTorsionBag,
)
from molpot.ir.nonbonded import NonbondedMethodRecord
from molpot.ir.potential_ir import PotentialIR
from molpot.ir.scaling import ExclusionScales, NonbondedScaling
from molpot.ir.units import CLASS_I_CANONICAL, UnitTag

__all__ = [
    "AngleBag",
    "BondBag",
    "CLASS_I_CANONICAL",
    "ChargeBag",
    "ExclusionScales",
    "ImproperHarmonicBag",
    "ImproperPeriodicBag",
    "LJBag",
    "NonbondedMethodRecord",
    "NonbondedScaling",
    "PotentialIR",
    "ProperTorsionBag",
    "UnitTag",
]
