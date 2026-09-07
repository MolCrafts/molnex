"""Potential Intermediate Representation (Class-I molecular mechanics).

Names interaction bags, canonical units, and nonbonded scaling for the
learnable classical force-field stack. Bags are parameter containers only —
evaluation lives in :mod:`molpot.potentials`.
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
    "NonbondedScaling",
    "PotentialIR",
    "ProperTorsionBag",
    "UnitTag",
]
