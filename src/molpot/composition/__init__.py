"""Modular potential composition (Sonata / generic PotentialComposer).

Class-I classical force fields live under :mod:`molpot.forcefield`
(:class:`~molpot.forcefield.ForceField`, :class:`~molpot.forcefield.ClassicalIMM`).
Parameter heads for Class-I terms are re-exported here for convenience.
"""

from molpot.composition.composer import PotentialComposer
from molpot.composition.heads import (
    ChargeHead,
    ChargeTransferParameterHead,
    LJParameterHead,
    RepulsionParameterHead,
    TSScalingHead,
)
from molpot.composition.mm_heads import (
    AngleParamHead,
    BondParamHead,
    ImproperParamHead,
    ProperTorsionParamHead,
)
from molpot.composition.multihead import MultiHead
from molpot.composition.sonata import Sonata, SonataSpec

__all__ = [
    "LJParameterHead",
    "RepulsionParameterHead",
    "ChargeTransferParameterHead",
    "ChargeHead",
    "TSScalingHead",
    "BondParamHead",
    "AngleParamHead",
    "ProperTorsionParamHead",
    "ImproperParamHead",
    "MultiHead",
    "PotentialComposer",
    "Sonata",
    "SonataSpec",
]
