"""Modular potential composition (Sonata / generic PotentialComposer).

Classical force fields live under :mod:`molpot.forcefield`
(:class:`~molpot.forcefield.ForceField`, :class:`~molpot.forcefield.OPLSAAForceField`);
the learnable Class-I heads are :mod:`molpot.heads.classical` and the torch
twins of the molrs styles are :mod:`molpot.potentials`.
"""

from molpot.composition.composer import PotentialComposer
from molpot.composition.heads import (
    ChargeHead,
    ChargeTransferParameterHead,
    LJParameterHead,
    RepulsionParameterHead,
    TSScalingHead,
)
from molpot.composition.multihead import MultiHead
from molpot.composition.sonata import Sonata, SonataSpec

__all__ = [
    "LJParameterHead",
    "RepulsionParameterHead",
    "ChargeTransferParameterHead",
    "ChargeHead",
    "TSScalingHead",
    "MultiHead",
    "PotentialComposer",
    "Sonata",
    "SonataSpec",
]
