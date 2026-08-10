"""Modular potential composition.

Build force fields by composing pooling, parameter heads, and potentials::

    pool = LayerPooling("mean")
    composer = PotentialComposer(
        head=LJParameterHead(feature_dim=16),
        potentials={"lj": LJ126()},
    )
    node_features = pool(encoder_output)
    outputs = composer(node_features=node_features, data=data)
"""

from molpot.composition.classical_mm import ClassicalMMComposer
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
    "ClassicalMMComposer",
    "Sonata",
    "SonataSpec",
]
