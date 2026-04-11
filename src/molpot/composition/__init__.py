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

from molpot.composition.composer import PotentialComposer
from molpot.composition.heads import (
    LJParameterHead,
    RepulsionParameterHead,
    ChargeTransferParameterHead,
    ChargeHead,
    TSScalingHead,
)
from molpot.composition.multi_head import MultiHead

__all__ = [
    "LJParameterHead",
    "RepulsionParameterHead",
    "ChargeTransferParameterHead",
    "ChargeHead",
    "TSScalingHead",
    "MultiHead",
    "PotentialComposer",
]
