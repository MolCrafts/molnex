"""PolarMACE representation blocks (no energy, no molpot).

Spin-resolved GTO source maps, per-layer feature mixing, SparseUvu field
updates, and the post-induction field readout. GTO electrostatics live in
molpot (chain 04).
"""

from molrep.interaction.polar.field import (
    AgnosticEmbeddedOneBodyVariableUpdate,
    OneBodyMLPFieldReadout,
)
from molrep.interaction.polar.mixer import MultiLayerFeatureMixer
from molrep.interaction.polar.source import EnvironmentDependentSpinSourceBlock
from molrep.interaction.polar.uvu import SparseUvuTensorProduct

__all__ = [
    "AgnosticEmbeddedOneBodyVariableUpdate",
    "EnvironmentDependentSpinSourceBlock",
    "MultiLayerFeatureMixer",
    "OneBodyMLPFieldReadout",
    "SparseUvuTensorProduct",
]
