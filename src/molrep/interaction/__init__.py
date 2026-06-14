"""Interaction network components for molrep.

Provides equivariant layers for message-passing and feature transformation.
"""

from .aggregation import MessageAggregation, MessageAggregationSpec
from .contraction import SymmetricContraction, SymmetricContractionSpec
from .element import ElementUpdate, ElementUpdateSpec
from .linear import EquivariantLinear
from .product import (
    ConvTP,
    ConvTPSpec,
    irreps_from_l_max,
    sh_irreps_from_l_max,
)
from .radial import RadialWeightMLP, RadialWeightMLPSpec

__all__ = [
    "MessageAggregation",
    "MessageAggregationSpec",
    "EquivariantLinear",
    "ConvTP",
    "ConvTPSpec",
    "irreps_from_l_max",
    "sh_irreps_from_l_max",
    "SymmetricContraction",
    "SymmetricContractionSpec",
    "ElementUpdate",
    "ElementUpdateSpec",
    "RadialWeightMLP",
    "RadialWeightMLPSpec",
]
