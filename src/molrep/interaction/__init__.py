"""Interaction network components for molrep.

Provides equivariant layers for message-passing and feature transformation.
"""

from .aggregation import MessageAggregation, MessageAggregationSpec
from .contraction import SymmetricContraction, SymmetricContractionSpec
from .element import ElementUpdate, ElementUpdateSpec
from .gate import GatedNonlinearity
from .linear import EquivariantLinear
from .product import (
    ConvTP,
    ConvTPSpec,
    irreps_from_l_max,
    sh_irreps_from_l_max,
)
from .product_basis import EquivariantProductBasis
from .radial import RadialMLP, RadialWeightMLP, RadialWeightMLPSpec
from .residual import ResidualInteraction

__all__ = [
    "GatedNonlinearity",
    "ResidualInteraction",
    "EquivariantProductBasis",
    "RadialMLP",
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
