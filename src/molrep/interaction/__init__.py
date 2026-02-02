"""Interaction network components for molrep.

Provides equivariant layers for message-passing and feature transformation.
"""

from .aggregation import MessageAggregation, MessageAggregationSpec
from .linear import EquivariantLinear
from .tensor_product import (
    ConvTP,
    ConvTPSpec,
    irreps_from_l_max,
    sh_irreps_from_l_max,
)
from .symmetric_contraction import SymmetricContraction, SymmetricContractionSpec
from .element_update import ElementUpdate, ElementUpdateSpec

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
]
