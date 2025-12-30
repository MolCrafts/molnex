"""TensorDictModule wrappers for custom operators.

This module provides TensorDictModule wrappers for all stateless custom operators,
enabling seamless integration with TensorDict-based pipelines.
"""

from .neighbors import GetNeighborPairs
from .scatter import ScatterSum, ScatterMean, ScatterMax, ScatterMin
from .segment import (
    SegmentSumCOO,
    SegmentMeanCOO,
    SegmentSumCSR,
    SegmentMeanCSR,
)

__all__ = [
    # Neighbors
    "GetNeighborPairs",
    # Scatter
    "ScatterSum",
    "ScatterMean",
    "ScatterMax",
    "ScatterMin",
    # Segment COO
    "SegmentSumCOO",
    "SegmentMeanCOO",
    # Segment CSR
    "SegmentSumCSR",
    "SegmentMeanCSR",
]
