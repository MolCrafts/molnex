"""Pooling modules for bridging encoder output to energy computation.

Provides adapters for different encoder output formats:
- LayerPooling: reduce multi-layer (N, L, D) -> (N, D)
- EdgeToNodePooling: aggregate edge features to nodes (E, D) -> (N, D)
- SumPooling / MeanPooling / MaxPooling: node-to-graph aggregation
"""

from molpot.pooling.edge_to_node import EdgeToNodePooling
from molpot.pooling.graph_pooling import MaxPooling, MeanPooling, SumPooling
from molpot.pooling.layer_pooling import LayerPooling

__all__ = [
    "LayerPooling",
    "EdgeToNodePooling",
    "SumPooling",
    "MeanPooling",
    "MaxPooling",
]
