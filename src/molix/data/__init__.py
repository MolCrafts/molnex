"""Data pipeline components for molix.

This module provides data processing nodes and pipelines that implement
the workflow protocol (GraphLike/OpLike), enabling composition with trainers
and other workflow components.

Also provides AtomicTD, the protocol-level TensorDict container for molecular data.
"""

from molix.data.atomic_td import AtomicTD, Config
from molix.data.node import DataNode
from molix.data.ops import CacheOp, FilterOp, NormalizeOp, TransformOp
from molix.data.pipeline import DataPipeline
from molix.data.topology import TopologyBuilder, GeometryPreprocessor, Normalizer

__all__ = [
    "AtomicTD",
    "Config",
    "DataNode",
    "DataPipeline",
    "TransformOp",
    "FilterOp",
    "NormalizeOp",
    "CacheOp",
    "TopologyBuilder",
    "GeometryPreprocessor",
    "Normalizer",
]
