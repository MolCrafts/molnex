"""Data processing components for molix.

This module provides data processing components including:
- Dataset base class with preprocess support
- DatasetPreprocessor base class and implementations
- AtomicTD, the protocol-level TensorDict container for molecular data
"""

from molix.data.atomic_td import AtomicTD, Config
from molix.data.dataset import Dataset
from molix.data.preprocess import (
    DatasetPreprocessor,
    AtomicDressPreprocessor,
    NeighborListPreprocessor,
    AtomicDressConfig,
    NeighborListConfig,
)
from molix.data.collate import collate_atomic_tds

__all__ = [
    "AtomicTD",
    "Config",
    "Dataset",
    "DatasetPreprocessor",
    "AtomicDressPreprocessor",
    "NeighborListPreprocessor",
    "AtomicDressConfig",
    "NeighborListConfig",
    "collate_atomic_tds",
]
