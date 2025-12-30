"""Functional API for custom operators.

This module provides a clean functional interface to all stateless custom operators.
All functions are pure and stateless, making them easy to use in functional programming contexts.
"""

from .binding.locality.neighbors import get_neighbor_pairs
from .binding.scatter.scatter_fn import (
    scatter,
    scatter_sum,
    scatter_add,
    scatter_mean,
    scatter_min,
    scatter_max,
    scatter_mul,
    batch_add,
    get_natoms_per_batch,
)
from .binding.scatter.segment_coo import (
    segment_coo,
    segment_sum_coo,
    segment_add_coo,
    segment_mean_coo,
    segment_min_coo,
    segment_max_coo,
    gather_coo,
)
from .binding.scatter.segment_csr import (
    segment_csr,
    segment_sum_csr,
    segment_add_csr,
    segment_mean_csr,
    segment_min_csr,
    segment_max_csr,
    gather_csr,
)

__all__ = [
    # Locality
    "get_neighbor_pairs",
    # Scatter operations
    "scatter",
    "scatter_sum",
    "scatter_add",
    "scatter_mean",
    "scatter_min",
    "scatter_max",
    "scatter_mul",
    # Segment COO operations
    "segment_coo",
    "segment_sum_coo",
    "segment_add_coo",
    "segment_mean_coo",
    "segment_min_coo",
    "segment_max_coo",
    "gather_coo",
    # Segment CSR operations
    "segment_csr",
    "segment_sum_csr",
    "segment_add_csr",
    "segment_mean_csr",
    "segment_min_csr",
    "segment_max_csr",
    "gather_csr",
    # Utilities
    "batch_add",
    "get_natoms_per_batch",
]
