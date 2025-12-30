"""Functional API for molix operations.

This module provides PyTorch-style functional API (molix.F) for common operations.
Similar to torch.nn.functional, this provides stateless functional interfaces.
"""

# Re-export from op.functional
from op.functional import (
    # Locality
    get_neighbor_pairs,
    # Scatter operations
    scatter,
    scatter_sum,
    scatter_add,
    scatter_mean,
    scatter_min,
    scatter_max,
    scatter_mul,
    # Segment COO operations
    segment_coo,
    segment_sum_coo,
    segment_add_coo,
    segment_mean_coo,
    segment_min_coo,
    segment_max_coo,
    gather_coo,
    # Segment CSR operations
    segment_csr,
    segment_sum_csr,
    segment_add_csr,
    segment_mean_csr,
    segment_min_csr,
    segment_max_csr,
    gather_csr,
    # Utilities
    batch_add,
    get_natoms_per_batch,
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
