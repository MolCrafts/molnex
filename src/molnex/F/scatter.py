"""
Functional API for scatter operations
"""

import torch


def scatter_sum(src, index, dim=0, dim_size=None):
    """
    Sum values from src into output at indices specified by index.
    
    Args:
        src: Tensor - source values
        index: LongTensor - indices to scatter into
        dim: int - dimension along which to scatter
        dim_size: int or None - size of output dimension
    
    Returns:
        Tensor with scattered sums
    """
    return torch.ops.molnex.scatter_sum(src, index, dim, dim_size)


def batch_add(src, batch, dim_size=None):
    """
    Add values within each batch.
    
    Args:
        src: Tensor - source values
        batch: LongTensor - batch assignment for each element
        dim_size: int or None - number of batches
    
    Returns:
        Tensor with per-batch sums
    """
    if dim_size is None:
        dim_size = int(batch.max().item()) + 1
    return scatter_sum(src, batch, dim=0, dim_size=dim_size)


__all__ = ["scatter_sum", "batch_add"]
