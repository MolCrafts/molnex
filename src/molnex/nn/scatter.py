"""
Module wrappers for scatter operations
"""

import torch
import torch.nn as nn
from ..F import scatter as F


class ScatterSum(nn.Module):
    """
    Module for scatter sum operations.
    
    Args:
        dim: int - dimension along which to scatter
    """
    
    def __init__(self, dim=0):
        super().__init__()
        self.dim = dim
    
    def forward(self, src, index, dim_size=None):
        """
        Args:
            src: Tensor - source values
            index: LongTensor - indices
            dim_size: int or None - output dimension size
        
        Returns:
            Tensor with scattered sums
        """
        return F.scatter_sum(src, index, self.dim, dim_size)
    
    def extra_repr(self):
        return f"dim={self.dim}"


class BatchAggregation(nn.Module):
    """
    Module for aggregating values within batches.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, src, batch, dim_size=None):
        """
        Args:
            src: Tensor - source values
            batch: LongTensor - batch assignment
            dim_size: int or None - number of batches
        
        Returns:
            Tensor with per-batch aggregation
        """
        return F.batch_add(src, batch, dim_size)


__all__ = ["ScatterSum", "BatchAggregation"]
