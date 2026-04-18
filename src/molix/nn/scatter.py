"""
Module wrappers for scatter operations (molix)
"""

import torch.nn as nn

from ..F import scatter as F


class ScatterSum(nn.Module):
    def __init__(self, dim=0):
        super().__init__()
        self.dim = dim

    def forward(self, src, index, dim_size=None):
        return F.scatter_sum(src, index, self.dim, dim_size)

    def extra_repr(self):
        return f"dim={self.dim}"


class BatchAggregation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, src, batch, dim_size=None):
        return F.batch_add(src, batch, dim_size)


__all__ = ["ScatterSum", "BatchAggregation"]
