"""
Functional API for scatter operations (molix)
"""

import torch


def scatter_sum(src, index, dim=0, dim_size=None):
    return torch.ops.molnex.scatter_sum(src, index, dim, dim_size)


def batch_add(src, batch, dim_size=None):
    if dim_size is None:
        dim_size = int(batch.max().item()) + 1
    return scatter_sum(src, batch, dim=0, dim_size=dim_size)
