"""
Module wrappers for locality operations (molix)
"""

import torch
import torch.nn as nn
from ..F import locality as F


class NeighborList(nn.Module):
    def __init__(self, cutoff, pbc=True):
        super().__init__()
        self.cutoff = cutoff
        self.pbc = pbc

    def forward(self, positions, cell):
        return F.get_neighbor_pairs(positions, self.cutoff, box_vectors=cell, check_errors=False)

    def extra_repr(self):
        return f"cutoff={self.cutoff}, pbc={self.pbc}"


__all__ = ["NeighborList"]
