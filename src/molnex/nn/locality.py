"""
Module wrappers for locality operations
"""

import torch
import torch.nn as nn
from ..F import locality as F


class NeighborList(nn.Module):
    """
    Module for computing neighbor lists with fixed cutoff.
    
    Args:
        cutoff: float - cutoff distance
        pbc: bool - whether to use periodic boundary conditions
    """
    
    def __init__(self, cutoff, pbc=True):
        super().__init__()
        self.cutoff = cutoff
        self.pbc = pbc
    
    def forward(self, positions, cell):
        """
        Args:
            positions: Tensor of shape (N, 3)
            cell: Tensor of shape (3, 3)
        
        Returns:
            Tuple of (edge_index, edge_vec, edge_dist)
        """
        return F.get_neighbor_pairs(positions, cell, self.cutoff, self.pbc)
    
    def extra_repr(self):
        return f"cutoff={self.cutoff}, pbc={self.pbc}"


__all__ = ["NeighborList"]
