"""
Functional API for locality operations
"""

import torch


def get_neighbor_pairs(positions, cell, cutoff, pbc=True):
    """
    Compute neighbor pairs within a cutoff distance.
    
    Args:
        positions: Tensor of shape (N, 3) - atomic positions
        cell: Tensor of shape (3, 3) - unit cell vectors
        cutoff: float - cutoff distance
        pbc: bool - whether to use periodic boundary conditions
    
    Returns:
        Tuple of (edge_index, edge_vec, edge_dist)
        - edge_index: LongTensor of shape (2, E) - pairs of atom indices
        - edge_vec: Tensor of shape (E, 3) - displacement vectors
        - edge_dist: Tensor of shape (E,) - distances
    """
    return torch.ops.molnex.get_neighbor_pairs(positions, cell, cutoff, pbc)


__all__ = ["get_neighbor_pairs"]
