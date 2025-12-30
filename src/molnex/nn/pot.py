"""
Module wrappers for potential operations
"""

import torch
import torch.nn as nn
from ..F import pot as F


class PMEElectrostatics(nn.Module):
    """
    Module for Particle Mesh Ewald electrostatic calculations.
    
    Args:
        cutoff: float - real-space cutoff
        alpha: float - Ewald splitting parameter
        kmax: tuple of 3 ints - maximum k-vectors
    """
    
    def __init__(self, cutoff, alpha, kmax):
        super().__init__()
        self.cutoff = cutoff
        self.alpha = alpha
        self.kmax = kmax
    
    def forward(self, positions, charges, cell):
        """
        Args:
            positions: Tensor of shape (N, 3)
            charges: Tensor of shape (N,)
            cell: Tensor of shape (3, 3)
        
        Returns:
            Tuple of (energy, forces)
        """
        return F.pme_kernel(positions, charges, cell, self.cutoff, self.alpha, self.kmax)
    
    def extra_repr(self):
        return f"cutoff={self.cutoff}, alpha={self.alpha}, kmax={self.kmax}"


__all__ = ["PMEElectrostatics"]
