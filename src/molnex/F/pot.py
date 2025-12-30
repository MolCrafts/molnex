"""
Functional API for potential operations
"""

import torch


def pme_kernel(positions, charges, cell, cutoff, alpha, kmax):
    """
    Compute Particle Mesh Ewald (PME) electrostatic interactions.
    
    Args:
        positions: Tensor of shape (N, 3) - atomic positions
        charges: Tensor of shape (N,) - atomic charges
        cell: Tensor of shape (3, 3) - unit cell vectors
        cutoff: float - real-space cutoff
        alpha: float - Ewald splitting parameter
        kmax: tuple of 3 ints - maximum k-vectors
    
    Returns:
        Tuple of (energy, forces)
        - energy: Tensor of shape () - total electrostatic energy
        - forces: Tensor of shape (N, 3) - forces on each atom
    """
    return torch.ops.molnex.pme_kernel(positions, charges, cell, cutoff, alpha, kmax)


__all__ = ["pme_kernel"]
