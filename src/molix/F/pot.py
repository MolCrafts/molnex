"""
Functional API for potential operations (molix)
"""

import torch


def pme_kernel(positions, charges, cell, cutoff, alpha, kmax):
    return torch.ops.molnex.pme_kernel(positions, charges, cell, cutoff, alpha, kmax)
