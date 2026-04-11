"""
Module wrappers for potential operations (molix)
"""

import torch.nn as nn

from ..F import pot as F


class PMEElectrostatics(nn.Module):
    def __init__(self, cutoff, alpha, kmax):
        super().__init__()
        self.cutoff = cutoff
        self.alpha = alpha
        self.kmax = kmax

    def forward(self, positions, charges, cell):
        return F.pme_kernel(positions, charges, cell, self.cutoff, self.alpha, self.kmax)

    def extra_repr(self):
        return f"cutoff={self.cutoff}, alpha={self.alpha}, kmax={self.kmax}"


__all__ = ["PMEElectrostatics"]
