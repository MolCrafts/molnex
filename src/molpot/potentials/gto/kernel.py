"""Smeared Coulomb kernel used by PolarMACE GTO electrostatics.

Official ACEsuit ``graph_longrange`` evaluates ``erf(r / (2σ)) / r``.
Paper Eq.37 writes ``erf(r / √2 σ) / r``; the dump disagrees, and this
module follows the dump.

Reference:
    Batatia et al. "MACE-POLAR-1" arXiv:2602.19411.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GTOCoulombKernel(nn.Module):
    """``erf(r / (2σ)) / r`` with σ in Å.

    Args:
        sigma: GTO width in Å. Polar-1-M dump uses 1.5 Å.
    """

    def __init__(self, *, sigma: float) -> None:
        super().__init__()
        if sigma <= 0.0:
            raise ValueError(f"sigma must be > 0 Å, got {sigma}")
        self.sigma = float(sigma)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Evaluate the kernel.

        Args:
            r: Pair distances in Å, any shape.

        Returns:
            Same shape as ``r``, units 1/Å.
        """
        return torch.erf(r / (2.0 * self.sigma)) / r.clamp(min=1e-6)
