"""Force derivation via autograd: F = -dE/dpos.

Single responsibility: compute atomic forces as the negative gradient
of energy with respect to atomic positions.

Example:
    >>> deriv = ForceDerivation()
    >>> pos = torch.randn(5, 3, requires_grad=True)
    >>> energy = some_energy_fn(pos)
    >>> forces = deriv(energy, pos)  # (5, 3)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ForceDerivation(nn.Module):
    """Compute forces via autograd as ``-dE/dpos``.

    Positions must have ``requires_grad=True`` before the energy computation.
    """

    def __init__(self):
        super().__init__()

    def forward(self, energy: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Compute forces as negative gradient of energy w.r.t. positions.

        Args:
            energy: Molecular energy (scalar or ``(B,)``).
            pos: Atomic positions ``(N, 3)`` with ``requires_grad=True``.

        Returns:
            Atomic forces ``(N, 3)``.
        """
        forces = -torch.autograd.grad(
            energy.sum(),
            pos,
            create_graph=self.training,
            retain_graph=self.training,
        )[0]

        return forces
