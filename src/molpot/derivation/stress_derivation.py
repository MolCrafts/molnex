"""Stress tensor derivation via autograd: sigma = (1/V) dE/dstrain.

Single responsibility: compute the stress tensor from the gradient of
energy with respect to a strain tensor.

Example:
    >>> deriv = StressDerivation()
    >>> strain = torch.zeros(3, 3, requires_grad=True)
    >>> energy = some_energy_fn(strain)
    >>> stress = deriv(energy, strain, cell)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StressDerivation(nn.Module):
    """Compute stress tensor via autograd as ``(1/V) * dE/dstrain``.

    Expects the upstream code to apply a symmetric strain displacement
    to positions before the energy computation. The strain tensor must
    have ``requires_grad=True``.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        energy: torch.Tensor,
        strain: torch.Tensor,
        cell: torch.Tensor,
    ) -> torch.Tensor:
        """Compute stress tensor from energy gradient w.r.t. strain.

        Args:
            energy: Molecular energy (scalar or ``(B,)``).
            strain: Strain tensor with ``requires_grad=True``.
            cell: Unit cell tensor.

        Returns:
            Stress tensor.
        """
        grad = torch.autograd.grad(
            energy.sum(),
            strain,
            create_graph=self.training,
            retain_graph=self.training,
        )[0]

        volume = torch.det(cell).abs()
        stress = grad / volume.view(-1, 1, 1)

        return stress
