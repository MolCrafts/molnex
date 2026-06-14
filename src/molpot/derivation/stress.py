"""Stress tensor derivation via functorch: ``σ = (1/V) ∂E/∂ε``.

Single responsibility: compute the stress tensor from the gradient of energy
with respect to a strain tensor, using ``torch.func.grad`` (functorch).

Example:
    >>> deriv = StressDerivation()
    >>> strain = torch.zeros_like(cell)        # linearization point ε = 0
    >>> stress = deriv(energy_fn, strain, cell)  # energy_fn(strain) -> scalar
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn


class StressDerivation(nn.Module):
    """Compute the stress tensor functionally as ``(1/V) ∂E/∂ε``.

    Takes the energy as a **pure function of a strain tensor** and differentiates
    it with ``torch.func.grad``, the compile-friendly counterpart of the old
    ``torch.autograd.grad`` path (traced into the forward graph, no
    double-backward barrier). ``energy_fn`` must apply the strain to positions
    (and cell) internally so the gradient flows ε → geometry → energy.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        energy_fn: Callable[[torch.Tensor], torch.Tensor],
        strain: torch.Tensor,
        cell: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the stress tensor from the energy gradient w.r.t. strain.

        Args:
            energy_fn: Maps a strain tensor (same shape as ``cell``) to a
                **scalar** total energy, applying the strain to positions/cell
                internally. Evaluated at ``strain``.
            strain: Strain tensor, typically zeros (the linearization point
                ε = 0). Does not need ``requires_grad`` — ``torch.func.grad``
                tracks the input itself.
            cell: Unit cell tensor; its determinant gives the volume.

        Returns:
            Stress tensor.
        """
        grad = torch.func.grad(energy_fn)(strain)

        volume = torch.det(cell).abs()
        stress = grad / volume.view(-1, 1, 1)

        return stress
