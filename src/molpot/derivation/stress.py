"""Stress tensor derivation via functorch: ``σ = (1/V) ∂E/∂ε``.

Single responsibility: compute the stress tensor from the gradient of energy
with respect to a strain tensor, using ``torch.func.grad`` (functorch).

This module also owns molnex's **strain-gradient convention**
(:func:`symmetrize_strain_grad`) — every virial / stress path in the repo
routes its raw ``∂E/∂ε`` through it, so there is exactly one definition of
what a shear component means.

Example:
    >>> deriv = StressDerivation()
    >>> strain = torch.zeros_like(cell)        # linearization point ε = 0
    >>> stress = deriv(energy_fn, strain, cell)  # energy_fn(strain) -> scalar
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn


def symmetrize_strain_grad(grad: torch.Tensor) -> torch.Tensor:
    """molnex strain-gradient protocol: ``W = ∂E/∂ε``, Cauchy ``σ = W / V``.

    ``ε`` is the **symmetric** strain tensor and ``W`` its matrix gradient, so
    ``dE = Σ_αβ W_αβ dε_αβ`` runs over all nine entries and a shear ``W_αβ``
    is *not* doubled. The gradient of a strain-symmetric energy is already
    symmetric; this only removes the asymmetry a finite-precision backward can
    leave behind.

    Doubling the off-diagonal instead is the engineering-shear convention
    (``γ_αβ = 2ε_αβ``, the ``(α,β)``/``(β,α)`` pair as one variable), which
    molnex used until 2026-09-07 and which made every shear component twice
    the Cauchy value — including in the numbers handed to LAMMPS and compared
    against mace-torch.

    Args:
        grad: Raw ``∂E/∂ε`` ``(..., 3, 3)``.

    Returns:
        The virial ``W`` ``(..., 3, 3)``, symmetric.
    """
    return 0.5 * (grad + grad.transpose(-1, -2))


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
            Cauchy stress ``σ = W / V``, symmetric — see
            :func:`symmetrize_strain_grad`.
        """
        grad = symmetrize_strain_grad(torch.func.grad(energy_fn)(strain))

        volume = torch.det(cell).abs()
        stress = grad / volume.unsqueeze(-1).unsqueeze(-1)

        return stress
