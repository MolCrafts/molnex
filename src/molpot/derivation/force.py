"""Force derivation via functorch: ``F = -∂E/∂pos``.

Single responsibility: compute atomic forces as the negative gradient of energy
with respect to atomic positions, using ``torch.func.grad`` (functorch).

Example:
    >>> deriv = ForceDerivation()
    >>> pos = torch.randn(5, 3)
    >>> forces = deriv(energy_fn, pos)  # (5, 3); energy_fn(pos) -> scalar
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn


class ForceDerivation(nn.Module):
    """Compute forces functionally as ``F = -torch.func.grad(energy_fn)(pos)``.

    Takes the energy as a **pure function of positions** and differentiates it
    with ``torch.func.grad``. That transform is *traced into the forward graph*,
    so the inner force derivative becomes ordinary forward ops and the whole
    ``energy → force → loss`` needs only ONE regular backward w.r.t. parameters.
    No double-backward barrier, so it composes with ``torch.compile(fullgraph=True)``.

    Validated by the Phase-3 spikes (``trainer-hardening-01-compile-strategy``):
    correct vs a plain ``autograd.grad`` reference to ~1e-8, fullgraph-compiles
    with zero graph breaks, and a single ``loss.backward()`` populates parameter
    grads — on both PiNet (no cuEq) and MACE (cuEquivariance).

    Note: this eliminates the *compilation barrier*, not the second-order compute —
    the mixed second-derivative FLOPs/memory are still incurred.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        energy_fn: Callable[[torch.Tensor], torch.Tensor],
        pos: torch.Tensor,
    ) -> torch.Tensor:
        """Compute forces as the negative functional gradient of energy.

        Args:
            energy_fn: Maps positions ``(N, 3)`` to a **scalar** total energy,
                closing over the model parameters and the rest of the batch.
                Must recompute any position-derived geometry (edge vectors,
                distances) *inside* itself so the gradient flows through them.
            pos: Atomic positions ``(N, 3)``. Does not need ``requires_grad`` —
                ``torch.func.grad`` tracks the input itself.

        Returns:
            Atomic forces ``(N, 3)``.
        """
        return -torch.func.grad(energy_fn)(pos)
