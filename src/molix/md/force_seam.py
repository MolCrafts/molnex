"""Force seam: wrap a PiNetPotential as a position-only ``force_fn`` for MD.

The integrator drives positions and needs *live* forces. Each call swaps the
candidate positions into a clone of the molecule template, runs
``PiNetPotential.forward(td, compute_forces=True)`` — which derives forces
functionally via ``torch.func.grad`` (no ``requires_grad`` bookkeeping needed on
the input) — and returns ``(energy, forces)`` detached for the integrator's
arithmetic.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from tensordict import TensorDict
from torch import nn

ForceFn = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def build_force_fn(model: nn.Module, template: TensorDict) -> ForceFn:
    """Return a ``force_fn(pos) -> (energy, forces)`` backed by ``model``.

    Args:
        model: A ``PiNetPotential`` (or any module with the same forward shape).
        template: A molecule TensorDict carrying topology (Z, edge_index, batch,
            graphs); its ``("atoms", "pos")`` is replaced per call.

    Returns:
        A callable mapping positions ``(N, 3)`` to ``(scalar energy, forces (N, 3))``,
        both detached.
    """

    def force_fn(pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = template.clone()
        batch["atoms", "pos"] = pos.detach()
        out = model(batch, compute_forces=True)
        return out["energy"].sum().detach(), out["forces"].detach()

    return force_fn
