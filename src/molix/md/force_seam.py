"""Force seam: wrap a PiNetPotential as a position-only ``force_fn`` for MD.

The integrator drives positions and needs *live* forces, so unlike the static
paired-delta path (which detaches the input), each call here rebuilds a fresh
grad-tracking ``pos`` leaf, runs ``PiNetPotential.forward(td, compute_forces=True)``
on a clone of the molecule template, and returns ``(energy, forces)`` detached
for the integrator's arithmetic. Rebuilding the leaf every step keeps the autograd
graph clean and supports PiNet's double-backward force path.
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
        leaf = pos.detach().clone().requires_grad_(True)
        batch["atoms", "pos"] = leaf
        out = model(batch, compute_forces=True)
        return out["energy"].sum().detach(), out["forces"].detach()

    return force_fn
