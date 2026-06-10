"""Generic weighted combination of multiple losses."""

from typing import Any

import torch
import torch.nn as nn


class WeightedLoss(nn.Module):
    """Weighted combination of multiple loss functions.

    Computes ``L = Σ wᵢ · Lᵢ(pred, target)`` over the supplied terms. Each
    term is a ``(weight, loss_fn)`` tuple.

    The weights are held as a registered buffer, so they move with
    :meth:`~torch.nn.Module.to` / ``.cuda()`` and ride along in
    ``state_dict`` — no per-step device copy. Every term is evaluated and
    summed; a term that raises (e.g. a missing target key) propagates
    rather than being silently skipped, because silently dropping a term
    means training a *different* objective than intended (energy-only when
    you asked for energy+force) with no signal.

    Args:
        losses: List of ``(weight, loss_fn)`` tuples.
    """

    def __init__(
        self,
        losses: list[tuple[float, nn.Module]],
    ):
        super().__init__()
        self.losses = nn.ModuleList([loss_fn for _, loss_fn in losses])
        self.register_buffer(
            "weights",
            torch.tensor([float(weight) for weight, _ in losses]),
        )
        self.weights: torch.Tensor

    def forward(
        self,
        pred: Any,
        target: Any,
    ) -> torch.Tensor:
        """Compute the weighted sum of all loss terms.

        Args:
            pred: Predictions (dict/dataclass or tensor).
            target: Targets (dict/dataclass or tensor).

        Returns:
            Weighted sum of all losses (scalar tensor).
        """
        total_loss = self.weights.new_zeros(())
        for weight, loss_fn in zip(self.weights, self.losses):
            total_loss = total_loss + weight * loss_fn(pred, target)
        return total_loss
