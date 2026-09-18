"""Generic weighted combination of multiple losses."""

from collections.abc import Callable
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

    A term is any callable ``(pred, target) -> Tensor``. Both kinds that
    occur here are accepted: an :class:`~torch.nn.Module` (which may carry
    buffers, e.g. :class:`ScaleNormalizedLoss`'s scale) and a plain closure
    (what :func:`~molix.core.losses.energy_loss` and
    :func:`~molix.core.losses.force_loss` return — those hold no state by
    design, because "molnex adds no loss class of its own"). The modules are
    registered so ``.to()`` and ``state_dict`` reach them; the closures have
    nothing to move. Rejecting closures would make the two halves of this
    package unusable together.

    Args:
        losses: List of ``(weight, loss_fn)`` tuples.

    Example::

        from torch import nn
        from molix.core.losses import WeightedLoss, energy_loss, force_loss

        smooth = lambda beta: nn.SmoothL1Loss(beta=beta, reduction="none")
        loss = WeightedLoss([
            (1.0, energy_loss("energy_residual", pred_key="energy_mm",
                              elementwise=smooth(20.0), weight_key="mask")),
            (1.0, force_loss("forces", elementwise=smooth(20.0),
                             weight_key="force_weight")),
        ])
    """

    def __init__(
        self,
        losses: list[tuple[float, Callable[[Any, Any], torch.Tensor]]],
    ):
        super().__init__()
        self._terms = [loss_fn for _, loss_fn in losses]
        # Only the stateful terms go in the ModuleList — a closure in there
        # raises. Call order lives in ``_terms``, which holds both kinds.
        self.losses = nn.ModuleList([fn for fn in self._terms if isinstance(fn, nn.Module)])
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
        for weight, loss_fn in zip(self.weights, self._terms):
            total_loss = total_loss + weight * loss_fn(pred, target)
        return total_loss
