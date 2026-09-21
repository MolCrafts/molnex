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

        Raises:
            RuntimeError: A term returned something other than a scalar. That was
                always the contract — this returns "a scalar tensor" — and the dot
                product below makes it explicit instead of broadcasting silently.
        """
        # One dot product, not a loop that multiplies each term by a slice of the
        # weight buffer. Iterating a tensor yields 0-dim **device** scalars, and
        # those — together with a 0-dim `new_zeros(())` accumulator — fused into a
        # reduction kernel that died in `cuPointerGetAttribute` with
        # `CUDA_ERROR_INVALID_VALUE` whenever the combined loss also consumed
        # `atoms.forces`, i.e. whenever its backward was second order. Either half
        # of stage 3's objective ran alone, and `energy(...) + force(...)` written
        # by hand ran too; only this combinator failed (probes 2755942, 2757760,
        # 2758109). The buffer is still what holds the weights, so `.to()` and
        # `state_dict` behave as before.
        terms = torch.stack([loss_fn(pred, target).reshape(()) for loss_fn in self._terms])
        return (self.weights.to(terms.dtype) * terms).sum()
