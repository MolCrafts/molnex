"""Generic weighted combination of multiple losses."""

from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn as nn


class WeightedLoss(nn.Module):
    """Weighted combination of multiple loss functions.

    Computes ``L = Σ wᵢ · Lᵢ(pred, target)`` over the supplied terms. Each
    term is a ``(weight, loss_fn)`` tuple.

    **The weights are plain Python floats, deliberately.** They are configuration,
    not learned state, and a float multiplying a device tensor folds into the kernel
    — where a weight tensor raises the one question this class has no answer to:
    which device is it on? Nobody calls ``.to()`` on a loss, so a registered buffer
    stays on the host, and a host 0-dim scalar multiplied into a CUDA tensor is
    accepted silently in eager and then dies under inductor —
    ``CUDA_ERROR_INVALID_VALUE from cuPointerGetAttribute`` inside the fused
    reduction. That took out every cell of two stage-3 calibration grids, and only
    when the combined loss also consumed ``atoms.forces``, i.e. when its backward
    was second order; either half alone, and a hand-written ``a(...) + b(...)``,
    ran clean (probes 2755942, 2757760, 2758109, 2758519).

    Every term is evaluated and summed; a term that raises (e.g. a missing target
    key) propagates rather than being silently skipped, because silently dropping a
    term means training a *different* objective than intended (energy-only when you
    asked for energy+force) with no signal.

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
        names: One short name per term, for the per-term scalars a step
            publishes. ``None`` derives them from each term's class name, or
            ``term<i>`` for a closure. A combined objective whose terms are not
            reported separately can only be watched as a sum, and a sum that
            climbs does not say which term is climbing — which is how stage 4 of
            the Class-I curriculum reached a minimum and walked back up at every
            learning rate with nothing to point at.

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
        *,
        names: Sequence[str] | None = None,
    ):
        super().__init__()
        self._terms = [loss_fn for _, loss_fn in losses]
        # Only the stateful terms go in the ModuleList — a closure in there
        # raises. Call order lives in ``_terms``, which holds both kinds.
        self.losses = nn.ModuleList([fn for fn in self._terms if isinstance(fn, nn.Module)])
        self.weights: tuple[float, ...] = tuple(float(weight) for weight, _ in losses)
        if names is not None and len(names) != len(self._terms):
            raise ValueError(
                f"names has {len(names)} entries for {len(self._terms)} terms; one name per term"
            )
        self.names: tuple[str, ...] = (
            tuple(str(name) for name in names)
            if names is not None
            else tuple(
                type(fn).__name__.removesuffix("Loss").lower()
                if isinstance(fn, nn.Module)
                else f"term{i}"
                for i, fn in enumerate(self._terms)
            )
        )

    def terms(self, pred: Any, target: Any) -> dict[str, torch.Tensor]:
        """Each term's **unweighted** value, by name.

        For reporting, not for the objective: :meth:`forward` is the weighted sum
        and stays the one thing differentiated. A caller watching a combined
        objective needs the parts, because their sum can fall while one of them
        rises.

        Args:
            pred: Forwarded to each term.
            target: Forwarded to each term.

        Returns:
            ``{name: value}`` in declaration order.
        """
        return {name: term(pred, target) for name, term in zip(self.names, self._terms)}

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
            Weighted sum of all losses, on the terms' own device and dtype.
        """
        # No accumulator seeded from a tensor of this class's own: the first term
        # decides device and dtype, and each weight is a Python scalar. That is the
        # shape a hand-written `a(pred, target) + b(pred, target)` has, which is the
        # one shape measured to survive the second-order backward (see the class
        # docstring).
        total: torch.Tensor | None = None
        for weight, loss_fn in zip(self.weights, self._terms, strict=True):
            term = loss_fn(pred, target) * weight
            total = term if total is None else total + term
        if total is None:
            raise ValueError("WeightedLoss has no terms; give it at least one")
        return total
