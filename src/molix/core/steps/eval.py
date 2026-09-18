"""Default evaluation step implementation."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import torch

from molix.config import config
from molix.core.steps.base import _mark_leaves

if TYPE_CHECKING:
    from molix.core.state import TrainState
    from molix.core.trainer import Trainer


class DefaultEvalStep:
    """Default evaluation step with optional AMP support.

    Precision is controlled globally via :meth:`molix.config.MolnexConfig.set_precision`.
    When ``config["use_amp"]`` is true the forward pass runs under
    :func:`torch.amp.autocast` with ``config["amp_dtype"]``. No
    ``GradScaler`` is used during evaluation.

    Args:
        no_grad: Wrap forward in ``torch.no_grad()`` (default ``False``).
            A force model derives ``F = -∂E/∂pos`` through
            :class:`~molix.autograd.Derivative`, which forces grad on for its
            own ``autograd.grad`` and so produces valid (detached) eval forces
            even under ``no_grad``. The ``enable_grad`` default is kept as a
            harmless safety margin for any consumer that still inspects the
            outer eval graph; set ``True`` for pure energy/property models
            where the small graph-construction memory saving matters.

    :meth:`compute_loss` is the sanctioned seam for subclasses that need a
    different *objective* while keeping the grad-context / AMP semantics (see
    :class:`~molix.core.steps.multitask.MultiTaskEvalStep`).

    :meth:`forward_loss` is an eager prologue plus :meth:`_traced_forward_loss`
    — forward pass plus objective, the region
    :meth:`molix.core.trainer.Trainer.compile` traces for evaluation
    (``step=True``, its default). The prologue marks the derivative op's
    leaves on the batch as the training step does, so the eval region is one
    graph whether it runs with grad or not; the grad and autocast contexts wrap
    the call from outside, so dynamo captures one eval graph under the
    configured ``no_grad`` and AMP state. Which losses trace is listed on
    :class:`~molix.core.steps.train.DefaultTrainStep`.
    """

    #: Why :meth:`forward_loss` cannot be traced, or ``None`` when it can —
    #: see :attr:`~molix.core.steps.train.DefaultTrainStep.untraceable`.
    untraceable: str | None = None

    def __init__(self, *, no_grad: bool = False) -> None:
        self._no_grad = no_grad

    def compute_loss(
        self,
        trainer: "Trainer",
        state: "TrainState",
        batch: Any,
        predictions: Any,
    ) -> torch.Tensor:
        """Evaluate the objective for one batch (override point, no cadence).

        Called from inside the grad-context / autocast block of
        :meth:`on_eval_batch`, so an override sees the same AMP dtype and the
        same ``no_grad`` setting as the forward pass. The default body
        delegates to ``trainer.loss_fn``; a subclass that owns its losses (e.g.
        per-task routing) may override this and run with
        ``Trainer(loss_fn=None)``.

        Args:
            trainer: The owning :class:`~molix.core.trainer.Trainer`.
            state: The :class:`~molix.core.state.TrainState` to record into.
            batch: The collated batch handed to the model.
            predictions: The model output for ``batch``.

        Returns:
            The scalar loss tensor for this batch.
        """
        assert trainer.loss_fn is not None
        return trainer.loss_fn(predictions, batch)

    def forward_loss(
        self,
        trainer: "Trainer",
        state: "TrainState",
        batch: Any,
    ) -> tuple[torch.Tensor, Any]:
        """The eager leaf prologue, then the traced body.

        Marks the derivative op's leaves on ``batch`` outside any compiled
        region (:func:`~molix.core.steps.base._mark_leaves`), then runs
        :meth:`_traced_forward_loss` through the instance attribute — the one
        ``Trainer.compile(step=True)`` replaces with the compiled wrapper.
        :meth:`on_eval_batch` calls this method.

        Args:
            trainer: The owning :class:`~molix.core.trainer.Trainer`
                (provides ``model`` and ``loss_fn``).
            state: The :class:`~molix.core.state.TrainState`, threaded to
                :meth:`compute_loss` unchanged.
            batch: The collated batch handed to the model.

        Returns:
            ``(loss, predictions)`` — the scalar loss for this batch and the
            model output the loss was evaluated on.
        """
        _mark_leaves(trainer.model, batch)
        return self._traced_forward_loss(trainer, state, batch)

    def _traced_forward_loss(
        self,
        trainer: "Trainer",
        state: "TrainState",
        batch: Any,
    ) -> tuple[torch.Tensor, Any]:
        """Forward pass and objective: the region ``Trainer.compile`` traces.

        ``predictions = trainer.model(batch)`` then :meth:`compute_loss`, and
        nothing else.

        Args:
            trainer: The owning :class:`~molix.core.trainer.Trainer`.
            state: The :class:`~molix.core.state.TrainState`, threaded to
                :meth:`compute_loss` unchanged.
            batch: The collated batch, its variable columns already marked.

        Returns:
            ``(loss, predictions)`` as :meth:`forward_loss`.
        """
        predictions = trainer.model(batch)
        return self.compute_loss(trainer, state, batch, predictions), predictions

    def on_train_batch(self, trainer: "Trainer", state: "TrainState", batch: Any) -> dict[str, Any]:
        """Not supported — :class:`DefaultEvalStep` handles eval batches only.

        Always raises :class:`NotImplementedError`; use
        :class:`~molix.core.steps.train.DefaultTrainStep` for training
        batches.
        """
        raise NotImplementedError(
            "DefaultEvalStep.on_train_batch() is not implemented. "
            "Use DefaultTrainStep for training batches."
        )

    def on_eval_batch(self, trainer: "Trainer", state: "TrainState", batch: Any) -> dict[str, Any]:
        """Run one evaluation batch: forward pass + loss, no parameter update.

        Runs the forward pass under ``torch.no_grad()`` (or
        ``torch.enable_grad()`` when ``no_grad=False``), optionally inside
        :func:`torch.amp.autocast` when ``config["use_amp"]`` is set, then
        evaluates the loss. Writes ``state["eval"]["loss"]`` as a detached
        tensor (no per-batch GPU sync); consumers materialize via
        ``float()``.

        Args:
            trainer: The owning :class:`~molix.core.trainer.Trainer`
                (provides ``model`` and ``loss_fn``).
            state: The :class:`~molix.core.state.TrainState` to record into.
            batch: A collated batch ``TensorDict`` for the model.

        Returns:
            ``{"loss": <Tensor>, "predictions": <model output>}``.
        """
        assert trainer.model is not None

        # Resolve autocast args only when AMP is on (see DefaultTrainStep).
        amp_enabled = bool(config["use_amp"])
        grad_ctx = torch.no_grad() if self._no_grad else torch.enable_grad()
        if amp_enabled:
            device_type = next(trainer.model.parameters()).device.type
            amp_ctx = torch.amp.autocast(device_type, dtype=config["amp_dtype"])
        else:
            amp_ctx = nullcontext()
        with grad_ctx, amp_ctx:
            loss, predictions = self.forward_loss(trainer, state, batch)

        # Detached tensor (not .item()) so eval doesn't force a GPU→CPU sync
        # every batch — mirrors train/loss; consumers materialize via float().
        state["eval"]["loss"] = loss.detach()
        return {"loss": loss, "predictions": predictions}
