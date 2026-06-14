"""Default evaluation step implementation."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import torch

from molix.config import config

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
            Force models now derive ``F = -∂E/∂pos`` via functorch
            (``torch.func.grad``), which manages its own differentiation and
            so produces valid eval forces even under ``no_grad``. The
            ``enable_grad`` default is kept as a harmless safety margin for any
            consumer that still inspects the outer eval graph; set ``True`` for
            pure energy/property models where the small graph-construction
            memory saving matters.
    """

    def __init__(self, *, no_grad: bool = False) -> None:
        self._no_grad = no_grad

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
        assert trainer.loss_fn is not None

        # Resolve autocast args only when AMP is on (see DefaultTrainStep).
        amp_enabled = bool(config["use_amp"])
        grad_ctx = torch.no_grad() if self._no_grad else torch.enable_grad()
        if amp_enabled:
            device_type = next(trainer.model.parameters()).device.type
            amp_ctx = torch.amp.autocast(device_type, dtype=config["amp_dtype"])
        else:
            amp_ctx = nullcontext()
        with grad_ctx, amp_ctx:
            predictions = trainer.model(batch)
            loss = trainer.loss_fn(predictions, batch)

        # Detached tensor (not .item()) so eval doesn't force a GPU→CPU sync
        # every batch — mirrors train/loss; consumers materialize via float().
        state["eval"]["loss"] = loss.detach()
        return {"loss": loss, "predictions": predictions}
