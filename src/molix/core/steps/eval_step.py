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

    Precision is controlled globally via :func:`molix.config.set_precision`.
    When ``config["use_amp"]`` is true the forward pass runs under
    :func:`torch.amp.autocast` with ``config["amp_dtype"]``. No
    ``GradScaler`` is used during evaluation.

    See Also:
        - :class:`DefaultTrainStep`
        - :func:`molix.config.set_precision`
    """

    def on_train_batch(
        self,
        trainer: "Trainer",
        state: "TrainState",
        batch: Any,
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "DefaultEvalStep.on_train_batch() is not implemented. "
            "Use DefaultTrainStep for training batches."
        )

    def on_eval_batch(
        self,
        trainer: "Trainer",
        state: "TrainState",
        batch: Any,
    ) -> dict[str, Any]:
        """Execute evaluation batch computation."""
        from molix.core.steps import batch_to_device, extract_model_inputs

        assert trainer.model is not None, "trainer.model must be set"
        assert trainer.loss_fn is not None, "trainer.loss_fn must be set"

        model_device = next(trainer.model.parameters()).device
        device_type = model_device.type
        amp_enabled = bool(config["use_amp"])
        amp_dtype = config["amp_dtype"]

        batch = batch_to_device(batch, model_device)

        ctx = (
            torch.amp.autocast(device_type, dtype=amp_dtype)
            if amp_enabled
            else nullcontext()
        )
        with torch.no_grad(), ctx:
            if isinstance(batch, dict):
                model_inputs = extract_model_inputs(batch)
                predictions = trainer.model(**model_inputs)
            else:
                predictions = trainer.model(batch)
            loss = trainer.loss_fn(predictions, batch)

        state["eval/loss"] = loss.item()

        return {
            "loss": loss,
            "predictions": predictions,
        }
