"""Default training step implementation."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import torch

from molix.config import config

if TYPE_CHECKING:
    from molix.core.state import TrainState
    from molix.core.trainer import Trainer


class DefaultTrainStep:
    """Default training step with optional AMP support.

    Precision is controlled globally via :func:`molix.config.set_precision`
    which writes ``use_amp`` and ``amp_dtype`` into the global
    :data:`molix.config.config`. When ``use_amp`` is true the forward pass
    runs under :func:`torch.amp.autocast` and backward uses the
    :class:`torch.amp.GradScaler` owned by the :class:`Trainer`.

    Flow:
        1. Forward pass (under autocast if AMP enabled)
        2. Loss computation
        3. Backward pass (with ``trainer.scaler`` if AMP enabled)
        4. ``on_after_backward`` hook point (gradients are unscaled)
        5. Optimizer step

    See Also:
        - :class:`DefaultEvalStep`
        - :func:`molix.config.set_precision`
    """

    def on_train_batch(
        self,
        trainer: "Trainer",
        state: "TrainState",
        batch: Any,
    ) -> dict[str, Any]:
        """Execute training batch computation."""
        from molix.core.steps import batch_to_device, extract_model_inputs

        assert trainer.model is not None, "trainer.model must be set"
        assert trainer.loss_fn is not None, "trainer.loss_fn must be set"
        assert trainer.optimizer is not None, "trainer.optimizer must be set"

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
        with ctx:
            if isinstance(batch, dict):
                model_inputs = extract_model_inputs(batch)
                predictions = trainer.model(**model_inputs)
            else:
                predictions = trainer.model(batch)
            loss = trainer.loss_fn(predictions, batch)

        trainer.optimizer.zero_grad()
        scaler = trainer.scaler if amp_enabled else None
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(trainer.optimizer)
        else:
            loss.backward()

        # Hook point: gradients are ready and unscaled
        trainer._call_hooks("on_after_backward", trainer, state)

        if scaler is not None:
            scaler.step(trainer.optimizer)
            scaler.update()
        else:
            trainer.optimizer.step()

        state["train/loss"] = loss.item()

        return {
            "loss": loss,
            "predictions": predictions,
        }

    def on_eval_batch(
        self,
        trainer: "Trainer",
        state: "TrainState",
        batch: Any,
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "DefaultTrainStep.on_eval_batch() is not implemented. "
            "Use DefaultEvalStep for evaluation batches."
        )
