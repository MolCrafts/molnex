"""Default training step implementation."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from molix.core.state import TrainState
    from molix.core.trainer import Trainer


class DefaultTrainStep:
    """Default training step with optional AMP support.

    This step implements the training computation flow:
    1. Forward pass (under autocast if AMP enabled)
    2. Loss computation
    3. Backward pass (with GradScaler if AMP enabled)
    4. ``on_after_backward`` hook point (gradients are unscaled)
    5. Optimizer step

    Args:
        amp_dtype: If provided (e.g. ``torch.float16`` or ``torch.bfloat16``),
            enables automatic mixed precision using ``torch.amp.autocast``
            and ``torch.amp.GradScaler``. When ``None`` (default), no AMP
            is used.

    Example:
        ```python
        from molix.core.steps import DefaultTrainStep

        # Without AMP (default, unchanged behavior)
        step = DefaultTrainStep()

        # With AMP
        step = DefaultTrainStep(amp_dtype=torch.bfloat16)

        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer_factory=lambda p: torch.optim.Adam(p),
            train_step=step,
        )
        ```

    See Also:
        - DefaultEvalStep: Corresponding evaluation step
        - Step: Protocol definition
    """

    def __init__(self, *, amp_dtype: torch.dtype | None = None):
        self._amp_dtype = amp_dtype
        self._amp_enabled = amp_dtype is not None
        self._scaler: torch.amp.GradScaler | None = None

    def on_train_batch(
        self,
        trainer: "Trainer",
        state: "TrainState",
        batch: Any,
    ) -> dict[str, Any]:
        """Execute training batch computation.

        Args:
            trainer: Trainer instance with model, optimizer, and loss_fn
            state: Current training state
            batch: Input batch data

        Returns:
            Dictionary with "loss" and "predictions" keys
        """
        from molix.core.steps import extract_model_inputs

        assert trainer.model is not None, "trainer.model must be set"
        assert trainer.loss_fn is not None, "trainer.loss_fn must be set"
        assert trainer.optimizer is not None, "trainer.optimizer must be set"

        device_type = next(trainer.model.parameters()).device.type

        # Lazy-init GradScaler on first call
        if self._amp_enabled and self._scaler is None:
            self._scaler = torch.amp.GradScaler(device_type)

        # Forward pass (under autocast if AMP enabled)
        ctx = (
            torch.amp.autocast(device_type, dtype=self._amp_dtype)
            if self._amp_enabled
            else nullcontext()
        )
        with ctx:
            if isinstance(batch, dict):
                model_inputs = extract_model_inputs(batch)
                predictions = trainer.model(**model_inputs)
            else:
                predictions = trainer.model(batch)

            # Compute loss
            loss = trainer.loss_fn(predictions, batch)

        # Backward pass
        trainer.optimizer.zero_grad()
        if self._amp_enabled:
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(trainer.optimizer)
        else:
            loss.backward()

        # Hook point: gradients are ready and unscaled
        trainer._call_hooks("on_after_backward", trainer, state)

        # Optimizer step
        if self._amp_enabled:
            self._scaler.step(trainer.optimizer)
            self._scaler.update()
        else:
            trainer.optimizer.step()

        # Write loss to state
        state["train/loss"] = loss.item()

        # Return predictions and loss for hooks/metrics
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
        """Not implemented - use DefaultEvalStep for evaluation.

        Raises:
            NotImplementedError: Always raises, this step is for training only
        """
        raise NotImplementedError(
            "DefaultTrainStep.on_eval_batch() is not implemented. "
            "Use DefaultEvalStep for evaluation batches."
        )
