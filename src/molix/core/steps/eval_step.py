"""Default evaluation step implementation."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from molix.core.state import TrainState
    from molix.core.trainer import Trainer


class DefaultEvalStep:
    """Default evaluation step with optional AMP support.

    This step implements the evaluation computation flow:
    1. Forward pass without gradient tracking (under autocast if AMP enabled)
    2. Loss computation (for logging/metrics)

    Args:
        amp_dtype: If provided (e.g. ``torch.float16`` or ``torch.bfloat16``),
            enables autocast for the forward pass. No GradScaler is used
            during evaluation. When ``None`` (default), no AMP is used.

    Example:
        ```python
        from molix.core.steps import DefaultEvalStep

        # Without AMP (default, unchanged behavior)
        step = DefaultEvalStep()

        # With AMP
        step = DefaultEvalStep(amp_dtype=torch.bfloat16)

        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer_factory=lambda p: torch.optim.Adam(p),
            eval_step=step,
        )
        ```

    See Also:
        - DefaultTrainStep: Corresponding training step
        - Step: Protocol definition
    """

    def __init__(self, *, amp_dtype: torch.dtype | None = None):
        self._amp_dtype = amp_dtype
        self._amp_enabled = amp_dtype is not None

    def on_train_batch(
        self,
        trainer: "Trainer",
        state: "TrainState",
        batch: Any,
    ) -> dict[str, Any]:
        """Not implemented - use DefaultTrainStep for training.

        Raises:
            NotImplementedError: Always raises, this step is for evaluation only
        """
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
        """Execute evaluation batch computation.

        Args:
            trainer: Trainer instance with model and loss_fn
            state: Current training state
            batch: Input batch data

        Returns:
            Dictionary with "loss" and "predictions" keys
        """
        from molix.core.steps import extract_model_inputs

        assert trainer.model is not None, "trainer.model must be set"
        assert trainer.loss_fn is not None, "trainer.loss_fn must be set"

        device_type = next(trainer.model.parameters()).device.type

        ctx = (
            torch.amp.autocast(device_type, dtype=self._amp_dtype)
            if self._amp_enabled
            else nullcontext()
        )
        with torch.no_grad(), ctx:
            if isinstance(batch, dict):
                model_inputs = extract_model_inputs(batch)
                predictions = trainer.model(**model_inputs)
            else:
                predictions = trainer.model(batch)

            # Compute loss
            loss = trainer.loss_fn(predictions, batch)

        # Write loss to state
        state["eval/loss"] = loss.item()

        # Return predictions and loss for hooks/metrics
        return {
            "loss": loss,
            "predictions": predictions,
        }
