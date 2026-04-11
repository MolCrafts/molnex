"""Default evaluation step implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from molix.core.state import TrainState
    from molix.core.trainer import Trainer


class DefaultEvalStep:
    """Default evaluation step implementation for standard supervised learning.

    This step implements the typical evaluation computation flow:
    1. Forward pass through model (without gradient tracking)
    2. Loss computation (for logging/metrics)

    This is the default step used when no custom eval_step is provided
    to the Trainer.

    Example:
        ```python
        from molix.core.trainer import Trainer
        from molix.core.steps import DefaultEvalStep

        # Explicit use (equivalent to default)
        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer_factory=lambda p: torch.optim.Adam(p),
            eval_step=DefaultEvalStep(),
        )

        # Implicit use (default behavior)
        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer_factory=lambda p: torch.optim.Adam(p),
        )
        ```

    See Also:
        - DefaultTrainStep: Corresponding training step
        - Step: Protocol definition
    """

    def on_train_batch(
        self,
        trainer: "Trainer",
        state: "TrainState",
        batch: Any,
    ) -> dict[str, Any]:
        """Not implemented - use DefaultTrainStep for training.

        DefaultEvalStep is only for evaluation batches. For training,
        use DefaultTrainStep or provide a custom train_step.

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
        """Execute standard evaluation batch computation.

        Args:
            trainer: Trainer instance with model and loss_fn
            state: Current training state
            batch: Input batch data

        Returns:
            Dictionary with "loss" and "predictions" keys
        """
        # Forward pass (no gradients)
        from molix.core.steps import extract_model_inputs

        assert trainer.model is not None, "trainer.model must be set"
        assert trainer.loss_fn is not None, "trainer.loss_fn must be set"

        with torch.no_grad():
            if isinstance(batch, dict):
                model_inputs = extract_model_inputs(batch)
                predictions = trainer.model(**model_inputs)
            else:
                predictions = trainer.model(batch)

            # Compute loss (loss_fn should handle batch format)
            loss = trainer.loss_fn(predictions, batch)

        # Write loss to state
        state["eval/loss"] = loss.item()

        # Return predictions and loss for hooks/metrics
        return {
            "loss": loss,
            "predictions": predictions,
        }
