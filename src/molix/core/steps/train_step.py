"""Default training step implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from molix.core.state import TrainState
    from molix.core.trainer import Trainer


class DefaultTrainStep:
    """Default training step implementation for standard supervised learning.

    This step implements the typical training computation flow:
    1. Forward pass through model
    2. Loss computation
    3. Backward pass (gradient computation)
    4. Optimizer step
    5. LR scheduler step (per-batch, if present)

    When ``trainer.scaler`` is set (AMP enabled), the forward pass runs
    inside ``torch.amp.autocast`` and gradients are scaled/unscaled via
    the ``GradScaler``.

    This is the default step used when no custom train_step is provided
    to the Trainer.

    Example:
        ```python
        from molix.core.trainer import Trainer
        from molix.core.steps import DefaultTrainStep

        # Explicit use (equivalent to default)
        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer_factory=lambda p: torch.optim.Adam(p),
            train_step=DefaultTrainStep(),
        )

        # Implicit use (default behavior)
        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer_factory=lambda p: torch.optim.Adam(p),
        )
        ```

    See Also:
        - DefaultEvalStep: Corresponding evaluation step
        - Step: Protocol definition
    """

    def on_train_batch(
        self,
        trainer: "Trainer",
        state: "TrainState",
        batch: Any,
    ) -> dict[str, Any]:
        """Execute standard training batch computation.

        Supports AMP (when ``trainer.scaler`` is set) and per-batch LR
        scheduling (when ``trainer.lr_scheduler`` is set).

        Args:
            trainer: Trainer instance with model, optimizer, and loss_fn
            state: Current training state
            batch: Input batch data

        Returns:
            Dictionary with "loss" and "predictions" keys
        """
        from molix.core.steps import extract_model_inputs

        if isinstance(batch, dict):
            model_inputs = extract_model_inputs(batch)
        else:
            model_inputs = None

        if trainer.scaler is not None:
            # AMP path
            device_type = next(trainer.model.parameters()).device.type
            with torch.amp.autocast(
                device_type=device_type, dtype=trainer.amp_dtype
            ):
                if model_inputs is not None:
                    predictions = trainer.model(**model_inputs)
                else:
                    predictions = trainer.model(batch)
                loss = trainer.loss_fn(predictions, batch)

            trainer.optimizer.zero_grad()
            trainer.scaler.scale(loss).backward()
            trainer.scaler.step(trainer.optimizer)
            trainer.scaler.update()
        else:
            # Standard path
            if model_inputs is not None:
                predictions = trainer.model(**model_inputs)
            else:
                predictions = trainer.model(batch)

            loss = trainer.loss_fn(predictions, batch)

            trainer.optimizer.zero_grad()
            loss.backward()
            trainer.optimizer.step()

        # LR scheduler step (per-batch)
        if trainer.lr_scheduler is not None:
            trainer.lr_scheduler.step()

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

        DefaultTrainStep is only for training batches. For evaluation,
        use DefaultEvalStep or provide a custom eval_step.

        Raises:
            NotImplementedError: Always raises, this step is for training only
        """
        raise NotImplementedError(
            "DefaultTrainStep.on_eval_batch() is not implemented. "
            "Use DefaultEvalStep for evaluation batches."
        )
