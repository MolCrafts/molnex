"""Step protocol definition for Molix.

The Step protocol defines the interface for training and evaluation
computation steps, separating computation logic from control flow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from molix.core.state import TrainState
    from molix.core.trainer import Trainer


class Step(Protocol):
    """Protocol for training and evaluation computation steps.

    Steps encapsulate the computation logic for processing batches during
    training and evaluation, while the Trainer handles control flow (loops,
    hooks, state management).

    Key Responsibilities:
    - Execute forward pass through model
    - Compute loss function
    - Perform backward pass and optimizer updates (training only)
    - Return outputs for logging/hooks

    The Trainer provides access to model, optimizer, loss function, and other
    resources through the `trainer` parameter.

    Gradient accumulation is **not** a Step concern — it is owned by the
    Trainer via ``Trainer(accumulate_grad_batches=N)``, which scales the loss
    and gates the optimizer / LR-scheduler / ``global_step`` advance on
    accumulation boundaries. A custom Step that re-implemented accumulation by
    skipping ``optimizer.step()`` would desync the scheduler (which the Trainer
    still advances per optimizer step). Implement a custom Step only to change
    the *computation* (e.g. an auxiliary loss term), not the step cadence.

    Example — a custom step that adds an L2 activation penalty::

        class PenalizedStep:
            def on_train_batch(self, trainer, state, batch):
                predictions = trainer.model(batch)
                loss = trainer.loss_fn(predictions, batch)
                loss = loss + 1e-4 * predictions["energy"].square().mean()
                trainer.optimizer.zero_grad()
                loss.backward()
                trainer._call_hooks("on_after_backward", trainer, state)
                trainer.optimizer.step()
                state["train"]["loss"] = loss.detach()
                return {"loss": loss, "predictions": predictions}

            def on_eval_batch(self, trainer, state, batch):
                predictions = trainer.model(batch)
                loss = trainer.loss_fn(predictions, batch)
                return {"loss": loss, "predictions": predictions}

    See Also:
        - DefaultTrainStep: Default training step (AMP + gradient accumulation)
        - DefaultEvalStep: Default evaluation step implementation
    """

    def on_train_batch(
        self,
        trainer: "Trainer",
        state: "TrainState",
        batch: Any,
    ) -> dict[str, Any]:
        """Execute training batch computation.

        This method is called by the Trainer for each training batch. It should:
        1. Perform forward pass through the model
        2. Compute the loss
        3. Perform backward pass (gradients)
        4. Update model parameters via optimizer
        5. Return outputs for hooks and logging

        Args:
            trainer: Trainer instance providing access to:
                - trainer.model: Neural network model
                - trainer.optimizer: Optimizer instance
                - trainer.loss_fn: Loss function
                - trainer.state: Training state (alternative to state param)
            state: Current training state with:
                - state.global_step: Current training step
                - state.epoch: Current epoch
                - state.stage: Current stage (TRAIN, EVAL, etc.)
            batch: Input batch data (typically dict or tensor)

        Returns:
            Dictionary with outputs, must include:
                - "loss": Loss tensor or numeric value
                - "predictions": Model predictions
            Additional keys allowed (e.g., "metrics", "auxiliary_outputs")

        Example:
            ```python
            def on_train_batch(self, trainer, state, batch):
                # Forward pass
                predictions = trainer.model(batch)

                # Compute loss (loss_fn reads nested targets from the batch,
                # e.g. batch["graphs", "energy"])
                loss = trainer.loss_fn(predictions, batch)

                # Backward pass
                trainer.optimizer.zero_grad()
                loss.backward()
                trainer.optimizer.step()

                return {"loss": loss, "predictions": predictions}
            ```
        """
        ...

    def on_eval_batch(
        self,
        trainer: "Trainer",
        state: "TrainState",
        batch: Any,
    ) -> dict[str, Any]:
        """Execute evaluation batch computation.

        This method is called by the Trainer for each evaluation batch. It should:
        1. Perform forward pass through the model (without gradients)
        2. Compute the loss (for logging/metrics)
        3. Return outputs for hooks and logging

        Note: No optimizer updates should be performed during evaluation.

        Args:
            trainer: Trainer instance providing access to:
                - trainer.model: Neural network model
                - trainer.loss_fn: Loss function
                - trainer.state: Training state
            state: Current training state
            batch: Input batch data (typically dict or tensor)

        Returns:
            Dictionary with outputs, must include:
                - "loss": Loss tensor or numeric value (for logging)
                - "predictions": Model predictions
            Additional keys allowed (e.g., "metrics", "auxiliary_outputs")

        Example:
            ```python
            def on_eval_batch(self, trainer, state, batch):
                # Forward pass. Do NOT wrap force models in torch.no_grad():
                # F = -dE/dx needs a live graph (see DefaultEvalStep.no_grad).
                predictions = trainer.model(batch)
                loss = trainer.loss_fn(predictions, batch)
                return {"loss": loss, "predictions": predictions}
            ```
        """
        ...
