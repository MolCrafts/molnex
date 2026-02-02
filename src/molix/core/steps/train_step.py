"""Default training step implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from molix.core.trainer import Trainer
    from molix.core.state import TrainState


class DefaultTrainStep:
    """Default training step implementation for standard supervised learning.
    
    This step implements the typical training computation flow:
    1. Forward pass through model
    2. Loss computation
    3. Backward pass (gradient computation)
    4. Optimizer step
    
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
        
        Args:
            trainer: Trainer instance with model, optimizer, and loss_fn
            state: Current training state
            batch: Input batch data
        
        Returns:
            Dictionary with "loss" and "predictions" keys
        """
        # Forward pass
        if hasattr(batch, "to_model_kwargs"):
            predictions = trainer.model(**batch.to_model_kwargs())
        else:
            predictions = trainer.model(batch)
        
        # Compute loss (loss_fn should handle batch format)
        loss = trainer.loss_fn(predictions, batch)
        
        # Backward pass
        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()
        
        # Write loss to state
        state["train/loss"] = loss.item()
        
        # Return only predictions in outputs
        return {
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
