"""Training step implementation."""

from typing import Any, Callable, Mapping, Optional

from molnex.core.state import TrainState


class TrainStep:
    """Training step - executes one training iteration.
    
    This step:
    - Represents one semantic training step
    - Is executed by the Trainer during training
    - Is structurally compatible with MolExp's Op protocol
    
    Attributes:
        op_name: Operation identifier
        forward_fn: Forward pass function (model)
        optimizer_fn: Optimizer step function
    """
    
    op_name: str = "train_step"
    
    def __init__(
        self,
        forward_fn: Optional[Callable[[Any], Any]] = None,
        optimizer_fn: Optional[Callable[[], None]] = None,
    ):
        """Initialize training step.
        
        Args:
            forward_fn: Function to compute forward pass (e.g., model(batch))
            optimizer_fn: Function to perform optimizer step (e.g., optimizer.step())
        """
        self.forward_fn = forward_fn
        self.optimizer_fn = optimizer_fn
    
    def input_schema(self) -> Mapping[str, Any]:
        """Return input schema specification.
        
        Returns:
            Mapping describing expected inputs
        """
        return {
            "train_state": "TrainState",
            "batch": "Any",
        }
    
    def output_schema(self) -> Mapping[str, Any]:
        """Return output schema specification.
        
        Returns:
            Mapping describing output structure
        """
        return {
            "loss": "Optional[Any]",
            "result": "Any",
            "logs": "Mapping[str, Any]",
        }
    
    def run(self, train_state: TrainState, *, batch: Any) -> Mapping[str, Any]:
        """Execute training step.
        
        Args:
            train_state: Current training state
            batch: Input batch data
            
        Returns:
            Mapping with loss, result, and logs
        """
        # Default implementation - can be overridden or configured via functions
        loss = None
        result = None
        
        if self.forward_fn is not None:
            result = self.forward_fn(batch)
            loss = result  # Assume result is loss for simple cases
        
        if self.optimizer_fn is not None:
            self.optimizer_fn()
        
        logs = {
            "step": train_state.global_step,
            "epoch": train_state.epoch,
        }
        
        return {
            "loss": loss,
            "result": result,
            "logs": logs,
        }
