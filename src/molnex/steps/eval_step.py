"""Evaluation step implementation."""

from typing import Any, Callable, Mapping, Optional

from molnex.core.state import TrainState


class EvalStep:
    """Evaluation step - executes one validation iteration.
    
    This step:
    - Represents one semantic evaluation step
    - Is executed by the Trainer during validation
    - Is structurally compatible with MolExp's Op protocol
    
    Attributes:
        op_name: Operation identifier
        forward_fn: Forward pass function (model in eval mode)
    """
    
    op_name: str = "eval_step"
    
    def __init__(self, forward_fn: Optional[Callable[[Any], Any]] = None):
        """Initialize evaluation step.
        
        Args:
            forward_fn: Function to compute forward pass (e.g., model.eval(); model(batch))
        """
        self.forward_fn = forward_fn
    
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
        """Execute evaluation step.
        
        Args:
            train_state: Current training state
            batch: Input batch data
            
        Returns:
            Mapping with loss, result, and logs
        """
        loss = None
        result = None
        
        if self.forward_fn is not None:
            result = self.forward_fn(batch)
            loss = result  # Assume result is loss for simple cases
        
        logs = {
            "step": train_state.global_step,
            "epoch": train_state.epoch,
        }
        
        return {
            "loss": loss,
            "result": result,
            "logs": logs,
        }
