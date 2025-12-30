"""Test step implementation."""

from typing import Any, Callable, Mapping, Optional

from molix.core.state import TrainState


class TestStep:
    """Test step - executes one test iteration.
    
    This step:
    - Represents one semantic test step
    - Is executed by the Trainer during testing
    - Is structurally compatible with MolExp's Op protocol
    
    Attributes:
        task_id: Operation identifier
        forward_fn: Forward pass function (model in test mode)
    """
    
    task_id: str = "test_step"
    
    def __init__(self, forward_fn: Optional[Callable[[Any], Any]] = None):
        """Initialize test step.
        
        Args:
            forward_fn: Function to compute forward pass
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
        """Execute test step.
        
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
            loss = result
        
        logs = {
            "step": train_state.global_step,
            "epoch": train_state.epoch,
        }
        
        return {
            "loss": loss,
            "result": result,
            "logs": logs,
        }
