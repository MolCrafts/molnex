"""Prediction step implementation."""

from typing import Any, Callable, Mapping, Optional

from molix.core.state import TrainState


class PredictStep:
    """Prediction step - executes one inference iteration.
    
    This step:
    - Represents one semantic prediction step
    - Is executed by the Trainer during inference
    - Is structurally compatible with MolExp's Op protocol
    
    Attributes:
        task_id: Operation identifier
        forward_fn: Forward pass function (model in inference mode)
    """
    
    task_id: str = "predict_step"
    
    def __init__(self, forward_fn: Optional[Callable[[Any], Any]] = None):
        """Initialize prediction step.
        
        Args:
            forward_fn: Function to compute predictions
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
        """Execute prediction step.
        
        Args:
            train_state: Current training state
            batch: Input batch data
            
        Returns:
            Mapping with predictions and logs
        """
        result = None
        
        if self.forward_fn is not None:
            result = self.forward_fn(batch)
        
        logs = {
            "step": train_state.global_step,
            "epoch": train_state.epoch,
        }
        
        return {
            "loss": None,  # No loss during prediction
            "result": result,
            "logs": logs,
        }
