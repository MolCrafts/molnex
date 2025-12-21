"""Training state and result types for MolNex."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional


class Stage(str, Enum):
    """Training stage enumeration.
    
    Defines the different phases of ML training.
    """
    
    TRAIN = "train"
    """Training phase - model learning from training data."""
    
    EVAL = "eval"
    """Evaluation/validation phase - assessing model performance."""
    
    TEST = "test"
    """Testing phase - final model evaluation on held-out data."""
    
    PREDICT = "predict"
    """Prediction/inference phase - generating predictions on new data."""


@dataclass
class TrainState:
    """Training state container.
    
    Tracks runtime counters and training progress.
    May be extended to carry model/optimizer references.
    
    Attributes:
        epoch: Current epoch number (0-indexed)
        global_step: Global step counter across all epochs
        stage: Current training stage
    """
    
    epoch: int = 0
    global_step: int = 0
    stage: Stage = Stage.TRAIN
    
    def increment_step(self) -> None:
        """Increment the global step counter."""
        self.global_step += 1
    
    def increment_epoch(self) -> None:
        """Increment the epoch counter."""
        self.epoch += 1
    
    def set_stage(self, stage: Stage) -> None:
        """Set the current training stage.
        
        Args:
            stage: New training stage
        """
        self.stage = stage


@dataclass
class StepResult:
    """Result from executing a training step.
    
    Attributes:
        loss: Optional loss value from the step
        result: Main result/output from the step
        logs: Additional logging information
    """
    
    loss: Optional[Any] = None
    result: Any = None
    logs: Mapping[str, Any] = field(default_factory=dict)
