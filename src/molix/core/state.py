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


class TrainState(dict):
    """Training state container based on dict.

    Tracks runtime counters, training progress,
    and all metrics/indicators produced during training using a unified
    namespace structure.

    Tracks:
        - epoch: Current epoch number (0-indexed)
        - global_step: Global step counter across all epochs
        - stage: Current training stage
        - steps_since_last_eval: Counter for steps since last step-based eval
        - train/*: Training metrics (loss, MAE, RMSE, etc.)
        - eval/*: Evaluation metrics (loss, MAE, RMSE, etc.)
        - performance/*: Performance metrics (step_per_second, etc.)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize core fields
        if "epoch" not in self:
            self["epoch"] = 0
        if "global_step" not in self:
            self["global_step"] = 0
        if "stage" not in self:
            self["stage"] = Stage.TRAIN
        if "steps_since_last_eval" not in self:
            self["steps_since_last_eval"] = 0

    def increment_step(self) -> None:
        """Increment the global step counter."""
        self["global_step"] = self["global_step"] + 1

    def increment_epoch(self) -> None:
        """Increment the epoch counter."""
        self["epoch"] = self["epoch"] + 1

    def set_stage(self, stage: Stage) -> None:
        """Set the current training stage.

        Args:
            stage: New training stage
        """
        self["stage"] = stage

    @property
    def epoch(self) -> int:
        """Get epoch counter."""
        return int(self.get("epoch", 0))

    @epoch.setter
    def epoch(self, value: int) -> None:
        """Set epoch counter."""
        self["epoch"] = value

    @property
    def global_step(self) -> int:
        """Get global step counter."""
        return int(self.get("global_step", 0))

    @global_step.setter
    def global_step(self, value: int) -> None:
        """Set global step counter."""
        self["global_step"] = value

    @property
    def stage(self) -> Stage:
        """Get training stage."""
        return self.get("stage", Stage.TRAIN)

    @stage.setter
    def stage(self, value: Stage) -> None:
        """Set training stage."""
        self["stage"] = value

    @property
    def steps_since_last_eval(self) -> int:
        """Get steps since last eval."""
        return int(self.get("steps_since_last_eval", 0))

    @steps_since_last_eval.setter
    def steps_since_last_eval(self, value: int) -> None:
        """Set steps since last eval."""
        self["steps_since_last_eval"] = value


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
