"""Training state and result types for MolNex."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


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
        return self["epoch"]

    @epoch.setter
    def epoch(self, value: int) -> None:
        self["epoch"] = value

    @property
    def global_step(self) -> int:
        return self["global_step"]

    @global_step.setter
    def global_step(self, value: int) -> None:
        self["global_step"] = value

    @property
    def stage(self) -> Stage:
        return self["stage"]

    @stage.setter
    def stage(self, value: Stage) -> None:
        self["stage"] = value

    @property
    def steps_since_last_eval(self) -> int:
        return self["steps_since_last_eval"]

    @steps_since_last_eval.setter
    def steps_since_last_eval(self, value: int) -> None:
        self["steps_since_last_eval"] = value

    @property
    def best_metric(self) -> float | None:
        """Get best tracked metric value (if set)."""
        return self.get("best_metric")

    @best_metric.setter
    def best_metric(self, value: float | None) -> None:
        """Set best tracked metric value."""
        self["best_metric"] = value


@dataclass
class StepResult:
    """Result from executing a training step.

    Attributes:
        loss: Optional loss value from the step
        result: Main result/output from the step
        logs: Additional logging information
    """

    loss: Any = None
    result: Any = None
    logs: Mapping[str, Any] = field(default_factory=dict)
