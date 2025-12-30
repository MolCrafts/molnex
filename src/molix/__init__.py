"""Molix: Unified modeling of molecular potentials and properties with physics-aware ML.

Molix is a standalone ML training system with structural protocol compatibility to MolExp.
"""

from molix.core.state import Stage, TrainState, StepResult
from molix.core.trainer import Trainer
from molix.steps.train_step import TrainStep
from molix.steps.eval_step import EvalStep
from molix.steps.test_step import TestStep
from molix.steps.predict_step import PredictStep

# Functional API (PyTorch-style)
from molix import F

__all__ = [
    "Stage",
    "TrainState",
    "StepResult",
    "Trainer",
    "TrainStep",
    "EvalStep",
    "TestStep",
    "PredictStep",
    "F",  # Functional API
]
