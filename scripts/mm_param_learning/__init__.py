"""MM parameter-learning validation workspace scaffolding (milestone 1).

Soft-depends on :mod:`molexp` for Workspace / Project / Experiment / Run and
workflow stubs. No hard dependency is added to molnex ``pyproject.toml``.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_WORKSPACE_ROOT",
    "EXPERIMENT_SLUGS",
    "MOLHUB_COORDINATES",
    "PROGRAM",
    "PROJECT_SLUG",
    "VALIDATION_STAGES",
    "MmParamLearningWorkspace",
    "MmParamWorkflows",
]

from scripts.mm_param_learning.constants import (
    DEFAULT_WORKSPACE_ROOT,
    EXPERIMENT_SLUGS,
    MOLHUB_COORDINATES,
    PROGRAM,
    PROJECT_SLUG,
    VALIDATION_STAGES,
)
from scripts.mm_param_learning.materialize_workspace import MmParamLearningWorkspace
from scripts.mm_param_learning.workflows import MmParamWorkflows
