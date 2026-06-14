"""Concrete hook implementations driven by :class:`molix.core.trainer.Trainer`.

The contract layer (:class:`Hook` Protocol, :class:`BaseHook`,
:class:`ScalarHook`) lives in :mod:`molix.core.hook`; this package
holds every concrete implementation. Naming convention: every
concrete class carries the ``Hook`` suffix
(``CheckpointHook``, ``JournalHook``, ``GradClipHook``, …) so the
boundary against the contract layer (whose three classes are
suffix-free) stays visually unambiguous.

Dependency direction: ``hooks/ → io/ + core/`` — never the reverse.
"""

from __future__ import annotations

from molix.hooks.checkpoint import CheckpointHook
from molix.hooks.early_stop import EarlyStop
from molix.hooks.gpu import GPUMemoryHook, GPUUtilsHook
from molix.hooks.journal import JournalHook
from molix.hooks.profiler import ProfilerHook
from molix.hooks.progress import Log, ProgressBarHook
from molix.hooks.scalar import MetricsHook, StepSpeedHook
from molix.hooks.tensorboard import TensorBoardHook
from molix.hooks.training import ActivationCheckpointingHook, GradClipHook

__all__ = [
    "ActivationCheckpointingHook",
    "CheckpointHook",
    "EarlyStop",
    "GPUMemoryHook",
    "GPUUtilsHook",
    "GradClipHook",
    "JournalHook",
    "Log",
    "MetricsHook",
    "ProfilerHook",
    "ProgressBarHook",
    "StepSpeedHook",
    "TensorBoardHook",
]
