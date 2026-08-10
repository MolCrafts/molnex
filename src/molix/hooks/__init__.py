"""Concrete hook implementations driven by :class:`molix.core.trainer.Trainer`.

The contract layer (:class:`~molix.core.hook.Hook` Protocol,
:class:`~molix.core.hook.BaseHook`, :class:`~molix.core.hook.ScalarHook`)
lives in the *singular* module :mod:`molix.core.hook`; this *plural*
package, :mod:`molix.hooks`, holds every concrete implementation. The
singular/plural module name is the boundary marker — class names are not,
since the ``Hook`` suffix appears on both sides (``BaseHook`` /
``ScalarHook`` in the contract layer; ``CheckpointHook`` / ``JournalHook`` /
``GradClipHook`` here) and some concrete hooks drop it entirely (``Log``,
``EarlyStop``).

Dependency direction: ``hooks/ → io/ + core/`` — never the reverse.
"""

from __future__ import annotations

from molix.hooks.checkpoint import CheckpointHook
from molix.hooks.early_stop import EarlyStop
from molix.hooks.gpu import GPUMemoryHook, GPUUtilsHook
from molix.hooks.journal import JournalHook
from molix.hooks.molrec_metrics import MolRecMetricsHook
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
    "MolRecMetricsHook",
    "ProfilerHook",
    "ProgressBarHook",
    "StepSpeedHook",
    "TensorBoardHook",
]
