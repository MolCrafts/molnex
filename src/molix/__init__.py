"""Molix: Unified modeling of molecular potentials and properties with physics-aware ML.

Molix is the canonical base package for shared NN utilities, ops, and training.
"""

import sys
from pathlib import Path

import torch

_lib_loaded = False


def _load_ops_library() -> None:
    """Load the C++ ops library (formerly loaded by molnex)."""
    global _lib_loaded
    if _lib_loaded:
        return
    if sys.platform == "win32":
        lib_name = "molnex_opLib.pyd"
    elif sys.platform == "darwin":
        lib_name = "libmolnex_opLib.dylib"
    else:
        lib_name = "libmolnex_opLib.so"

    # Library resides under molix/op in this monorepo
    candidate = Path(__file__).resolve().parents[0] / "op" / lib_name
    if candidate.exists():
        torch.ops.load_library(str(candidate))
        _lib_loaded = True


_load_ops_library()

from molix import logger
from molix.compile import maybe_compile
from molix.config import config
from molix.core.checkpoint import Checkpoint, CheckpointBackend, TorchSaveBackend
from molix.core.hooks import ProfilerHook
from molix.core.losses import MAELoss, MSELoss, WeightedLoss
from molix.core.state import Stage, StepResult, TrainState
from molix.core.trainer import Trainer

__all__ = [
    "Stage",
    "TrainState",
    "Checkpoint",
    "StepResult",
    "Trainer",
    "CheckpointBackend",
    "TorchSaveBackend",
    "MSELoss",
    "MAELoss",
    "WeightedLoss",
    "config",
    "logger",
    "maybe_compile",
    "ProfilerHook",
]
