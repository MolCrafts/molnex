"""Molix: Unified modeling of molecular potentials and properties with physics-aware ML.

Molix is the canonical base package for shared NN utilities, ops, and training.
"""

import sys
from pathlib import Path

import torch

_lib_loaded = False


def _load_ops_library() -> None:
    """Load the C++ ops library. Raises ImportError with build instructions if missing."""
    global _lib_loaded
    if _lib_loaded:
        return
    if sys.platform == "win32":
        lib_name = "molnex_opLib.pyd"
    elif sys.platform == "darwin":
        lib_name = "libmolnex_opLib.dylib"
    else:
        lib_name = "libmolnex_opLib.so"

    candidate = Path(__file__).resolve().parents[0] / "op" / lib_name
    if not candidate.exists():
        op_src = candidate.parents[0]
        raise ImportError(
            f"molix native op library not found at {candidate}.\n"
            f"Build it with:\n"
            f"  cmake -S {op_src} -B {op_src}/build -DMOLNEX_OP_ENABLE_CUDA=ON\n"
            f"  cmake --build {op_src}/build -j\n"
            f"(drop -DMOLNEX_OP_ENABLE_CUDA=ON for CPU-only builds.)"
        )
    torch.ops.load_library(str(candidate))
    _lib_loaded = True


def ensure_op_registered(op_name: str) -> None:
    """Ensure a named ``torch.ops.molix`` op is registered in this process.

    This is a defensive helper for editable installs, worker subprocesses,
    and import paths that reach ``molix.F`` modules before callers have
    imported the top-level ``molix`` package explicitly.
    """
    _load_ops_library()
    if hasattr(torch.ops.molix, op_name):
        return

    candidate = Path(__file__).resolve().parents[0] / "op"
    raise RuntimeError(
        f"molix native op 'molix::{op_name}' is not registered after loading "
        f"the native library from {candidate}. "
        "The shared library is likely stale relative to the Python sources. "
        "Rebuild it with:\n"
        f"  cmake -S {candidate} -B {candidate}/build -DMOLNEX_OP_ENABLE_CUDA=ON\n"
        f"  cmake --build {candidate}/build -j\n"
        "(drop -DMOLNEX_OP_ENABLE_CUDA=ON for CPU-only builds.)"
    )


_load_ops_library()

from molix import logger, logging
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
    "logging",
    "maybe_compile",
    "ProfilerHook",
]
