"""Molix: Unified modeling of molecular potentials and properties with physics-aware ML.

Molix is the canonical base package for shared NN utilities, ops, and training.
"""

import platform
import sys
from pathlib import Path

import torch

_lib_loaded = False


def _op_lib_candidates() -> list[Path]:
    """Candidate op-library paths, most-specific first.

    The build (``op/CMakeLists.txt``) tags the artifact with the target
    architecture — ``libmolnex_opLib.<machine>.so`` (e.g. ``...x86_64.so``,
    ``...aarch64.so``) — so per-arch builds coexist in one in-source ``op/``
    directory. Prefer the file matching the running machine; fall back to the
    legacy un-tagged name so pre-tagging builds keep loading.
    """
    if sys.platform == "win32":
        prefix, ext = "", "pyd"
    elif sys.platform == "darwin":
        prefix, ext = "lib", "dylib"
    else:
        prefix, ext = "lib", "so"
    op_dir = Path(__file__).resolve().parents[0] / "op"
    arch = platform.machine()  # 'x86_64', 'aarch64', ... — matches CMAKE_SYSTEM_PROCESSOR
    return [
        op_dir / f"{prefix}molnex_opLib.{arch}.{ext}",  # arch-tagged (current builds)
        op_dir / f"{prefix}molnex_opLib.{ext}",  # legacy un-tagged (older builds)
    ]


def _load_ops_library() -> None:
    """Load the C++ ops library. Raises ImportError with build instructions if missing."""
    global _lib_loaded
    if _lib_loaded:
        return

    candidates = _op_lib_candidates()
    candidate = next((c for c in candidates if c.exists()), None)
    if candidate is None:
        op_src = candidates[0].parents[0]
        tried = "\n".join(f"  - {c}" for c in candidates)
        raise ImportError(
            f"molix native op library not found for machine '{platform.machine()}'. Tried:\n"
            f"{tried}\n"
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
from molix.compile import Compiler
from molix.config import config
from molix.core.checkpoint import Checkpoint, CheckpointBackend, TorchSaveBackend
from molix.core.losses import MAELoss, MSELoss, WeightedLoss
from molix.core.state import Stage, StepResult, TrainState
from molix.core.trainer import Trainer
from molix.export import Exporter
from molix.hooks import JournalHook, ProfilerHook

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
    "Exporter",
    "Compiler",
    "ProfilerHook",
    "JournalHook",
]
