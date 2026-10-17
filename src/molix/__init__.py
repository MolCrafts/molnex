"""Molix: Unified modeling of molecular potentials and properties with physics-aware ML.

Molix is the canonical base package for shared NN utilities, ops, and training.
"""

import os
import platform
import sys
from pathlib import Path

import torch

_lib_loaded = False

#: Rebuild instructions shown when the native library is missing or stale.
_BUILD_HINT = (
    "Build it from csrc/ (out of source) and put the build directory on "
    "MOLNEX_OP_LIBRARY_PATH:\n"
    "  cmake -S csrc -B build/csrc-$(uname -m) -DMOLNEX_OP_ENABLE_CUDA=ON "
    "-DTorch_DIR=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')/Torch\n"
    "  cmake --build build/csrc-$(uname -m) -j   # -> build/csrc-<arch>/molix/\n"
    "(scripts/env/activate.sh sets MOLNEX_OP_LIBRARY_PATH; drop "
    "-DMOLNEX_OP_ENABLE_CUDA=ON for a CPU-only build.)"
)


def _op_library_dirs() -> list[Path]:
    """Where the native op library is looked for, in order.

    Each directory of ``MOLNEX_OP_LIBRARY_PATH`` (``os.pathsep``-separated —
    the variable molnex's C++ runtime reads too), then this package's ``lib/``,
    where a wheel installs it.
    """
    listed = os.environ.get("MOLNEX_OP_LIBRARY_PATH", "")
    return [Path(d) for d in listed.split(os.pathsep) if d] + [
        Path(__file__).resolve().parent / "lib"
    ]


def _op_lib_path() -> Path | None:
    """The native op library for the running machine, or ``None`` if not built.

    ``csrc/molix/CMakeLists.txt`` tags the file with the target architecture —
    ``libmolnex_opLib.<machine>.so`` — so builds for several machines can share
    a directory.
    """
    if sys.platform == "win32":
        prefix, ext = "", "pyd"
    elif sys.platform == "darwin":
        prefix, ext = "lib", "dylib"
    else:
        prefix, ext = "lib", "so"
    name = f"{prefix}molnex_opLib.{platform.machine()}.{ext}"
    for directory in _op_library_dirs():
        if (directory / name).is_file():
            return directory / name
    return None


def _load_ops_library() -> None:
    """Load the native op library when present and ABI-compatible.

    Pure-Python call paths (e.g. ClassicalIMM energy training) do not need the
    native neighbour/MD ops. A missing or ABI-mismatched library (common when
    switching between CPU and CUDA torch wheels) therefore **warns** rather than
    hard-failing import — :func:`ensure_op_registered` still raises for callers
    that actually need a named op.
    """
    global _lib_loaded
    if _lib_loaded:
        return

    candidate = _op_lib_path()
    if candidate is None:
        import warnings

        searched = ", ".join(str(d) for d in _op_library_dirs())
        warnings.warn(
            f"molix native op library not found for machine '{platform.machine()}' "
            f"(searched {searched}). Pure-Python paths still work. " + _BUILD_HINT,
            stacklevel=2,
        )
        return
    try:
        torch.ops.load_library(str(candidate))
    except OSError as exc:
        import warnings

        warnings.warn(
            f"molix native op library at {candidate} failed to load ({exc}). "
            "Likely built against a different torch ABI — pure-Python paths still "
            "work; rebuild the op against this interpreter's torch to enable "
            "molix C++/CUDA ops.",
            stacklevel=2,
        )
        return
    _lib_loaded = True


def ensure_op_registered(op_name: str) -> None:
    """Ensure a named ``torch.ops.molix`` op is registered in this process.

    This is a defensive helper for worker subprocesses and import paths that
    reach ``molix.F`` modules before callers have imported the top-level
    ``molix`` package explicitly.
    """
    _load_ops_library()
    if hasattr(torch.ops.molix, op_name):
        return
    raise RuntimeError(
        f"molix native op 'molix::{op_name}' is not registered after loading "
        f"the native library ({_op_lib_path()}). It is missing or stale relative "
        "to the Python sources. " + _BUILD_HINT
    )


_load_ops_library()

from molix import logger, logging
from molix.autograd import Derivative
from molix.compile import Compiler, ExplainSummary, register_inductor_fallbacks
from molix.core.checkpoint import Checkpoint, CheckpointBackend, TorchSaveBackend
from molix.core.losses import MAELoss, MSELoss, WeightedLoss
from molix.core.state import Stage, StepResult, TrainState
from molix.core.trainer import Trainer
from molix.hooks import JournalHook, ProfilerHook

__all__ = [
    "Checkpoint",
    "CheckpointBackend",
    "Compiler",
    "Derivative",
    "ExplainSummary",
    "JournalHook",
    "MAELoss",
    "MSELoss",
    "ProfilerHook",
    "Stage",
    "StepResult",
    "TorchSaveBackend",
    "TrainState",
    "Trainer",
    "WeightedLoss",
    "logger",
    "logging",
    "register_inductor_fallbacks",
]
