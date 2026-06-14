"""Environment self-check for MolNex (the ``molnex check`` command).

Probes the runtime for everything a MolNex training / inference session
needs and reports a per-item verdict:

* Python and PyTorch versions against the project minimums.
* CUDA runtime availability, device count, and compute capability.
* The native C++ ops library (``torch.ops.molix.*``) — loaded and registered.
* ``cuequivariance`` (required for equivariant models).
* GPU telemetry via ``nvidia-ml-py`` (``pynvml``).
* Core runtime dependencies (tensordict, molpy, mollog, molcfg, …).

The checks never mutate global state and each returns an immutable
:class:`CheckResult`; rendering lives in :mod:`molix.cli`.
"""

from __future__ import annotations

import importlib.metadata as _md
import importlib.util as _util
import platform
import re
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Final

# Project minimums — kept in sync with ``pyproject.toml``.
MIN_PYTHON: Final[tuple[int, int]] = (3, 10)
MIN_TORCH: Final[tuple[int, int]] = (2, 10)

# Required runtime deps probed by :meth:`Doctor._check_core_deps`.
# (import name, distribution name shown on failure).
_CORE_DEPS: Final[tuple[tuple[str, str], ...]] = (
    ("tensordict", "tensordict"),
    ("numpy", "numpy"),
    ("pydantic", "pydantic"),
    ("zarr", "zarr"),
    ("molpy", "molcrafts-molpy"),
    ("mollog", "molcrafts-mollog"),
    ("molcfg", "molcrafts-molcfg"),
)

# Canonical native op used to confirm the C++ library loaded + registered.
_PROBE_OP: Final[str] = "get_neighbor_pairs"


class Status(Enum):
    """Outcome of a single check, ordered from best to worst."""

    OK = "ok"
    WARN = "warn"
    FAIL = "fail"

    @property
    def symbol(self) -> str:
        return {"ok": "✓", "warn": "⚠", "fail": "✗"}[self.value]

    @property
    def color(self) -> str:
        return {"ok": "green", "warn": "yellow", "fail": "red"}[self.value]


# Severity ranking for computing the worst outcome / exit code.
_SEVERITY: Final[dict[Status, int]] = {Status.OK: 0, Status.WARN: 1, Status.FAIL: 2}


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Immutable outcome of one environment check."""

    name: str
    status: Status
    detail: str
    hint: str | None = None


def _parse_version(text: str) -> tuple[int, ...]:
    """Extract the leading dotted-integer release from a version string.

    ``"2.12.0+cpu"`` -> ``(2, 12, 0)``. Returns ``()`` if unparseable.
    """
    match = re.match(r"\s*(\d+(?:\.\d+)*)", text or "")
    if not match:
        return ()
    return tuple(int(part) for part in match.group(1).split("."))


def _module_version(module: str, dist: str | None = None) -> str | None:
    """Best-effort installed version for an import name, or ``None``.

    ``dist`` is the distribution name when it differs from the import name
    (e.g. import ``molpy`` ships as ``molcrafts-molpy``).
    """
    candidates = (dist, module, module.replace("_", "-"))
    for name in candidates:
        if name is None:
            continue
        try:
            return _md.version(name)
        except _md.PackageNotFoundError:
            continue
    try:  # fall back to the imported module's ``__version__``
        mod = __import__(module)
        version = getattr(mod, "__version__", None)
        return str(version) if version is not None else None
    except Exception:
        return None


class Doctor:
    """Runs the full suite of MolNex environment checks.

    Each ``_check_*`` method is pure: it inspects the interpreter / installed
    packages and returns a :class:`CheckResult` without side effects.
    """

    def run(self) -> tuple[CheckResult, ...]:
        """Run every check in display order."""
        return (
            self._check_python(),
            self._check_torch(),
            self._check_cuda(),
            self._check_native_ops(),
            self._check_cuequivariance(),
            self._check_gpu_telemetry(),
            self._check_core_deps(),
        )

    @staticmethod
    def worst_status(results: tuple[CheckResult, ...]) -> Status:
        """The most severe status across ``results`` (``OK`` if empty)."""
        return max(
            (r.status for r in results),
            key=lambda s: _SEVERITY[s],
            default=Status.OK,
        )

    # -- individual checks -------------------------------------------------

    def _check_python(self) -> CheckResult:
        current = sys.version_info[:2]
        detail = platform.python_version()
        if current >= MIN_PYTHON:
            return CheckResult("Python", Status.OK, detail)
        return CheckResult(
            "Python",
            Status.FAIL,
            detail,
            hint=f"MolNex requires Python >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}.",
        )

    def _check_torch(self) -> CheckResult:
        if _util.find_spec("torch") is None:
            return CheckResult(
                "PyTorch", Status.FAIL, "not installed", hint="pip install 'torch>=2.10'"
            )
        import torch

        version = torch.__version__
        if _parse_version(version)[:2] >= MIN_TORCH:
            return CheckResult("PyTorch", Status.OK, version)
        return CheckResult(
            "PyTorch",
            Status.FAIL,
            version,
            hint=f"MolNex requires torch >= {MIN_TORCH[0]}.{MIN_TORCH[1]}.",
        )

    def _check_cuda(self) -> CheckResult:
        if _util.find_spec("torch") is None:
            return CheckResult("CUDA", Status.FAIL, "PyTorch missing")
        import torch

        built_cuda = torch.version.cuda
        if built_cuda is None:
            return CheckResult(
                "CUDA",
                Status.WARN,
                "CPU-only PyTorch build",
                hint="Install a CUDA build of PyTorch for GPU acceleration.",
            )
        if not torch.cuda.is_available():
            return CheckResult(
                "CUDA",
                Status.WARN,
                f"built for CUDA {built_cuda}, but no GPU detected at runtime",
                hint="Check the driver / `nvidia-smi` and that a GPU is visible.",
            )
        count = torch.cuda.device_count()
        names = []
        for i in range(count):
            major, minor = torch.cuda.get_device_capability(i)
            names.append(f"{torch.cuda.get_device_name(i)} (sm_{major}{minor})")
        detail = f"CUDA {built_cuda} · {count} device(s): " + ", ".join(names)
        return CheckResult("CUDA", Status.OK, detail)

    def _check_native_ops(self) -> CheckResult:
        try:
            import torch

            import molix

            molix.ensure_op_registered(_PROBE_OP)
            registered = hasattr(torch.ops.molix, _PROBE_OP)
        except ImportError as exc:
            return CheckResult(
                "Native ops",
                Status.FAIL,
                "C++ library not built",
                hint=str(exc).splitlines()[0] if str(exc) else None,
            )
        except Exception as exc:  # stale lib / registration mismatch
            return CheckResult(
                "Native ops",
                Status.FAIL,
                f"{type(exc).__name__}: {exc}",
                hint="Rebuild the native op library (see molix/op/CMakeLists.txt).",
            )
        if registered:
            return CheckResult("Native ops", Status.OK, f"torch.ops.molix.{_PROBE_OP} registered")
        return CheckResult(
            "Native ops",
            Status.FAIL,
            "library loaded but op not registered",
            hint="The shared library is likely stale relative to the Python sources.",
        )

    def _check_cuequivariance(self) -> CheckResult:
        required = ("cuequivariance", "cuequivariance_torch")
        missing = [m for m in required if _util.find_spec(m) is None]
        if missing:
            return CheckResult(
                "cuEquivariance",
                Status.FAIL,
                f"missing: {', '.join(missing)}",
                hint="pip install cuequivariance cuequivariance-torch",
            )
        version = _module_version("cuequivariance")
        return CheckResult("cuEquivariance", Status.OK, version or "installed")

    def _check_gpu_telemetry(self) -> CheckResult:
        if _util.find_spec("pynvml") is None:
            return CheckResult(
                "GPU telemetry",
                Status.WARN,
                "nvidia-ml-py (pynvml) not installed",
                hint="pip install nvidia-ml-py for GPU memory telemetry.",
            )
        try:
            import torch

            cuda_available = torch.cuda.is_available()
        except Exception:
            cuda_available = False
        if not cuda_available:
            return CheckResult("GPU telemetry", Status.OK, "pynvml installed (no GPU to query)")
        try:
            import pynvml

            pynvml.nvmlInit()
            try:
                count = pynvml.nvmlDeviceGetCount()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gib = mem.total / (1024**3)
                detail = f"{count} GPU(s) · device 0: {gib:.1f} GiB total"
            finally:
                pynvml.nvmlShutdown()
        except Exception as exc:
            return CheckResult(
                "GPU telemetry",
                Status.WARN,
                f"pynvml query failed: {type(exc).__name__}: {exc}",
            )
        return CheckResult("GPU telemetry", Status.OK, detail)

    def _check_core_deps(self) -> CheckResult:
        missing: list[str] = []
        present: list[str] = []
        for module, dist in _CORE_DEPS:
            if _util.find_spec(module) is None:
                missing.append(dist)
            else:
                version = _module_version(module, dist)
                present.append(f"{module} {version}" if version else module)
        if missing:
            return CheckResult(
                "Core deps",
                Status.FAIL,
                f"missing: {', '.join(missing)}",
                hint="pip install -e '.[dev]' to restore the full dependency set.",
            )
        return CheckResult("Core deps", Status.OK, ", ".join(present))
