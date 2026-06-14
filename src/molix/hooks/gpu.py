"""GPU memory + utilization scalar hooks."""

from __future__ import annotations

from collections.abc import Sequence

from molix.core.hook import ScalarHook
from molix.core.state import Path


class GPUMemoryHook(ScalarHook):
    """Record CUDA memory usage (GiB) on every training batch.

    Available metrics (pick any subset via ``metrics=``):

    ============  ==========================================================
    name          source
    ============  ==========================================================
    ``alloc``     ``torch.cuda.memory_allocated`` → ``gpu/alloc_gib``
    ``resv``      ``torch.cuda.memory_reserved``  → ``gpu/resv_gib``
    ``peak``      ``torch.cuda.max_memory_allocated`` → ``gpu/peak_gib``;
                  ``reset_peak_memory_stats`` is called after reading so the
                  value always reflects the *window* between consecutive
                  reads — most useful for locating OOM hotspots.
    ============  ==========================================================

    Args:
        metrics: Names from ``{"alloc", "resv", "peak"}``. Empty / unknown
            names raise.

    No-op when ``torch.cuda.is_available()`` is ``False``; the requested
    keys are still written (as ``0.0``) so headers stay aligned.
    """

    _AVAILABLE: dict[str, str] = {
        "alloc": "alloc_gib",
        "resv": "resv_gib",
        "peak": "peak_gib",
    }

    _GB = 1024**3

    def __init__(self, metrics: Sequence[str] = ("alloc", "resv", "peak")) -> None:
        metrics = tuple(metrics)
        if not metrics:
            raise ValueError("GPUMemoryHook needs at least one metric.")
        unknown = [m for m in metrics if m not in self._AVAILABLE]
        if unknown:
            raise ValueError(
                f"GPUMemoryHook: unknown metric(s) {unknown}. Available: {sorted(self._AVAILABLE)}."
            )
        self.metrics: tuple[str, ...] = metrics

    @property
    def scalar_keys(self) -> tuple[Path, ...]:
        """The ``("gpu", <name>)`` state keys this hook writes, one per metric."""
        return tuple(("gpu", self._AVAILABLE[m]) for m in self.metrics)

    def on_train_start(self, trainer, state):
        """Reset CUDA peak-memory stats when the ``peak`` metric is tracked."""
        import torch

        if "peak" in self.metrics and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_batch_end(self, trainer, state, batch, outputs):
        """Record the selected memory metrics (GiB) into ``state["gpu"]``.

        Writes ``0.0`` for every metric when CUDA is unavailable. When
        ``peak`` is tracked, the peak counter is reset after reading so the
        value reflects the most recent step only.
        """
        import torch

        gpu = state["gpu"]
        cuda = torch.cuda.is_available()
        for m in self.metrics:
            key = self._AVAILABLE[m]
            if not cuda:
                gpu[key] = 0.0
                continue
            if m == "alloc":
                gpu[key] = torch.cuda.memory_allocated() / self._GB
            elif m == "resv":
                gpu[key] = torch.cuda.memory_reserved() / self._GB
            elif m == "peak":
                gpu[key] = torch.cuda.max_memory_allocated() / self._GB
        if cuda and "peak" in self.metrics:
            torch.cuda.reset_peak_memory_stats()


class GPUUtilsHook(ScalarHook):
    """Record GPU SM / memory-bandwidth utilization (%) via NVML per batch.

    Available metrics (pick any subset via ``metrics=``):

    ==============  ========================================================
    name            source
    ==============  ========================================================
    ``util``        SM utilization (NVML ``rates.gpu``)    → ``gpu/util_pct``
    ``mem_util``    Memory-bandwidth utilization
                    (NVML ``rates.memory``)               → ``gpu/mem_util_pct``
    ==============  ========================================================

    Backed by NVIDIA's official ``nvidia-ml-py`` PyPI package. Raises at
    ``on_train_start`` if CUDA or ``nvidia-ml-py`` is unavailable.

    Args:
        metrics: Names from ``{"util", "mem_util"}``.
    """

    _AVAILABLE: dict[str, str] = {
        "util": "util_pct",
        "mem_util": "mem_util_pct",
    }

    def __init__(self, metrics: Sequence[str] = ("util", "mem_util")) -> None:
        metrics = tuple(metrics)
        if not metrics:
            raise ValueError("GPUUtilsHook needs at least one metric.")
        unknown = [m for m in metrics if m not in self._AVAILABLE]
        if unknown:
            raise ValueError(
                f"GPUUtilsHook: unknown metric(s) {unknown}. Available: {sorted(self._AVAILABLE)}."
            )
        self.metrics: tuple[str, ...] = metrics

    @property
    def scalar_keys(self) -> tuple[Path, ...]:
        """The ``("gpu", <name>)`` state keys this hook writes, one per metric."""
        return tuple(("gpu", self._AVAILABLE[m]) for m in self.metrics)

    def on_train_start(self, trainer, state):
        """Initialize NVML and bind the current CUDA device handle.

        Raises:
            RuntimeError: If CUDA is unavailable.
            ImportError: If the ``nvidia-ml-py`` package is not installed.
        """
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("GPUUtilsHook requires CUDA but torch.cuda.is_available() is False.")
        try:
            import pynvml as nvml
        except ImportError as exc:
            raise ImportError(
                "GPUUtilsHook requires the official `nvidia-ml-py` package "
                "(`pip install nvidia-ml-py`)."
            ) from exc

        nvml.nvmlInit()
        idx = torch.cuda.current_device()
        self._nvml = nvml
        self._handle = nvml.nvmlDeviceGetHandleByIndex(idx)

    def on_train_batch_end(self, trainer, state, batch, outputs):
        """Record SM / memory-bandwidth utilization (%) into ``state["gpu"]``."""
        rates = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
        gpu = state["gpu"]
        for m in self.metrics:
            key = self._AVAILABLE[m]
            if m == "util":
                gpu[key] = float(rates.gpu)
            elif m == "mem_util":
                gpu[key] = float(rates.memory)

    def on_train_end(self, trainer, state):
        """Shut down NVML and release the device handle (best-effort)."""
        nvml = getattr(self, "_nvml", None)
        if nvml is not None:
            try:
                nvml.nvmlShutdown()
            except Exception:
                pass
        self._nvml = None
        self._handle = None
