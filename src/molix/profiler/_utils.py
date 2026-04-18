"""Shared utilities for the molix profiler suite."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------


class Timer:
    """Wall-clock context manager using ``time.perf_counter``.

    Attributes:
        elapsed: Elapsed time in seconds (set after ``__exit__``).

    Example:
        ```python
        with Timer() as t:
            do_work()
        print(t.elapsed * 1000, "ms")
        ```
    """

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed = time.perf_counter() - self._start


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------


def reset_peak_memory() -> None:
    """Reset CUDA peak memory tracker (no-op on CPU)."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb() -> float:
    """Return CUDA peak memory since last reset, in MB.

    Returns:
        Peak allocated CUDA memory in MB, or ``0.0`` on CPU.
    """
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    return 0.0


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass
class TimingStat:
    """Timing statistics over a set of measurements.

    All values in milliseconds.
    """

    mean_ms: float
    std_ms: float
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float

    @classmethod
    def from_list(cls, values_ms: list[float]) -> "TimingStat":
        """Compute statistics from a list of timings (in ms).

        Args:
            values_ms: List of timing measurements in milliseconds.

        Returns:
            ``TimingStat`` with computed statistics.
        """
        arr = np.asarray(values_ms, dtype=np.float64)
        return cls(
            mean_ms=float(arr.mean()),
            std_ms=float(arr.std()),
            p50_ms=float(np.percentile(arr, 50)),
            p95_ms=float(np.percentile(arr, 95)),
            min_ms=float(arr.min()),
            max_ms=float(arr.max()),
        )


@dataclass
class ValueStat:
    """Statistics over scalar values (counts, sizes, etc.)."""

    mean: float
    std: float
    p50: float
    p95: float

    @classmethod
    def from_list(cls, values: list[float | int]) -> "ValueStat":
        """Compute statistics from a list of values.

        Args:
            values: List of scalar measurements.

        Returns:
            ``ValueStat`` with computed statistics.
        """
        arr = np.asarray(values, dtype=np.float64)
        return cls(
            mean=float(arr.mean()),
            std=float(arr.std()),
            p50=float(np.percentile(arr, 50)),
            p95=float(np.percentile(arr, 95)),
        )


# ---------------------------------------------------------------------------
# ASCII table formatter
# ---------------------------------------------------------------------------


def _fmt_table(rows: list[dict], cols: list[str], col_width: int = 12) -> str:
    """Render a fixed-width ASCII table.

    Args:
        rows: List of dicts mapping column name to cell value.
        cols: Ordered list of column names to display.
        col_width: Minimum column width.

    Returns:
        Multi-line string with aligned columns.
    """
    # Compute column widths
    widths = {c: max(col_width, len(c)) for c in cols}
    for row in rows:
        for c in cols:
            widths[c] = max(widths[c], len(str(row.get(c, ""))))

    sep = "  "
    header = sep.join(c.ljust(widths[c]) for c in cols)
    rule = sep.join("-" * widths[c] for c in cols)
    lines = [header, rule]
    for row in rows:
        lines.append(sep.join(str(row.get(c, "")).ljust(widths[c]) for c in cols))
    return "\n".join(lines)
