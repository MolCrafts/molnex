"""Training-run IO for MolNex.

Surfaces:

* **MolRec metrics (JSONL)** — :class:`MetricsWriter` / :func:`read_metrics`
  implement the molrec append binding under ``metrics/metrics.jsonl``.
  This is the **interoperable** training-curve path (molexp / molplot).
  Provisional in molnex; layout is stable, implementation may move to molpy.
* **Run package scaffold** — :func:`ensure_run_package` writes minimal
  ``meta/`` + ``status/`` for a Run-shaped record.
* **Internal journal (Zarr)** — :class:`JournalWriter` /
  :class:`JournalReader` for high-volume HPC journaling. Not the MolRec
  metrics interchange format.
* **Extended-XYZ codec** — :func:`parse_extxyz_frames` /
  :func:`write_extxyz_frames` over :class:`ExtxyzFrame`. The in-tree extxyz
  reader/writer (numpy only, no ASE) that MD trajectories are written with and
  that the numerical regression fixtures are read with.
"""

from __future__ import annotations

from molix.io.extxyz import ExtxyzFrame, parse_extxyz_frames, write_extxyz_frames
from molix.io.metrics import (
    METRICS_DIRNAME,
    METRICS_FILENAME,
    METRICS_INDEX_FILENAME,
    MetricReadResult,
    MetricsWriter,
    read_metrics,
    rebuild_metrics_index,
)
from molix.io.reader import JournalReader
from molix.io.run_record import ensure_run_package, write_meta, write_status
from molix.io.writer import JournalWriter

__all__ = [
    "ExtxyzFrame",
    "JournalReader",
    "JournalWriter",
    "METRICS_DIRNAME",
    "METRICS_FILENAME",
    "METRICS_INDEX_FILENAME",
    "MetricReadResult",
    "MetricsWriter",
    "ensure_run_package",
    "parse_extxyz_frames",
    "read_metrics",
    "rebuild_metrics_index",
    "write_extxyz_frames",
    "write_meta",
    "write_status",
]
