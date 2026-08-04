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
"""

from __future__ import annotations

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
    "METRICS_DIRNAME",
    "METRICS_FILENAME",
    "METRICS_INDEX_FILENAME",
    "JournalReader",
    "JournalWriter",
    "MetricReadResult",
    "MetricsWriter",
    "ensure_run_package",
    "read_metrics",
    "rebuild_metrics_index",
    "write_meta",
    "write_status",
]
