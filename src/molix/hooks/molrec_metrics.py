"""Persist TrainState scalars as a MolRec metrics JSONL stream.

:class:`MolRecMetricsHook` is the canonical training-metrics sink for
interoperability with molexp / molplot: it writes the molrec JSONL binding
under a record root (see :mod:`molix.io.metrics`) and maintains a minimal
Run-shaped package (``meta`` + ``status``).

Compared with :class:`~molix.hooks.tensorboard.TensorBoardHook` (external
tfevents) and :class:`~molix.hooks.journal.JournalHook` (internal Zarr
journal), this hook is the one other MolCrafts tools are expected to read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from molix import logger as _logger_mod
from molix.core.hook import BaseHook
from molix.hooks._utils import _as_scalar
from molix.io.metrics import MetricsWriter
from molix.io.run_record import ensure_run_package, write_status

logger = _logger_mod.getLogger(__name__)


class MolRecMetricsHook(BaseHook):
    """Mirror TrainState scalar namespaces into MolRec ``metrics/metrics.jsonl``.

    On each train-batch end (throttled by ``every_n_steps``) scans
    ``train/*``, ``performance/*``, and ``gpu/*``; on each eval completion
    scans ``eval/*``. Values that coerce to a finite scalar are appended;
    non-scalars are skipped.

    Args:
        record_root: MolRec package root (directory). Created if missing.
        every_n_steps: Train-phase logging cadence.
        start_step: Suppress train/performance/gpu writes before this step.
            Eval scalars are still written from the beginning.
        creator_name: ``meta.creator.name`` for the Run package scaffold.
        creator_version: optional ``meta.creator.version``.
        flush_every_n_appends: Persist ``index.json`` every N successful
            appends (and always on train end). ``0`` means only on train end.
        status_every_n_steps: Train-phase cadence of the ``status/`` heartbeat.
            Each heartbeat is a read-modify-write of ``status.json``, so it is
            throttled independently of the metric appends; eval completion and
            train start/end always write one regardless.
    """

    TRAIN_NAMESPACES: tuple[str, ...] = ("train", "performance", "gpu")
    EVAL_NAMESPACES: tuple[str, ...] = ("eval",)

    def __init__(
        self,
        record_root: str | Path,
        every_n_steps: int = 1,
        *,
        start_step: int = 0,
        creator_name: str = "molnex",
        creator_version: str | None = None,
        flush_every_n_appends: int = 50,
        status_every_n_steps: int = 1000,
    ) -> None:
        if every_n_steps <= 0:
            raise ValueError("every_n_steps must be positive")
        if status_every_n_steps <= 0:
            raise ValueError("status_every_n_steps must be positive")
        if start_step < 0:
            raise ValueError("start_step must be non-negative")
        if flush_every_n_appends < 0:
            raise ValueError("flush_every_n_appends must be non-negative")

        self.record_root = Path(record_root)
        self.every_n_steps = every_n_steps
        self.start_step = start_step
        self.creator_name = creator_name
        self.creator_version = creator_version
        self.flush_every_n_appends = flush_every_n_appends
        self.status_every_n_steps = status_every_n_steps

        self._writer: MetricsWriter | None = None
        self._appends_since_flush = 0

    def on_train_start(self, trainer: Any, state: Any) -> None:
        """Scaffold Run package and open the metrics writer."""
        ensure_run_package(
            self.record_root,
            creator_name=self.creator_name,
            creator_version=self.creator_version,
            state="running",
            stage="train",
        )
        self._writer = MetricsWriter(self.record_root)
        self._appends_since_flush = 0
        logger.info(f"MolRecMetricsHook: writing metrics under {self.record_root}")

    def on_train_batch_end(self, trainer: Any, state: Any, batch: Any, outputs: Any) -> None:
        """Append train/performance/gpu scalars at the configured cadence."""
        if self._writer is None:
            return
        if state.global_step < self.start_step:
            return
        if state.global_step % self.every_n_steps != 0:
            return
        self._log_namespaces(state, self.TRAIN_NAMESPACES, stage="train")

    def on_eval_step_complete(self, trainer: Any, state: Any) -> None:
        """Append eval/* scalars once per eval phase."""
        if self._writer is None:
            return
        self._log_namespaces(state, self.EVAL_NAMESPACES, stage="eval")

    def on_train_end(self, trainer: Any, state: Any) -> None:
        """Flush index and mark status succeeded."""
        if self._writer is not None:
            self._writer.flush()
        write_status(
            self.record_root,
            state="succeeded",
            stage="train",
            global_step=state.global_step,
        )
        self._writer = None

    def _log_namespaces(self, state: Any, namespaces: tuple[str, ...], *, stage: str) -> None:
        assert self._writer is not None
        step = int(state.global_step)
        # The heartbeat re-reads and rewrites status.json; on a long run that is
        # one filesystem round trip per logged step on the training thread, so
        # the train phase gets its own stride. Eval completions are rare enough
        # to publish every time.
        if stage != "train" or step % self.status_every_n_steps == 0:
            write_status(
                self.record_root,
                state="running",
                stage=stage,
                global_step=step,
            )
        for ns in namespaces:
            try:
                ns_dict = state[ns]
            except Exception:
                continue
            if not isinstance(ns_dict, dict):
                continue
            for key, value in ns_dict.items():
                scalar = _as_scalar(value)
                if scalar is None:
                    continue
                self._writer.scalar(f"{ns}/{key}", scalar, step=step)
                self._appends_since_flush += 1
                if (
                    self.flush_every_n_appends
                    and self._appends_since_flush >= self.flush_every_n_appends
                ):
                    self._writer.flush()
                    self._appends_since_flush = 0


__all__ = ["MolRecMetricsHook"]
