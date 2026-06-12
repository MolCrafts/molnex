"""Append-only persistence sink for the training-event journal.

:class:`JournalHook` is the producer side of
:class:`molix.io.JournalWriter`: it mirrors the
:class:`~molix.core.state.TrainState` namespaces (``train/``,
``performance/``, ``gpu/`` on the train phase; ``eval/`` on the eval
phase) into the writer at a configured cadence, and emits optional
``hparams`` / weight + gradient histogram records at the
appropriate lifecycle moments.

Async mode (``async_writes=True``)
----------------------------------
The default (synchronous) path calls ``float(tensor)`` on every mirrored
scalar, which forces a CPU↔GPU sync and serialises the launch-bound
training pipeline. With ``async_writes=True`` and CUDA available, each GPU
scalar is instead copied into a *pinned* CPU buffer with
``non_blocking=True`` and tagged with a :class:`torch.cuda.Event`; a single
background thread (:class:`_AsyncJournalWriter`) drains a queue, writing each
record to the :class:`~molix.io.JournalWriter` only once its event reports
ready (``event.query()``), and re-checking — without advancing past it — when
it is not. The training loop therefore never blocks on the GPU. The cost is
that a record may land in the store one or two steps late; per-key step
ordering is still preserved because CUDA events recorded on one stream
complete in submission order and the drain loop never reorders.

When CUDA is unavailable the async machinery is not started and the hook
falls back to the synchronous path, exactly as :mod:`molix.hooks.gpu`
hooks no-op without CUDA.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import TYPE_CHECKING, Any, Callable

from molix.core.hook import BaseHook
from molix.hooks._utils import _as_scalar

if TYPE_CHECKING:
    from molix.core.state import TrainState
    from molix.core.trainer import Trainer
    from molix.io import JournalWriter

# Sentinel pushed onto the async queue by ``shutdown`` to terminate the drain
# loop after every preceding record has been written.
_SHUTDOWN = object()


class _AsyncJournalWriter:
    """Single background thread that owns all writer access in async mode.

    Records are submitted as ``(event, factory)`` pairs. ``factory`` is a
    zero-arg callable returning the ``JournalWriter.append`` kwargs; deferring
    it lets a pinned-buffer scalar materialise ``float(buf)`` only once its
    event is ready (a cheap host read, no sync). ``event`` is ``None`` for
    records that are ready immediately (hparams JSON, histograms, CPU scalars).

    The drain loop processes items in strict FIFO order and **never advances
    past a not-ready item** — it holds the item and re-checks after a short
    poll. Because CUDA events recorded on a single stream become ready in
    submission order, the head of the queue is always the next event to
    complete, so this both preserves per-key step ordering and avoids
    busy-spinning on later work.

    The :class:`~molix.io.JournalWriter` is touched *only* from this thread,
    which is the single-writer contract the writer relies on (it has no
    internal lock). The owning :class:`JournalHook` calls :meth:`shutdown`
    (which drains and joins) before it closes the writer from the main thread.

    Args:
        store: The :class:`~molix.io.JournalWriter` to append to.
        poll_interval_s: Seconds to wait before re-checking a not-ready event.
    """

    def __init__(self, store: "JournalWriter", *, poll_interval_s: float = 1e-4) -> None:
        if poll_interval_s <= 0:
            raise ValueError(f"poll_interval_s must be positive, got {poll_interval_s}")
        self._store = store
        self._poll_interval_s = poll_interval_s
        self._queue: queue.Queue[Any] = queue.Queue()
        self._thread = threading.Thread(
            target=self._drain_loop, name="molix-journal-writer", daemon=True
        )
        self._started = False

    def _ensure_started(self) -> None:
        """Start the background thread on first submit (main-thread only)."""
        if not self._started:
            self._thread.start()
            self._started = True

    def submit(self, event: Any | None, factory: Callable[[], dict[str, Any]]) -> None:
        """Enqueue one record; starts the drain thread on first call.

        Args:
            event: A CUDA completion event, or ``None`` if the record is ready
                to write immediately.
            factory: Zero-arg callable returning ``JournalWriter.append`` kwargs.
        """
        self._ensure_started()
        self._queue.put((event, factory))

    def shutdown(self) -> None:
        """Drain every pending record, then join the background thread.

        A no-op when the thread was never started (no records submitted).
        """
        if not self._started:
            return
        self._queue.put(_SHUTDOWN)
        self._thread.join()

    def _drain_loop(self) -> None:
        pending: tuple[Any | None, Callable[[], dict[str, Any]]] | None = None
        while True:
            item = pending if pending is not None else self._queue.get()
            pending = None
            if item is _SHUTDOWN:
                return
            event, factory = item
            if event is not None and not event.query():
                # Earliest-recorded event not ready yet: hold it at the front
                # (do not pull later items) and re-check after a short poll.
                pending = item
                time.sleep(self._poll_interval_s)
                continue
            self._store.append(**factory())


class JournalHook(BaseHook):
    """Append-only persistence sink for the training-event journal.

    Mirrors the same :class:`~molix.core.state.TrainState` namespaces as
    :class:`molix.hooks.tensorboard.TensorBoardHook` (``train/``,
    ``performance/``, ``gpu/`` on the train phase; ``eval/`` on the eval
    phase) but writes records to a :class:`~molix.io.JournalWriter`
    backend instead of TensorBoard event files.

    Args:
        every_n_steps: Mirror frequency for ``train``/``performance``/
            ``gpu`` namespace scalars during the train phase. Must be
            positive.
        store: A :class:`~molix.io.JournalWriter` (kept named
            ``store`` for source-compat with the legacy ``Journal``
            constructor; the writer is the only first-party concrete
            backend).
        async_writes: When ``True`` and CUDA is available, mirror GPU
            scalars without blocking the training hot path — copy each to a
            pinned CPU buffer (``non_blocking=True``), tag it with a
            :class:`torch.cuda.Event`, and let a single background thread
            write it to ``store`` once the event is ready. Records may lag a
            step or two; per-key step ordering is preserved. When CUDA is
            unavailable this falls back to the synchronous path. Default
            ``False`` (synchronous; every scalar forces a CPU↔GPU sync).
        poll_interval_s: Background-thread re-check interval for a not-ready
            event, in seconds. Only used when ``async_writes`` is active.
            Default ``1e-4``.
        log_hparams: When ``True`` and ``hparams`` is provided, emit
            one ``type="json", key="hparams", step=0`` record at
            ``on_train_start``.
        log_histograms: When ``True``, emit ``type="histogram"``
            records for ``Weights/<name>`` and ``Gradients/<name>`` on
            every ``histogram_freq``-th epoch end. Bins are computed
            via :func:`numpy.histogram` (per-record).
        hparams: Hyperparameter mapping; required when
            ``log_hparams=True``.
        histogram_freq: Histogram emission cadence in epochs. Default 1.
        histogram_bins: Number of histogram bins. Default 64.
    """

    TRAIN_NAMESPACES: tuple[str, ...] = ("train", "performance", "gpu")
    EVAL_NAMESPACES: tuple[str, ...] = ("eval",)

    def __init__(
        self,
        every_n_steps: int,
        store: "JournalWriter",
        *,
        async_writes: bool = False,
        poll_interval_s: float = 1e-4,
        log_hparams: bool = False,
        log_histograms: bool = False,
        hparams: dict[str, Any] | None = None,
        histogram_freq: int = 1,
        histogram_bins: int = 64,
    ) -> None:
        if every_n_steps <= 0:
            raise ValueError(f"every_n_steps must be positive, got {every_n_steps}")
        if histogram_freq <= 0:
            raise ValueError(f"histogram_freq must be positive, got {histogram_freq}")

        self._store = store
        self._every_n_steps = every_n_steps
        self._log_hparams = log_hparams
        self._log_histograms = log_histograms
        self._hparams = hparams or {}
        self._histogram_freq = histogram_freq
        self._histogram_bins = histogram_bins

        # Async writer is created only when requested *and* CUDA is present;
        # otherwise every record routes through the synchronous path. Keeping
        # the import lazy/guarded mirrors the CUDA no-op fallback in
        # ``molix.hooks.gpu`` and keeps a CPU-only environment dependency-free.
        self._async: _AsyncJournalWriter | None = None
        if async_writes:
            try:
                import torch

                if torch.cuda.is_available():
                    self._async = _AsyncJournalWriter(store, poll_interval_s=poll_interval_s)
            except Exception:
                self._async = None

    @staticmethod
    def _now_ns() -> int:
        import time

        return time.time_ns()

    def on_train_start(self, trainer: "Trainer | None", state: "TrainState") -> None:
        """Emit the ``hparams`` JSON record if enabled."""
        if self._log_hparams and self._hparams:
            self._emit(
                type="json",
                key="hparams",
                step=0,
                wall_time_ns=self._now_ns(),
                value=dict(self._hparams),
            )

    def on_train_batch_end(
        self,
        trainer: "Trainer | None",
        state: "TrainState",
        batch: Any,
        outputs: Any,
    ) -> None:
        """Mirror ``train/`` ``performance/`` ``gpu/`` scalars at the configured cadence."""
        global_step = int(state.get("global_step", 0))
        if global_step % self._every_n_steps != 0:
            return
        self._mirror_namespaces(state, self.TRAIN_NAMESPACES, global_step)

    def on_eval_step_complete(self, trainer: "Trainer | None", state: "TrainState") -> None:
        """Mirror ``eval/`` scalars whenever an eval phase completes."""
        global_step = int(state.get("global_step", 0))
        self._mirror_namespaces(state, self.EVAL_NAMESPACES, global_step)

    def on_epoch_end(self, trainer: "Trainer | None", state: "TrainState") -> None:
        """Emit weight / gradient histogram records when enabled."""
        if not self._log_histograms or trainer is None:
            return
        epoch = int(state.get("epoch", 0))
        if (epoch + 1) % self._histogram_freq != 0:
            return
        self._emit_histograms(trainer, state)

    def on_train_end(self, trainer: "Trainer | None", state: "TrainState") -> None:
        """Drain the async queue (if any), then close the underlying writer.

        Draining before close honours the writer's use-after-close contract
        (``JournalWriter.append`` raises ``RuntimeError`` once closed): the
        background thread is joined first, so the main thread is the sole
        accessor by the time ``close()`` runs.
        """
        if self._async is not None:
            self._async.shutdown()
        self._store.close()

    def _mirror_namespaces(
        self, state: "TrainState", namespaces: tuple[str, ...], global_step: int
    ) -> None:
        wall = self._now_ns()
        for ns in namespaces:
            sub = state[ns]
            if not isinstance(sub, dict):
                continue
            for k, value in sub.items():
                self._emit_scalar(f"{ns}/{k}", value, global_step, wall)

    def _emit_scalar(self, key: str, value: Any, step: int, wall_time_ns: int) -> None:
        """Mirror one scalar, async (pinned-buffer + event) when possible.

        A 0-d CUDA tensor is copied into a pinned CPU buffer with
        ``non_blocking=True`` and handed to the background writer with a
        completion event — no host sync on the training thread. Everything
        else (Python scalars, CPU tensors, non-scalars) takes the synchronous
        coercion path via :func:`_as_scalar`.
        """
        if self._async is not None:
            import torch

            if isinstance(value, torch.Tensor) and value.is_cuda and value.ndim == 0:
                buf = torch.empty((), dtype=value.dtype, device="cpu", pin_memory=True)
                buf.copy_(value.detach(), non_blocking=True)
                event = torch.cuda.Event()
                event.record()
                self._async.submit(
                    event,
                    lambda b=buf, k=key, s=step, w=wall_time_ns: {
                        "type": "scalar",
                        "key": k,
                        "step": s,
                        "wall_time_ns": w,
                        "value": float(b),
                    },
                )
                return

        scalar = _as_scalar(value)
        if scalar is None:
            return
        self._emit(
            type="scalar",
            key=key,
            step=step,
            wall_time_ns=wall_time_ns,
            value=float(scalar),
        )

    def _emit(self, **record: Any) -> None:
        """Write one ready record — through the async queue or synchronously.

        In async mode even ready records (no event) are routed through the
        background thread so the writer is touched by exactly one thread.
        """
        if self._async is not None:
            self._async.submit(None, lambda r=record: r)
        else:
            self._store.append(**record)

    def _emit_histograms(self, trainer: "Trainer", state: "TrainState") -> None:
        import numpy as np

        epoch = int(state.get("epoch", 0))
        wall = self._now_ns()
        model = getattr(trainer, "model", None)
        if model is None:
            return
        for name, param in model.named_parameters():
            data = param.detach().cpu().numpy().ravel()
            counts, bins = np.histogram(data, bins=self._histogram_bins)
            self._emit(
                type="histogram",
                key=f"Weights/{name}",
                step=epoch,
                wall_time_ns=wall,
                value={"bins": bins.tolist(), "counts": counts.tolist()},
            )
            if param.grad is not None:
                gdata = param.grad.detach().cpu().numpy().ravel()
                gcounts, gbins = np.histogram(gdata, bins=self._histogram_bins)
                self._emit(
                    type="histogram",
                    key=f"Gradients/{name}",
                    step=epoch,
                    wall_time_ns=wall,
                    value={"bins": gbins.tolist(), "counts": gcounts.tolist()},
                )
