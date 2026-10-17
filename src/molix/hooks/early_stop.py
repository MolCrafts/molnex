"""Early-stop hook — end training on divergence, on plateau, or on a time budget."""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from molix.core.hook import BaseHook

logger = logging.getLogger("molix.hooks.early_stop")


class EarlyStop(BaseHook):
    """Abort training when a stop condition is met.

    Args:
        if_nan: Abort when ``state["train"]["loss"]`` becomes non-finite
            (NaN or ±inf). On detection, saves ``<out_dir>/nan_checkpoint.pt``
            and raises ``RuntimeError`` so the Trainer unwinds cleanly. The
            caller can catch it and translate to a distinct exit code.
        param_check_every_n_steps: Also scan *every* model parameter for
            non-finiteness, but only every N steps. ``0`` (default) disables
            the parameter scan entirely — it launches one reduction kernel
            **per parameter tensor** every step, which throttles a
            launch-bound training loop; the loss check catches the same
            divergence one step later in practice. Set e.g. ``50`` to keep a
            periodic parameter audit at negligible cost.
        patience: End training when *monitor* has not improved for this many
            consecutive evaluations. ``None`` (default) never stops on a plateau
            — which is what left a run with ``epochs: 1_000_000`` and no stopping
            rule at all. Unlike the NaN conditions this is not a failure: the
            loop is asked to finish through :meth:`~molix.core.Trainer.stop`, so
            the epoch's eval and checkpoint still happen.
        monitor: Path into the state to watch, as ``"eval/loss"`` or
            ``("eval", "loss")``. Defaults to the checkpoint hook's own
            ``best_metric_name``, so the metric that selects the checkpoint is
            the metric that ends the run — a plateau declared on one quantity
            while the checkpoint tracks another is how a run stops on a number
            nobody kept.
        min_delta: How much *monitor* must fall to count as an improvement, in
            the metric's own unit. ``0.0`` (default) counts any decrease.
        max_seconds: End training after this much wall clock. ``None`` (default)
            is no budget. A run bounded only by its Slurm wall is killed
            mid-epoch, which loses the epoch and can lose the checkpoint the next
            curriculum stage needs; asked to stop, it exits the batch loop at
            once and still writes one.
        regression_factor: Stop when *monitor* stays above this multiple of its
            best for ``regression_patience`` consecutive evaluations. ``None``
            (default) never stops on a spike. One evaluation past the line does
            not count — a single reading the next one undoes (56 then 1.27 on a
            Hessian eval) is noise; two in a row is a run leaving its basin
            (511 then 844, and the next reading 51918).

    Example::

        EarlyStop(if_nan=True, model=model, out_dir=run, patience=10,
                  max_seconds=20 * 3600)
    """

    def __init__(
        self,
        *,
        if_nan: bool = False,
        model: nn.Module | None = None,
        out_dir: str | Path | None = None,
        param_check_every_n_steps: int = 0,
        patience: int | None = None,
        monitor: str | tuple[str, ...] | None = None,
        min_delta: float = 0.0,
        max_seconds: float | None = None,
        regression_factor: float | None = None,
        regression_patience: int = 2,
    ) -> None:
        if if_nan and model is None:
            raise ValueError("EarlyStop(if_nan=True) requires `model=` for the NaN checkpoint.")
        if param_check_every_n_steps < 0:
            raise ValueError("param_check_every_n_steps must be >= 0.")
        if patience is not None and patience < 1:
            raise ValueError(f"patience must be >= 1 evaluations, got {patience}")
        if min_delta < 0.0:
            raise ValueError(f"min_delta must be >= 0, got {min_delta}")
        if max_seconds is not None and max_seconds <= 0.0:
            raise ValueError(f"max_seconds must be > 0, got {max_seconds}")
        if regression_factor is not None and not (regression_factor > 1.0):
            raise ValueError(f"regression_factor must be > 1, got {regression_factor}")
        if regression_patience < 1:
            raise ValueError(f"regression_patience must be >= 1, got {regression_patience}")
        self._if_nan = if_nan
        self._model = model
        self._out_dir = Path(out_dir) if out_dir is not None else None
        self._param_check_every = param_check_every_n_steps
        self._batch_count = 0
        self._patience = patience
        self._monitor = monitor
        self._min_delta = float(min_delta)
        self._max_seconds = max_seconds
        self._best: float | None = None
        self._stale = 0
        self._started: float | None = None
        self._regression_factor = None if regression_factor is None else float(regression_factor)
        self._regression_patience = int(regression_patience)
        self._regressed = 0

    def on_train_start(self, trainer, state) -> None:
        """Start the clock. Nothing else — the budget is wall time, not steps."""
        self._started = time.perf_counter()

    def _monitor_path(self, trainer) -> tuple[str, ...]:
        """Which metric to watch: the one given, else the checkpoint's own."""
        monitor = self._monitor
        if monitor is None:
            checkpoint = getattr(trainer, "_checkpoint", None)
            monitor = getattr(checkpoint, "best_metric_name", None) or ("eval", "loss")
        return tuple(monitor.split("/")) if isinstance(monitor, str) else tuple(monitor)

    def on_eval_step_complete(self, trainer, state) -> None:
        """Count evaluations without improvement, and stop at *patience*.

        Reads the metric the same way the checkpoint hook does, at the same point
        in the phase, so the two cannot disagree about whether this eval improved.
        A missing metric is not a plateau — it is counted as nothing at all, since
        a stage whose eval has not reported yet has told us nothing.
        """
        if self._patience is None and self._regression_factor is None:
            return
        path = self._monitor_path(trainer)
        value = state.get(path)
        if value is None:
            return
        current = float(value)
        if not math.isfinite(current):
            return
        if self._best is None or current < self._best - self._min_delta:
            self._best, self._stale, self._regressed = current, 0, 0
            return
        if self._regression_factor is not None and current > self._regression_factor * self._best:
            self._regressed += 1
            if self._regressed >= self._regression_patience:
                trainer.stop(
                    f"{'/'.join(path)} is {current:.6g}, more than "
                    f"{self._regression_factor:g}x its best {self._best:.6g} for "
                    f"{self._regressed} evaluations"
                )
                return
        else:
            self._regressed = 0
        if self._patience is None:
            return
        self._stale += 1
        if self._stale >= self._patience:
            trainer.stop(
                f"{'/'.join(path)} has not improved by more than {self._min_delta} "
                f"in {self._stale} evaluations (best {self._best})"
            )

    def on_train_batch_end(self, trainer, state, batch, outputs) -> None:
        """Abort training if a non-finite loss (or, optionally, parameter) is seen.

        No-op unless ``if_nan=True``. Always checks ``state["train"]["loss"]``;
        scans all parameters only on the ``param_check_every_n_steps`` cadence
        (off by default — the per-step scan is a launch-bound bottleneck).
        """
        if self._max_seconds is not None and self._started is not None:
            elapsed = time.perf_counter() - self._started
            if elapsed >= self._max_seconds:
                trainer.stop(f"time budget reached: {elapsed:.0f}s of {self._max_seconds:.0f}s")
        if not self._if_nan:
            return
        self._batch_count += 1

        loss = state["train"].get("loss")
        if loss is not None and not math.isfinite(float(loss)):
            self._abort(state, reason=f"non-finite loss={float(loss)}")
            return

        if self._param_check_every and self._batch_count % self._param_check_every == 0:
            assert self._model is not None
            for name, p in self._model.named_parameters():
                if not torch.isfinite(p).all():
                    self._abort(state, reason=f"non-finite parameter {name}")
                    return

    def _abort(self, state, *, reason: str) -> None:
        step = int(state.get("global_step", 0))
        logger.error("NaN detected at step=%d — %s", step, reason)
        if self._out_dir is not None and self._model is not None:
            torch.save(self._model.state_dict(), self._out_dir / "nan_checkpoint.pt")
        raise RuntimeError(f"NaN detected — early stopping ({reason})")
