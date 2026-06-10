"""Checkpoint persistence hook."""

from __future__ import annotations

from molix import logger as _logger_mod
from molix import logging as _logging_mod
from molix.core.hook import BaseHook
from molix.core.state import Path, resolve

logger = _logger_mod.getLogger(__name__)


class CheckpointHook(BaseHook):
    """Saves model checkpoints during training.

    Delegates to :meth:`trainer._checkpoint.state_dict` so that the saved
    payload always carries the *complete* set of resume-critical state
    (model, optimizer, lr_scheduler, AMP scaler, RNG states, counters,
    best-metric tracking).

    Trigger model — all step-based, one event per trigger, no implicit
    epoch-end firing:

    - ``save_every_n_steps`` (default 0 = off) fires on
      :meth:`on_train_batch_end` and writes ``step_<N>.pt``.
    - ``save_last`` fires on :meth:`on_eval_step_complete`, writing
      ``last.pt`` so the freshest evaluated state is always recoverable.
      The eval cadence is set by ``trainer.eval_every_n_steps``.
    - ``save_best`` fires on :meth:`on_eval_step_complete`, writing
      ``best.pt`` when the tracked metric improves.
    - A final ``last.pt`` is always written on :meth:`on_train_end`.

    Args:
        checkpoint_dir: Directory to save checkpoints into.
        save_every_n_steps: Write ``step_<N>.pt`` every N training steps.
            ``0`` disables periodic snapshots. Use a multiple of
            ``eval_every_n_steps`` so each snapshot lines up with an eval.
        save_last: Write ``last.pt`` at every eval completion and at the
            end of training (default ``True``). Cadence = eval cadence.
        save_best: When ``True``, watch ``best_metric_name`` at every eval
            completion and write ``best.pt`` on improvement.
        best_metric_name: Path into ``state`` to track for ``save_best``.
            Default ``("eval", "loss")``.
        best_metric_mode: ``"min"`` (smaller = better) or ``"max"``.
        register_artifacts: Register each written checkpoint via
            ``trainer.ctx.save_artifact`` (if present).
    """

    _MINUS_INF = float("-inf")
    _PLUS_INF = float("inf")

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        save_every_n_steps: int = 0,
        save_last: bool = True,
        save_best: bool = False,
        best_metric_name: Path = ("eval", "loss"),
        best_metric_mode: str = "min",
        register_artifacts: bool = False,
    ):
        import os

        import torch

        if best_metric_mode not in ("min", "max"):
            raise ValueError(f"best_metric_mode must be 'min' or 'max', got {best_metric_mode!r}")
        if save_every_n_steps < 0:
            raise ValueError(f"save_every_n_steps must be >= 0, got {save_every_n_steps}")

        self.os = os
        self.torch = torch

        self.checkpoint_dir = checkpoint_dir
        self.save_every_n_steps = save_every_n_steps
        self.save_last = save_last
        self.save_best = save_best
        self.best_metric_name = best_metric_name
        self.best_metric_mode = best_metric_mode
        self.register_artifacts = register_artifacts

        self._best_value: float | None = None
        self._last_announced: tuple[str, int] | None = None

    def on_train_start(self, trainer, state):
        """Create checkpoint directory and sync best-metric metadata.

        On resume, seed ``self._best_value`` from the restored checkpoint's
        ``best_metric`` so the first post-resume eval is compared against the
        true historical best — otherwise ``_best_value`` is ``None`` and that
        eval always counts as an "improvement", overwriting ``best.pt`` with a
        possibly-worse model.
        """
        self.os.makedirs(self.checkpoint_dir, exist_ok=True)
        ckpt = getattr(trainer, "_checkpoint", None)
        if ckpt is not None:
            ckpt.best_metric_name = self.best_metric_name
            if self._best_value is None and ckpt.best_metric is not None:
                self._best_value = float(ckpt.best_metric)

    def on_train_batch_end(self, trainer, state, batch, outputs):
        """Write step-indexed periodic snapshot.

        ``on_train_batch_end`` fires *before* the trainer increments
        ``global_step``; the logical step that this batch represents is
        therefore ``global_step + 1`` (matches :class:`Log`'s convention).
        """
        if self.save_every_n_steps <= 0:
            return
        step = int(state.get("global_step", 0)) + 1
        if step % self.save_every_n_steps != 0:
            return
        self._save_checkpoint(trainer, state, f"step_{step}.pt", step=step)

    def on_eval_step_complete(self, trainer, state):
        """Refresh ``last.pt`` / ``best.pt`` against the just-finished eval."""
        if self.save_best:
            self._maybe_save_best(trainer, state)
        if self.save_last:
            self._save_checkpoint(trainer, state, "last.pt")

    def on_train_end(self, trainer, state):
        """Always write a final ``last.pt`` so the run is resumable.

        If an eval-driven ``last.pt`` was already written at this same
        ``global_step``, the file's counters (``epoch`` in particular) are
        stale by one because ``state.increment_epoch()`` ran in between.
        Silently rewrite the file in that case so the on-disk payload
        matches the post-training state, but skip the announce so the
        eval+train_end pair still reads as a single event.
        """
        if not self.save_last:
            return
        step = int(state.get("global_step", 0))
        if self._last_announced == ("last.pt", step):
            self._rewrite_checkpoint(trainer, state, "last.pt")
            return
        self._save_checkpoint(trainer, state, "last.pt")

    def _is_improvement(self, candidate: float) -> bool:
        if self._best_value is None:
            return True
        if self.best_metric_mode == "min":
            return candidate < self._best_value
        return candidate > self._best_value

    def _maybe_save_best(self, trainer, state):
        raw = resolve(state, self.best_metric_name)
        if raw is None:
            return
        value = float(raw)
        if not self._is_improvement(value):
            return
        self._best_value = value
        ckpt = getattr(trainer, "_checkpoint", None)
        if ckpt is not None:
            ckpt.best_metric = value
        self._save_checkpoint(trainer, state, "best.pt")

    def _build_state_dict(self, trainer, state) -> dict:
        """Produce the complete resume-ready state dict."""
        ckpt = getattr(trainer, "_checkpoint", None)
        if ckpt is not None:
            ckpt.epoch = int(state.epoch)
            ckpt.global_step = int(state.global_step)
            if self.save_best and self._best_value is not None:
                ckpt.best_metric = self._best_value
            return ckpt.state_dict()
        sd: dict = {
            "epoch": int(state.epoch),
            "global_step": int(state.global_step),
        }
        if trainer.model is not None:
            sd["model_state_dict"] = trainer.model.state_dict()
        if trainer.optimizer is not None:
            sd["optimizer_state_dict"] = trainer.optimizer.state_dict()
        return sd

    def _rewrite_checkpoint(self, trainer, state, filename):
        """Overwrite ``filename`` with the current state — no announce, no dedupe.

        Used by :meth:`on_train_end` to refresh a same-step ``last.pt``
        whose counters drifted after ``state.increment_epoch()``.
        """
        filepath = self.os.path.join(self.checkpoint_dir, filename)
        checkpoint = self._build_state_dict(trainer, state)
        self.torch.save(checkpoint, filepath)
        logger.info(f"Refreshed checkpoint at {filepath}")

    def _save_checkpoint(self, trainer, state, filename, *, step: int | None = None):
        """Save checkpoint to disk; dedupe redundant writes + announces.

        Two triggers can fire at the same step (e.g. ``on_eval_step_complete``
        and ``on_train_end`` at end-of-training). The marker check skips
        both the redundant ``torch.save`` and the announce. ``step``
        overrides the value read from ``state`` — needed by
        :meth:`on_train_batch_end` which fires before the trainer
        increments ``global_step``.
        """
        if step is None:
            step = int(state.get("global_step", 0))
        marker = (filename, step)
        if self._last_announced == marker:
            return
        self._last_announced = marker

        filepath = self.os.path.join(self.checkpoint_dir, filename)
        checkpoint = self._build_state_dict(trainer, state)
        self.torch.save(checkpoint, filepath)

        logger.info(f"Saved checkpoint to {filepath}")
        message = f"ckpt: {filename} @ step={step}"
        evt = _logging_mod.events_logger()
        width = _logging_mod.get_table_width()
        if _logging_mod.has_effective_handlers(evt):
            evt.info(
                message,
                kind="announce",
                table_width=width,
                category="checkpoint",
                filename=filename,
                step=step,
                filepath=filepath,
            )
        else:
            prefix = f"─── {message} "
            pad = max(3, width - len(prefix))
            print(prefix + "─" * pad, flush=True)

        if self.register_artifacts and hasattr(trainer, "ctx"):
            ctx = trainer.ctx
            if ctx:
                from pathlib import Path as _PathLib

                checkpoint_path = _PathLib(filepath)
                if checkpoint_path.exists():
                    ctx.save_artifact(
                        name=f"checkpoint_{filename}",
                        src=checkpoint_path,
                    )
                    logger.info(f"Registered checkpoint as artifact: {checkpoint_path}")
