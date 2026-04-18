"""Hook system for Molix Trainer.

This module provides an extensible hook system that allows users to inject
custom logic at various points in the training lifecycle.

Hooks execute in registration order by default. Use (hook, priority) tuples
to override execution order (lower priority = earlier execution).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

import torch.nn as nn

from molix import logger as _logger_mod

if TYPE_CHECKING:
    from molix.core.state import TrainState
    from molix.core.trainer import Trainer

logger = _logger_mod.getLogger(__name__)


class Hook(Protocol):
    """Protocol for training hooks.

    Hooks receive notifications at various points in the training lifecycle.
    All methods are optional - implement only the hooks you need.

    Hook execution order:
    - By default, hooks execute in registration order
    - Use (hook, priority) tuples to override execution order
    - Lower priority values execute earlier (default priority = 100)
    - Hooks with same priority execute in registration order
    - If a hook raises an exception, it is logged but training continues

    Example:
        ```python
        class MyHook:
            def on_epoch_end(self, trainer, state):
                print(f"Epoch {state.epoch} completed")

        # Registration order
        trainer = Trainer(hooks=[MyHook(), OtherHook()])

        # With priority
        trainer = Trainer(hooks=[(MyHook(), 10), OtherHook()])
        ```
    """

    def on_train_start(self, trainer: "Trainer", state: "TrainState") -> None:
        """Called once at the beginning of training.

        Args:
            trainer: The trainer instance
            state: Current training state (epoch=0, global_step=0)
        """
        ...

    def on_train_end(self, trainer: "Trainer", state: "TrainState") -> None:
        """Called once at the end of training.

        Args:
            trainer: The trainer instance
            state: Final training state
        """
        ...

    def on_epoch_start(self, trainer: "Trainer", state: "TrainState") -> None:
        """Called at the start of each epoch.

        Args:
            trainer: The trainer instance
            state: Current training state
        """
        ...

    def on_epoch_end(self, trainer: "Trainer", state: "TrainState") -> None:
        """Called at the end of each epoch (after validation).

        Args:
            trainer: The trainer instance
            state: Current training state
        """
        ...

    def on_train_batch_start(self, trainer: "Trainer", state: "TrainState", batch: Any) -> None:
        """Called before processing each training batch.

        Args:
            trainer: The trainer instance
            state: Current training state
            batch: The current batch data
        """
        ...

    def on_train_batch_end(
        self, trainer: "Trainer", state: "TrainState", batch: Any, outputs: Any
    ) -> None:
        """Called after processing each training batch.

        Args:
            trainer: The trainer instance
            state: Current training state
            batch: The current batch data
            outputs: Outputs from the training step (loss, predictions, etc.)
        """
        ...

    def on_eval_batch_start(self, trainer: "Trainer", state: "TrainState", batch: Any) -> None:
        """Called before processing each validation batch.

        Args:
            trainer: The trainer instance
            state: Current training state
            batch: The current batch data
        """
        ...

    def on_after_backward(self, trainer: "Trainer", state: "TrainState") -> None:
        """Called after backward pass, before optimizer step.

        Gradients are available and unscaled (if AMP is active, unscale
        has already been applied). Use this hook for gradient manipulation
        (clipping, logging, etc.).

        Args:
            trainer: The trainer instance
            state: Current training state
        """
        ...

    def on_eval_batch_end(
        self, trainer: "Trainer", state: "TrainState", batch: Any, outputs: Any
    ) -> None:
        """Called after processing each validation batch.

        Args:
            trainer: The trainer instance
            state: Current training state
            batch: The current batch data
            outputs: Outputs from the evaluation step (loss, metrics, etc.)
        """
        ...

    def on_eval_step_complete(self, trainer: "Trainer", state: "TrainState") -> None:
        """Called after step-based evaluation completes (not on epoch-end eval).

        This hook is only triggered when eval runs due to the eval_every_n_steps
        parameter being reached. Epoch-end evals do not trigger this hook.

        Args:
            trainer: The trainer instance
            state: Current training state (steps_since_last_eval reset to 0)
        """
        ...


class BaseHook:
    """Base hook with no-op implementations.

    Inherit from this class and override only the methods you need.
    This provides better IDE support than implementing the Protocol directly.

    Example:
        ```python
        class MyHook(BaseHook):
            def on_epoch_end(self, trainer, state):
                print(f"Epoch {state.epoch} completed")
        ```
    """

    def on_train_start(self, trainer: "Trainer", state: "TrainState") -> None:
        """Called once at the beginning of training."""
        pass

    def on_train_end(self, trainer: "Trainer", state: "TrainState") -> None:
        """Called once at the end of training."""
        pass

    def on_epoch_start(self, trainer: "Trainer", state: "TrainState") -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(self, trainer: "Trainer", state: "TrainState") -> None:
        """Called at the end of each epoch (after validation)."""
        pass

    def on_train_batch_start(self, trainer: "Trainer", state: "TrainState", batch: Any) -> None:
        """Called before processing each training batch."""
        pass

    def on_train_batch_end(
        self, trainer: "Trainer", state: "TrainState", batch: Any, outputs: Any
    ) -> None:
        """Called after processing each training batch."""
        pass

    def on_after_backward(self, trainer: "Trainer", state: "TrainState") -> None:
        """Called after backward pass, before optimizer step."""
        pass

    def on_eval_batch_start(self, trainer: "Trainer", state: "TrainState", batch: Any) -> None:
        """Called before processing each validation batch."""
        pass

    def on_eval_batch_end(
        self, trainer: "Trainer", state: "TrainState", batch: Any, outputs: Any
    ) -> None:
        """Called after processing each validation batch."""
        pass

    def on_eval_step_complete(self, trainer: "Trainer", state: "TrainState") -> None:
        """Called after step-based evaluation completes (not on epoch-end eval)."""
        pass


class ScalarHook(BaseHook):
    """Hook that writes scalar values into ``state``.

    Subclasses advertise the state keys they populate via ``scalar_keys``.
    Container hooks such as :class:`Log` read this attribute to discover
    which keys to render, avoiding hard-coded column lists.

    For hooks whose keys depend on runtime configuration (e.g. metric
    names), override ``scalar_keys`` as a ``@property``.
    """

    scalar_keys: tuple[str, ...] = ()


# Built-in Hooks


class TensorBoardHook(BaseHook):
    """Logs whatever scalar metrics other hooks have written into ``state``.

    The hook does not require a key list.  On each train-batch end it scans
    ``state`` for keys under the ``train/*`` and ``performance/*`` namespaces;
    on each eval completion it scans ``eval/*``.  Any value that is a number
    or a 0-d tensor is written to TensorBoard under its state key; non-scalar
    entries are skipped.

    This means TensorBoard mirrors whatever the trainer/other hooks have
    decided to populate — no duplicated key declarations, no drift between
    the logger and the producers.

    Args:
        every_n_steps: Logging frequency for train-step scalars.
        log_dir: Directory to save TensorBoard event files.
        log_hparams: Log hyperparameters for the HParams dashboard.
        log_histograms: Log weight/gradient histograms each epoch.
        hparams: Hyperparameter dict (required when ``log_hparams=True``).
        histogram_freq: Log histograms every N epochs (default: 1).

    Example:
        ```python
        hooks = [
            MetricsHook(...),
            step_speed,
            gpu,
            Log(50, keys=[step_speed, gpu, metrics_hook]),
            TensorBoardHook(50, "./runs/exp1"),   # same (n, dir, ...) form
        ]
        ```
    """

    TRAIN_PREFIXES: tuple[str, ...] = ("train/", "performance/")
    EVAL_PREFIXES: tuple[str, ...] = ("eval/",)

    def __init__(
        self,
        every_n_steps: int,
        log_dir: str,
        *,
        log_hparams: bool = False,
        log_histograms: bool = False,
        hparams: dict | None = None,
        histogram_freq: int = 1,
    ):
        if every_n_steps <= 0:
            raise ValueError("every_n_steps must be positive")

        from torch.utils.tensorboard import SummaryWriter

        self.SummaryWriter = SummaryWriter
        self.log_dir = log_dir
        self.every_n_steps = every_n_steps
        self.log_hparams = log_hparams
        self.log_histograms = log_histograms
        self.hparams = hparams or {}
        self.histogram_freq = histogram_freq

        self.writer = None
        self._graph_logged = False

    def on_train_start(self, trainer, state):
        """Open the SummaryWriter."""
        self.writer = self.SummaryWriter(self.log_dir)
        if self.log_hparams and self.hparams:
            logger.info(f"TensorBoardHook: logging hyperparameters {self.hparams}")

    def on_train_batch_end(self, trainer, state, batch, outputs):
        """Log every ``train/*`` and ``performance/*`` scalar currently in state."""
        if state.global_step % self.every_n_steps != 0:
            return
        self._log_prefixed(state, self.TRAIN_PREFIXES)

    def on_eval_step_complete(self, trainer, state):
        """Log every ``eval/*`` scalar currently in state."""
        self._log_prefixed(state, self.EVAL_PREFIXES)

    def _log_prefixed(self, state, prefixes: tuple[str, ...]) -> None:
        for key, value in state.items():
            if not key.startswith(prefixes):
                continue
            scalar = _as_scalar(value)
            if scalar is None:
                continue
            self.writer.add_scalar(key, scalar, state.global_step)

    def on_epoch_end(self, trainer, state):
        """Log weight/gradient histograms."""
        if self.log_histograms and (state.epoch + 1) % self.histogram_freq == 0:
            self._log_histograms(trainer, state)

    def on_train_end(self, trainer, state):
        """Log final metrics with hyperparameters and close writer."""
        # Log hyperparameters with final metrics
        if self.log_hparams and self.hparams:
            final_metrics = self._extract_final_metrics(trainer)
            if final_metrics:
                self.writer.add_hparams(self.hparams, final_metrics)
                logger.info(f"Logged hyperparameters with final metrics: {final_metrics}")

        # Close writer
        if self.writer:
            self.writer.close()

    def _log_histograms(self, trainer, state):
        """Log weight and gradient histograms."""
        for name, param in trainer.model.named_parameters():
            # Log weights
            self.writer.add_histogram(f"Weights/{name}", param.data, state.epoch)

            # Log gradients
            if param.grad is not None:
                self.writer.add_histogram(f"Gradients/{name}", param.grad.data, state.epoch)

    def _extract_final_metrics(self, trainer):
        """Extract final metrics for hparams logging."""
        final_metrics = {}

        # Find MetricsHook and extract final metrics
        for hook in trainer.hooks:
            hook_obj = hook[0] if isinstance(hook, tuple) else hook
            if hook_obj.__class__.__name__ == "MetricsHook":
                for metric in hook_obj.metrics:
                    value = metric.compute()
                    metric_name = metric.__class__.__name__
                    final_metrics[f"final_{metric_name}"] = value

        return final_metrics


class CheckpointHook(BaseHook):
    """Saves model checkpoints during training.

    Delegates to :meth:`trainer._checkpoint.state_dict` so that the saved
    payload always carries the *complete* set of resume-critical state
    (model, optimizer, lr_scheduler, AMP scaler, RNG states, counters,
    best-metric tracking). That matches what
    :meth:`trainer._load_checkpoint` consumes on resume, so the save/load
    round-trip round-trips — no silent drift between what the hook writes
    and what the Trainer reads.

    Args:
        checkpoint_dir: Directory to save checkpoints into.
        save_every_n_epochs: Save an ``epoch_{N}.pt`` snapshot every N epochs.
            Use a large number (e.g. ``10_000``) to disable periodic saves
            and rely solely on ``save_last`` / ``save_best``.
        save_last: Write ``last.pt`` at end of training and on every epoch-end
            save (so a kill -9'd run still has a resumable snapshot).
        save_best: Track a scalar metric in :class:`TrainState` and write
            ``best.pt`` whenever it improves.
        best_metric_name: Key in ``state`` to track for ``save_best``
            (default ``"eval/loss"``).
        best_metric_mode: ``"min"`` (smaller = better) or ``"max"``.
        register_artifacts: Register each written checkpoint via
            ``trainer.ctx.save_artifact`` (if present).

    Example:
        ```python
        hook = CheckpointHook(
            checkpoint_dir="./ckpt",
            save_every_n_epochs=5,
            save_best=True,
            best_metric_name="eval/MAE",
            best_metric_mode="min",
        )
        trainer = Trainer(model=model, hooks=[hook])
        ```
    """

    _MINUS_INF = float("-inf")
    _PLUS_INF = float("inf")

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        save_every_n_epochs: int = 1,
        save_last: bool = True,
        save_best: bool = False,
        best_metric_name: str = "eval/loss",
        best_metric_mode: str = "min",
        register_artifacts: bool = False,
    ):
        import os

        import torch

        if best_metric_mode not in ("min", "max"):
            raise ValueError(
                f"best_metric_mode must be 'min' or 'max', got {best_metric_mode!r}"
            )

        self.os = os
        self.torch = torch

        self.checkpoint_dir = checkpoint_dir
        self.save_every_n_epochs = save_every_n_epochs
        self.save_last = save_last
        self.save_best = save_best
        self.best_metric_name = best_metric_name
        self.best_metric_mode = best_metric_mode
        self.register_artifacts = register_artifacts

        self._best_value: float | None = None

    def on_train_start(self, trainer, state):
        """Create checkpoint directory and sync best-metric metadata."""
        self.os.makedirs(self.checkpoint_dir, exist_ok=True)
        # Push tracking config into trainer._checkpoint so the saved payload
        # self-describes its own best-metric semantics.
        ckpt = getattr(trainer, "_checkpoint", None)
        if ckpt is not None:
            ckpt.best_metric_name = self.best_metric_name

    def on_epoch_end(self, trainer, state):
        """Save periodic / best / last checkpoints at epoch end."""
        if self.save_every_n_epochs > 0 and (state.epoch + 1) % self.save_every_n_epochs == 0:
            self._save_checkpoint(trainer, state, f"epoch_{state.epoch}.pt")
        if self.save_best:
            self._maybe_save_best(trainer, state)
        if self.save_last:
            self._save_checkpoint(trainer, state, "last.pt")

    def on_train_end(self, trainer, state):
        """Save final ``last.pt``."""
        if self.save_last:
            self._save_checkpoint(trainer, state, "last.pt")

    def _is_improvement(self, candidate: float) -> bool:
        if self._best_value is None:
            return True
        if self.best_metric_mode == "min":
            return candidate < self._best_value
        return candidate > self._best_value

    def _maybe_save_best(self, trainer, state):
        try:
            raw = state[self.best_metric_name]
        except KeyError:
            return  # metric not populated yet
        value = float(raw)
        if not self._is_improvement(value):
            return
        self._best_value = value
        ckpt = getattr(trainer, "_checkpoint", None)
        if ckpt is not None:
            ckpt.best_metric = value
        self._save_checkpoint(trainer, state, "best.pt")

    def _build_state_dict(self, trainer, state) -> dict:
        """Produce the complete resume-ready state dict.

        Prefers the Trainer-owned :class:`Checkpoint` aggregate (includes
        RNG, AMP scaler, lr_scheduler, best-metric tracking); falls back
        to a minimal snapshot if the trainer doesn't expose one (e.g.
        a stub trainer used in a focused unit test).
        """
        ckpt = getattr(trainer, "_checkpoint", None)
        if ckpt is not None:
            ckpt.epoch = int(state.epoch)
            ckpt.global_step = int(state.global_step)
            if self.save_best and self._best_value is not None:
                ckpt.best_metric = self._best_value
            return ckpt.state_dict()
        # Fallback: minimal payload.
        sd: dict = {
            "epoch": int(state.epoch),
            "global_step": int(state.global_step),
        }
        if trainer.model is not None:
            sd["model_state_dict"] = trainer.model.state_dict()
        if trainer.optimizer is not None:
            sd["optimizer_state_dict"] = trainer.optimizer.state_dict()
        return sd

    def _save_checkpoint(self, trainer, state, filename):
        """Save checkpoint to file."""
        filepath = self.os.path.join(self.checkpoint_dir, filename)
        checkpoint = self._build_state_dict(trainer, state)
        self.torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")

        # Register checkpoint as artifact
        if self.register_artifacts and hasattr(trainer, "ctx"):
            ctx = trainer.ctx
            if ctx:
                from pathlib import Path

                checkpoint_path = Path(filepath)
                if checkpoint_path.exists():
                    ctx.save_artifact(
                        name=f"checkpoint_{filename}",
                        src=checkpoint_path,
                    )
                    logger.info(f"Registered checkpoint as artifact: {checkpoint_path}")


class ProgressBarHook(BaseHook):
    """Displays training progress with tqdm.

    Args:
        desc: Description for the progress bar (default: "Training")
        leave: Leave progress bar after completion (default: True)

    Example:
        ```python
        from molix.core.hooks import ProgressBarHook

        hook = ProgressBarHook(desc="My Training")
        trainer = Trainer(model=model, hooks=[hook])
        ```
    """

    def __init__(self, desc: str = "Training", leave: bool = True):
        from tqdm import tqdm

        self.tqdm = tqdm

        self.desc = desc
        self.leave = leave
        self.pbar = None

    def on_train_start(self, trainer, state):
        """Initialize progress bar."""
        # We don't know total steps yet, will update on first epoch
        self.pbar = None

    def on_epoch_start(self, trainer, state):
        """Update progress bar for new epoch."""
        if self.pbar is None:
            # Create progress bar on first epoch
            self.pbar = self.tqdm(desc=f"{self.desc} Epoch {state.epoch}", leave=self.leave)
        else:
            self.pbar.set_description(f"{self.desc} Epoch {state.epoch}")
            self.pbar.reset()

    def on_train_batch_end(self, trainer, state, batch, outputs):
        """Update progress bar after each batch."""
        if self.pbar is not None:
            postfix = {}
            if isinstance(outputs, dict) and "loss" in outputs:
                loss_value = (
                    outputs["loss"].item() if hasattr(outputs["loss"], "item") else outputs["loss"]
                )
                postfix["loss"] = f"{loss_value:.4f}"
            self.pbar.set_postfix(postfix)
            self.pbar.update(1)

    def on_train_end(self, trainer, state):
        """Close progress bar."""
        if self.pbar is not None:
            self.pbar.close()


class MetricsHook(ScalarHook):
    """Track training and validation metrics.

    Integrates metrics into the training loop, automatically updating them
    on each batch and computing final values at epoch end.

    Supports both built-in metrics (from molix.core.metrics) and torchmetrics.

    Args:
        metrics: List of metric instances
        pred_key: Key or tuple of keys to extract predictions from outputs
        target_key: Key or tuple of keys to extract targets from batch
        prefix_train: Prefix for training metric names (default: "train")
        prefix_val: Prefix for validation metric names (default: "eval")

    Example:
        ```python
        from molix.core.hooks import MetricsHook
        from molix.core.metrics import MAE, RMSE

        hook = MetricsHook(
            metrics=[MAE(), RMSE()],
            pred_key=("pred", "scalar"),
            target_key=("target", "U0"),
        )
        trainer = Trainer(model=model, hooks=[hook])
        ```
    """

    def __init__(
        self,
        metrics: list[Any],
        pred_key: str | tuple = "predictions",
        target_key: str | tuple = "targets",
        prefix_train: str = "train",
        prefix_val: str = "eval",
    ):
        self.metrics = metrics
        self.pred_key = pred_key if isinstance(pred_key, tuple) else (pred_key,)
        self.target_key = target_key if isinstance(target_key, tuple) else (target_key,)
        self.prefix_train = prefix_train
        self.prefix_val = prefix_val

        # Separate storage for train and val metrics
        self.train_preds: list[Any] = []
        self.train_targets: list[Any] = []
        self.val_preds: list[Any] = []
        self.val_targets: list[Any] = []

    @property
    def scalar_keys(self) -> tuple[str, ...]:
        names = [m.__class__.__name__ for m in self.metrics]
        return tuple(
            f"{prefix}/{n}"
            for prefix in (self.prefix_train, self.prefix_val)
            for n in names
        )

    def _extract_value(self, data: Any, keys: tuple) -> Any:
        """Extract value from nested dict/dataclass using key path."""
        value = data
        for key in keys:
            if isinstance(value, dict):
                value = value[key]
            elif hasattr(value, key):
                value = getattr(value, key)
            elif hasattr(value, "__getitem__"):
                value = value[key]
            else:
                raise KeyError(f"Cannot extract key {key} from {type(value)}")
        return value

    def on_epoch_start(self, trainer, state):
        """Reset all metrics at epoch start."""
        for metric in self.metrics:
            metric.reset()
        self.train_preds = []
        self.train_targets = []
        self.val_preds = []
        self.val_targets = []

    def on_train_batch_end(self, trainer, state, batch, outputs):
        """Update metrics with training batch and write to state."""
        preds = self._extract_value(outputs, self.pred_key)
        targets = self._extract_value(batch, self.target_key)

        for metric in self.metrics:
            metric.update(preds, targets)
            # Write current metric value to state
            value = metric.compute()
            metric_name = metric.__class__.__name__
            state[f"{self.prefix_train}/{metric_name}"] = value

    def on_eval_batch_end(self, trainer, state, batch, outputs):
        """Update metrics with validation batch and write accumulated metrics to state."""
        preds = self._extract_value(outputs, self.pred_key)
        targets = self._extract_value(batch, self.target_key)

        # Store for accumulated validation metrics computation
        self.val_preds.append(preds.detach().cpu())
        self.val_targets.append(targets.detach().cpu())

        # Compute accumulated eval metrics and write to state
        import torch

        if self.val_preds:
            for metric in self.metrics:
                metric.reset()
                preds_cat = torch.cat(self.val_preds)
                targets_cat = torch.cat(self.val_targets)
                metric.update(preds_cat, targets_cat)
                value = metric.compute()
                metric_name = metric.__class__.__name__
                # Write to state
                state[f"{self.prefix_val}/{metric_name}"] = value

    def on_epoch_end(self, trainer, state):
        """No-op: metrics are written to state incrementally in on_eval_batch_end."""


class StepSpeedHook(ScalarHook):
    """Track training step speed and write to outputs.

    Measures steps per second during training and writes to
    outputs["performance"]["step_per_second"].
    This can then be logged by TensorBoardHook or other hooks.

    Args:
        window_size: Number of steps to average over (default: 10)

    Example:
        ```python
        from molix.core.hooks import StepSpeedHook, TensorBoardHook

        hooks = [
            StepSpeedHook(window_size=10),
            TensorBoardHook(10, "./runs"),  # auto-picks up performance/* keys
        ]
        ```
    """

    scalar_keys = ("performance/step_per_second",)

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self._step_start_time = None
        self._steps_in_window = 0

    def on_train_start(self, trainer, state):
        """Initialize timing."""
        import time

        self._step_start_time = time.time()
        self._steps_in_window = 0

    def on_train_batch_end(self, trainer, state, batch, outputs):
        """Compute step speed and write to state."""
        import time

        self._steps_in_window += 1

        # Compute speed every window_size steps
        if self._steps_in_window >= self.window_size:
            if self._step_start_time is not None:
                elapsed = time.time() - self._step_start_time
                steps_per_sec = self._steps_in_window / elapsed

                # Write to state
                state["performance/step_per_second"] = steps_per_sec

                # Reset for next window
                self._step_start_time = time.time()
                self._steps_in_window = 0


class ProfilerHook(BaseHook):
    """PyTorch Profiler integration for performance analysis.

    Profiles training performance and exports results as Chrome Trace Viewer
    format (trace.json) and optionally TensorBoard format. Supports artifact
    registration for molexp workflow integration.

    Args:
        output_dir: Directory for profiler outputs (default: "./profiler_output")
        schedule_wait: Steps to wait before profiling (default: 1)
        schedule_warmup: Warmup steps not recorded (default: 1)
        schedule_active: Steps to actively profile (default: 3)
        schedule_repeat: Repeat profiling every N steps, 0=no repeat (default: 0)
        activities: Profiling activities, None=CPU+CUDA (default: None)
        profile_memory: Profile memory allocations (default: False)
        with_stack: Include Python stack traces (default: False)
        with_flops: Estimate FLOPs (default: False)
        with_modules: Record module hierarchy (default: False)
        record_shapes: Record tensor shapes (default: False)
        export_chrome_trace: Export trace.json (default: True)
        export_tensorboard: Export to TensorBoard format (default: False)
        register_artifacts: Register outputs as artifacts (default: False)

    Example:
        ```python
        from molix.core.hooks import ProfilerHook

        # Basic profiling
        hook = ProfilerHook(output_dir="./profiler")

        # With artifact registration
        hook = ProfilerHook(
            output_dir="./profiler",
            register_artifacts=True,
        )

        # Detailed profiling
        hook = ProfilerHook(
            output_dir="./profiler",
            schedule_wait=5,
            schedule_warmup=2,
            schedule_active=10,
            profile_memory=True,
            with_stack=True,
            with_modules=True,
            export_chrome_trace=True,
            export_tensorboard=True,
            register_artifacts=True,
        )
        ```
    """

    def __init__(
        self,
        output_dir: str = "./profiler_output",
        schedule_wait: int = 1,
        schedule_warmup: int = 1,
        schedule_active: int = 3,
        schedule_repeat: int = 0,
        activities: list | None = None,
        profile_memory: bool = False,
        with_stack: bool = False,
        with_flops: bool = False,
        with_modules: bool = False,
        record_shapes: bool = False,
        export_chrome_trace: bool = True,
        export_tensorboard: bool = False,
        register_artifacts: bool = False,
    ):
        from pathlib import Path

        self.output_dir = Path(output_dir)
        self.schedule_wait = schedule_wait
        self.schedule_warmup = schedule_warmup
        self.schedule_active = schedule_active
        self.schedule_repeat = schedule_repeat
        self.activities = activities
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_flops = with_flops
        self.with_modules = with_modules
        self.record_shapes = record_shapes
        self.export_chrome_trace = export_chrome_trace
        self.export_tensorboard = export_tensorboard
        self.register_artifacts = register_artifacts

        self.profiler = None
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_train_start(self, trainer, state):
        """Initialize and start profiler."""
        import torch

        # Set up activities
        if self.activities is None:
            activities = [
                torch.profiler.ProfilerActivity.CPU,
            ]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
        else:
            activities = self.activities

        # Create profiler schedule
        schedule = torch.profiler.schedule(
            wait=self.schedule_wait,
            warmup=self.schedule_warmup,
            active=self.schedule_active,
            repeat=self.schedule_repeat,
        )

        # Initialize profiler
        self.profiler = torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=self._on_trace_ready,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
            with_flops=self.with_flops,
            with_modules=self.with_modules,
            record_shapes=self.record_shapes,
        )

        # Start profiler
        self.profiler.__enter__()
        logger.info(f"Started profiler with output_dir={self.output_dir}")

    def on_train_batch_end(self, trainer, state, batch, outputs):
        """Step profiler after each batch."""
        if self.profiler:
            self.profiler.step()

    def on_train_end(self, trainer, state):
        """Stop profiler and register artifacts."""
        # Stop profiler
        if self.profiler:
            self.profiler.__exit__(None, None, None)
            logger.info("Stopped profiler")

        # Register artifacts if context available
        if self.register_artifacts and hasattr(trainer, "ctx"):
            ctx = trainer.ctx
            if ctx:
                # Register trace.json
                trace_path = self.output_dir / "trace.json"
                if trace_path.exists():
                    ctx.save_artifact(
                        name="profiler_trace.json",
                        src=trace_path,
                    )
                    logger.info(f"Registered profiler trace as artifact: {trace_path}")

                # Register TensorBoard logs
                if self.export_tensorboard:
                    tb_dir = self.output_dir / "tensorboard"
                    if tb_dir.exists():
                        ctx.save_artifact(
                            name="profiler_tensorboard",
                            src=tb_dir,
                        )
                        logger.info(f"Registered profiler TensorBoard logs as artifact: {tb_dir}")

    def _on_trace_ready(self, prof):
        """Callback when trace is ready - export to files."""
        # Export Chrome Trace
        if self.export_chrome_trace:
            trace_path = self.output_dir / "trace.json"
            prof.export_chrome_trace(str(trace_path))
            logger.info(f"Exported Chrome Trace to {trace_path}")

        # Export TensorBoard
        if self.export_tensorboard:
            tb_dir = self.output_dir / "tensorboard"
            tb_dir.mkdir(parents=True, exist_ok=True)

            # export_stacks requires with_stack=True
            if self.with_stack:
                prof.export_stacks(str(tb_dir / "profiler.pt.trace.json"), "self_cuda_time_total")
                logger.info(f"Exported TensorBoard profiler stacks to {tb_dir}")
            else:
                logger.warning(
                    "Skipping export_stacks() because with_stack=False. "
                    "Set with_stack=True to enable stack trace export."
                )


class GradClipHook(ScalarHook):
    """Clip gradient norms after backward pass.

    Applies ``torch.nn.utils.clip_grad_norm_`` at the ``on_after_backward``
    stage and writes the pre-clip gradient norm to ``state["train/grad_norm"]``.

    Args:
        max_norm: Maximum norm of the gradients.
        norm_type: Type of the used p-norm (default: 2.0, i.e. L2).

    Example:
        ```python
        from molix.core.hooks import GradClipHook

        hook = GradClipHook(max_norm=1.0)
        trainer = Trainer(model=model, hooks=[hook])
        ```
    """

    scalar_keys = ("train/grad_norm",)

    def __init__(self, max_norm: float, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type

    def on_after_backward(self, trainer, state):
        """Clip gradients and record pre-clip norm."""
        import torch

        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainer.model.parameters(),
            self.max_norm,
            norm_type=self.norm_type,
        )
        state["train/grad_norm"] = grad_norm.item()


class ActivationCheckpointingHook(BaseHook):
    """Apply activation checkpointing to model layers at training start.

    Wraps matching submodules with ``torch.utils.checkpoint.checkpoint``
    using ``use_reentrant=False`` (recommended for DDP compatibility).

    Args:
        check_fn: Predicate selecting which submodules to wrap.
            If None, wraps all direct children of the model.

    Example:
        ```python
        from molix.core.hooks import ActivationCheckpointingHook

        hook = ActivationCheckpointingHook(
            check_fn=lambda m: isinstance(m, InteractionBlock),
        )
        trainer = Trainer(model=model, hooks=[hook])
        ```
    """

    def __init__(self, check_fn: Callable[[nn.Module], bool] | None = None):
        self.check_fn = check_fn

    def on_train_start(self, trainer, state):
        """Wrap matching modules with activation checkpointing."""
        _apply_activation_checkpointing(trainer.model, self.check_fn)


def _apply_activation_checkpointing(
    model: nn.Module,
    check_fn: Callable[[nn.Module], bool] | None = None,
) -> None:
    """Wrap matching submodules with activation checkpointing.

    Args:
        model: The model to apply checkpointing to.
        check_fn: Predicate that selects which submodules to wrap.
            If None, wraps all direct children.
    """
    from torch.utils.checkpoint import checkpoint

    if check_fn is None:
        targets = set(model.children())
    else:
        targets = {m for m in model.modules() if m is not model and check_fn(m)}

    for module in targets:
        original_forward = module.forward

        def _make_checkpointed(fn):
            def checkpointed_forward(*args, **kwargs):
                return checkpoint(fn, *args, use_reentrant=False, **kwargs)

            return checkpointed_forward

        module.forward = _make_checkpointed(original_forward)  # type: ignore[method-assign]


class GPUMemoryHook(ScalarHook):
    """Record CUDA memory usage on every training batch.

    Writes three scalars to ``state`` each ``on_train_batch_end``:

    - ``gpu/alloc_gib``: currently allocated memory (``memory_allocated``).
    - ``gpu/resv_gib``:  reserved memory (``memory_reserved``).
    - ``gpu/peak_gib``:  peak ``memory_allocated`` since the last batch;
      ``reset_peak_memory_stats`` is called after reading so the value always
      reflects the *window* between consecutive reads — most useful for
      locating OOM hotspots inside :class:`Log`.

    Key names are slash/underscore only (no brackets), so they round-trip
    cleanly through TensorBoard tags, Parquet column names, and filesystem
    paths.

    No-op when ``torch.cuda.is_available()`` is ``False``; the keys are still
    written (as ``0.0``) so headers stay aligned.
    """

    scalar_keys = ("gpu/alloc_gib", "gpu/resv_gib", "gpu/peak_gib")

    _GB = 1024 ** 3  # GiB, matches the ``_gib`` suffix on the keys

    def on_train_start(self, trainer, state):
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_batch_end(self, trainer, state, batch, outputs):
        import torch

        if torch.cuda.is_available():
            state["gpu/alloc_gib"] = torch.cuda.memory_allocated() / self._GB
            state["gpu/resv_gib"] = torch.cuda.memory_reserved() / self._GB
            state["gpu/peak_gib"] = torch.cuda.max_memory_allocated() / self._GB
            torch.cuda.reset_peak_memory_stats()
        else:
            state["gpu/alloc_gib"] = 0.0
            state["gpu/resv_gib"] = 0.0
            state["gpu/peak_gib"] = 0.0


class GPUUtilizationHook(ScalarHook):
    """Record GPU SM / memory-bandwidth utilization (%) via NVML per batch.

    Writes two scalars to ``state`` on ``on_train_batch_end``:

    - ``gpu/util_pct``:     SM utilization (instantaneous NVML sample).
    - ``gpu/mem_util_pct``: Memory-bandwidth utilization.

    Backed by ``nvidia-ml-py`` (imported as ``pynvml``). The call takes
    ~100µs and does not trigger a CUDA synchronization, so it is safe to
    run every step.

    Requires CUDA and ``nvidia-ml-py`` — install via the ``profile`` or
    ``dev`` extras (``pip install -e ".[profile]"``). Raises at
    ``on_train_start`` if either is unavailable rather than silently
    reporting zeros.
    """

    scalar_keys = ("gpu/util_pct", "gpu/mem_util_pct")

    def on_train_start(self, trainer, state):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPUUtilizationHook requires CUDA but torch.cuda.is_available() is False."
            )
        try:
            import pynvml
        except ImportError as exc:
            raise ImportError(
                "GPUUtilizationHook requires nvidia-ml-py. "
                'Install with `pip install -e ".[profile]"` or `pip install nvidia-ml-py`.'
            ) from exc

        pynvml.nvmlInit()
        idx = torch.cuda.current_device()
        self._pynvml = pynvml
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(idx)

    def on_train_batch_end(self, trainer, state, batch, outputs):
        rates = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
        state["gpu/util_pct"] = float(rates.gpu)
        state["gpu/mem_util_pct"] = float(rates.memory)

    def on_train_end(self, trainer, state):
        pynvml = getattr(self, "_pynvml", None)
        if pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        self._pynvml = None
        self._handle = None


def _parse_fmt_width(fmt: str) -> int:
    """Extract the width component from a format spec like ``"{:>12.4g}"``."""
    import re

    m = re.search(r"(\d+)", fmt)
    return int(m.group(1)) if m else 12


def _as_scalar(value) -> float | int | None:
    """Coerce ``value`` to a Python scalar or return ``None`` if non-scalar.

    Accepts Python numbers and 0-d tensors / numpy scalars. Anything else
    (vectors, matrices, strings, enums, None) returns ``None`` so callers
    can skip silently instead of raising.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (RuntimeError, ValueError):
            return None
    return None


def _collect_keys(items) -> list[str]:
    """Flatten a mix of raw key strings and :class:`ScalarHook` instances.

    Accepts::

        keys=["train/loss", step_speed_hook, "gpu/peak_gib"]

    where any ``ScalarHook`` is expanded to its ``scalar_keys``. Duplicates
    are preserved in order of first occurrence.
    """
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if isinstance(item, ScalarHook):
            names = tuple(item.scalar_keys)
        elif isinstance(item, str):
            names = (item,)
        else:
            raise TypeError(
                f"Log keys must be str or ScalarHook, got {type(item).__name__}"
            )
        for n in names:
            if n not in seen:
                seen.add(n)
                out.append(n)
    return out


class Log(BaseHook):
    """Periodic LAMMPS-thermo-style stdout logger.

    Reads scalar values from ``state`` and prints a formatted row every
    ``every_n_steps`` training batches. ``Log`` does **not** own the hooks
    that populate those scalars — register those hooks normally in the
    trainer's top-level ``hooks=[...]`` list so they can be shared with
    other consumers (e.g. :class:`TensorBoardHook`).

    Keys may be given either as plain state paths (``"train/loss"``) or as
    :class:`ScalarHook` instances (the hook's ``scalar_keys`` are expanded).
    Passing the hook itself keeps key names out of the call site and lets
    the hook evolve its advertised keys independently.

    Args:
        every_n_steps: Print a row every N training batches.
        keys: State keys to print. Each entry is either a ``str`` path or a
            :class:`ScalarHook` instance whose ``scalar_keys`` will be used.
            ``step`` and ``epoch`` are always prepended.
        fmt: Format spec for each scalar column (default ``"{:>12.4g}"``).

    Example:
        ```python
        step_speed = StepSpeedHook(window_size=20)
        gpu = GPUMemoryHook()
        hooks = [
            MetricsHook(...),
            step_speed,
            gpu,
            Log(50, keys=[step_speed, gpu, "train/loss"]),
            TensorBoardHook(...),   # also reads the same state keys
        ]
        ```
    """

    def __init__(
        self,
        every_n_steps: int,
        keys,
        *,
        fmt: str = "{:>12.4g}",
    ):
        if every_n_steps <= 0:
            raise ValueError("every_n_steps must be positive")
        self.every_n_steps = every_n_steps
        self.keys = _collect_keys(keys)
        self.fmt = fmt

    def _columns(self) -> list[str]:
        return ["step", "epoch", *self.keys]

    def on_train_start(self, trainer, state):
        width = _parse_fmt_width(self.fmt)
        header = " ".join(f"{c:>{width}}" for c in self._columns())
        print(header, flush=True)

    def on_train_batch_end(self, trainer, state, batch, outputs):
        # global_step is incremented *after* this hook fires, so add 1 to get
        # the 1-based step number and trigger at clean multiples (10, 20, 30…).
        step = int(state.get("global_step", 0)) + 1
        if step % self.every_n_steps != 0:
            return

        width = _parse_fmt_width(self.fmt)
        parts = [
            f"{step:>{width}d}",
            f"{int(state.get('epoch', 0)):>{width}d}",
        ]
        for key in self.keys:
            val = state.get(key)
            if isinstance(val, (int, float)):
                parts.append(self.fmt.format(val))
            else:
                parts.append(f"{'nan':>{width}}")
        print(" ".join(parts), flush=True)
