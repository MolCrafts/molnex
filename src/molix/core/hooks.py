"""Hook system for Molix Trainer.

This module provides an extensible hook system that allows users to inject
custom logic at various points in the training lifecycle.

Hooks execute in registration order by default. Use (hook, priority) tuples
to override execution order (lower priority = earlier execution).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

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


# Built-in Hooks


class TensorBoardHook(BaseHook):
    """Logs training metrics to TensorBoard from trainer state.

    Reads metrics from configured key paths in state and logs them
    to TensorBoard. Supports model graph visualization, weight/gradient histograms,
    and hyperparameter tracking.

    Args:
        log_dir: Directory to save TensorBoard logs (default: "./runs")
        log_every_n_steps: Log scalars every N steps (default: 1)
        metric_paths: List of key paths to read from trainer state.
                     Each path is a list of keys, TensorBoard name is auto-generated
                     by joining with '/'. (default: train/eval loss/MAE/RMSE)
        log_hparams: Log hyperparameters for HParams dashboard (default: False)
        log_graph: Log model graph on first batch (default: False)
        log_histograms: Log weight/gradient histograms (default: False)
        hparams: Hyperparameters dict (required if log_hparams=True)
        histogram_freq: Log histograms every N epochs (default: 1)
        register_artifacts: Register logs as artifacts (default: False)

    Example:
        ```python
        from molix.core.hooks import TensorBoardHook, MetricsHook
        from molix.core.metrics import MAE, RMSE

        # Basic usage with default paths
        hook = TensorBoardHook(log_dir="./runs")

        # Custom metric paths (TensorBoard names auto-generated)
        hook = TensorBoardHook(
            log_dir="./runs",
            metric_paths=[
                ["train", "loss"],    # -> "train/loss"
                ["train", "MAE"],     # -> "train/MAE"
                ["eval", "RMSE"],     # -> "eval/RMSE"
            ],
        )

        # Full featured
        hook = TensorBoardHook(
            log_dir="./runs",
            log_hparams=True,
            log_graph=True,
            log_histograms=True,
            hparams={"lr": 0.001, "batch_size": 32},
            histogram_freq=5,
        )
        ```
    """

    def __init__(
        self,
        log_dir: str = "./runs",
        log_every_n_steps: int = 1,
        metric_paths: list[list[str]] | None = None,
        log_hparams: bool = False,
        log_graph: bool = False,
        log_histograms: bool = False,
        hparams: dict | None = None,
        histogram_freq: int = 1,
        register_artifacts: bool = False,
    ):
        """Initialize TensorBoardHook.

        Args:
            log_dir: Directory for TensorBoard logs
            log_every_n_steps: Log frequency
            metric_paths: List of key paths to read from trainer state.
                         Each path is a list of keys. TensorBoard name is auto-generated
                         by joining path with '/'. Example:
                         [["train", "loss"],      # -> "train/loss"
                          ["train", "MAE"],       # -> "train/MAE"
                          ["eval", "RMSE"]]       # -> "eval/RMSE"
            log_hparams: Log hyperparameters
            log_graph: Log model graph
            log_histograms: Log weight/gradient histograms
            hparams: Hyperparameter dict
            histogram_freq: Histogram logging frequency
            register_artifacts: Register logs as artifacts
        """
        from torch.utils.tensorboard import SummaryWriter

        self.SummaryWriter = SummaryWriter

        self.log_dir = log_dir
        self.log_every_n_steps = log_every_n_steps
        self.metric_paths = metric_paths or [
            ["train", "loss"],
            ["train", "MAE"],
            ["train", "RMSE"],
            ["eval", "loss"],
            ["eval", "MAE"],
            ["eval", "RMSE"],
        ]
        self.log_hparams = log_hparams
        self.log_graph = log_graph
        self.log_histograms = log_histograms
        self.hparams = hparams or {}
        self.histogram_freq = histogram_freq
        self.register_artifacts = register_artifacts

        self.writer = None
        self.trainer = None
        self._graph_logged = False

    def on_train_start(self, trainer, state):
        """Initialize TensorBoard writer and log hyperparameters."""
        self.writer = self.SummaryWriter(self.log_dir)
        self.trainer = trainer

        # Log hyperparameters at start (metrics will be logged at end)
        if self.log_hparams and self.hparams:
            # Create placeholder for metrics (will be updated at end)
            logger.info(f"Logging hyperparameters: {self.hparams}")

    def on_train_batch_start(self, trainer, state, batch):
        """Log model graph on first batch."""
        if self.log_graph and not self._graph_logged:
            import torch

            input_tensor = None
            if isinstance(batch, dict):
                model_inputs = batch.get("model_inputs", batch)
                if isinstance(model_inputs, dict):
                    for value in model_inputs.values():
                        if isinstance(value, torch.Tensor):
                            input_tensor = value
                            break
            if input_tensor is None and isinstance(batch, dict):
                for key in batch.keys():
                    value = batch[key]
                    if isinstance(value, torch.Tensor):
                        input_tensor = value
                        break
            elif isinstance(batch, torch.Tensor):
                input_tensor = batch

            if input_tensor is not None:
                # trainer.model might expect multiple args, SummaryWriter.add_graph
                # might need adjustments if the model takes a dict or multiple args.
                # However, for now we just try with one tensor if possible.
                try:
                    self.writer.add_graph(trainer.model, input_tensor)
                    logger.info("Logged model graph to TensorBoard")
                except Exception as e:
                    logger.warning(f"Failed to log model graph: {e}")
            self._graph_logged = True

    def on_train_batch_end(self, trainer, state, batch, outputs):
        """Log training scalars from state dict."""
        if state.global_step % self.log_every_n_steps == 0:
            # Log metrics from configured paths
            for path in self.metric_paths:
                if path[0] == "train":  # Only log train metrics here
                    value = self._extract_from_path(state, path)
                    if value is not None:
                        value = value.item() if hasattr(value, "item") else value
                        tb_name = "/".join(path)
                        self.writer.add_scalar(tb_name, value, state.global_step)

    def on_eval_batch_end(self, trainer, state, batch, outputs):
        """Log evaluation scalars from state dict."""
        # Log metrics from configured paths
        for path in self.metric_paths:
            if path[0] == "eval":  # Only log eval metrics here
                value = self._extract_from_path(state, path)
                if value is not None:
                    value = value.item() if hasattr(value, "item") else value
                    tb_name = "/".join(path)
                    self.writer.add_scalar(tb_name, value, state.global_step)

    def _extract_from_path(self, data: Any, path: list[str]) -> Any:
        """Extract value from nested dictionary-like data via key path.

        Args:
            data: The data structure to extract from
            path: List of keys to traverse

        Returns:
            The value at the path, or None if not found
        """
        try:
            value = data
            for key in path:
                if hasattr(value, "get"):
                    value = value.get(key)
                    if value is None:
                        return None
                elif hasattr(value, "__getitem__"):
                    value = value[key]
                else:
                    return None
            return value
        except (KeyError, IndexError, TypeError):
            return None

    def on_epoch_end(self, trainer, state):
        """Log histograms."""
        # Log histograms
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

        # Register TensorBoard logs as artifact
        if self.register_artifacts and hasattr(trainer, "ctx"):
            ctx = trainer.ctx
            if ctx:
                from pathlib import Path

                log_dir_path = Path(self.log_dir)
                if log_dir_path.exists():
                    ctx.save_artifact(
                        name="tensorboard_logs",
                        src=log_dir_path,
                    )
                    logger.info(f"Registered TensorBoard logs as artifact: {log_dir_path}")

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

    Args:
        checkpoint_dir: Directory to save checkpoints (default: "./checkpoints")
        save_every_n_epochs: Save checkpoint every N epochs (default: 1)
        save_last: Always save the last checkpoint (default: True)

    Example:
        ```python
        from molix.core.hooks import CheckpointHook

        hook = CheckpointHook(checkpoint_dir="./ckpt", save_every_n_epochs=5)
        trainer = Trainer(model=model, hooks=[hook])
        ```
    """

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        save_every_n_epochs: int = 1,
        save_last: bool = True,
        register_artifacts: bool = False,
    ):
        import os

        import torch

        self.os = os
        self.torch = torch

        self.checkpoint_dir = checkpoint_dir
        self.save_every_n_epochs = save_every_n_epochs
        self.save_last = save_last
        self.register_artifacts = register_artifacts

    def on_train_start(self, trainer, state):
        """Create checkpoint directory."""
        self.os.makedirs(self.checkpoint_dir, exist_ok=True)

    def on_epoch_end(self, trainer, state):
        """Save checkpoint at specified intervals."""
        if (state.epoch + 1) % self.save_every_n_epochs == 0:
            self._save_checkpoint(trainer, state, f"epoch_{state.epoch}.pt")

    def on_train_end(self, trainer, state):
        """Save final checkpoint."""
        if self.save_last:
            self._save_checkpoint(trainer, state, "last.pt")

    def _save_checkpoint(self, trainer, state, filename):
        """Save checkpoint to file."""
        filepath = self.os.path.join(self.checkpoint_dir, filename)
        checkpoint = {
            "epoch": state.epoch,
            "global_step": state.global_step,
            "model_state_dict": trainer.model.state_dict() if trainer.model else None,
            "optimizer_state_dict": trainer.optimizer.state_dict() if trainer.optimizer else None,
        }
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


class MetricsHook(BaseHook):
    """Track training and validation metrics.

    Integrates metrics into the training loop, automatically updating them
    on each batch and computing final values at epoch end.

    Supports both built-in metrics (from molix.core.metrics) and torchmetrics.

    Args:
        metrics: List of metric instances
        pred_key: Key or tuple of keys to extract predictions from outputs
        target_key: Key or tuple of keys to extract targets from batch
        prefix_train: Prefix for training metric names (default: "train")
        prefix_val: Prefix for validation metric names (default: "val")

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
        prefix_val: str = "val",
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
        """Compute and log all metrics at epoch end."""
        import torch

        print(f"\nEpoch {state.epoch + 1} Metrics:")

        # Compute training metrics
        for metric in self.metrics:
            value = metric.compute()
            metric_name = metric.__class__.__name__
            print(f"  {self.prefix_train}/{metric_name}: {value:.4f}")

        # Compute validation metrics if we have validation data
        if self.val_preds:
            # Reset metrics and compute on validation data
            for metric in self.metrics:
                metric.reset()
                preds = torch.cat(self.val_preds)
                targets = torch.cat(self.val_targets)
                metric.update(preds, targets)

                value = metric.compute()
                metric_name = metric.__class__.__name__
                print(f"  {self.prefix_val}/{metric_name}: {value:.4f}")


class StepSpeedHook(BaseHook):
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
            TensorBoardHook(
                metric_paths=[
                    ["performance", "step_per_second"],
                ],
            ),
        ]
        ```
    """

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

    @classmethod
    def for_diagnosis(cls, output_dir: str = "./profiler_output") -> "ProfilerHook":
        """Create a profiler configured for full performance diagnosis.

        Enables all options needed to answer: data-slow, operator-slow,
        communication-slow, or graph-break-slow.

        Args:
            output_dir: Directory for profiler outputs.

        Returns:
            Configured ProfilerHook instance.
        """
        return cls(
            output_dir=output_dir,
            schedule_wait=2,
            schedule_warmup=2,
            schedule_active=6,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True,
            record_shapes=True,
            export_chrome_trace=True,
            export_tensorboard=True,
        )

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


class DataLoaderProfilingHook(BaseHook):
    """Measures wall-clock time between batches to detect data loading bottlenecks.

    All diagnostic state is internal to this hook. Access results via the
    hook instance directly, not through TrainState.

    Args:
        log_every_n_steps: Log timing summary every N steps.

    Attributes:
        median_data_wait_ms: Median time waiting for next batch (ms).
        median_compute_ms: Median time in forward/backward/optimizer (ms).
        is_data_bottleneck: True when data wait exceeds compute time.

    Example:
        ```python
        dl_hook = DataLoaderProfilingHook(log_every_n_steps=50)
        trainer = Trainer(model=model, hooks=[dl_hook])
        trainer.train(dm, max_epochs=5)

        # After training, inspect:
        print(dl_hook.median_data_wait_ms)
        print(dl_hook.is_data_bottleneck)
        ```
    """

    def __init__(self, log_every_n_steps: int = 50):
        self.log_every_n_steps = log_every_n_steps

        self._batch_end_time: float | None = None
        self._batch_start_time: float = 0.0
        self._data_wait_times: list[float] = []
        self._compute_times: list[float] = []

        # Public read-only diagnostics
        self.median_data_wait_ms: float = 0.0
        self.median_compute_ms: float = 0.0
        self.is_data_bottleneck: bool = False

    def on_train_start(self, trainer, state):
        """Reset timing state."""
        self._batch_end_time = None
        self._data_wait_times.clear()
        self._compute_times.clear()

    def on_train_batch_start(self, trainer, state, batch):
        """Record data wait time (gap since last batch_end)."""
        import time

        now = time.perf_counter()
        if self._batch_end_time is not None:
            self._data_wait_times.append((now - self._batch_end_time) * 1000)
        self._batch_start_time = now

    def on_train_batch_end(self, trainer, state, batch, outputs):
        """Record compute time and periodically update diagnostics."""
        import time
        import statistics

        now = time.perf_counter()
        self._batch_end_time = now
        self._compute_times.append((now - self._batch_start_time) * 1000)

        step = state.global_step
        if (
            step > 0
            and step % self.log_every_n_steps == 0
            and self._data_wait_times
        ):
            self.median_data_wait_ms = statistics.median(self._data_wait_times)
            self.median_compute_ms = statistics.median(self._compute_times)
            self.is_data_bottleneck = self.median_data_wait_ms > self.median_compute_ms
            logger.info(
                "Step %d: data_wait=%.1fms compute=%.1fms (data_bottleneck=%s)",
                step,
                self.median_data_wait_ms,
                self.median_compute_ms,
                self.is_data_bottleneck,
            )
            self._data_wait_times.clear()
            self._compute_times.clear()
