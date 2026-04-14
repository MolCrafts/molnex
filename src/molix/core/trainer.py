"""Trainer implementation for Molix."""

from __future__ import annotations

import platform
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

import torch
import torch.nn as nn

from molix import logger as _logger_mod
from molix.config import config
from molix.config import set_precision as _set_precision_global
from molix.core.checkpoint import Checkpoint, CheckpointBackend, TorchSaveBackend
from molix.core.hooks import Hook
from molix.core.state import Stage, TrainState
from molix.core.steps import DefaultEvalStep, DefaultTrainStep, Step
from molix.data.datamodule import DataModuleProtocol

logger = _logger_mod.getLogger(__name__)


class Trainer:
    """ML training system for MolNex.

    The Trainer:
    - Owns execution and control flow
    - Owns training state
    - Executes epoch/step loops
    - Provides graph export for introspection

    Attributes:
        train_step: Training step object (for step-based training)
        eval_step: Evaluation step object (for step-based training)
        model: Neural network model (for direct training)
        loss_fn: Loss function (for direct training)
        optimizer: Optimizer instance (for direct training)
        state: Current training state
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        optimizer_factory: Callable,
        lr_scheduler_factory: Callable | None = None,
        train_step: Step | None = None,
        eval_step: Step | None = None,
        hooks: list[Hook | tuple[Hook, int]] | None = None,
        eval_every_n_steps: int | None = None,
        resume_from_checkpoint: str | Path | None = None,
        checkpoint_backend: CheckpointBackend | None = None,
    ):
        """Initialize trainer.

        Args:
            model: Neural network model.
            loss_fn: Loss function.
            optimizer_factory: Factory to create optimizer from parameters.
            lr_scheduler_factory: Factory to create lr scheduler from optimizer.
                Called as ``lr_scheduler_factory(optimizer)``.
            train_step: Training step implementing Step protocol. If None, uses
                DefaultTrainStep.
            eval_step: Evaluation step implementing Step protocol. If None, uses
                DefaultEvalStep.
            hooks: List of hooks or (hook, priority) tuples. Hooks execute in
                   registration order by default. Use tuples to override priority
                   (lower priority = earlier execution, default = 100).
            eval_every_n_steps: Run evaluation every N training steps (in addition
                   to epoch-end eval). If None (default), only epoch-end eval runs.
                   Must be > 0 if provided.
            resume_from_checkpoint: Path to a checkpoint file to resume from,
                   or ``"auto"`` to detect torchrun elastic snapshots.
            checkpoint_backend: Backend for checkpoint I/O. Defaults to
                   :class:`TorchSaveBackend`.

        Raises:
            ValueError: If eval_every_n_steps is <= 0
        """
        if eval_every_n_steps is not None and eval_every_n_steps <= 0:
            raise ValueError(f"eval_every_n_steps must be > 0, got {eval_every_n_steps}")
        self.eval_every_n_steps = eval_every_n_steps

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer_factory(model.parameters())

        # lr scheduler
        self.lr_scheduler = (
            lr_scheduler_factory(self.optimizer)
            if lr_scheduler_factory
            else None
        )

        # Scaler reflects current global config. Call :meth:`set_precision`
        # to change it after construction.
        self.scaler = torch.amp.GradScaler() if config["use_amp"] else None

        # Checkpoint backend
        self._checkpoint_backend = checkpoint_backend or TorchSaveBackend()
        self._resume_from_checkpoint = resume_from_checkpoint

        # Steps
        self.train_step = train_step or DefaultTrainStep()
        self.eval_step = eval_step or DefaultEvalStep()

        self.state = TrainState()

        # Checkpoint serialisation aggregate
        self._checkpoint = Checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            scaler=self.scaler,
        )

        # Hooks with priority sorting
        self.hooks: list[Hook] = []
        if hooks:
            normalized = []
            for idx, item in enumerate(hooks):
                if isinstance(item, tuple):
                    hook, priority = item
                    normalized.append((hook, priority, idx))
                else:
                    normalized.append((item, 100, idx))
            normalized.sort(key=lambda x: (x[1], x[2]))
            self.hooks = [hook for hook, _, _ in normalized]

    def set_precision(self, mode: str) -> None:
        """Configure training precision and sync the AMP ``GradScaler``.

        Delegates to :func:`molix.config.set_precision` (which writes
        ``ftype`` / ``use_amp`` / ``amp_dtype`` into the global
        :data:`molix.config.config`), then (re)creates ``self.scaler`` to
        match ``config["use_amp"]`` and updates the checkpoint aggregate so
        the new scaler is saved/loaded.

        Must be called **before** ``trainer.train()``. For ``"fp64"`` the
        model must additionally be constructed (or cast) with the new
        ``ftype``, because layers bake in ``config.ftype`` at ``__init__``.

        Args:
            mode: One of ``"fp32"``, ``"fp64"``, ``"fp16-mixed"``,
                ``"bf16-mixed"``.

        Raises:
            ValueError: If ``mode`` is not a supported preset.
        """
        _set_precision_global(mode)
        self.scaler = torch.amp.GradScaler() if config["use_amp"] else None
        self._checkpoint.scaler = self.scaler

    def train(
        self,
        datamodule: DataModuleProtocol,
        max_epochs: int | None = None,
        max_steps: int | None = None,
    ) -> TrainState:
        """Execute training loop.

        When both ``max_epochs`` and ``max_steps`` are provided, training
        stops at whichever limit is reached first.

        Args:
            datamodule: Data module providing train/val dataloaders
            max_epochs: Maximum number of epochs to train.
            max_steps: Maximum number of training steps (batches) across
                all epochs.

        Returns:
            Final training state

        Raises:
            ValueError: If neither ``max_epochs`` nor ``max_steps`` is set,
                or if either value is <= 0.
        """
        if max_epochs is None and max_steps is None:
            raise ValueError(
                "At least one of max_epochs or max_steps must be specified"
            )
        if max_epochs is not None and max_epochs <= 0:
            raise ValueError(f"max_epochs must be > 0, got {max_epochs}")
        if max_steps is not None and max_steps <= 0:
            raise ValueError(f"max_steps must be > 0, got {max_steps}")
        return self._train(datamodule, max_epochs, max_steps)

    def _call_hooks(self, hook_name: str, *args, **kwargs) -> None:
        """Call a hook on all registered hooks.

        Args:
            hook_name: Name of the hook method to call
            *args: Positional arguments to pass to hook
            **kwargs: Keyword arguments to pass to hook
        """
        for hook in self.hooks:
            try:
                method = getattr(hook, hook_name, None)
                if method is not None and callable(method):
                    method(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Error in hook {hook.__class__.__name__}.{hook_name}: {e}", exc_info=True
                )

    def _load_checkpoint(self, path: str | Path) -> None:
        """Load checkpoint and restore all training state.

        Args:
            path: Checkpoint file path, or ``"auto"`` for torchrun elastic.
        """
        resolved = self._resolve_checkpoint_path(path)
        if resolved is None:
            logger.info("No checkpoint found to resume from.")
            return

        state_dict = self._checkpoint_backend.load(resolved, map_location="cpu")
        self._checkpoint.load_state_dict(state_dict)

        # Sync counters from Checkpoint → TrainState
        self.state.epoch = self._checkpoint.epoch
        self.state.global_step = self._checkpoint.global_step
        if self._checkpoint.best_metric is not None:
            self.state["best_metric"] = self._checkpoint.best_metric

        logger.info(
            f"Resumed from checkpoint {resolved} "
            f"(epoch={self.state.epoch}, step={self.state.global_step})"
        )

    @staticmethod
    def _resolve_checkpoint_path(path: str | Path) -> Path | None:
        """Resolve checkpoint path, handling ``"auto"`` for torchrun elastic.

        Args:
            path: File path or ``"auto"``.

        Returns:
            Resolved :class:`Path` if a checkpoint exists, else ``None``.
        """
        import os

        if str(path) == "auto":
            run_id = os.environ.get("TORCHELASTIC_RUN_ID")
            snapshot_dir = os.environ.get(
                "TORCHELASTIC_SNAPSHOT_DIR", "./snapshots"
            )
            if run_id:
                snapshot_path = Path(snapshot_dir) / run_id / "snapshot.pt"
                if snapshot_path.exists():
                    return snapshot_path
            return None

        p = Path(path)
        return p if p.exists() else None

    def _train(
        self,
        datamodule: DataModuleProtocol,
        max_epochs: int | None,
        max_steps: int | None,
    ) -> TrainState:
        """Execute training loop.

        Stops at whichever limit (epochs or steps) is reached first.
        """
        epoch_limit = max_epochs if max_epochs is not None else float("inf")
        step_limit = max_steps if max_steps is not None else float("inf")

        datamodule.setup("fit")

        if self._resume_from_checkpoint is not None:
            self._load_checkpoint(self._resume_from_checkpoint)

        self._log_setup_banner(datamodule, max_epochs, max_steps)

        self._call_hooks("on_train_start", self, self.state)

        epoch = self.state.epoch
        start_epoch = epoch
        start_global_step = self.state.global_step
        start_time = time.perf_counter()
        step_limit_reached = False

        while epoch < epoch_limit and not step_limit_reached:
            datamodule.on_epoch_start(epoch)
            self._call_hooks("on_epoch_start", self, self.state)

            # Training phase
            self.state.set_stage(Stage.TRAIN)
            self.model.train()

            for batch in datamodule.train_dataloader():
                self._call_hooks("on_train_batch_start", self, self.state, batch)
                outputs = self.train_step.on_train_batch(self, self.state, batch)
                self._call_hooks("on_train_batch_end", self, self.state, batch, outputs)

                self.state.increment_step()
                self.state.steps_since_last_eval += 1

                if (
                    self.eval_every_n_steps is not None
                    and self.state.steps_since_last_eval >= self.eval_every_n_steps
                ):
                    self._run_eval_phase(datamodule)
                    self.state.steps_since_last_eval = 0
                    self._call_hooks("on_eval_step_complete", self, self.state)

                if self.state.global_step >= step_limit:
                    step_limit_reached = True
                    break

            # Validation phase
            self.state.set_stage(Stage.EVAL)
            self.model.eval()

            for batch in datamodule.val_dataloader():
                self._call_hooks("on_eval_batch_start", self, self.state, batch)
                outputs = self.eval_step.on_eval_batch(self, self.state, batch)
                self._call_hooks("on_eval_batch_end", self, self.state, batch, outputs)

            self._call_hooks("on_epoch_end", self, self.state)
            self.state.increment_epoch()
            epoch += 1

        self._call_hooks("on_train_end", self, self.state)

        elapsed = time.perf_counter() - start_time
        self._log_summary_banner(
            elapsed=elapsed,
            epochs_done=epoch - start_epoch,
            steps_done=self.state.global_step - start_global_step,
        )
        return self.state

    def compile(
        self,
        *,
        backend: str = "inductor",
        fullgraph: bool = False,
        dynamic: bool | None = None,
        mode: str | None = None,
    ) -> "Trainer":
        """Compile the model with ``torch.compile`` for optimized execution.

        Can be chained after construction::

            trainer = Trainer(model, ...).compile(mode="max-autotune")

        Args:
            backend: Compile backend (default: ``"inductor"``).
            fullgraph: If True, require single graph (error on graph breaks).
            dynamic: Enable dynamic shape tracing.
            mode: Compile mode (``"default"``, ``"reduce-overhead"``,
                ``"max-autotune"``).

        Returns:
            ``self`` for method chaining.
        """
        from molix.compile import maybe_compile

        self.model = maybe_compile(
            self.model,
            compile=True,
            backend=backend,
            fullgraph=fullgraph,
            dynamic=dynamic,
            mode=mode,
        )
        # Keep checkpoint in sync so state_dict() sees the compiled wrapper
        self._checkpoint.model = self.model
        return self

    def _log_setup_banner(
        self,
        datamodule: DataModuleProtocol,
        max_epochs: int | None,
        max_steps: int | None,
    ) -> None:
        """Print a LAMMPS-style setup banner before training starts."""
        n_params = sum(p.numel() for p in self.model.parameters())
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        model_device = next((p.device for p in self.model.parameters()), torch.device("cpu"))
        model_dtype = next((p.dtype for p in self.model.parameters()), config["ftype"])

        use_amp = bool(config["use_amp"])
        amp_dtype = config["amp_dtype"] if use_amp else None

        opt_name = type(self.optimizer).__name__
        lrs = [g.get("lr") for g in self.optimizer.param_groups]
        lr_str = ", ".join(f"{lr:.3e}" for lr in lrs if lr is not None) or "n/a"
        sched_name = type(self.lr_scheduler).__name__ if self.lr_scheduler else "None"
        hooks_str = ", ".join(h.__class__.__name__ for h in self.hooks) or "None"
        batch_size = getattr(datamodule, "batch_size", "n/a")

        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            gpu_name = torch.cuda.get_device_name(0)
            n_gpu = torch.cuda.device_count()
            gpu_line = f"GPUs           : {n_gpu}x {gpu_name}"
        else:
            gpu_line = "GPUs           : none (CPU only)"

        max_epochs_str = str(max_epochs) if max_epochs is not None else "unlimited"
        max_steps_str = str(max_steps) if max_steps is not None else "unlimited"
        started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            "=" * 72,
            "  Molix Trainer — run starting",
            "=" * 72,
            f"  Started at     : {started_at}",
            f"  Host / Python  : {platform.node()} / Python {platform.python_version()}",
            f"  PyTorch        : {torch.__version__}",
            f"  {gpu_line}",
            "-" * 72,
            "  Model",
            f"    class        : {type(self.model).__name__}",
            f"    device       : {model_device}",
            f"    dtype        : {model_dtype}",
            f"    parameters   : {n_params:,} ({n_trainable:,} trainable)",
            "-" * 72,
            "  Precision",
            f"    ftype        : {config['ftype']}",
            f"    use_amp      : {use_amp}" + (f"  (amp_dtype={amp_dtype})" if use_amp else ""),
            f"    matmul       : {torch.get_float32_matmul_precision()}",
            "-" * 72,
            "  Optimization",
            f"    optimizer    : {opt_name}",
            f"    lr           : {lr_str}",
            f"    scheduler    : {sched_name}",
            f"    loss_fn      : {getattr(self.loss_fn, '__name__', type(self.loss_fn).__name__)}",
            "-" * 72,
            "  Data / Run",
            f"    datamodule   : {type(datamodule).__name__}",
            f"    batch_size   : {batch_size}",
            f"    max_epochs   : {max_epochs_str}",
            f"    max_steps    : {max_steps_str}",
            f"    eval_every   : "
            + (f"{self.eval_every_n_steps} steps" if self.eval_every_n_steps else "epoch-end"),
            f"    resume_epoch : {self.state.epoch}, global_step: {self.state.global_step}",
            f"    hooks        : {hooks_str}",
            "=" * 72,
        ]
        for line in lines:
            logger.info(line)

    def _log_summary_banner(
        self,
        elapsed: float,
        epochs_done: int,
        steps_done: int,
    ) -> None:
        """Print a LAMMPS-style summary banner after training ends."""
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        per_epoch = (elapsed / epochs_done) if epochs_done > 0 else 0.0
        per_step = (elapsed / steps_done) if steps_done > 0 else 0.0
        steps_per_sec = (steps_done / elapsed) if elapsed > 0 else 0.0
        finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        peak_mem_line = None
        if torch.cuda.is_available():
            try:
                peak = torch.cuda.max_memory_allocated() / (1024**3)
                peak_mem_line = f"    peak GPU mem : {peak:.2f} GiB"
            except Exception:
                peak_mem_line = None

        lines = [
            "=" * 72,
            "  Molix Trainer — run finished",
            "=" * 72,
            f"  Finished at    : {finished_at}",
            "  Timing",
            f"    wall time    : {elapsed_str}  ({elapsed:.2f} s)",
            f"    epochs done  : {epochs_done}",
            f"    steps done   : {steps_done}",
            f"    s / epoch    : {per_epoch:.3f}",
            f"    s / step     : {per_step:.4f}",
            f"    steps / s    : {steps_per_sec:.2f}",
            "  Final state",
            f"    epoch        : {self.state.epoch}",
            f"    global_step  : {self.state.global_step}",
        ]
        best = self.state.get("best_metric") if hasattr(self.state, "get") else None
        if best is not None:
            lines.append(f"    best_metric  : {best}")
        if peak_mem_line is not None:
            lines.append("  Memory")
            lines.append(peak_mem_line)
        lines.append("=" * 72)
        for line in lines:
            logger.info(line)

    def _run_eval_phase(self, datamodule: DataModuleProtocol) -> None:
        """Run evaluation phase (internal helper)."""
        prev_stage = self.state.stage
        self.state.set_stage(Stage.EVAL)
        self.model.eval()

        for batch in datamodule.val_dataloader():
            self._call_hooks("on_eval_batch_start", self, self.state, batch)
            outputs = self.eval_step.on_eval_batch(self, self.state, batch)
            self._call_hooks("on_eval_batch_end", self, self.state, batch, outputs)

        self.model.train()
        self.state.set_stage(prev_stage)
