"""Trainer implementation for Molix."""

from __future__ import annotations

import copy
import platform
import time
from collections.abc import Callable
from contextlib import nullcontext
from datetime import datetime, timedelta
from pathlib import Path

import torch
import torch.nn as nn

from molix import logger as _logger_mod
from molix.config import config
from molix.core.checkpoint import Checkpoint, CheckpointBackend, TorchSaveBackend
from molix.core.hook import Hook
from molix.core.state import Stage, TrainState, resolve
from molix.core.steps import DefaultEvalStep, DefaultTrainStep, Step, batch_to
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
        accumulate_grad_batches: int = 1,
        resume_from_checkpoint: str | Path | None = None,
        checkpoint_backend: CheckpointBackend | None = None,
        device: str | torch.device | None = None,
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
            eval_every_n_steps: Run evaluation every N *optimizer* steps. When
                   set, this is the exclusive eval cadence — epoch-end eval is
                   suppressed so short epochs (e.g. revmd17's ~30-step epoch)
                   do not silently override the configured schedule. If None
                   (default), only epoch-end eval runs.
                   Must be > 0 if provided.
            accumulate_grad_batches: Number of micro-batches whose gradients
                   are accumulated before one optimizer step. ``1`` (default)
                   = step every batch. With ``N``, the loss is scaled by
                   ``1/N`` and the optimizer / LR scheduler / ``global_step``
                   advance once per ``N`` micro-batches — so ``global_step``,
                   ``max_steps`` and ``eval_every_n_steps`` all count optimizer
                   steps, letting you trade a launch-bound small physical batch
                   for a larger effective batch without the memory cost.
                   Must be > 0.
            resume_from_checkpoint: Path to a checkpoint file to resume from,
                   or ``"auto"`` to detect torchrun elastic snapshots.
            checkpoint_backend: Backend for checkpoint I/O. Defaults to
                   :class:`TorchSaveBackend`.
            device: Target device for the model (e.g. ``"cuda"``, ``"cuda:0"``,
                   ``"cpu"``). When set, the model is moved to this device at the
                   start of :meth:`train`. Each batch is then automatically moved
                   to the same device by the trainer loop (before hooks and the
                   step see it), so hooks and steps always observe a
                   device-aligned batch. If ``None`` (default), the model is
                   left on its current device and batches are still moved to
                   whatever device the model currently sits on.

        Raises:
            ValueError: If eval_every_n_steps is <= 0
        """
        if eval_every_n_steps is not None and eval_every_n_steps <= 0:
            raise ValueError(f"eval_every_n_steps must be > 0, got {eval_every_n_steps}")
        if accumulate_grad_batches <= 0:
            raise ValueError(f"accumulate_grad_batches must be > 0, got {accumulate_grad_batches}")
        self.eval_every_n_steps = eval_every_n_steps
        self.accumulate_grad_batches = accumulate_grad_batches
        # Monotonic micro-batch counter driving accumulation-window boundaries.
        self._micro_step = 0
        self.device = torch.device(device) if device is not None else None

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer_factory(model.parameters())

        # lr scheduler
        self.lr_scheduler = lr_scheduler_factory(self.optimizer) if lr_scheduler_factory else None

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

        Delegates to :meth:`molix.config.MolnexConfig.set_precision` (which
        writes ``ftype`` / ``use_amp`` / ``amp_dtype`` into the global
        :data:`molix.config.config`), then (re)creates ``self.scaler`` to
        match ``config["use_amp"]`` and updates the checkpoint aggregate so
        the new scaler is saved/loaded.

        Must be called **before** ``trainer.train()``. For ``"fp64"`` the
        model must additionally be constructed (or cast) with the new
        ``ftype``, because layers bake in ``config["ftype"]`` at ``__init__``.

        Args:
            mode: One of ``"fp32"``, ``"fp64"``, ``"fp16-mixed"``,
                ``"bf16-mixed"``.

        Raises:
            ValueError: If ``mode`` is not a supported preset.
        """
        config.set_precision(mode)
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
            raise ValueError("At least one of max_epochs or max_steps must be specified")
        if max_epochs is not None and max_epochs <= 0:
            raise ValueError(f"max_epochs must be > 0, got {max_epochs}")
        if max_steps is not None and max_steps <= 0:
            raise ValueError(f"max_steps must be > 0, got {max_steps}")
        return self._train(datamodule, max_epochs, max_steps)

    def _call_hooks(self, hook_name: str, *args, **kwargs) -> None:
        """Call ``hook_name`` on every registered hook.

        Hook errors propagate to the caller. Swallowing them silently hid a
        NaNGuardHook fatal signal behind 15k lines of repeated tracebacks;
        fail-loud is the safer default — a hook that needs to tolerate its
        own errors should catch them itself.
        """
        for hook in self.hooks:
            method = getattr(hook, hook_name, None)
            if method is not None and callable(method):
                try:
                    method(*args, **kwargs)
                except Exception:
                    logger.error(
                        f"Fatal error in hook {hook.__class__.__name__}.{hook_name}",
                        exc_info=True,
                    )
                    raise

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
            snapshot_dir = os.environ.get("TORCHELASTIC_SNAPSHOT_DIR", "./snapshots")
            if run_id:
                snapshot_path = Path(snapshot_dir) / run_id / "snapshot.pt"
                if snapshot_path.exists():
                    return snapshot_path
            return None

        p = Path(path)
        return p if p.exists() else None

    def smoke(self, datamodule: DataModuleProtocol, steps: int = 3) -> None:
        """Run an unlogged smoke pass — call this **before** :meth:`train`.

        Deliberately a standalone, caller-invoked method, NOT folded into
        :meth:`train`: the smoke pass is a separate concern (pay one-off costs +
        sanity-check data) and keeping it out of the training loop avoids
        entangling its state snapshot/restore and compile side effects with the
        recorded run. Typical use::

            trainer.smoke(datamodule, steps=3)
            trainer.train(datamodule, max_steps=...)

        What it does:

        - **Compilation / autotune.** A few full train steps (forward → loss →
          backward → optimizer step) on real batches trigger ``torch.compile``
          graph capture and cuDNN / inductor autotuning, so the first *logged*
          step — and the ``step_per_second`` it feeds — reflects steady-state
          throughput instead of being dominated by one-off compile latency.
        - **Real-data check.** Running actual dataloader batches surfaces
          shape / dtype / NaN problems immediately, before training starts. A
          non-finite warmup loss aborts loudly.

        The pass fires **no hooks** and does **not** advance ``global_step``, so
        nothing reaches the journal, metrics, checkpoints or the step counters.
        Model parameters, optimizer state, AMP-scaler state and the (CPU + CUDA)
        RNG state are snapshotted before and restored after, so a subsequent
        :meth:`train` is bit-identical to one launched without warmup — step-0
        weights and data ordering are unchanged. A best-effort eval-graph warmup
        compiles the ``model.eval()`` path too; any failure there is logged and
        skipped (it never blocks training).

        Args:
            datamodule: Provides ``train_dataloader`` (and optionally
                ``val_dataloader``) supplying real batches to warm up on.
            steps: Number of full train steps to run. ``<= 0`` is a no-op.
        """
        if steps <= 0:
            return

        if self.device is not None:
            self.model = self.model.to(self.device)
            self._checkpoint.model = self.model

        logger.info(
            f"Smoke: {steps} step(s) on real data — "
            "compiling + checking (unlogged; state restored after)."
        )
        t0 = time.perf_counter()

        # Snapshot every mutable piece so the warmup is invisible to the real run.
        model_snap = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        opt_snap = copy.deepcopy(self.optimizer.state_dict())
        scaler_snap = copy.deepcopy(self.scaler.state_dict()) if self.scaler is not None else None
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        model_device = next(self.model.parameters()).device
        amp_enabled = bool(config["use_amp"])
        amp_dtype = config["amp_dtype"]
        device_type = model_device.type

        try:
            self.model.train()
            loader = datamodule.train_dataloader()
            it = iter(loader)
            for i in range(steps):
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(loader)
                    batch = next(it)
                batch = batch_to(batch, device=model_device)

                ctx = (
                    torch.amp.autocast(device_type, dtype=amp_dtype)
                    if amp_enabled
                    else nullcontext()
                )
                self.optimizer.zero_grad(set_to_none=True)
                with ctx:
                    predictions = self.model(batch)
                    loss = self.loss_fn(predictions, batch)
                if not torch.isfinite(loss).all():
                    raise RuntimeError(
                        f"Smoke warmup produced a non-finite loss ({loss.detach()}) on "
                        f"warmup step {i}; aborting before the real run starts."
                    )
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

            # Best-effort eval-graph warmup (a separate compile under eval()).
            val_dl = getattr(datamodule, "val_dataloader", None)
            if callable(val_dl):
                try:
                    self.model.eval()
                    vbatch = batch_to(next(iter(val_dl())), device=model_device)
                    self.eval_step.on_eval_batch(self, self.state, vbatch)
                except Exception as exc:  # noqa: BLE001
                    logger.info(f"Smoke warmup: eval-graph warmup skipped ({exc!r}).")
        finally:
            # Restore: the real run must start from the exact pre-warmup state.
            self.model.load_state_dict(model_snap)
            self.optimizer.load_state_dict(opt_snap)
            if self.scaler is not None and scaler_snap is not None:
                self.scaler.load_state_dict(scaler_snap)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            self.optimizer.zero_grad(set_to_none=True)
            self.model.train()

        logger.info(f"Smoke done in {time.perf_counter() - t0:.1f}s (state restored).")

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

        if self.device is not None:
            self.model = self.model.to(self.device)
            self._checkpoint.model = self.model

        # ``setup`` / ``on_epoch_start`` are optional on the datamodule
        # contract so ad-hoc / test harnesses don't need to spell out empty
        # method bodies just to satisfy the Trainer.
        setup = getattr(datamodule, "setup", None)
        if callable(setup):
            setup("fit")

        if self._resume_from_checkpoint is not None:
            self._load_checkpoint(self._resume_from_checkpoint)

        self._log_setup_banner(datamodule, max_epochs, max_steps)

        # AMP without grad clip → occasional outlier-gradient metric spikes.
        if config["use_amp"]:
            from molix.hooks.training import GradClipHook as _GradClipHook

            if not any(isinstance(h, _GradClipHook) for h in self.hooks):
                logger.warning(
                    "AMP is enabled but no GradClipHook is registered. "
                    "Consider adding GradClipHook(max_norm=1.0) to hooks."
                )

        self._call_hooks("on_train_start", self, self.state)

        epoch = self.state.epoch
        start_epoch = epoch
        start_global_step = self.state.global_step
        start_time = time.perf_counter()
        step_limit_reached = False

        while epoch < epoch_limit and not step_limit_reached:
            on_epoch_start = getattr(datamodule, "on_epoch_start", None)
            if callable(on_epoch_start):
                on_epoch_start(epoch)
            self._call_hooks("on_epoch_start", self, self.state)

            # Training phase
            self.state.set_stage(Stage.TRAIN)
            self.model.train()

            model_device = next(self.model.parameters()).device
            accum = self.accumulate_grad_batches
            for batch in datamodule.train_dataloader():
                batch = batch_to(batch, device=model_device)
                self._call_hooks("on_train_batch_start", self, self.state, batch)
                outputs = self.train_step.on_train_batch(self, self.state, batch)
                self._call_hooks("on_train_batch_end", self, self.state, batch, outputs)

                self._micro_step += 1
                # Optimizer-step boundary: only here do global_step, the LR
                # scheduler, the eval cadence and the step limit advance, so
                # they all count optimizer steps rather than micro-batches.
                if self._micro_step % accum != 0:
                    continue

                self.state.increment_step()
                self.state.steps_since_last_eval += 1

                # Step step-based schedulers per optimizer step. Skip the
                # advance when an AMP inf/nan-skipped step didn't apply, and
                # skip ``ReduceLROnPlateau`` (metric-driven, stepped at epoch
                # end after evaluation — see below).
                if (
                    self.lr_scheduler is not None
                    and not isinstance(
                        self.lr_scheduler,
                        torch.optim.lr_scheduler.ReduceLROnPlateau,
                    )
                    and outputs.get("optimizer_applied", True)
                ):
                    self.lr_scheduler.step()

                if (
                    self.eval_every_n_steps is not None
                    and self.state.steps_since_last_eval >= self.eval_every_n_steps
                ):
                    self._run_eval_phase(datamodule)
                    self.state.steps_since_last_eval = 0

                if self.state.global_step >= step_limit:
                    step_limit_reached = True
                    break

            # Epoch-end validation. When eval_every_n_steps is set, that
            # schedule is *exclusive* — epoch-end eval is suppressed so a
            # short epoch (e.g. revmd17 aspirin's 30-step epoch under
            # batch_size=32) does not silently override the configured
            # cadence. With eval_every_n_steps=None (default), epoch-end
            # eval is the only schedule and always runs.
            if self.eval_every_n_steps is None and self.state.steps_since_last_eval != 0:
                self._run_eval_phase(datamodule)
                self.state.steps_since_last_eval = 0

            # ReduceLROnPlateau is metric-driven and only advances at epoch
            # boundaries. Pull the metric the Checkpoint aggregate is tracking
            # (``best_metric_name``, defaults to ``eval/loss``) so the
            # scheduler sees the same signal the user's checkpoint hook does.
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_name = self._checkpoint.best_metric_name
                metric = resolve(self.state, metric_name)
                if metric is not None:
                    self.lr_scheduler.step(float(metric))

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
        cuda_graphs: bool = False,
        backend: str = "inductor",
        fullgraph: bool = False,
        dynamic: bool | None = None,
        mode: str | None = None,
    ) -> "Trainer":
        """Compile the model with ``torch.compile`` for optimized execution.

        Can be chained after construction::

            # Force training (energy → functorch force → loss is one fullgraph):
            # the benchmarked winning config — ~10x eager on PiNet/GH200.
            trainer = Trainer(model, ...).compile(cuda_graphs=True)

            # Custom config:
            trainer = Trainer(model, ...).compile(mode="max-autotune")

        Args:
            cuda_graphs: Apply :data:`molix.compile.CUDA_GRAPH_PRESET`
                (``backend="inductor", fullgraph=True, dynamic=False,
                mode="reduce-overhead"``) — inductor fusion + CUDA graphs, the
                fastest force-training config. Overrides the four params below.
                REQUIRES static shapes: register
                :class:`molix.data.tasks.PadMolecularBatch` and set the train
                loader's ``drop_last=True`` (see
                ``docs/molix/explanation/throughput-and-compilation.md``).
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
            cuda_graphs=cuda_graphs,
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
            "    eval_every   : "
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

    def evaluate(self, datamodule: DataModuleProtocol) -> dict[str, Any]:
        """Run one full evaluation phase and return the published eval metrics.

        Runs the *same* eval path the train loop uses (:meth:`_run_eval_phase` ->
        ``on_eval_step_complete``), so the registered eval hooks (``MetricsHook``,
        ...) populate ``state["eval"]`` exactly as they do mid-training — no
        separate eval loop, model reload, or metric re-implementation at the call
        site. Evaluates the current in-memory model on ``datamodule.val_dataloader()``.

        Args:
            datamodule: Provides ``val_dataloader()`` with the eval batches.

        Returns:
            A shallow copy of ``state["eval"]`` — the metric scalars published by
            the eval hooks (e.g. ``{"E_MAE": ..., "F_MAE": ..., "loss": ...}``),
            in the model's native units. Empty if no eval hook published anything.
        """
        self._run_eval_phase(datamodule)
        evals = self.state.get("eval")
        return dict(evals) if evals else {}

    def _run_eval_phase(self, datamodule: DataModuleProtocol) -> None:
        """Run evaluation phase and fire ``on_eval_step_complete``.

        Used by both step-based eval (inside the train loop) and the
        epoch-end val pass, so eval-publishing hooks (MetricsHook,
        TensorBoardHook) fire exactly once per eval phase regardless
        of which trigger caused it.
        """
        prev_stage = self.state.stage
        self.state.set_stage(Stage.EVAL)
        self.model.eval()

        # Fire BEFORE the batch loop so accumulating hooks reset their
        # eval-side buffers; makes every eval phase self-contained
        # regardless of trigger (step-based or epoch-end).
        self._call_hooks("on_eval_phase_start", self, self.state)

        model_device = next(self.model.parameters()).device
        for batch in datamodule.val_dataloader():
            batch = batch_to(batch, device=model_device)
            self._call_hooks("on_eval_batch_start", self, self.state, batch)
            outputs = self.eval_step.on_eval_batch(self, self.state, batch)
            self._call_hooks("on_eval_batch_end", self, self.state, batch, outputs)

        self._call_hooks("on_eval_step_complete", self, self.state)

        self.model.train()
        self.state.set_stage(prev_stage)
