"""PyTorch Profiler integration for performance analysis."""

from __future__ import annotations

from molix import logger as _logger_mod
from molix.core.hook import BaseHook

logger = _logger_mod.getLogger(__name__)


class ProfilerHook(BaseHook):
    """PyTorch Profiler integration for performance analysis.

    Profiles training performance and exports results as Chrome Trace Viewer
    format (``trace.json``) and optionally TensorBoard format. Supports
    artifact registration for molexp workflow integration.

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
        print_summary: bool = True,
        summary_row_limit: int = 15,
    ):
        from pathlib import Path

        self.output_dir = Path(output_dir)
        self.print_summary = print_summary
        self.summary_row_limit = summary_row_limit
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

        if self.activities is None:
            activities = [
                torch.profiler.ProfilerActivity.CPU,
            ]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
        else:
            activities = self.activities

        schedule = torch.profiler.schedule(
            wait=self.schedule_wait,
            warmup=self.schedule_warmup,
            active=self.schedule_active,
            repeat=self.schedule_repeat,
        )

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

        self.profiler.__enter__()
        logger.info(f"Started profiler with output_dir={self.output_dir}")

    def on_train_batch_end(self, trainer, state, batch, outputs):
        """Step profiler after each batch."""
        if self.profiler:
            self.profiler.step()

    def on_train_end(self, trainer, state):
        """Stop profiler and register artifacts."""
        if self.profiler:
            self.profiler.__exit__(None, None, None)
            logger.info("Stopped profiler")

        if self.register_artifacts and hasattr(trainer, "ctx"):
            ctx = trainer.ctx
            if ctx:
                trace_path = self.output_dir / "trace.json"
                if trace_path.exists():
                    ctx.save_artifact(
                        name="profiler_trace.json",
                        src=trace_path,
                    )
                    logger.info(f"Registered profiler trace as artifact: {trace_path}")

                if self.export_tensorboard:
                    tb_dir = self.output_dir / "tensorboard"
                    if tb_dir.exists():
                        ctx.save_artifact(
                            name="profiler_tensorboard",
                            src=tb_dir,
                        )
                        logger.info(f"Registered profiler TensorBoard logs as artifact: {tb_dir}")

    def _print_summary(self, prof):
        """Print an op-level summary to stdout so the bottleneck is visible
        without opening the Chrome trace.

        Shows the top ops by self-CUDA time plus aggregate CPU/CUDA totals and
        the total op-call count over the profiled window. A CPU total that
        dwarfs the CUDA total (and a high op count) is the signature of a
        launch-bound step — the GPU is idle waiting on CPU kernel dispatch.
        """
        import torch

        ka = prof.key_averages()
        sort_key = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
        logger.info(
            "Profiler op summary (top %d by %s):\n%s",
            self.summary_row_limit,
            sort_key,
            ka.table(sort_by=sort_key, row_limit=self.summary_row_limit),
        )

        def _us(attr: str) -> float:
            return sum(getattr(e, attr, 0) for e in ka) / 1e3  # → ms

        cpu_ms = _us("self_cpu_time_total")
        cuda_ms = _us("self_device_time_total") or _us("self_cuda_time_total")
        n_calls = sum(e.count for e in ka)
        verdict = "LAUNCH-BOUND (GPU idle on CPU dispatch)" if cpu_ms > cuda_ms else "compute-bound"
        logger.info(
            "Totals over window: CPU self=%.1f ms, CUDA self=%.1f ms, op-calls=%d  →  %s",
            cpu_ms,
            cuda_ms,
            n_calls,
            verdict,
        )

    def _on_trace_ready(self, prof):
        """Callback when trace is ready - export to files."""
        if self.print_summary:
            try:
                self._print_summary(prof)
            except Exception as exc:  # never let a summary error abort training
                logger.warning("Profiler summary failed: %s", exc)

        if self.export_chrome_trace:
            trace_path = self.output_dir / "trace.json"
            prof.export_chrome_trace(str(trace_path))
            logger.info(f"Exported Chrome Trace to {trace_path}")

        if self.export_tensorboard:
            tb_dir = self.output_dir / "tensorboard"
            tb_dir.mkdir(parents=True, exist_ok=True)

            if self.with_stack:
                prof.export_stacks(str(tb_dir / "profiler.pt.trace.json"), "self_cuda_time_total")
                logger.info(f"Exported TensorBoard profiler stacks to {tb_dir}")
            else:
                logger.warning(
                    "Skipping export_stacks() because with_stack=False. "
                    "Set with_stack=True to enable stack trace export."
                )
