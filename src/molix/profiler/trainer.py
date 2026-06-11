"""Trainer-loop overhead profiler.

Unlike :class:`~molix.profiler.module.ModuleProfiler` (which times a model in
isolation), :class:`TrainerProfiler` profiles the **Trainer's own per-step
machinery** — Step-protocol dispatch, hook calls, ``batch_to``, TrainState
writes, the optimizer/scheduler/eval-cadence bookkeeping — by driving a real
:class:`~molix.core.trainer.Trainer` over a near-zero-compute
:class:`~molix.profiler.mock.MockModel`. With model FLOPs out of the way, a
``cProfile`` window over ``Trainer.train`` attributes wall time to concrete
functions, surfacing framework hotspots.

Example::

    from molix.profiler import TrainerProfiler

    result = TrainerProfiler().run(n_steps=2000)
    result.print_report()        # per-step wall + top framework hotspots

    # price hook dispatch by adding hooks
    from molix.hooks import StepSpeedHook
    TrainerProfiler(hooks=[StepSpeedHook()]).run(n_steps=2000).print_report()
"""

from __future__ import annotations

import cProfile
import pstats
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from molix.profiler._utils import _fmt_table
from molix.profiler.mock import MockBatch, MockModel, mock_node_feature_loss


@dataclass
class TrainerResult:
    """Trainer-loop profiling results.

    Attributes:
        n_steps: Number of optimizer steps measured.
        device: Device the loop ran on.
        n_hooks: Number of hooks registered during the run.
        wall_ms_per_step: End-to-end wall-clock per step (no profiler attached).
        steps_per_sec: ``1000 / wall_ms_per_step``.
        hotspots: Per-step hotspot rows sorted by self-time, each a dict with
            ``func`` / ``self_us`` / ``cum_us`` / ``calls``.
        model_name: ``type(model).__name__``.
        data_description: Human-readable batch description.
    """

    n_steps: int
    device: str
    n_hooks: int
    wall_ms_per_step: float
    steps_per_sec: float
    hotspots: list[dict] = field(default_factory=list)
    model_name: str = "MockModel"
    data_description: str = ""

    def print_report(self) -> None:
        """Print a human-readable Trainer-overhead report to stdout."""
        print(
            f"\nTrainer loop  |  model={self.model_name}  |  device={self.device}  "
            f"|  hooks={self.n_hooks}  |  n_steps={self.n_steps}"
        )
        print(f"Data  : {self.data_description}")
        print("─" * 72)
        print(f"  Wall: {self.wall_ms_per_step:.4f} ms/step   ({self.steps_per_sec:,.0f} step/s)")
        print("\n  Top framework hotspots (per step, by self-time):")
        print(_fmt_table(self.hotspots, ["func", "self_us", "cum_us", "calls"], col_width=10))
        print("─" * 72)


class _ReplayDataModule:
    """One-epoch datamodule replaying a fixed batch *n_steps* times."""

    def __init__(self, batch, n_steps: int) -> None:
        self._batches = [batch] * n_steps

    def setup(self, stage: str = "fit") -> None:  # noqa: D102
        pass

    def on_epoch_start(self, epoch: int) -> None:  # noqa: D102
        pass

    def train_dataloader(self):  # noqa: D102
        return iter(self._batches)

    def val_dataloader(self):  # noqa: D102
        return iter(self._batches[:1])


class TrainerProfiler:
    """Profile the Trainer loop's per-step framework overhead.

    Args:
        model: A batch-consuming ``nn.Module``; defaults to
            :class:`~molix.profiler.mock.MockModel` (compute ≈ 0) so the
            measurement reflects loop machinery, not model FLOPs.
        loss_fn: ``loss_fn(predictions, batch) -> scalar``; defaults to
            :func:`~molix.profiler.mock.mock_node_feature_loss`.
        device: Device to run on (``"cpu"`` / ``"cuda"``).
        hooks: Hooks to register — pass the set you run with to price hook
            dispatch, or leave empty to measure the bare loop.
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        loss_fn: Callable | None = None,
        device: str | torch.device = "cpu",
        hooks: list | None = None,
    ) -> None:
        self.model = model if model is not None else MockModel()
        self.loss_fn = loss_fn if loss_fn is not None else mock_node_feature_loss
        self.device = torch.device(device)
        self.hooks = hooks if hooks is not None else []

    def _build_trainer(self):
        # Imported lazily so the profiler module stays importable without
        # pulling the full core.trainer graph until a run is requested.
        from molix.core.trainer import Trainer

        return Trainer(
            model=self.model,
            loss_fn=self.loss_fn,
            optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.0),
            hooks=list(self.hooks),
            device=str(self.device),
        )

    def run(
        self,
        n_steps: int = 2000,
        n_warmup: int = 50,
        batch=None,
        top: int = 15,
    ) -> TrainerResult:
        """Run the Trainer loop and attribute per-step wall time to functions.

        Args:
            n_steps: Optimizer steps to measure (one epoch).
            n_warmup: Warmup steps before the timed run.
            batch: A pre-built batch ``TensorDict``; defaults to a
                :class:`~molix.profiler.mock.MockBatch` on ``device``.
            top: Number of hotspot rows to keep.

        Returns:
            A :class:`TrainerResult`.
        """
        if batch is None:
            batch = MockBatch(n_atoms=32, n_edges=128, n_graphs=4, device=str(self.device))()
        desc = "MockBatch(n_atoms=32, n_edges=128, n_graphs=4)"

        if n_warmup > 0:
            self._build_trainer().train(datamodule=_ReplayDataModule(batch, n_warmup), max_epochs=1)

        # --- wall timing (no profiler attached) ---
        trainer = self._build_trainer()
        dm = _ReplayDataModule(batch, n_steps)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        trainer.train(datamodule=dm, max_epochs=1)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t0) * 1000.0 / n_steps

        # --- cProfile attribution (separate run; profiler perturbs absolute time) ---
        pr = cProfile.Profile()
        prof_trainer = self._build_trainer()
        prof_dm = _ReplayDataModule(batch, n_steps)
        pr.enable()
        prof_trainer.train(datamodule=prof_dm, max_epochs=1)
        pr.disable()

        hotspots = self._extract_hotspots(pr, n_steps, top)

        return TrainerResult(
            n_steps=n_steps,
            device=str(self.device),
            n_hooks=len(self.hooks),
            wall_ms_per_step=wall_ms,
            steps_per_sec=(1000.0 / wall_ms if wall_ms > 0 else 0.0),
            hotspots=hotspots,
            model_name=type(self.model).__name__,
            data_description=desc,
        )

    @staticmethod
    def _extract_hotspots(pr: cProfile.Profile, n_steps: int, top: int) -> list[dict]:
        """Top functions by total self-time, normalized per step."""
        stats = pstats.Stats(pr)
        rows: list[dict] = []
        for (fname, lineno, func), (_cc, nc, tt, ct, _callers) in stats.stats.items():
            short = fname.rsplit("/", 1)[-1]
            label = f"{short}:{lineno}({func})" if short else f"{func}"
            rows.append(
                {
                    "func": label[:48],
                    "self_us": f"{tt / n_steps * 1e6:.2f}",
                    "cum_us": f"{ct / n_steps * 1e6:.2f}",
                    "calls": f"{nc / n_steps:.1f}",
                    "_tt": tt,
                }
            )
        rows.sort(key=lambda r: r["_tt"], reverse=True)
        for r in rows:
            del r["_tt"]
        return rows[:top]
