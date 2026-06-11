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
        hotspots: Per-step framework-hotspot rows sorted by baseline-
            subtracted self-time, each a dict with ``func`` / ``added_us``
            (Trainer self-time minus raw-loop baseline) / ``gross_us``
            (raw Trainer self-time) / ``calls``.
        model_name: ``type(model).__name__``.
        data_description: Human-readable batch description.
        baseline_ms_per_step: Raw-loop (no Trainer) wall per step.
        overhead_ms_per_step: ``wall_ms_per_step - baseline_ms_per_step``.
    """

    n_steps: int
    device: str
    n_hooks: int
    wall_ms_per_step: float
    steps_per_sec: float
    hotspots: list[dict] = field(default_factory=list)
    model_name: str = "MockModel"
    data_description: str = ""
    baseline_ms_per_step: float = 0.0
    overhead_ms_per_step: float = 0.0

    def print_report(self) -> None:
        """Print a human-readable Trainer-overhead report to stdout."""
        print(
            f"\nTrainer loop  |  model={self.model_name}  |  device={self.device}  "
            f"|  hooks={self.n_hooks}  |  n_steps={self.n_steps}"
        )
        print(f"Data  : {self.data_description}")
        print("─" * 72)
        sps = f"{self.steps_per_sec:,.0f}"
        pct = (
            self.overhead_ms_per_step / self.wall_ms_per_step * 100.0
            if self.wall_ms_per_step > 0
            else 0.0
        )
        oh = f"{self.overhead_ms_per_step:.4f}"
        print(f"  Raw loop (baseline) : {self.baseline_ms_per_step:.4f} ms/step")
        print(f"  Trainer loop        : {self.wall_ms_per_step:.4f} ms/step   ({sps} step/s)")
        print(f"  Trainer overhead    : {oh} ms/step   ({pct:.1f}% of loop)")
        print("\n  Framework hotspots — Trainer self-time minus raw-loop baseline (per step):")
        print(_fmt_table(self.hotspots, ["func", "added_us", "gross_us", "calls"], col_width=10))
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

        # Warm up both the raw-loop baseline and the Trainer path.
        if n_warmup > 0:
            self._raw_loop(self._fresh_model(), [batch] * n_warmup)
            self._build_trainer().train(datamodule=_ReplayDataModule(batch, n_warmup), max_epochs=1)

        # baseline: irreducible per-step work, no Trainer/Step/hook machinery
        base_wall, base_pr = self._timed_raw_loop(batch, n_steps)
        # Trainer loop wall + cProfile (separate runs; profiler perturbs time)
        trainer_wall = self._timed_trainer(batch, n_steps)
        trainer_pr = self._profiled_trainer(batch, n_steps)

        # Subtract the baseline per-function self-time so the table shows only
        # what the Trainer *adds* over a bare loop — run_backward, the model
        # forward, optimizer.step appear in both and cancel, leaving framework
        # machinery (batch_to, hooks, the Step wrapper, state writes, _train).
        hotspots = self._extract_hotspots(trainer_pr, base_pr, n_steps, top)
        overhead = max(0.0, trainer_wall - base_wall)

        return TrainerResult(
            n_steps=n_steps,
            device=str(self.device),
            n_hooks=len(self.hooks),
            wall_ms_per_step=trainer_wall,
            steps_per_sec=(1000.0 / trainer_wall if trainer_wall > 0 else 0.0),
            hotspots=hotspots,
            model_name=type(self.model).__name__,
            data_description=desc,
            baseline_ms_per_step=base_wall,
            overhead_ms_per_step=overhead,
        )

    def _fresh_model(self) -> nn.Module:
        """Model on the target device for the baseline loop.

        Re-instantiates the default :class:`MockModel` (cheap); otherwise
        reuses the configured model so the baseline shares its forward path.
        """
        if isinstance(self.model, MockModel):
            return MockModel(n_features=self.model.n_features).to(self.device)
        return self.model

    def _raw_loop(self, model: nn.Module, batches: list) -> None:
        """The irreducible step: zero_grad / forward / loss / backward / step."""
        opt = torch.optim.SGD(model.parameters(), lr=0.0)
        model.train()
        for batch in batches:
            opt.zero_grad()
            loss = self.loss_fn(model(batch), batch)
            loss.backward()
            opt.step()

    def _timed_raw_loop(self, batch, n_steps: int) -> tuple[float, cProfile.Profile]:
        model = self._fresh_model()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        self._raw_loop(model, [batch] * n_steps)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) * 1000.0 / n_steps
        pr = cProfile.Profile()
        pr.enable()
        self._raw_loop(self._fresh_model(), [batch] * n_steps)
        pr.disable()
        return wall, pr

    def _timed_trainer(self, batch, n_steps: int) -> float:
        trainer = self._build_trainer()
        dm = _ReplayDataModule(batch, n_steps)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        trainer.train(datamodule=dm, max_epochs=1)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0 / n_steps

    def _profiled_trainer(self, batch, n_steps: int) -> cProfile.Profile:
        pr = cProfile.Profile()
        trainer = self._build_trainer()
        dm = _ReplayDataModule(batch, n_steps)
        pr.enable()
        trainer.train(datamodule=dm, max_epochs=1)
        pr.disable()
        return pr

    @staticmethod
    def _extract_hotspots(
        trainer_pr: cProfile.Profile,
        base_pr: cProfile.Profile,
        n_steps: int,
        top: int,
    ) -> list[dict]:
        """Top functions by *baseline-subtracted* self-time, per step.

        ``added`` = Trainer self-time − raw-loop self-time for the same
        ``(file, line, function)``. Trainer-only functions keep their full
        self-time; functions common to both (autograd, forward, optimizer)
        largely cancel and fall out of the top rows.
        """
        base_self = {
            key: tt for key, (_cc, _nc, tt, _ct, _cl) in pstats.Stats(base_pr).stats.items()
        }
        rows: list[dict] = []
        for (fname, lineno, func), (_cc, nc, tt, _ct, _cl) in pstats.Stats(
            trainer_pr
        ).stats.items():
            added = tt - base_self.get((fname, lineno, func), 0.0)
            if added <= 0:
                continue
            short = fname.rsplit("/", 1)[-1]
            label = f"{short}:{lineno}({func})" if short else f"{func}"
            rows.append(
                {
                    "func": label[:48],
                    "added_us": f"{added / n_steps * 1e6:.2f}",
                    "gross_us": f"{tt / n_steps * 1e6:.2f}",
                    "calls": f"{nc / n_steps:.1f}",
                    "_added": added,
                }
            )
        rows.sort(key=lambda r: r["_added"], reverse=True)
        for r in rows:
            del r["_added"]
        return rows[:top]
