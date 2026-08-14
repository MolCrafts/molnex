"""Unit tests for :mod:`molix.hooks.audit`.

Mirrors ``src/molix/hooks/audit.py``. Unit under test:
:class:`GradientBalanceHook` — the cadence wrapper that fires a
:class:`~molix.core.audit.TaskGradientProbe` at initialization and at the
configured optimizer steps, and publishes the shared-group imbalance into
``state["train"]`` through the nested write form only.

The model fixture is the same analytic shared-trunk pair used by
``tests/test_molix/test_core/test_audit.py`` (trunk ``w=2.0``, two unit
heads, ``x = [[1.0]]``, ``L_a = a**2``, ``L_b = 100*b**2``), whose closed
form gives ``I_trunk = 400/4 = 100.0`` with both heads task-specific. It is
restated here rather than shared so this module stands alone.
"""

from __future__ import annotations

from typing import Mapping

import pytest
import torch
import torch.nn as nn

import molix.hooks
import molix.hooks.audit
from molix.core.audit import TaskGradientProbe
from molix.core.state import TrainState
from molix.hooks import GradientBalanceHook

#: Absolute tolerance for the analytic golden imbalance.
TOL = 1e-9


class _TwoTaskModel(nn.Module):
    """Shared trunk feeding two single-weight task heads (float64)."""

    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Linear(1, 1, bias=False, dtype=torch.float64)
        self.head_a = nn.Linear(1, 1, bias=False, dtype=torch.float64)
        self.head_b = nn.Linear(1, 1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.trunk.weight.fill_(2.0)
            self.head_a.weight.fill_(1.0)
            self.head_b.weight.fill_(1.0)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return ``{"a": head_a(trunk(x)), "b": head_b(trunk(x))}``."""
        hidden = self.trunk(batch["x"])
        return {"a": self.head_a(hidden), "b": self.head_b(hidden)}


class _TrainerStub:
    """Minimal stand-in for :class:`molix.core.trainer.Trainer`."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model


def _loss_a(
    predictions: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    """``L_a = a**2`` — golden 4.0 at ``a = 2.0``."""
    del batch
    return (predictions["a"] ** 2).sum()


def _loss_b(
    predictions: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    """``L_b = 100 * b**2`` — golden 400.0 at ``b = 2.0``."""
    del batch
    return 100.0 * (predictions["b"] ** 2).sum()


@pytest.fixture
def trainer() -> _TrainerStub:
    """Trainer stub exposing only the model the probe shares."""
    return _TrainerStub(_TwoTaskModel())


@pytest.fixture
def batches() -> list[tuple[str, dict[str, torch.Tensor]]]:
    """The fixed ``(task, batch)`` stream reused at every probe point."""
    sample = {"x": torch.tensor([[1.0]], dtype=torch.float64)}
    return [("a", sample), ("b", sample)]


@pytest.fixture
def hook(
    trainer: _TrainerStub, batches: list[tuple[str, dict[str, torch.Tensor]]]
) -> GradientBalanceHook:
    """Hook at the default cadence ``at_steps=(0, 100)``."""
    probe = TaskGradientProbe(trainer.model, {"a": _loss_a, "b": _loss_b})
    return GradientBalanceHook(probe, batches)


class TestGradientBalanceHook:
    """Cadence + nested-scalar contract for the gradient-balance audit."""

    def test_on_train_start_records_step_zero_snapshot(
        self, hook: GradientBalanceHook, trainer: _TrainerStub
    ) -> None:
        """The initialization probe point is labelled ``step_0``."""
        hook.on_train_start(trainer, TrainState())
        assert [entry.label for entry in hook.report.snapshots] == ["step_0"]

    def test_on_train_batch_end_records_nothing_before_configured_step(
        self, hook: GradientBalanceHook, trainer: _TrainerStub
    ) -> None:
        """Steps 1..99 are not probe points at ``at_steps=(0, 100)``."""
        state = TrainState()
        hook.on_train_start(trainer, state)
        for step in range(1, 100):
            state["global_step"] = step
            hook.on_train_batch_end(trainer, state, batch=None, outputs=None)
        assert [entry.label for entry in hook.report.snapshots] == ["step_0"]

    def test_on_train_batch_end_records_configured_step_once(
        self, hook: GradientBalanceHook, trainer: _TrainerStub
    ) -> None:
        """Step 100 adds exactly one further snapshot, and no later step does."""
        state = TrainState()
        hook.on_train_start(trainer, state)
        for step in range(1, 151):
            state["global_step"] = step
            hook.on_train_batch_end(trainer, state, batch=None, outputs=None)
        assert [entry.label for entry in hook.report.snapshots] == ["step_0", "step_100"]

    def test_on_train_start_writes_nested_grad_imbalance(
        self, hook: GradientBalanceHook, trainer: _TrainerStub
    ) -> None:
        """The scalar lands inside the ``train`` namespace sub-dict."""
        state = TrainState()
        hook.on_train_start(trainer, state)
        assert "grad_imbalance" in state["train"]

    def test_grad_imbalance_is_max_shared_group_imbalance(
        self, hook: GradientBalanceHook, trainer: _TrainerStub
    ) -> None:
        """Only ``trunk`` is shared, so the published scalar is ``I_trunk``."""
        state = TrainState()
        hook.on_train_start(trainer, state)
        assert state["train"]["grad_imbalance"] == pytest.approx(100.0, abs=TOL)

    def test_scalar_keys_declares_the_nested_train_path(self, hook: GradientBalanceHook) -> None:
        """Container hooks discover the column via ``scalar_keys``."""
        assert hook.scalar_keys == (("train", "grad_imbalance"),)

    def test_hook_is_reexported_from_molix_hooks(self) -> None:
        """``from molix.hooks import GradientBalanceHook`` resolves the impl."""
        assert GradientBalanceHook is molix.hooks.audit.GradientBalanceHook
        assert "GradientBalanceHook" in molix.hooks.__all__
