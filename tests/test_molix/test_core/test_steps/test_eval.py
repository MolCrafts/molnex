"""Tests for the ``compute_loss`` seam of :class:`molix.core.steps.eval.DefaultEvalStep`.

Grad-context semantics (``no_grad``) and AMP behaviour are covered by
``tests/test_molix/test_core/test_step_protocol.py``; this module targets
only the seam that lets a subclass change *what* is computed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import pytest
import torch
import torch.nn as nn

from molix.config import config
from molix.core.state import TrainState
from molix.core.steps.eval import DefaultEvalStep
from molix.core.trainer import Trainer

Batch = Mapping[str, Mapping[str, torch.Tensor]]
Predictions = dict[str, torch.Tensor]
LossFn = Callable[[Predictions, Batch], torch.Tensor]


@pytest.fixture(autouse=True)
def _reset_precision() -> None:
    """Precision is a global singleton — restore fp32 around every test."""
    config.set_precision("fp32")
    yield
    config.set_precision("fp32")


class _StubModel(nn.Module):
    """Positions-dependent stub encoder: ``pos`` -> ``{"energy": (N,)}``."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, batch: Batch) -> Predictions:
        return {"energy": self.linear(batch["atoms"]["pos"]).squeeze(-1)}


class _RecordingLoss:
    """Records its call arguments and returns a parameterized scalar."""

    def __init__(self) -> None:
        self.calls: list[tuple[Predictions, Batch]] = []

    def __call__(self, predictions: Predictions, batch: Batch) -> torch.Tensor:
        self.calls.append((predictions, batch))
        return predictions["energy"].square().mean()


class _SpyEvalStep(DefaultEvalStep):
    """Records every ``compute_loss`` call and the autocast state it saw."""

    def __init__(self, *, no_grad: bool = False) -> None:
        super().__init__(no_grad=no_grad)
        self.calls: list[tuple[Batch, Predictions]] = []
        self.autocast_flags: list[bool] = []

    def compute_loss(
        self,
        trainer: Trainer,
        state: TrainState,
        batch: Batch,
        predictions: Predictions,
    ) -> torch.Tensor:
        self.calls.append((batch, predictions))
        self.autocast_flags.append(torch.is_autocast_enabled("cpu"))
        return super().compute_loss(trainer, state, batch, predictions)


def _batch() -> dict[str, dict[str, torch.Tensor]]:
    pos = torch.tensor(
        [(0.0, 0.0, 0.0), (1.0, 0.5, -0.5), (0.25, -0.75, 0.5)],
        dtype=torch.float32,
    )
    return {"atoms": {"pos": pos}}


def _make_trainer(*, loss_fn: LossFn | None, eval_step: object | None = None) -> Trainer:
    torch.manual_seed(0)
    return Trainer(
        model=_StubModel(),
        loss_fn=loss_fn,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=0.1),
        eval_step=eval_step,
    )


class TestDefaultEvalStep:
    """The ``compute_loss`` seam: default body, assertion, and call site."""

    def test_compute_loss_delegates_to_trainer_loss_fn(self) -> None:
        loss_fn = _RecordingLoss()
        trainer = _make_trainer(loss_fn=loss_fn)
        batch = _batch()
        predictions = trainer.model(batch)

        loss = DefaultEvalStep().compute_loss(trainer, TrainState(), batch, predictions)

        assert len(loss_fn.calls) == 1
        assert loss_fn.calls[0][0] is predictions
        assert loss_fn.calls[0][1] is batch
        assert torch.equal(loss, predictions["energy"].square().mean())

    def test_compute_loss_asserts_without_a_loss_fn(self) -> None:
        trainer = _make_trainer(loss_fn=None)
        batch = _batch()
        predictions = trainer.model(batch)

        with pytest.raises(AssertionError):
            DefaultEvalStep().compute_loss(trainer, TrainState(), batch, predictions)

    def test_on_eval_batch_computes_the_loss_through_the_seam(self) -> None:
        step = _SpyEvalStep()
        trainer = _make_trainer(loss_fn=_RecordingLoss(), eval_step=step)
        batch = _batch()

        outputs = step.on_eval_batch(trainer, TrainState(), batch)

        assert len(step.calls) == 1
        assert step.calls[0][0] is batch
        assert step.calls[0][1] is outputs["predictions"]

    def test_seam_runs_outside_autocast_without_amp(self) -> None:
        step = _SpyEvalStep()
        trainer = _make_trainer(loss_fn=_RecordingLoss(), eval_step=step)

        step.on_eval_batch(trainer, TrainState(), _batch())

        assert step.autocast_flags == [bool(config["use_amp"])]
        assert step.autocast_flags == [False]

    def test_seam_runs_inside_autocast_with_amp(self) -> None:
        step = _SpyEvalStep()
        trainer = _make_trainer(loss_fn=_RecordingLoss(), eval_step=step)
        trainer.set_precision("bf16-mixed")

        step.on_eval_batch(trainer, TrainState(), _batch())

        assert step.autocast_flags == [bool(config["use_amp"])]
        assert step.autocast_flags == [True]

    def test_seam_sees_the_configured_grad_context(self) -> None:
        """``no_grad=True`` must still reach the seam (loss built without a graph)."""
        step = _SpyEvalStep(no_grad=True)
        trainer = _make_trainer(loss_fn=_RecordingLoss(), eval_step=step)

        outputs = step.on_eval_batch(trainer, TrainState(), _batch())

        assert len(step.calls) == 1
        assert outputs["loss"].requires_grad is False
