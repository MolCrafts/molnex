"""Tests for :meth:`molix.core.trainer.Trainer.smoke` objective routing.

``smoke`` must warm up the *configured* objective: when the train Step owns
its losses (``compute_loss`` seam, ``Trainer(loss_fn=None)``) the warmup has
to go through the Step, not through ``trainer.loss_fn``. The non-finite abort
is preserved either way.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping

import pytest
import torch
import torch.nn as nn

from molix.config import config
from molix.core.state import TrainState
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

    def __init__(self, non_finite: bool = False) -> None:
        self.non_finite = non_finite
        self.calls: list[tuple[Predictions, Batch]] = []

    def __call__(self, predictions: Predictions, batch: Batch) -> torch.Tensor:
        self.calls.append((predictions, batch))
        loss = predictions["energy"].square().mean()
        return loss * float("nan") if self.non_finite else loss


class _SeamStep:
    """Duck-typed Step that owns its objective via the ``compute_loss`` seam."""

    def __init__(self, non_finite: bool = False) -> None:
        self.non_finite = non_finite
        self.compute_calls = 0

    def compute_loss(
        self,
        trainer: Trainer,
        state: TrainState,
        batch: Batch,
        predictions: Predictions,
    ) -> torch.Tensor:
        self.compute_calls += 1
        loss = predictions["energy"].square().mean()
        return loss * float("nan") if self.non_finite else loss

    def on_train_batch(
        self, trainer: Trainer, state: TrainState, batch: Batch
    ) -> dict[str, object]:
        raise AssertionError("smoke must not drive the full train step")

    def on_eval_batch(self, trainer: Trainer, state: TrainState, batch: Batch) -> dict[str, object]:
        raise NotImplementedError


class _SeamlessStep:
    """Duck-typed Step without the seam — third-party steps must keep working."""

    def on_train_batch(
        self, trainer: Trainer, state: TrainState, batch: Batch
    ) -> dict[str, object]:
        raise AssertionError("smoke must not drive the full train step")

    def on_eval_batch(self, trainer: Trainer, state: TrainState, batch: Batch) -> dict[str, object]:
        raise NotImplementedError


class _StubDataModule:
    """Two fixed batches, no ``val_dataloader`` (eval warmup is out of scope)."""

    def train_dataloader(self) -> Iterator[dict[str, dict[str, torch.Tensor]]]:
        for shift in (0.0, 0.5):
            pos = torch.tensor(
                [(0.0, 0.0, 0.0), (1.0, 0.5, -0.5), (0.25, -0.75, 0.5)],
                dtype=torch.float32,
            )
            yield {"atoms": {"pos": pos + shift}}


def _make_trainer(*, loss_fn: LossFn | None, train_step: object) -> Trainer:
    torch.manual_seed(0)
    return Trainer(
        model=_StubModel(),
        loss_fn=loss_fn,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=0.1),
        train_step=train_step,
    )


class TestTrainerSmoke:
    """Warmup objective routing through the Step seam."""

    def test_routes_through_the_step_compute_loss(self) -> None:
        step = _SeamStep()
        loss_fn = _RecordingLoss()
        trainer = _make_trainer(loss_fn=loss_fn, train_step=step)

        trainer.smoke(_StubDataModule(), steps=2)

        assert step.compute_calls == 2
        assert loss_fn.calls == []

    def test_warms_up_a_step_that_owns_its_losses(self) -> None:
        """``Trainer(loss_fn=None)`` + seam step: no ``TypeError`` on warmup."""
        step = _SeamStep()
        trainer = _make_trainer(loss_fn=None, train_step=step)

        trainer.smoke(_StubDataModule(), steps=1)

        assert step.compute_calls == 1

    def test_falls_back_to_loss_fn_without_the_seam(self) -> None:
        loss_fn = _RecordingLoss()
        trainer = _make_trainer(loss_fn=loss_fn, train_step=_SeamlessStep())

        trainer.smoke(_StubDataModule(), steps=2)

        assert len(loss_fn.calls) == 2

    def test_still_aborts_on_a_non_finite_loss(self) -> None:
        trainer = _make_trainer(
            loss_fn=_RecordingLoss(non_finite=True),
            train_step=_SeamStep(non_finite=True),
        )

        with pytest.raises(RuntimeError, match="non-finite"):
            trainer.smoke(_StubDataModule(), steps=1)
