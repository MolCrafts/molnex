"""``EarlyStop`` — the three conditions that end a run, and what each promises.

Divergence aborts (it is a failure); a plateau and a time budget *ask* the loop to
finish through :meth:`~molix.core.Trainer.stop`, so the epoch's eval and checkpoint
still happen. A staged run depends on that difference: raising leaves the next stage
with no checkpoint to continue from.

The loop-level behaviour is asserted against the real :class:`Trainer` with a
two-parameter model, not a mock, because "the loop actually ends" is the claim.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from molix.core.trainer import Trainer
from molix.hooks import EarlyStop


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, batch):
        return self.weight.expand_as(batch)


def _loss(pred, batch):
    return ((pred - batch) ** 2).mean()


class _Data:
    """Two batches per epoch, unbounded epochs — so only a stop can end it."""

    def __init__(self, batches: int = 2) -> None:
        self._batches = batches

    def train_dataloader(self):
        for _ in range(self._batches):
            yield torch.ones(2, 1)

    def val_dataloader(self):
        yield torch.ones(2, 1)


def _trainer(**kwargs) -> Trainer:
    return Trainer(
        model=_Model(),
        loss_fn=_loss,
        optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.0),
        **kwargs,
    )


class _Metric:
    """Publishes a scripted ``eval/loss`` at each eval, as a metrics hook would."""

    def __init__(self, values: list[float]) -> None:
        self._values = list(values)

    def on_eval_step_complete(self, trainer, state) -> None:
        if self._values:
            state["eval"]["loss"] = self._values.pop(0)


class TestEarlyStopPlateau:
    """``patience`` — ending a run that has stopped improving."""

    def test_it_stops_after_patience_evaluations_without_improvement(self) -> None:
        """The condition that was missing entirely: a run had no stopping rule.

        Three evals at the same value after the first is one improvement and three
        stale ones, so a patience of 3 ends it — and the reason says so.
        """
        trainer = _trainer(hooks=[_Metric([1.0, 1.0, 1.0, 1.0]), EarlyStop(patience=3)])
        trainer.train(_Data(), max_epochs=10)
        assert trainer.stop_reason is not None
        assert "has not improved" in trainer.stop_reason
        assert int(trainer.state["epoch"]) < 10

    def test_an_improvement_resets_the_count(self) -> None:
        """Otherwise patience counts evaluations, not stagnation."""
        hook = EarlyStop(patience=2)
        trainer = _trainer(hooks=[_Metric([5.0, 5.0, 4.0, 4.0, 4.0]), hook])
        trainer.train(_Data(), max_epochs=5)
        # 5, 5(stale 1), 4(reset), 4(stale 1), 4(stale 2) -> stops on the fifth.
        assert trainer.stop_reason is not None
        assert int(trainer.state["epoch"]) == 5

    def test_an_improvement_smaller_than_min_delta_does_not_count(self) -> None:
        """A metric creeping by 1e-9 an eval is a plateau, not progress."""
        trainer = _trainer(
            hooks=[_Metric([1.0, 0.999, 0.998, 0.997]), EarlyStop(patience=2, min_delta=0.01)]
        )
        trainer.train(_Data(), max_epochs=10)
        assert trainer.stop_reason is not None

    def test_a_monitor_nothing_writes_is_not_a_plateau(self) -> None:
        """An absent metric has told us nothing, and nothing is not stagnation.

        Counting it as "no improvement" would end a run for the one reason that
        carries no information about it — a metrics hook not yet mounted, or a
        stage whose eval reports under another name. (``eval/loss`` itself is
        always present: the Trainer's own eval phase writes it.)
        """
        trainer = _trainer(hooks=[EarlyStop(patience=1, monitor="eval/not_reported")])
        trainer.train(_Data(), max_epochs=3)
        assert trainer.stop_reason is None
        assert int(trainer.state["epoch"]) == 3

    def test_without_patience_it_never_stops_on_a_plateau(self) -> None:
        """The default must stay the old behaviour for every existing caller."""
        trainer = _trainer(hooks=[_Metric([1.0] * 5), EarlyStop()])
        trainer.train(_Data(), max_epochs=3)
        assert trainer.stop_reason is None
        assert int(trainer.state["epoch"]) == 3

    def test_it_watches_the_checkpoints_own_metric_by_default(self) -> None:
        """The metric that selects the checkpoint is the metric that ends the run.

        A plateau declared on one quantity while the checkpoint tracks another is
        how a run stops on a number nobody kept.
        """
        hook = EarlyStop(patience=1)
        trainer = _trainer()
        assert hook._monitor_path(trainer) == tuple(trainer._checkpoint.best_metric_name)

    @pytest.mark.parametrize("patience", [0, -1])
    def test_a_patience_below_one_is_refused(self, patience: int) -> None:
        """Zero would end the run at its first eval, which is not a plateau."""
        with pytest.raises(ValueError, match="patience must be >= 1"):
            EarlyStop(patience=patience)


class TestEarlyStopRegression:
    """A run that leaves its best by an order of magnitude, twice."""

    def test_two_evals_past_ten_times_the_best_stop_the_run(self) -> None:
        trainer = _trainer(
            hooks=[
                _Metric([1.0, 1.2, 20.0, 30.0]),
                EarlyStop(patience=50, regression_factor=10.0, regression_patience=2),
            ]
        )
        trainer.train(_Data(), max_epochs=10)
        assert trainer.stop_reason is not None
        assert "more than 10x" in trainer.stop_reason

    def test_one_spike_does_not_stop_the_run(self) -> None:
        trainer = _trainer(
            hooks=[
                _Metric([1.0, 50.0, 1.1, 1.1]),
                EarlyStop(patience=10, regression_factor=10.0, regression_patience=2),
            ]
        )
        trainer.train(_Data(), max_epochs=4)
        assert trainer.stop_reason is None


class TestEarlyStopTimeBudget:
    """``max_seconds`` — ending before something else does it less gently."""

    def test_it_stops_once_the_budget_is_spent(self) -> None:
        """A run bounded only by its Slurm wall is killed mid-epoch.

        That loses the epoch and can lose the checkpoint the next curriculum stage
        needs. Asked to stop, it leaves the batch loop and still writes one.
        """
        trainer = _trainer(hooks=[EarlyStop(max_seconds=1e-9)])
        trainer.train(_Data(batches=4), max_epochs=10)
        assert trainer.stop_reason is not None
        assert "time budget" in trainer.stop_reason
        assert int(trainer.state["epoch"]) < 10

    def test_no_budget_by_default(self) -> None:
        """Every existing caller keeps its behaviour."""
        trainer = _trainer(hooks=[EarlyStop()])
        trainer.train(_Data(), max_epochs=2)
        assert trainer.stop_reason is None

    def test_a_non_positive_budget_is_refused(self) -> None:
        """Zero seconds would end the run before its first batch."""
        with pytest.raises(ValueError, match="max_seconds must be > 0"):
            EarlyStop(max_seconds=0.0)


class TestTrainerStop:
    """The clean-stop path the plateau and budget conditions use."""

    def test_the_first_reason_is_the_one_reported(self) -> None:
        """It is the condition that actually ended the run; later ones are noise."""
        trainer = _trainer()
        trainer.stop("first")
        trainer.stop("second")
        assert trainer.stop_reason == "first"

    def test_a_fresh_run_clears_the_previous_reason(self) -> None:
        """A curriculum reuses a Trainer shape per stage; a stale reason would end
        the next stage before it began."""
        trainer = _trainer()
        trainer.stop("stale")
        trainer.train(_Data(), max_epochs=1)
        assert trainer.stop_reason is None
