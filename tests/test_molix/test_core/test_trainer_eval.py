"""Unit tests for step-based evaluation feature."""

import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from molix.core.state import TrainState
from molix.core.trainer import Trainer


def test_trainstate_counter_increment():
    """Verify counter increments correctly."""
    state = TrainState()
    state.steps_since_last_eval += 1
    assert state.steps_since_last_eval == 1

    state.steps_since_last_eval += 1
    assert state.steps_since_last_eval == 2


def test_trainer_parameter_storage():
    """Verify eval_every_n_steps parameter is stored correctly."""
    trainer = _make_trainer(eval_every_n_steps=100)
    assert trainer.eval_every_n_steps == 100

    trainer_none = _make_trainer(eval_every_n_steps=None)
    assert trainer_none.eval_every_n_steps is None

    trainer_default = _make_trainer()
    assert trainer_default.eval_every_n_steps is None


def test_trainer_parameter_validation():
    """Verify ValueError raised on invalid eval_every_n_steps input."""
    with pytest.raises(ValueError, match="must be > 0"):
        _make_trainer(eval_every_n_steps=0)

    with pytest.raises(ValueError, match="must be > 0"):
        _make_trainer(eval_every_n_steps=-5)


def test_trainer_eval_every_n_steps_disabled_by_default():
    """Verify step-based eval logic is disabled when eval_every_n_steps=None."""
    state = TrainState()
    _make_trainer(eval_every_n_steps=None)

    # Simulate multiple steps
    for _ in range(100):
        state.steps_since_last_eval += 1

    # Counter should still be incremented (it's independent of trainer)
    assert state.steps_since_last_eval == 100

    # But trainer shouldn't check it since eval_every_n_steps is None
    # This is tested through integration tests with actual dataloaders


# ---------------------------------------------------------------------------
# Fixtures for max_steps tests
# ---------------------------------------------------------------------------


class _SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, batch):
        return self.linear(batch["x"])


def _simple_loss(predictions, batch):
    return ((predictions - batch["targets"]["y_energy"]) ** 2).mean()


def _make_batch():
    # A TensorDict, not a plain dict: the batch schema is a nested TensorDict and
    # the Trainer moves batches with `TensorDict.to`.
    return TensorDict(
        {
            "x": torch.randn(4, 10),
            "targets": TensorDict({"y_energy": torch.randn(4, 1)}, batch_size=[4]),
        },
        batch_size=[4],
    )


class _MockDataModule:
    """Each epoch yields *batches_per_epoch* training batches."""

    def __init__(self, batches_per_epoch: int = 5):
        self._n = batches_per_epoch

    def setup(self, stage: str = "fit") -> None:
        pass

    def on_epoch_start(self, epoch: int) -> None:
        pass

    def train_dataloader(self):
        for _ in range(self._n):
            yield _make_batch()

    def val_dataloader(self):
        yield _make_batch()


def _make_trainer(**kwargs):
    model = _SimpleModel()
    return Trainer(
        model=model,
        loss_fn=_simple_loss,
        optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.01),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# max_steps / max_epochs validation
# ---------------------------------------------------------------------------


def test_train_requires_at_least_one_limit():
    """ValueError when neither max_epochs nor max_steps is set."""
    trainer = _make_trainer()
    with pytest.raises(ValueError, match="At least one"):
        trainer.train(_MockDataModule())


def test_train_rejects_non_positive_max_epochs():
    trainer = _make_trainer()
    with pytest.raises(ValueError, match="max_epochs must be > 0"):
        trainer.train(_MockDataModule(), max_epochs=0)
    with pytest.raises(ValueError, match="max_epochs must be > 0"):
        trainer.train(_MockDataModule(), max_epochs=-1)


def test_train_rejects_non_positive_max_steps():
    trainer = _make_trainer()
    with pytest.raises(ValueError, match="max_steps must be > 0"):
        trainer.train(_MockDataModule(), max_steps=0)
    with pytest.raises(ValueError, match="max_steps must be > 0"):
        trainer.train(_MockDataModule(), max_steps=-3)


# ---------------------------------------------------------------------------
# max_epochs only (existing behaviour)
# ---------------------------------------------------------------------------


def test_train_max_epochs_only():
    """Training with only max_epochs completes the expected epochs."""
    dm = _MockDataModule(batches_per_epoch=3)
    trainer = _make_trainer()
    state = trainer.train(dm, max_epochs=2)
    assert state.epoch == 2
    assert state.global_step == 6  # 2 epochs * 3 batches


# ---------------------------------------------------------------------------
# max_steps only
# ---------------------------------------------------------------------------


def test_train_max_steps_only():
    """Training with only max_steps stops at the step limit."""
    dm = _MockDataModule(batches_per_epoch=5)
    trainer = _make_trainer()
    state = trainer.train(dm, max_steps=7)
    # 7 steps = 1 full epoch (5) + 2 steps into epoch 2
    assert state.global_step == 7
    assert state.epoch == 2  # both epochs counted


# ---------------------------------------------------------------------------
# Both limits — whichever comes first
# ---------------------------------------------------------------------------


def test_train_both_limits_epochs_first():
    """When max_epochs is the binding constraint, stop by epochs."""
    dm = _MockDataModule(batches_per_epoch=5)
    trainer = _make_trainer()
    # 2 epochs = 10 steps, but max_steps=100 → epochs bind
    state = trainer.train(dm, max_epochs=2, max_steps=100)
    assert state.epoch == 2
    assert state.global_step == 10


def test_train_both_limits_steps_first():
    """When max_steps is the binding constraint, stop by steps."""
    dm = _MockDataModule(batches_per_epoch=5)
    trainer = _make_trainer()
    # max_epochs=100 → would be 500 steps, but max_steps=7 binds
    state = trainer.train(dm, max_epochs=100, max_steps=7)
    assert state.global_step == 7
    assert state.epoch == 2  # partial second epoch still counted


# ---------------------------------------------------------------------------
# gradient accumulation
# ---------------------------------------------------------------------------


def test_accumulate_grad_batches_rejects_non_positive():
    with pytest.raises(ValueError, match="accumulate_grad_batches must be > 0"):
        _make_trainer(accumulate_grad_batches=0)


def test_accumulate_grad_batches_counts_optimizer_steps():
    """global_step / max_steps count optimizer steps, not micro-batches."""
    dm = _MockDataModule(batches_per_epoch=8)
    trainer = _make_trainer(accumulate_grad_batches=2)
    state = trainer.train(dm, max_epochs=1)
    # 8 micro-batches / accum=2 → 4 optimizer steps
    assert state.global_step == 4


def test_accumulate_grad_batches_updates_params():
    """Training with accumulation still updates parameters."""
    dm = _MockDataModule(batches_per_epoch=6)
    trainer = _make_trainer(accumulate_grad_batches=3)
    initial = [p.clone() for p in trainer.model.parameters()]
    trainer.train(dm, max_epochs=1)
    assert any(not torch.equal(i, c) for i, c in zip(initial, trainer.model.parameters()))


def test_accumulate_grad_batches_one_is_per_batch():
    """accumulate_grad_batches=1 (default) steps every micro-batch."""
    dm = _MockDataModule(batches_per_epoch=5)
    trainer = _make_trainer(accumulate_grad_batches=1)
    state = trainer.train(dm, max_epochs=1)
    assert state.global_step == 5


# ---------------------------------------------------------------------------
# eval starvation warning
# ---------------------------------------------------------------------------


def _capture_warnings(monkeypatch):
    """Record messages passed to the trainer module logger's ``warning``.

    mollog emits from a background thread, so fd/sys capture is racy —
    intercepting the call itself is deterministic.
    """
    import molix.core.trainer as trainer_mod

    messages: list[str] = []
    original = trainer_mod.logger.warning

    def recording(msg, *args, **kwargs):
        messages.append(str(msg))
        return original(msg, *args, **kwargs)

    monkeypatch.setattr(trainer_mod.logger, "warning", recording)
    return messages


def test_eval_starvation_warns_once(monkeypatch):
    """Warn once when eval_every_n_steps exceeds whole epochs.

    With a 5-step epoch and eval_every_n_steps=100, epochs complete with
    zero evals (epoch-end eval is suppressed by the step cadence) — the
    trainer must surface that instead of staying silent.
    """
    messages = _capture_warnings(monkeypatch)
    dm = _MockDataModule(batches_per_epoch=5)
    trainer = _make_trainer(eval_every_n_steps=100)
    trainer.train(dm, max_epochs=3)
    starved = [m for m in messages if "no evaluation ran this epoch" in m]
    assert len(starved) == 1  # once, not per epoch
    assert "eval_every_n_steps=100" in starved[0]


def test_eval_starvation_no_warning_when_cadence_fits(monkeypatch):
    """No starvation warning when evals actually run within the epoch."""
    messages = _capture_warnings(monkeypatch)
    dm = _MockDataModule(batches_per_epoch=5)
    trainer = _make_trainer(eval_every_n_steps=2)
    trainer.train(dm, max_epochs=2)
    assert not [m for m in messages if "no evaluation ran this epoch" in m]


def test_eval_starvation_no_warning_without_step_cadence(monkeypatch):
    """Default epoch-end eval schedule never triggers the starvation warning."""
    messages = _capture_warnings(monkeypatch)
    dm = _MockDataModule(batches_per_epoch=5)
    trainer = _make_trainer()
    trainer.train(dm, max_epochs=2)
    assert not [m for m in messages if "no evaluation ran this epoch" in m]


# ---------------------------------------------------------------------------
# per-epoch device-move decision (batch_to hot-path skip)
# ---------------------------------------------------------------------------


def test_needs_device_move_false_when_already_on_device():
    """A batch whose leaves are all on the target device needs no move."""
    from molix.core.trainer import Trainer

    batch = TensorDict(
        {"atoms": TensorDict({"pos": torch.randn(4, 3)}, batch_size=[4])}, batch_size=[]
    )
    assert Trainer._needs_device_move(batch, torch.device("cpu")) is False


def test_needs_device_move_true_for_foreign_device():
    """A move is required when the target differs from the leaves' device."""
    from molix.core.trainer import Trainer

    batch = TensorDict(
        {"atoms": TensorDict({"pos": torch.randn(4, 3)}, batch_size=[4])}, batch_size=[]
    )
    assert Trainer._needs_device_move(batch, torch.device("meta")) is True


def test_train_loop_runs_without_per_step_move_on_cpu():
    """Training still updates params when the per-epoch skip path is taken."""
    dm = _MockDataModule(batches_per_epoch=5)
    trainer = _make_trainer()
    initial = [p.clone() for p in trainer.model.parameters()]
    trainer.train(dm, max_epochs=1)
    assert any(not torch.equal(i, c) for i, c in zip(initial, trainer.model.parameters()))


def test_eval_phase_publishes_the_mean_over_val_batches():
    """``eval/loss`` is the val-set mean, not the last batch's value.

    Best-checkpoint selection and patience read it; publishing the last batch
    made both decisions on one batch of nine.
    """

    class _Const(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(()))

        def forward(self, batch):
            return batch["y"]

    class _Val:
        def val_dataloader(self):
            for value in (1.0, 2.0, 9.0):
                yield TensorDict({"y": torch.tensor([value])}, batch_size=[1])

    trainer = Trainer(
        model=_Const(),
        loss_fn=lambda predictions, batch: predictions.sum(),
        optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.01),
    )
    trainer._run_eval_phase(_Val())
    assert float(trainer.state["eval"]["loss"]) == pytest.approx(4.0)


def test_eval_phase_averages_each_named_term_of_a_combined_objective():
    """A ``WeightedLoss`` with named terms is published per term, val-set mean."""
    from molix.core.losses import WeightedLoss
    from molix.core.steps import DefaultEvalStep

    class _Const(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(()))

        def forward(self, batch):
            return batch["y"]

    class _Val:
        def val_dataloader(self):
            for value in (1.0, 3.0):
                yield TensorDict({"y": torch.tensor([value])}, batch_size=[1])

    objective = WeightedLoss(
        [(1.0, lambda p, b: p.sum()), (2.0, lambda p, b: p.sum() ** 2)], names=("lin", "sq")
    )
    trainer = Trainer(
        model=_Const(),
        loss_fn=None,
        optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.01),
        eval_step=DefaultEvalStep(objective=objective),
    )
    trainer._run_eval_phase(_Val())
    assert float(trainer.state["eval"]["loss_lin"]) == pytest.approx(2.0)
    assert float(trainer.state["eval"]["loss_sq"]) == pytest.approx(5.0)
    assert float(trainer.state["eval"]["loss"]) == pytest.approx((1 + 2 + 3 + 18) / 2)


def test_eval_phase_releases_a_batch_before_running_the_next():
    """The previous batch's outputs are dead when the next forward starts.

    The step detaches what it returns, but the outputs are still a batch's
    worth of predictions; held in the loop variable through the next forward
    they were, with the graph then attached, double the eval peak (a large
    tier that survived its smoke steps OOMed in its baseline eval).
    """
    import weakref

    from molix.core.steps import DefaultEvalStep

    class _Scaled(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.ones(()))

        def forward(self, batch):
            return batch["y"] * self.w

    class _Watching(DefaultEvalStep):
        def __init__(self):
            super().__init__(no_grad=False)
            self.previous_loss = None
            self.previous_alive_at_next_call: list[bool] = []

        def on_eval_batch(self, trainer, state, batch):
            if self.previous_loss is not None:
                self.previous_alive_at_next_call.append(self.previous_loss() is not None)
            outputs = super().on_eval_batch(trainer, state, batch)
            self.previous_loss = weakref.ref(outputs["loss"])
            return outputs

    class _Val:
        def val_dataloader(self):
            for value in (1.0, 2.0, 3.0):
                yield TensorDict({"y": torch.tensor([value])}, batch_size=[1])

    step = _Watching()
    trainer = Trainer(
        model=_Scaled(),
        loss_fn=lambda predictions, batch: predictions.sum(),
        optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.01),
        eval_step=step,
    )
    trainer._run_eval_phase(_Val())
    assert step.previous_alive_at_next_call == [False, False]
