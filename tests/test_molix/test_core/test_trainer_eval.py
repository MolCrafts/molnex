"""Unit tests for step-based evaluation feature."""

import pytest
import torch
import torch.nn as nn

from molix.core.state import TrainState
from molix.core.trainer import Trainer
from molix.core.hooks import BaseHook
from molix.core.steps import DefaultTrainStep, DefaultEvalStep, extract_model_inputs


def test_trainstate_counter_init():
    """Verify steps_since_last_eval starts at 0."""
    state = TrainState()
    assert state.steps_since_last_eval == 0


def test_trainstate_counter_increment():
    """Verify counter increments correctly."""
    state = TrainState()
    state.steps_since_last_eval += 1
    assert state.steps_since_last_eval == 1
    
    state.steps_since_last_eval += 1
    assert state.steps_since_last_eval == 2


def test_trainer_parameter_storage():
    """Verify eval_every_n_steps parameter is stored correctly."""
    trainer = Trainer(eval_every_n_steps=100)
    assert trainer.eval_every_n_steps == 100
    
    trainer_none = Trainer(eval_every_n_steps=None)
    assert trainer_none.eval_every_n_steps is None
    
    trainer_default = Trainer()
    assert trainer_default.eval_every_n_steps is None


def test_trainer_parameter_validation():
    """Verify ValueError raised on invalid eval_every_n_steps input."""
    with pytest.raises(ValueError, match="must be > 0"):
        Trainer(eval_every_n_steps=0)
    
    with pytest.raises(ValueError, match="must be > 0"):
        Trainer(eval_every_n_steps=-5)


def test_hook_on_eval_step_complete_exists():
    """Verify on_eval_step_complete method exists on BaseHook."""
    hook = BaseHook()
    assert hasattr(hook, "on_eval_step_complete")
    assert callable(getattr(hook, "on_eval_step_complete"))


def test_hook_on_eval_step_complete_callable():
    """Verify on_eval_step_complete can be overridden."""
    class CustomHook(BaseHook):
        def __init__(self):
            self.called = False
        
        def on_eval_step_complete(self, trainer, state):
            self.called = True
    
    hook = CustomHook()
    hook.on_eval_step_complete(None, TrainState())
    assert hook.called is True


def test_trainer_eval_every_n_steps_disabled_by_default():
    """Verify step-based eval logic is disabled when eval_every_n_steps=None."""
    state = TrainState()
    trainer = Trainer(eval_every_n_steps=None)

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

    def forward(self, x, **_kwargs):
        return self.linear(x)


def _simple_loss(predictions, batch):
    return ((predictions - batch["targets"]["y_energy"]) ** 2).mean()


def _make_batch():
    return {"x": torch.randn(4, 10), "targets": {"y_energy": torch.randn(4, 1)}}


class _MockDataModule:
    """Each epoch yields *batches_per_epoch* training batches."""

    def __init__(self, batches_per_epoch: int = 5):
        self._n = batches_per_epoch

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
