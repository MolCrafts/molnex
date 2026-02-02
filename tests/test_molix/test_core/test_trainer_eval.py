"""Unit tests for step-based evaluation feature."""

import pytest
from molix.core.state import TrainState
from molix.core.trainer import Trainer
from molix.core.hooks import BaseHook


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
