"""Test suite for Trainer execution."""

import pytest


class DummyDataModule:
    """Dummy data module for testing."""
    
    def __init__(self, train_batches=3, val_batches=2):
        self.train_batches = train_batches
        self.val_batches = val_batches
    
    def train_dataloader(self):
        """Return dummy training batches."""
        for i in range(self.train_batches):
            yield {"data": i, "label": i * 2}
    
    def val_dataloader(self):
        """Return dummy validation batches."""
        for i in range(self.val_batches):
            yield {"data": i + 100, "label": (i + 100) * 2}


def test_trainer_initialization():
    """Test that Trainer initializes correctly."""
    from molnex.core.trainer import Trainer
    
    trainer = Trainer()
    assert trainer.train_step is not None
    assert trainer.eval_step is not None
    assert trainer.state is not None
    assert trainer.state.epoch == 0
    assert trainer.state.global_step == 0


def test_trainer_updates_epoch():
    """Test that Trainer updates epoch counter."""
    from molnex.core.trainer import Trainer
    
    trainer = Trainer()
    datamodule = DummyDataModule(train_batches=2, val_batches=1)
    
    final_state = trainer.train(datamodule, max_epochs=3)
    
    assert final_state.epoch == 3


def test_trainer_updates_global_step():
    """Test that Trainer updates global_step counter."""
    from molnex.core.trainer import Trainer
    
    trainer = Trainer()
    datamodule = DummyDataModule(train_batches=3, val_batches=2)
    
    final_state = trainer.train(datamodule, max_epochs=2)
    
    # 3 train batches per epoch * 2 epochs = 6 steps
    assert final_state.global_step == 6


def test_trainer_stage_transitions():
    """Test that Trainer transitions between TRAIN and EVAL stages."""
    from molnex.core.state import Stage
    from molnex.core.trainer import Trainer
    
    trainer = Trainer()
    datamodule = DummyDataModule(train_batches=1, val_batches=1)
    
    # Track stage changes
    stages_seen = []
    
    original_train_run = trainer.train_step.run
    original_eval_run = trainer.eval_step.run
    
    def track_train_run(state, *, batch):
        stages_seen.append(state.stage)
        return original_train_run(state, batch=batch)
    
    def track_eval_run(state, *, batch):
        stages_seen.append(state.stage)
        return original_eval_run(state, batch=batch)
    
    trainer.train_step.run = track_train_run
    trainer.eval_step.run = track_eval_run
    
    trainer.train(datamodule, max_epochs=1)
    
    # Should see TRAIN then EVAL
    assert Stage.TRAIN in stages_seen
    assert Stage.EVAL in stages_seen
    assert stages_seen.index(Stage.TRAIN) < stages_seen.index(Stage.EVAL)


def test_trainer_executes_all_batches():
    """Test that Trainer executes all training and validation batches."""
    from molnex.core.trainer import Trainer
    
    trainer = Trainer()
    datamodule = DummyDataModule(train_batches=5, val_batches=3)
    
    train_count = 0
    eval_count = 0
    
    original_train_run = trainer.train_step.run
    original_eval_run = trainer.eval_step.run
    
    def count_train_run(state, *, batch):
        nonlocal train_count
        train_count += 1
        return original_train_run(state, batch=batch)
    
    def count_eval_run(state, *, batch):
        nonlocal eval_count
        eval_count += 1
        return original_eval_run(state, batch=batch)
    
    trainer.train_step.run = count_train_run
    trainer.eval_step.run = count_eval_run
    
    trainer.train(datamodule, max_epochs=2)
    
    # 5 train batches * 2 epochs = 10
    assert train_count == 10
    # 3 val batches * 2 epochs = 6
    assert eval_count == 6


def test_trainer_with_custom_steps():
    """Test that Trainer works with custom step implementations."""
    from molnex.core.trainer import Trainer
    from molnex.steps.train_step import TrainStep
    from molnex.steps.eval_step import EvalStep
    
    # Custom steps that track execution
    executed = []
    
    def custom_forward(batch):
        executed.append(("forward", batch))
        return 0.5
    
    def custom_optimizer():
        executed.append(("optimizer",))
    
    train_step = TrainStep(forward_fn=custom_forward, optimizer_fn=custom_optimizer)
    eval_step = EvalStep(forward_fn=custom_forward)
    
    trainer = Trainer(train_step=train_step, eval_step=eval_step)
    datamodule = DummyDataModule(train_batches=1, val_batches=1)
    
    trainer.train(datamodule, max_epochs=1)
    
    # Should have executed forward and optimizer
    assert len(executed) > 0
    assert any(item[0] == "forward" for item in executed)
    assert any(item[0] == "optimizer" for item in executed)
