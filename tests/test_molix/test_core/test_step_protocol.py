"""Tests for Step protocol and default implementations."""

import pytest
import torch
import torch.nn as nn

from molix.core.trainer import Trainer
from molix.core.state import TrainState, Stage
from molix.core.steps import Step, DefaultTrainStep, DefaultEvalStep


# Simple test model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, batch):
        if isinstance(batch, dict):
            x = batch.get("x", batch.get("pos", torch.randn(5, 10)))
        else:
            x = batch
        return self.linear(x)


def simple_loss_fn(predictions, batch):
    """Simple MSE loss that handles batch dictionary or tensor."""
    if isinstance(batch, dict):
        targets = batch.get("y_energy")
    else:
        targets = batch
    return ((predictions - targets) ** 2).mean()


def simple_optimizer_factory(params):
    """Simple optimizer factory."""
    return torch.optim.SGD(params, lr=0.01)


# Mock datamodule
class MockDataModule:
    def train_dataloader(self):
        for _ in range(3):
            yield {"x": torch.randn(5, 10), "y_energy": torch.randn(5, 1)}
    
    def val_dataloader(self):
        for _ in range(2):
            yield {"x": torch.randn(5, 10), "y_energy": torch.randn(5, 1)}


# Test Step Protocol
def test_default_train_step_satisfies_protocol():
    """Verify DefaultTrainStep satisfies Step protocol."""
    step = DefaultTrainStep()
    
    # Check methods exist
    assert hasattr(step, "on_train_batch")
    assert hasattr(step, "on_eval_batch")
    assert callable(step.on_train_batch)
    assert callable(step.on_eval_batch)


def test_default_eval_step_satisfies_protocol():
    """Verify DefaultEvalStep satisfies Step protocol."""
    step = DefaultEvalStep()
    
    # Check methods exist
    assert hasattr(step, "on_train_batch")
    assert hasattr(step, "on_eval_batch")
    assert callable(step.on_train_batch)
    assert callable(step.on_eval_batch)


def test_trainer_accepts_default_steps():
    """Verify Trainer accepts default steps (implicit and explicit)."""
    model = SimpleModel()
    
    # Implicit default steps
    trainer1 = Trainer(
        model=model,
        loss_fn=simple_loss_fn,
        optimizer_factory=simple_optimizer_factory,
    )
    assert isinstance(trainer1.train_step, DefaultTrainStep)
    assert isinstance(trainer1.eval_step, DefaultEvalStep)
    
    # Explicit default steps
    trainer2 = Trainer(
        model=model,
        loss_fn=simple_loss_fn,
        optimizer_factory=simple_optimizer_factory,
        train_step=DefaultTrainStep(),
        eval_step=DefaultEvalStep(),
    )
    assert isinstance(trainer2.train_step, DefaultTrainStep)
    assert isinstance(trainer2.eval_step, DefaultEvalStep)


def test_trainer_accepts_custom_step():
    """Verify Trainer accepts custom step implementations."""
    
    class CustomStep:
        def __init__(self):
            self.train_called = False
            self.eval_called = False
        
        def on_train_batch(self, trainer, state, batch):
            self.train_called = True
            predictions = trainer.model(batch)
            targets = batch.get("y_energy") if isinstance(batch, dict) else batch
            loss = trainer.loss_fn(predictions, targets)
            trainer.optimizer.zero_grad()
            loss.backward()
            trainer.optimizer.step()
            return {"loss": loss, "predictions": predictions}
        
        def on_eval_batch(self, trainer, state, batch):
            self.eval_called = True
            with torch.no_grad():
                predictions = trainer.model(batch)
                targets = batch.get("y_energy") if isinstance(batch, dict) else batch
                loss = trainer.loss_fn(predictions, targets)
            return {"loss": loss, "predictions": predictions}
    
    custom_step = CustomStep()
    model = SimpleModel()
    
    trainer = Trainer(
        model=model,
        loss_fn=simple_loss_fn,
        optimizer_factory=simple_optimizer_factory,
        train_step=custom_step,
        eval_step=custom_step,
    )
    
    assert trainer.train_step is custom_step
    assert trainer.eval_step is custom_step


def test_default_train_step_computes_loss_and_updates():
    """Verify DefaultTrainStep performs forward, backward, and optimizer step."""
    model = SimpleModel()
    optimizer = simple_optimizer_factory(model.parameters())
    
    trainer = Trainer(
        model=model,
        loss_fn=simple_loss_fn,
        optimizer_factory=lambda p: optimizer,
    )
    
    state = TrainState()
    batch = {"x": torch.randn(5, 10), "y_energy": torch.randn(5, 1)}
    
    # Get initial parameters
    initial_params = [p.clone() for p in model.parameters()]
    
    # Execute training step
    outputs = trainer.train_step.on_train_batch(trainer, state, batch)
    
    # Check outputs
    assert "loss" in outputs
    assert "predictions" in outputs
    assert isinstance(outputs["loss"], torch.Tensor)
    assert isinstance(outputs["predictions"], torch.Tensor)
    
    # Check parameters updated (optimizer step executed)
    for initial, current in zip(initial_params, model.parameters()):
        assert not torch.equal(initial, current), "Parameters should have been updated"


def test_default_eval_step_no_gradient():
    """Verify DefaultEvalStep does not compute gradients."""
    model = SimpleModel()
    
    trainer = Trainer(
        model=model,
        loss_fn=simple_loss_fn,
        optimizer_factory=simple_optimizer_factory,
    )
    
    state = TrainState()
    batch = {"x": torch.randn(5, 10), "y_energy": torch.randn(5, 1)}
    
    # Execute eval step
    outputs = trainer.eval_step.on_eval_batch(trainer, state, batch)
    
    # Check outputs
    assert "loss" in outputs
    assert "predictions" in outputs
    assert isinstance(outputs["loss"], torch.Tensor)
    assert isinstance(outputs["predictions"], torch.Tensor)
    
    # Verify no gradients computed
    assert not outputs["loss"].requires_grad
    assert not outputs["predictions"].requires_grad


def test_trainer_delegates_to_train_step():
    """Verify Trainer delegates training computation to train_step."""
    
    class MockStep:
        def __init__(self):
            self.train_calls = 0
            self.eval_calls = 0
        
        def on_train_batch(self, trainer, state, batch):
            self.train_calls += 1
            predictions = trainer.model(batch)
            targets = batch["y_energy"]
            loss = trainer.loss_fn(predictions, targets)
            trainer.optimizer.zero_grad()
            loss.backward()
            trainer.optimizer.step()
            return {"loss": loss, "predictions": predictions}
        
        def on_eval_batch(self, trainer, state, batch):
            self.eval_calls += 1
            with torch.no_grad():
                predictions = trainer.model(batch)
                targets = batch["y_energy"]
                loss = trainer.loss_fn(predictions, targets)
            return {"loss": loss, "predictions": predictions}
    
    mock_step = MockStep()
    model = SimpleModel()
    
    trainer = Trainer(
        model=model,
        loss_fn=simple_loss_fn,
        optimizer_factory=simple_optimizer_factory,
        train_step=mock_step,
        eval_step=mock_step,
    )
    
    datamodule = MockDataModule()
    
    # Train for 1 epoch (3 train batches, 2 eval batches)
    trainer.train(datamodule, max_epochs=1)
    
    # Verify step methods were called
    assert mock_step.train_calls == 3, "train_step.on_train_batch should be called 3 times"
    assert mock_step.eval_calls == 2, "eval_step.on_eval_batch should be called 2 times"


def test_custom_step_gradient_accumulation():
    """Verify custom step with gradient accumulation works."""
    
    class GradientAccumulationStep:
        def __init__(self, accumulation_steps: int = 2):
            self.accumulation_steps = accumulation_steps
            self.accumulated = 0
        
        def on_train_batch(self, trainer, state, batch):
            predictions = trainer.model(batch)
            targets = batch["y_energy"]
            loss = trainer.loss_fn(predictions, targets) / self.accumulation_steps
            
            loss.backward()
            self.accumulated += 1
            
            if self.accumulated >= self.accumulation_steps:
                trainer.optimizer.step()
                trainer.optimizer.zero_grad()
                self.accumulated = 0
            
            return {"loss": loss * self.accumulation_steps, "predictions": predictions}
        
        def on_eval_batch(self, trainer, state, batch):
            with torch.no_grad():
                predictions = trainer.model(batch)
                targets = batch["y_energy"]
                loss = trainer.loss_fn(predictions, targets)
            return {"loss": loss, "predictions": predictions}
    
    model = SimpleModel()
    grad_accum_step = GradientAccumulationStep(accumulation_steps=2)
    
    trainer = Trainer(
        model=model,
        loss_fn=simple_loss_fn,
        optimizer_factory=simple_optimizer_factory,
        train_step=grad_accum_step,
        eval_step=DefaultEvalStep(),
    )
    
    datamodule = MockDataModule()
    
    # Should run without errors
    state = trainer.train(datamodule, max_epochs=1)
    
    # Verify training completed
    assert state.epoch == 1
    assert state.global_step == 3  # 3 training batches


def test_step_return_format():
    """Verify steps return correct output format."""
    model = SimpleModel()
    
    trainer = Trainer(
        model=model,
        loss_fn=simple_loss_fn,
        optimizer_factory=simple_optimizer_factory,
    )
    
    state = TrainState()
    batch = {"x": torch.randn(5, 10), "y_energy": torch.randn(5, 1)}
    
    # Test train step output
    train_outputs = trainer.train_step.on_train_batch(trainer, state, batch)
    assert isinstance(train_outputs, dict)
    assert "loss" in train_outputs
    assert "predictions" in train_outputs
    
    # Test eval step output
    eval_outputs = trainer.eval_step.on_eval_batch(trainer, state, batch)
    assert isinstance(eval_outputs, dict)
    assert "loss" in eval_outputs
    assert "predictions" in eval_outputs
