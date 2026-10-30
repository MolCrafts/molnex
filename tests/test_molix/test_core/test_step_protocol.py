"""Tests for Step protocol and default implementations."""

import torch
import torch.nn as nn
from tensordict import TensorDict

from molix.core.hook import BaseHook
from molix.core.state import TrainState
from molix.core.steps import DefaultEvalStep, DefaultTrainStep
from molix.core.trainer import Trainer


# Simple test model following canonical MolNex contract: forward(batch)
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, batch):
        return self.linear(batch["x"])


def simple_loss_fn(predictions, batch):
    """Simple MSE loss following canonical contract: loss_fn(predictions, batch)."""
    targets = batch["targets"]["y_energy"]
    return ((predictions - targets) ** 2).mean()


def simple_optimizer_factory(params):
    """Simple optimizer factory."""
    return torch.optim.SGD(params, lr=0.01)


def _make_batch():
    """Create a batch following the canonical schema — a nested TensorDict.

    Not a plain dict: the batch schema is a nested TensorDict and the Trainer
    moves batches with ``TensorDict.to``, so a dict stand-in tested a shape
    production never produces.
    """
    return TensorDict(
        {
            "x": torch.randn(5, 10),
            "targets": TensorDict({"y_energy": torch.randn(5, 1)}, batch_size=[5]),
        },
        batch_size=[5],
    )


# Mock datamodule
class MockDataModule:
    def setup(self, stage: str = "fit") -> None:
        pass

    def on_epoch_start(self, epoch: int) -> None:
        pass

    def train_dataloader(self):
        for _ in range(3):
            yield _make_batch()

    def val_dataloader(self):
        for _ in range(2):
            yield _make_batch()


# Test Step Protocol
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
    batch = _make_batch()

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


def test_default_eval_step_no_grad_opt_in():
    """``DefaultEvalStep(no_grad=True)`` suppresses the autograd graph."""
    model = SimpleModel()

    trainer = Trainer(
        model=model,
        loss_fn=simple_loss_fn,
        optimizer_factory=simple_optimizer_factory,
        eval_step=DefaultEvalStep(no_grad=True),
    )

    state = TrainState()
    batch = _make_batch()

    outputs = trainer.eval_step.on_eval_batch(trainer, state, batch)

    assert "loss" in outputs
    assert "predictions" in outputs
    assert isinstance(outputs["loss"], torch.Tensor)
    assert isinstance(outputs["predictions"], torch.Tensor)

    # no_grad=True → no graph
    assert not outputs["loss"].requires_grad
    assert not outputs["predictions"].requires_grad


def test_trainer_delegates_to_train_step():
    """Verify Trainer delegates training computation to train_step."""

    class MockStep:
        scalar_keys = ()

        def __init__(self):
            self.train_calls = 0
            self.eval_calls = 0

        def on_train_batch(self, trainer, state, batch):
            self.train_calls += 1
            predictions = trainer.model(batch)
            loss = trainer.loss_fn(predictions, batch)
            trainer.optimizer.zero_grad()
            loss.backward()
            trainer.optimizer.step()
            return {"loss": loss, "predictions": predictions}

        def on_eval_batch(self, trainer, state, batch):
            self.eval_calls += 1
            with torch.no_grad():
                predictions = trainer.model(batch)
                loss = trainer.loss_fn(predictions, batch)
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
            loss = trainer.loss_fn(predictions, batch) / self.accumulation_steps

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
                loss = trainer.loss_fn(predictions, batch)
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


# ---- AMP tests ----


def test_default_train_step_with_amp_bfloat16():
    """Verify DefaultTrainStep with bf16-mixed precision trains correctly on CPU."""
    model = SimpleModel()

    trainer = Trainer(
        model=model,
        loss_fn=simple_loss_fn,
        optimizer_factory=simple_optimizer_factory,
        train_step=DefaultTrainStep(),
        autocast=torch.bfloat16,
    )

    state = TrainState()
    batch = _make_batch()

    initial_params = [p.clone() for p in model.parameters()]
    outputs = trainer.train_step.on_train_batch(trainer, state, batch)

    assert "loss" in outputs
    assert "predictions" in outputs
    # Parameters should have been updated
    for initial, current in zip(initial_params, model.parameters()):
        assert not torch.equal(initial, current)


def test_default_train_step_amp_backward_compatible():
    """Verify DefaultTrainStep() with no args works identically to before."""
    model = SimpleModel()

    trainer = Trainer(
        model=model,
        loss_fn=simple_loss_fn,
        optimizer_factory=simple_optimizer_factory,
        train_step=DefaultTrainStep(),
    )

    state = TrainState()
    batch = _make_batch()

    outputs = trainer.train_step.on_train_batch(trainer, state, batch)
    assert "loss" in outputs
    assert "predictions" in outputs
    assert "loss" in state["train"]


def test_default_eval_step_with_amp_bfloat16():
    """Verify DefaultEvalStep with bf16-mixed precision works on CPU."""
    model = SimpleModel()

    trainer = Trainer(
        model=model,
        loss_fn=simple_loss_fn,
        optimizer_factory=simple_optimizer_factory,
        eval_step=DefaultEvalStep(no_grad=True),
        autocast=torch.bfloat16,
    )

    state = TrainState()
    batch = _make_batch()

    outputs = trainer.eval_step.on_eval_batch(trainer, state, batch)
    assert "loss" in outputs
    assert "predictions" in outputs
    # no_grad=True path → no autograd graph even under autocast
    assert not outputs["loss"].requires_grad


# ---- on_after_backward hook point tests ----


def test_on_after_backward_fires_with_grads_present():
    """``DefaultTrainStep.on_train_batch`` fires ``on_after_backward`` once the
    accumulation window closes, with gradients populated."""
    call_log = []

    class AfterBackwardHook(BaseHook):
        def on_after_backward(self, trainer, state):
            has_grads = any(p.grad is not None for p in trainer.model.parameters())
            call_log.append(has_grads)

    trainer = Trainer(
        model=SimpleModel(),
        loss_fn=simple_loss_fn,
        optimizer_factory=simple_optimizer_factory,
        hooks=[AfterBackwardHook()],
    )

    trainer.train_step.on_train_batch(trainer, TrainState(), _make_batch())

    assert call_log == [True]


def test_on_after_backward_fires_with_amp():
    """Same firing contract holds with AMP enabled (grads unscaled before the hook)."""
    call_log = []

    class AfterBackwardHook(BaseHook):
        def on_after_backward(self, trainer, state):
            call_log.append(any(p.grad is not None for p in trainer.model.parameters()))

    trainer = Trainer(
        model=SimpleModel(),
        loss_fn=simple_loss_fn,
        optimizer_factory=simple_optimizer_factory,
        train_step=DefaultTrainStep(),
        hooks=[AfterBackwardHook()],
        autocast=torch.bfloat16,
    )

    trainer.train_step.on_train_batch(trainer, TrainState(), _make_batch())

    assert call_log == [True]
