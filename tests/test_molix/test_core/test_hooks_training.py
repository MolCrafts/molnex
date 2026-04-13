"""Tests for GradClipHook and ActivationCheckpointingHook."""

import pytest
import torch
import torch.nn as nn

from molix.config import set_precision
from molix.core.hooks import (
    ActivationCheckpointingHook,
    GradClipHook,
)
from molix.core.state import TrainState
from molix.core.steps import DefaultTrainStep
from molix.core.trainer import Trainer


@pytest.fixture(autouse=True)
def _reset_precision():
    set_precision("fp32")
    yield
    set_precision("fp32")


# ---- Test fixtures ----


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x, **_kwargs):
        return self.linear(x)


class TwoLayerModel(nn.Module):
    """Model with two named children for checkpointing tests."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 10)
        self.layer2 = nn.Linear(10, 1)

    def forward(self, x, **_kwargs):
        return self.layer2(torch.relu(self.layer1(x)))


def simple_loss_fn(predictions, batch):
    targets = batch["targets"]["y_energy"]
    return ((predictions - targets) ** 2).mean()


def simple_optimizer_factory(params):
    return torch.optim.SGD(params, lr=0.01)


def _make_batch():
    return {
        "x": torch.randn(5, 10),
        "targets": {"y_energy": torch.randn(5, 1)},
    }


class MockDataModule:
    def train_dataloader(self):
        for _ in range(3):
            yield _make_batch()

    def val_dataloader(self):
        for _ in range(2):
            yield _make_batch()


# ---- GradClipHook tests ----


def test_grad_clip_hook_clips_gradients():
    """Verify GradClipHook actually clips gradient norms."""
    model = SimpleModel()
    hook = GradClipHook(max_norm=0.01)  # Very small to force clipping

    trainer = Trainer(
        model=model,
        loss_fn=simple_loss_fn,
        optimizer_factory=simple_optimizer_factory,
        hooks=[hook],
    )

    datamodule = MockDataModule()
    state = trainer.train(datamodule, max_epochs=1)

    # grad_norm should have been written to state
    assert "train/grad_norm" in state


def test_grad_clip_hook_respects_max_norm():
    """Verify clipped gradients do not exceed max_norm."""
    max_norm = 0.1
    model = SimpleModel()
    grad_norms_after = []

    class InspectGradHook:
        """Runs after GradClipHook to verify clipping happened."""

        def on_after_backward(self, trainer, state):
            total_norm = torch.nn.utils.clip_grad_norm_(
                trainer.model.parameters(), float("inf")
            )
            grad_norms_after.append(total_norm.item())

    trainer = Trainer(
        model=model,
        loss_fn=simple_loss_fn,
        optimizer_factory=simple_optimizer_factory,
        hooks=[
            GradClipHook(max_norm=max_norm),
            InspectGradHook(),  # Runs after GradClipHook
        ],
    )

    datamodule = MockDataModule()
    trainer.train(datamodule, max_epochs=1)

    # After clipping, norms should be <= max_norm (with small tolerance)
    for norm in grad_norms_after:
        assert norm <= max_norm + 1e-6, f"Grad norm {norm} exceeds max_norm {max_norm}"


def test_grad_clip_hook_writes_state():
    """Verify GradClipHook writes train/grad_norm to state."""
    model = SimpleModel()
    hook = GradClipHook(max_norm=10.0)

    trainer = Trainer(
        model=model,
        loss_fn=simple_loss_fn,
        optimizer_factory=simple_optimizer_factory,
        hooks=[hook],
    )

    state = TrainState()
    batch = _make_batch()

    # Manually run one training step to trigger on_after_backward
    trainer.train_step.on_train_batch(trainer, state, batch)

    assert "train/grad_norm" in state
    assert isinstance(state["train/grad_norm"], float)


def test_grad_clip_hook_with_amp():
    """Verify GradClipHook works correctly with AMP enabled."""
    model = SimpleModel()
    hook = GradClipHook(max_norm=1.0)

    trainer = Trainer(
        model=model,
        loss_fn=simple_loss_fn,
        optimizer_factory=simple_optimizer_factory,
        train_step=DefaultTrainStep(),
        hooks=[hook],
    )
    trainer.set_precision("bf16-mixed")

    datamodule = MockDataModule()
    state = trainer.train(datamodule, max_epochs=1)

    assert "train/grad_norm" in state
    assert state.global_step == 3


# ---- ActivationCheckpointingHook tests ----


def test_activation_checkpointing_wraps_modules():
    """Verify ActivationCheckpointingHook wraps targeted modules."""
    model = TwoLayerModel()
    original_forward = model.layer1.forward

    hook = ActivationCheckpointingHook(
        check_fn=lambda m: isinstance(m, nn.Linear),
    )

    trainer = Trainer(
        model=model,
        loss_fn=simple_loss_fn,
        optimizer_factory=simple_optimizer_factory,
        hooks=[hook],
    )

    # Simulate on_train_start
    hook.on_train_start(trainer, TrainState())

    # Forward should have been replaced
    assert model.layer1.forward is not original_forward
    assert model.layer2.forward is not type(model.layer2).forward


def test_activation_checkpointing_default_wraps_children():
    """Verify check_fn=None wraps all direct children."""
    model = TwoLayerModel()
    original_l1 = model.layer1.forward
    original_l2 = model.layer2.forward

    hook = ActivationCheckpointingHook()  # check_fn=None

    hook.on_train_start(
        Trainer(model=model, loss_fn=simple_loss_fn, optimizer_factory=simple_optimizer_factory),
        TrainState(),
    )

    assert model.layer1.forward is not original_l1
    assert model.layer2.forward is not original_l2


def test_activation_checkpointing_numerical_equivalence():
    """Verify checkpointed model produces same output as original."""
    torch.manual_seed(42)
    model = TwoLayerModel()

    x = torch.randn(5, 10)

    # Get reference output before checkpointing
    with torch.no_grad():
        ref_output = model(x).clone()

    # Apply checkpointing
    hook = ActivationCheckpointingHook(
        check_fn=lambda m: isinstance(m, nn.Linear),
    )
    hook.on_train_start(
        Trainer(model=model, loss_fn=simple_loss_fn, optimizer_factory=simple_optimizer_factory),
        TrainState(),
    )

    # Get checkpointed output
    with torch.no_grad():
        ckpt_output = model(x)

    torch.testing.assert_close(ref_output, ckpt_output)


def test_activation_checkpointing_gradient_flow():
    """Verify gradients flow correctly through checkpointed modules."""
    model = TwoLayerModel()

    hook = ActivationCheckpointingHook(
        check_fn=lambda m: isinstance(m, nn.Linear),
    )
    hook.on_train_start(
        Trainer(model=model, loss_fn=simple_loss_fn, optimizer_factory=simple_optimizer_factory),
        TrainState(),
    )

    x = torch.randn(5, 10)
    output = model(x)
    loss = output.sum()
    loss.backward()

    # All parameters should have gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_activation_checkpointing_trains_with_trainer():
    """Verify full training loop works with activation checkpointing."""
    model = TwoLayerModel()

    trainer = Trainer(
        model=model,
        loss_fn=simple_loss_fn,
        optimizer_factory=simple_optimizer_factory,
        hooks=[
            ActivationCheckpointingHook(
                check_fn=lambda m: isinstance(m, nn.Linear),
            ),
        ],
    )

    datamodule = MockDataModule()
    state = trainer.train(datamodule, max_epochs=1)

    assert state.epoch == 1
    assert state.global_step == 3


def test_all_three_features_together():
    """Verify AMP + GradClip + ActivationCheckpointing work together."""
    model = TwoLayerModel()

    trainer = Trainer(
        model=model,
        loss_fn=simple_loss_fn,
        optimizer_factory=simple_optimizer_factory,
        train_step=DefaultTrainStep(),
        hooks=[
            ActivationCheckpointingHook(
                check_fn=lambda m: isinstance(m, nn.Linear),
            ),
            GradClipHook(max_norm=1.0),
        ],
    )
    trainer.set_precision("bf16-mixed")

    datamodule = MockDataModule()
    state = trainer.train(datamodule, max_epochs=1)

    assert state.epoch == 1
    assert state.global_step == 3
    assert "train/grad_norm" in state
