"""Tests for GradClipHook and ActivationCheckpointingHook."""

import pytest
import torch
import torch.nn as nn

from molix.config import config
from molix.core.state import TrainState
from molix.core.trainer import Trainer
from molix.hooks import (
    ActivationCheckpointingHook,
    GradClipHook,
)


@pytest.fixture(autouse=True)
def _reset_precision():
    config.set_precision("fp32")
    yield
    config.set_precision("fp32")


# ---- Test fixtures ----


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, batch):
        return self.linear(batch["x"])


class TwoLayerModel(nn.Module):
    """Model with two named children for checkpointing tests."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 10)
        self.layer2 = nn.Linear(10, 1)

    def forward(self, batch):
        return self.layer2(torch.relu(self.layer1(batch["x"])))


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


# ---- GradClipHook tests ----


def _trainer_with(*hooks):
    return Trainer(
        model=SimpleModel(),
        loss_fn=simple_loss_fn,
        optimizer_factory=simple_optimizer_factory,
        hooks=list(hooks),
    )


def test_grad_clip_hook_caps_grad_norm():
    """``GradClipHook.on_after_backward`` clips the global grad norm to max_norm."""
    torch.manual_seed(0)
    trainer = _trainer_with()
    # Populate large gradients, then clip via the hook directly (no train loop).
    (trainer.model(_make_batch()).sum() * 1000.0).backward()
    max_norm = 0.1
    GradClipHook(max_norm=max_norm).on_after_backward(trainer, TrainState())
    post = torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), float("inf"))
    assert float(post) <= max_norm + 1e-6


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

    # Stored as a 0-d device tensor (not float()-ed) so the optimizer step
    # incurs no CPU<->GPU sync; consumers materialise on their own cadence.
    assert "grad_norm" in state["train"]
    grad_norm = state["train"]["grad_norm"]
    assert isinstance(grad_norm, torch.Tensor)
    assert grad_norm.ndim == 0
    assert float(grad_norm) >= 0.0


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

    batch = {"x": torch.randn(5, 10)}

    # Get reference output before checkpointing
    with torch.no_grad():
        ref_output = model(batch).clone()

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
        ckpt_output = model(batch)

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

    output = model({"x": torch.randn(5, 10)})
    loss = output.sum()
    loss.backward()

    # All parameters should have gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


