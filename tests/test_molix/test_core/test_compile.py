"""Tests for Trainer.compile()."""

from __future__ import annotations

import torch
import torch.nn as nn

from molix.core.trainer import Trainer


class _SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, x, **_kwargs):
        return self.linear(x)


def _make_trainer():
    model = _SimpleModel()
    loss_fn = lambda preds, batch: ((preds - batch["targets"]["y"]) ** 2).mean()
    optimizer_factory = lambda params: torch.optim.SGD(params, lr=1e-3)
    return Trainer(model, loss_fn, optimizer_factory), model


def test_compile_returns_self():
    trainer, _ = _make_trainer()
    result = trainer.compile()
    assert result is trainer


def test_compile_wraps_model():
    trainer, original_model = _make_trainer()
    trainer.compile()
    # torch.compile returns an OptimizedModule, not the original
    assert trainer.model is not original_model


def test_compile_checkpoint_sync():
    trainer, _ = _make_trainer()
    trainer.compile()
    assert trainer._checkpoint.model is trainer.model


def test_compile_chaining():
    model = _SimpleModel()
    loss_fn = lambda preds, batch: preds.mean()
    optimizer_factory = lambda params: torch.optim.SGD(params, lr=1e-3)
    trainer = Trainer(model, loss_fn, optimizer_factory).compile()
    assert trainer._checkpoint.model is trainer.model
