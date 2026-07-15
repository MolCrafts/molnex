"""Tests for Trainer.compile()."""

from __future__ import annotations

from unittest import mock

import torch
import torch.nn as nn

from molix.compile import Compiler
from molix.core.trainer import Trainer


class _SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, batch):
        return self.linear(batch["x"])


def _mse_loss(preds, batch):
    return ((preds - batch["targets"]["y"]) ** 2).mean()


def _sgd_factory(params):
    return torch.optim.SGD(params, lr=1e-3)


def _make_trainer():
    model = _SimpleModel()
    return Trainer(model, _mse_loss, _sgd_factory), model


def test_compile_returns_self():
    trainer, _ = _make_trainer()
    with mock.patch("molix.compile.torch.compile", side_effect=lambda m, **k: m):
        result = trainer.compile()
    assert result is trainer


def test_compile_wraps_model():
    trainer, original_model = _make_trainer()
    sentinel = _SimpleModel()
    with mock.patch("molix.compile.torch.compile", return_value=sentinel):
        trainer.compile()
    # compile() installs whatever torch.compile returns in place of the original.
    assert trainer.model is sentinel
    assert trainer.model is not original_model


def test_compile_checkpoint_sync():
    trainer, _ = _make_trainer()
    sentinel = _SimpleModel()
    with mock.patch("molix.compile.torch.compile", return_value=sentinel):
        trainer.compile()
    assert trainer._checkpoint.model is trainer.model is sentinel


def test_cuda_graphs_preset_values():
    """The named preset is exactly the benchmarked winning force-training config."""
    assert Compiler.CUDA_GRAPH_PRESET == {
        "backend": "inductor",
        "fullgraph": True,
        "dynamic": False,
        "mode": "reduce-overhead",
    }


def test_compile_cuda_graphs_applies_preset():
    """``compile(cuda_graphs=True)`` forwards exactly the preset to torch.compile."""
    trainer, model = _make_trainer()
    with mock.patch("molix.compile.torch.compile", return_value=model) as mc:
        trainer.compile(cuda_graphs=True)
    _, kwargs = mc.call_args
    assert kwargs["backend"] == "inductor"
    assert kwargs["fullgraph"] is True
    assert kwargs["dynamic"] is False
    assert kwargs["mode"] == "reduce-overhead"


def test_compile_cuda_graphs_overrides_explicit_args():
    """The preset wins over conflicting explicit flags (it is a named preset)."""
    trainer, model = _make_trainer()
    with mock.patch("molix.compile.torch.compile", return_value=model) as mc:
        trainer.compile(cuda_graphs=True, dynamic=True, mode="max-autotune")
    _, kwargs = mc.call_args
    assert kwargs["dynamic"] is False
    assert kwargs["mode"] == "reduce-overhead"
