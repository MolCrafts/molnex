"""Tests for ``Trainer.compile`` and the ``_traced_forward_loss`` seam it swaps."""

from __future__ import annotations

from typing import Any
from unittest import mock

import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from molix.core.steps import (
    DefaultEvalStep,
    DefaultTrainStep,
    MultiTaskEvalStep,
    MultiTaskTrainStep,
    TaskRouter,
    TaskSpec,
)
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


def _make_trainer(**kwargs: Any) -> tuple[Trainer, _SimpleModel]:
    model = _SimpleModel()
    return Trainer(model, _mse_loss, _sgd_factory, **kwargs), model


def _batch() -> dict[str, Any]:
    # A TensorDict, not a plain dict: the batch schema is a nested TensorDict and
    # the Trainer moves batches with `TensorDict.to`, which a dict has no answer
    # for. A dict stand-in here meant these tests exercised a shape production
    # never produces.
    return TensorDict(
        {"x": torch.randn(5, 4), "targets": TensorDict({"y": torch.randn(5, 1)}, batch_size=[5])},
        batch_size=[5],
    )


class _DataModule:
    def train_dataloader(self):
        for _ in range(3):
            yield _batch()

    def val_dataloader(self):
        for _ in range(1):
            yield _batch()


class _RecordingForwardLoss:
    """Counts calls through the one attribute both ``smoke`` and ``train`` must use."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def __call__(self, *args):
        self.calls += 1
        return self.inner(*args)


def test_compile_returns_self():
    trainer, _ = _make_trainer()
    with mock.patch("molix.compile.torch.compile", side_effect=lambda m, **k: m):
        result = trainer.compile()
    assert result is trainer


def test_compile_step_swaps_the_traced_body_on_both_steps_and_leaves_the_model_alone():
    """The traced body is swapped; ``forward_loss`` (the eager leaf prologue) is not."""
    trainer, model = _make_trainer()
    wrapped: dict[tuple[object, object], object] = {}

    def fake_compile(target, **_):
        # A bound method is a fresh object per attribute access: key on what it binds.
        wrapped[(target.__self__, target.__func__)] = sentinel = mock.Mock(name="compiled")
        return sentinel

    with mock.patch("molix.compile.torch.compile", side_effect=fake_compile):
        trainer.compile()

    train_step, eval_step = trainer.train_step, trainer.eval_step
    assert (
        train_step._traced_forward_loss
        is wrapped[(train_step, DefaultTrainStep._traced_forward_loss)]
    )
    assert (
        eval_step._traced_forward_loss is wrapped[(eval_step, DefaultEvalStep._traced_forward_loss)]
    )
    assert train_step.forward_loss.__func__ is DefaultTrainStep.forward_loss
    assert eval_step.forward_loss.__func__ is DefaultEvalStep.forward_loss
    assert trainer.model is model
    assert trainer._checkpoint.model is model


def test_compile_model_only_wraps_the_model_and_syncs_the_checkpoint():
    trainer, original_model = _make_trainer()
    sentinel = _SimpleModel()
    with mock.patch("molix.compile.torch.compile", return_value=sentinel):
        trainer.compile(step=False)
    # step=False installs whatever torch.compile returns in place of the model.
    assert trainer.model is sentinel
    assert trainer.model is not original_model
    assert trainer._checkpoint.model is sentinel
    assert trainer.train_step._traced_forward_loss.__func__ is DefaultTrainStep._traced_forward_loss


def test_compile_cuda_graphs_applies_preset():
    """``compile(cuda_graphs=True)`` forwards exactly the preset to torch.compile."""
    trainer, _ = _make_trainer()
    with mock.patch("molix.compile.torch.compile", side_effect=lambda m, **k: m) as mc:
        trainer.compile(cuda_graphs=True)
    _, kwargs = mc.call_args
    assert kwargs["backend"] == "inductor"
    assert kwargs["fullgraph"] is True
    assert kwargs["dynamic"] is False
    assert kwargs["mode"] == "reduce-overhead"


@pytest.mark.parametrize("side", ["train_step", "eval_step"])
def test_compile_step_traces_the_routed_body(side: str):
    """The task-label read stays in the prologue; compile wraps the body."""
    router = TaskRouter([TaskSpec("energy", 0, _mse_loss)])
    step = MultiTaskTrainStep(router) if side == "train_step" else MultiTaskEvalStep(router)
    trainer, _ = _make_trainer(**{side: step})
    sentinel = object()
    with mock.patch("molix.compile.torch.compile", return_value=sentinel):
        trainer.compile()
    assert getattr(trainer, side)._traced_forward_loss is sentinel
    assert step.forward_loss.__func__ is type(step).forward_loss


def test_wrap_distributed_is_a_no_op_without_a_process_group():
    trainer, model = _make_trainer()
    trainer._wrap_distributed()
    assert trainer.model is model


def test_wrap_distributed_refuses_a_compiled_step(monkeypatch: pytest.MonkeyPatch):
    import torch.distributed as dist

    trainer, _ = _make_trainer()
    trainer.train_step._traced_forward_loss = lambda *_args, **_kwargs: None
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    with pytest.raises(ValueError, match="compile\\(step=True\\)"):
        trainer._wrap_distributed()


def test_wrap_distributed_puts_ddp_around_the_module(monkeypatch: pytest.MonkeyPatch):
    import torch.distributed as dist

    class _FakeDDP(nn.Module):
        def __init__(self, module: nn.Module, **kwargs: object) -> None:
            super().__init__()
            self.module = module
            self.kwargs = kwargs

    trainer, model = _make_trainer()
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", _FakeDDP)
    trainer._wrap_distributed()
    assert isinstance(trainer.model, _FakeDDP)
    assert trainer.model.module is model
    assert trainer.model.kwargs["find_unused_parameters"] is True
    assert trainer._checkpoint.model is trainer.model


def test_smoke_and_train_run_the_same_forward_loss():
    """One attribute, ``train_step.forward_loss``, serves both — what ``compile`` swaps."""
    trainer, _ = _make_trainer()
    recording = _RecordingForwardLoss(trainer.train_step.forward_loss)
    trainer.train_step.forward_loss = recording  # type: ignore[method-assign]

    trainer.smoke(_DataModule(), steps=2)
    assert recording.calls == 2

    trainer.train(_DataModule(), max_steps=3)
    assert recording.calls == 5


class _ExtraStateModel(_SimpleModel):
    def get_extra_state(self) -> dict[str, int]:
        return {"seed_tables": 1}

    def set_extra_state(self, state: object) -> None:
        del state


def test_smoke_snapshots_non_tensor_extra_state() -> None:
    """Class-I state_dicts carry nested dicts; smoke must not call ``.detach()`` on them."""
    model = _ExtraStateModel()
    trainer = Trainer(model, _mse_loss, _sgd_factory)
    trainer.smoke(_DataModule(), steps=1)


class _LateBindModel(_SimpleModel):
    """Mirrors Class-I: compiled smoke ``add_module``s live tables onto ``_physics``."""

    def forward(self, batch):
        if "_physics" not in self._modules:
            physics = nn.Linear(1, 1)
            physics.register_buffer("k", torch.ones(3))
            self.add_module("_physics", physics)
        return super().forward(batch)


def test_smoke_restores_when_warmup_registers_new_module() -> None:
    """Live OPLS tables registered during smoke must not block the snapshot restore."""
    model = _LateBindModel()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    assert not any(k.startswith("_physics") for k in before)
    trainer = Trainer(model, _mse_loss, _sgd_factory)
    trainer.smoke(_DataModule(), steps=1)
    after = model.state_dict()
    assert not any(k.startswith("_physics") for k in after)
    assert before.keys() == after.keys()
    for key, value in before.items():
        assert torch.equal(after[key], value)
