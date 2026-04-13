"""Tests for Log / ScalarHook / GPUMemoryHook."""

from __future__ import annotations

import pytest
import torch

from molix.core.hooks import (
    GPUMemoryHook,
    Log,
    ScalarHook,
    StepSpeedHook,
)
from molix.core.state import TrainState


def test_log_prints_header_on_train_start(capsys):
    log = Log(10, keys=["train/loss", "performance/step_per_second"])
    state = TrainState()

    log.on_train_start(trainer=None, state=state)

    header = capsys.readouterr().out.strip().split()
    assert header == ["step", "epoch", "train/loss", "performance/step_per_second"]


def test_log_accepts_scalar_hook_instances(capsys):
    speed = StepSpeedHook()
    gpu = GPUMemoryHook()
    log = Log(10, keys=[speed, gpu, "train/loss"])

    log.on_train_start(trainer=None, state=TrainState())
    header = capsys.readouterr().out.strip().split()
    assert header == [
        "step",
        "epoch",
        "performance/step_per_second",
        "gpu/alloc_gib",
        "gpu/resv_gib",
        "gpu/peak_gib",
        "train/loss",
    ]


def test_log_deduplicates_keys():
    speed = StepSpeedHook()
    log = Log(1, keys=[speed, "performance/step_per_second", speed])
    assert log.keys == ["performance/step_per_second"]


def test_log_prints_row_every_n_steps(capsys):
    log = Log(3, keys=["train/loss"])
    state = TrainState()

    log.on_train_start(trainer=None, state=state)
    capsys.readouterr()  # drop header

    for i in range(7):
        state["global_step"] = i
        state["train/loss"] = float(i)
        log.on_train_batch_end(trainer=None, state=state, batch=None, outputs=None)

    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    # 7 calls, every_n=3 => prints on calls 3 and 6
    assert len(lines) == 2
    assert "2" in lines[0].split()[-1]  # loss=2.0 at step=2
    assert "5" in lines[1].split()[-1]  # loss=5.0 at step=5


def test_log_handles_missing_keys_as_nan(capsys):
    log = Log(1, keys=["not/present"])
    state = TrainState()

    log.on_train_start(trainer=None, state=state)
    capsys.readouterr()
    log.on_train_batch_end(trainer=None, state=state, batch=None, outputs=None)
    row = capsys.readouterr().out.strip()
    assert "nan" in row


def test_log_rejects_bad_key_type():
    with pytest.raises(TypeError):
        Log(1, keys=[42])


def test_log_rejects_non_positive_interval():
    with pytest.raises(ValueError):
        Log(0, keys=["x"])


def test_gpu_memory_hook_noop_on_cpu():
    hook = GPUMemoryHook()
    state = TrainState()

    hook.on_train_start(trainer=None, state=state)
    hook.on_train_batch_end(trainer=None, state=state, batch=None, outputs=None)

    if torch.cuda.is_available():
        assert state["gpu/alloc_gib"] >= 0.0
    else:
        assert state["gpu/alloc_gib"] == 0.0
        assert state["gpu/resv_gib"] == 0.0
        assert state["gpu/peak_gib"] == 0.0


def test_step_speed_hook_is_scalar_hook():
    h = StepSpeedHook()
    assert isinstance(h, ScalarHook)
    assert h.scalar_keys == ("performance/step_per_second",)


def test_metrics_hook_scalar_keys_are_dynamic():
    # MetricsHook.scalar_keys is a @property; values depend on metrics list.
    from molix.core.hooks import MetricsHook

    class _FakeMetric:
        def reset(self): pass

    class _MAE(_FakeMetric): pass
    class _RMSE(_FakeMetric): pass

    hook = MetricsHook(
        metrics=[_MAE(), _RMSE()],
        pred_key="p",
        target_key="t",
        prefix_train="train",
        prefix_val="val",
    )
    assert hook.scalar_keys == ("train/_MAE", "train/_RMSE", "val/_MAE", "val/_RMSE")
