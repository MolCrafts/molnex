"""Tests for Log / ScalarHook / GPUMemoryHook."""

from __future__ import annotations

import pytest
import torch

from molix.core.hook import ScalarHook
from molix.core.state import TrainState
from molix.hooks import (
    GPUMemoryHook,
    Log,
    StepSpeedHook,
)


def test_log_prints_header_on_train_start(capsys):
    log = Log(10, keys=[("train", "loss"), ("performance", "step_per_second")])
    state = TrainState()

    log.on_train_start(trainer=None, state=state)

    # 2-row header: top = category, bottom = item.
    lines = capsys.readouterr().out.rstrip("\n").split("\n")
    assert len(lines) == 2
    assert lines[0].split() == ["train", "performance"]
    assert lines[1].split() == ["step", "epoch", "loss", "step_per_second"]


def test_log_accepts_scalar_hook_instances(capsys):
    speed = StepSpeedHook()
    gpu = GPUMemoryHook()
    log = Log(10, keys=[speed, gpu, ("train", "loss")])

    log.on_train_start(trainer=None, state=TrainState())
    lines = capsys.readouterr().out.rstrip("\n").split("\n")
    assert len(lines) == 2
    assert lines[0].split() == ["performance", "gpu", "gpu", "gpu", "train"]
    assert lines[1].split() == [
        "step",
        "epoch",
        "step_per_second",
        "alloc_gib",
        "resv_gib",
        "peak_gib",
        "loss",
    ]


def test_log_deduplicates_keys():
    speed = StepSpeedHook()
    log = Log(1, keys=[speed, ("performance", "step_per_second"), speed])
    assert log.keys == [("performance", "step_per_second")]


def test_log_prints_row_every_n_steps(capsys):
    log = Log(3, keys=[("train", "loss")])
    state = TrainState()

    log.on_train_start(trainer=None, state=state)
    capsys.readouterr()  # drop header

    for i in range(7):
        state["global_step"] = i
        state["train"]["loss"] = float(i)
        log.on_train_batch_end(trainer=None, state=state, batch=None, outputs=None)

    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    # 7 calls, every_n=3 => prints on calls 3 and 6
    assert len(lines) == 2
    assert "2" in lines[0].split()[-1]  # loss=2.0 at step=2
    assert "5" in lines[1].split()[-1]  # loss=5.0 at step=5


def test_log_renders_missing_path_as_dash(capsys):
    """Missing paths render as ``—``, not ``nan`` — keeps real numerical NaN
    visually unambiguous in the log."""
    log = Log(1, keys=[("not", "present")])
    state = TrainState()

    log.on_train_start(trainer=None, state=state)
    capsys.readouterr()
    log.on_train_batch_end(trainer=None, state=state, batch=None, outputs=None)
    row = capsys.readouterr().out.strip()
    assert "—" in row
    assert "nan" not in row


def test_log_renders_real_nan_as_nan(capsys):
    """A real numerical NaN must still render as ``"nan"`` so divergence is
    visible. Distinct from the missing-path marker."""
    log = Log(1, keys=[("train", "loss")])
    state = TrainState()

    log.on_train_start(trainer=None, state=state)
    capsys.readouterr()
    state["global_step"] = 0
    state["train"]["loss"] = float("nan")
    log.on_train_batch_end(trainer=None, state=state, batch=None, outputs=None)
    row = capsys.readouterr().out.strip()
    assert "nan" in row
    assert "—" not in row


def test_log_accepts_slash_string_keys(capsys):
    """A slash-separated string key (``"train/loss"``) is parsed into a
    tuple path so it resolves the same as ``("train", "loss")``."""
    log = Log(1, keys=["train/loss", "performance/step_per_second"])
    assert log.keys == [("train", "loss"), ("performance", "step_per_second")]


def test_log_validates_keys_against_advertised_paths():
    """Startup validation rejects keys not advertised by any registered hook
    nor in the built-in set — catches the silent-nan failure mode."""

    class _FakeTrainer:
        hooks = []  # no hooks registered → only built-ins are advertised

    # ``("train", "loss")`` is a built-in (DefaultTrainStep) — accepted.
    Log(1, keys=[("train", "loss")]).on_train_start(trainer=_FakeTrainer(), state=TrainState())

    # ``("performance", "step_per_second")`` requires StepSpeedHook —
    # not registered → ValueError.
    log = Log(1, keys=[("performance", "step_per_second")])
    with pytest.raises(ValueError, match="not advertised"):
        log.on_train_start(trainer=_FakeTrainer(), state=TrainState())


def test_log_validation_accepts_advertised_hook_paths(capsys):
    """A key advertised by a registered ScalarHook passes validation."""
    speed = StepSpeedHook()

    class _FakeTrainer:
        hooks = [speed]

    log = Log(1, keys=["performance/step_per_second"])
    log.on_train_start(trainer=_FakeTrainer(), state=TrainState())  # no raise


def test_log_validation_skipped_when_trainer_none():
    """``trainer=None`` disables validation so unit tests can drive Log
    without a real Trainer."""
    log = Log(1, keys=[("not", "advertised")])
    log.on_train_start(trainer=None, state=TrainState())  # no raise


def test_log_rejects_bad_key_type():
    with pytest.raises(TypeError):
        Log(1, keys=[42])


def test_log_rejects_non_positive_interval():
    with pytest.raises(ValueError):
        Log(0, keys=["x"])


def test_log_rejects_non_positive_header_interval():
    with pytest.raises(ValueError):
        Log(1, keys=["x"], header_every_n_rows=0)


def test_log_reprints_header_every_n_rows(capsys):
    """P2: header is reprinted every ``header_every_n_rows`` data rows."""
    log = Log(1, keys=[("train", "loss")], header_every_n_rows=3)
    state = TrainState()

    log.on_train_start(trainer=None, state=state)
    for i in range(7):
        state["global_step"] = i
        state["train"]["loss"] = float(i)
        log.on_train_batch_end(trainer=None, state=state, batch=None, outputs=None)

    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    # 1 start-header + 7 rows + 2 reprinted headers (after rows 3 and 6).
    header_line = lines[0]
    header_occurrences = sum(1 for ln in lines if ln == header_line)
    assert header_occurrences == 3, lines


def test_log_draws_epoch_separator(capsys):
    """P3: epoch transition draws a thin separator before the next row."""
    log = Log(1, keys=[("train", "loss")], header_every_n_rows=1000)
    state = TrainState()

    log.on_train_start(trainer=None, state=state)
    capsys.readouterr()  # discard start-of-run header

    # Log a row in epoch 0, cross to epoch 1, log another row.
    state["global_step"] = 0
    state["epoch"] = 0
    state["train"]["loss"] = 1.0
    log.on_train_batch_end(trainer=None, state=state, batch=None, outputs=None)

    state["epoch"] = 1
    log.on_epoch_end(trainer=None, state=state)

    state["global_step"] = 1
    state["train"]["loss"] = 2.0
    log.on_train_batch_end(trainer=None, state=state, batch=None, outputs=None)

    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.strip()]
    # Expect: [row0, separator, header_top (train), header_bot (step…), row1]
    assert lines[0].strip().split()[0] == "1"
    assert set(lines[1]) == {"─"}
    assert "train" in lines[2]
    assert "step" in lines[3]
    assert lines[4].strip().split()[0] == "2"


def test_log_epoch_separator_can_be_disabled(capsys):
    log = Log(1, keys=[("train", "loss")], epoch_separator=False)
    state = TrainState()
    log.on_train_start(trainer=None, state=state)
    capsys.readouterr()

    state["epoch"] = 1
    log.on_epoch_end(trainer=None, state=state)

    out = capsys.readouterr().out
    assert "─" not in out


def test_log_announce_renders_inline_separator(capsys):
    """P4: announce() inlines an event without breaking column alignment."""
    log = Log(1, keys=[("train", "loss")])
    state = TrainState()

    log.on_train_start(trainer=None, state=state)
    capsys.readouterr()

    log.announce("ckpt: last.pt @ step=3000")

    out = capsys.readouterr().out.rstrip("\n")
    assert out.startswith("─── ckpt: last.pt @ step=3000 ")
    assert out.endswith("─")
    assert len(out) == log._table_width()


def test_log_announce_before_start_is_noop(capsys):
    """Announcing before :meth:`on_train_start` must not print a stray line."""
    log = Log(1, keys=[("train", "loss")])
    log.announce("anything")
    assert capsys.readouterr().out == ""


def test_log_announce_forces_header_reprint_on_next_row(capsys):
    log = Log(1, keys=[("train", "loss")], header_every_n_rows=1000)
    state = TrainState()
    log.on_train_start(trainer=None, state=state)
    capsys.readouterr()

    log.announce("lr reduced")

    state["global_step"] = 0
    state["train"]["loss"] = 0.5
    log.on_train_batch_end(trainer=None, state=state, batch=None, outputs=None)

    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert lines[0].startswith("─── lr reduced ")
    assert "train" in lines[1]
    assert "step" in lines[2]
    assert lines[3].strip().split()[0] == "1"


def test_gpu_memory_hook_noop_on_cpu():
    hook = GPUMemoryHook()
    state = TrainState()

    hook.on_train_start(trainer=None, state=state)
    hook.on_train_batch_end(trainer=None, state=state, batch=None, outputs=None)

    if torch.cuda.is_available():
        assert state["gpu"]["alloc_gib"] >= 0.0
    else:
        assert state["gpu"]["alloc_gib"] == 0.0
        assert state["gpu"]["resv_gib"] == 0.0
        assert state["gpu"]["peak_gib"] == 0.0


def test_gpu_memory_hook_selects_metrics():
    hook = GPUMemoryHook(metrics=("peak",))
    assert hook.scalar_keys == (("gpu", "peak_gib"),)

    state = TrainState()
    hook.on_train_start(trainer=None, state=state)
    hook.on_train_batch_end(trainer=None, state=state, batch=None, outputs=None)

    # Only the requested key is written; others are absent.
    assert "peak_gib" in state["gpu"]
    assert "alloc_gib" not in state["gpu"]
    assert "resv_gib" not in state["gpu"]


def test_gpu_memory_hook_rejects_unknown_metrics():
    with pytest.raises(ValueError, match="unknown metric"):
        GPUMemoryHook(metrics=("alloc", "bogus"))
    with pytest.raises(ValueError, match="at least one metric"):
        GPUMemoryHook(metrics=())


def test_step_speed_hook_is_scalar_hook():
    h = StepSpeedHook()
    assert isinstance(h, ScalarHook)
    assert h.scalar_keys == (("performance", "step_per_second"),)


def test_metrics_hook_scalar_keys_are_dynamic():
    # MetricsHook.scalar_keys is a @property; values depend on metrics list.
    from molix.hooks import MetricsHook

    class _FakeMetric:
        def reset(self):
            pass

    class _MAE(_FakeMetric):
        pass

    class _RMSE(_FakeMetric):
        pass

    hook = MetricsHook(
        metrics=[_MAE(), _RMSE()],
        pred_key="p",
        target_key="t",
        prefix_train="train",
        prefix_val="val",
    )
    assert hook.scalar_keys == (
        ("train", "_MAE"),
        ("train", "_RMSE"),
        ("val", "_MAE"),
        ("val", "_RMSE"),
    )
