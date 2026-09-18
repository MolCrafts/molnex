"""Tests for MolRecMetricsHook."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from molix.hooks.molrec_metrics import MolRecMetricsHook
from molix.io.metrics import read_metrics
from molix.io.run_record import read_meta, read_status


class _FakeState:
    def __init__(self) -> None:
        self.global_step = 0
        self._ns: dict[str, dict[str, Any]] = {
            "train": {},
            "eval": {},
            "performance": {},
            "gpu": {},
        }

    def __getitem__(self, key: str) -> dict[str, Any]:
        return self._ns[key]


def test_hook_writes_jsonl_and_status(tmp_path: Path) -> None:
    root = tmp_path / "record"
    hook = MolRecMetricsHook(root, every_n_steps=1, flush_every_n_appends=0)
    state = _FakeState()
    trainer = SimpleNamespace()

    hook.on_train_start(trainer, state)
    state.global_step = 1
    state["train"]["loss"] = 0.4
    state["performance"]["step_per_second"] = 12.0
    hook.on_train_batch_end(trainer, state, batch=None, outputs=None)

    state.global_step = 2
    state["eval"]["MAE"] = 0.05
    hook.on_eval_step_complete(trainer, state)
    hook.on_train_end(trainer, state)

    result = read_metrics(root, metric_type="scalar")
    keys = {r["k"] for r in result.records}
    assert "train/loss" in keys
    assert "performance/step_per_second" in keys
    assert "eval/MAE" in keys

    status = read_status(root)
    assert status["state"] == "succeeded"
    meta = read_meta(root)
    assert meta["molrec_version"] == 1


def test_hook_respects_every_n_steps(tmp_path: Path) -> None:
    root = tmp_path / "record"
    hook = MolRecMetricsHook(root, every_n_steps=2, flush_every_n_appends=0)
    state = _FakeState()
    trainer = SimpleNamespace()
    hook.on_train_start(trainer, state)

    state.global_step = 1
    state["train"]["loss"] = 1.0
    hook.on_train_batch_end(trainer, state, None, None)
    state.global_step = 2
    state["train"]["loss"] = 0.5
    hook.on_train_batch_end(trainer, state, None, None)
    hook.on_train_end(trainer, state)

    result = read_metrics(root, key="train/loss")
    assert len(result.records) == 1
    assert result.records[0]["v"] == 0.5


def test_hook_throttles_status_heartbeat(tmp_path: Path) -> None:
    """Train-phase status writes follow their own stride, not every logged step."""
    root = tmp_path / "record"
    hook = MolRecMetricsHook(root, every_n_steps=1, flush_every_n_appends=0, status_every_n_steps=4)
    state = _FakeState()
    trainer = SimpleNamespace()
    hook.on_train_start(trainer, state)

    for step in range(1, 7):
        state.global_step = step
        state["train"]["loss"] = float(step)
        hook.on_train_batch_end(trainer, state, None, None)

    # Six logged steps, but the last heartbeat is the last multiple of 4.
    assert read_status(root)["global_step"] == 4
    assert len(read_metrics(root, key="train/loss").records) == 6
