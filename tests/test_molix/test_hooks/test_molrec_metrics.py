"""Tests for MolRecMetricsHook."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from molix.hooks.molrec_metrics import MolRecMetricsHook
from molix.io.metrics import read_metrics


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

    status = json.loads((root / "status" / "status.json").read_text())
    assert status["state"] == "succeeded"
    meta = json.loads((root / "meta" / "meta.json").read_text())
    assert meta["record_schema_version"] == 1


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
