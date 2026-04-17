"""Tests for :mod:`molix.data.execute` — run / transform / task-state helpers."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from molix.data.execute import (
    call_task,
    collect_task_states,
    load_task_states,
    run,
    transform,
)
from molix.data.pipeline import Pipeline
from molix.data.source import InMemorySource
from molix.data.task import BatchTask, DatasetTask, SampleTask


# ---------------------------------------------------------------------------
# Instrumented tasks
# ---------------------------------------------------------------------------


class CountingSample(SampleTask):
    def __init__(self, name: str = "counter") -> None:
        self._name = name
        self.calls = 0

    @property
    def task_id(self) -> str:
        return f"counting:{self._name}"

    def execute(self, data: dict) -> dict:
        self.calls += 1
        return {**data, self._name: True}


class MeanShift(DatasetTask):
    def __init__(self, key: str = "y") -> None:
        self.key = key
        self.mean = 0.0
        self.fit_calls = 0
        self.exec_calls = 0

    @property
    def task_id(self) -> str:
        return f"mean_shift:{self.key}"

    def fit(self, samples: list[dict]) -> None:
        self.fit_calls += 1
        ys = [float(s[self.key].item()) for s in samples]
        self.mean = sum(ys) / len(ys)

    def execute(self, data: dict) -> dict:
        self.exec_calls += 1
        return {**data, self.key: data[self.key] - self.mean}

    def state_dict(self) -> dict[str, Any]:
        return {"mean": torch.tensor(self.mean, dtype=torch.float64)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.mean = float(state["mean"].item())


def _samples(n: int = 8) -> list[dict]:
    return [
        {
            "Z": torch.tensor([1, 6]),
            "pos": torch.zeros(2, 3),
            "y": torch.tensor([float(i)]),
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# call_task
# ---------------------------------------------------------------------------


class TestCallTask:
    def test_dispatches_runnable(self):
        t = CountingSample()
        out = call_task(t, {"y": torch.tensor([0.0])})
        assert out["counter"] is True
        assert t.calls == 1

    def test_dispatches_bare_callable(self):
        out = call_task(lambda s: {**s, "tag": True}, {"y": 0})
        assert out["tag"] is True

    def test_rejects_non_callable(self):
        with pytest.raises(TypeError):
            call_task(42, {})


# ---------------------------------------------------------------------------
# transform
# ---------------------------------------------------------------------------


class TestTransform:
    def test_applies_prepare_tasks_in_order(self):
        counter = CountingSample("a")
        spec = Pipeline("p").add(counter).build()
        out = transform(spec, {"y": torch.tensor([0.0])})
        assert out["a"] is True
        assert counter.calls == 1

    def test_skips_batch_tasks(self):
        class NoopBatch(BatchTask):
            def __init__(self):
                self.calls = 0

            def execute(self, data: dict) -> dict:
                self.calls += 1
                return data

        counter = CountingSample("a")
        batch = NoopBatch()
        spec = Pipeline("p").add(counter).add(batch).build()
        transform(spec, {"y": torch.tensor([0.0])})
        assert counter.calls == 1
        assert batch.calls == 0


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


class TestRun:
    def test_sample_task_once_per_sample(self):
        counter = CountingSample()
        spec = Pipeline("p").add(counter).build()
        out = list(run(spec, InMemorySource(_samples(10))))
        assert len(out) == 10
        assert counter.calls == 10

    def test_dataset_task_fit_on_full_source(self):
        shift = MeanShift()
        spec = Pipeline("p").add(shift).build()
        list(run(spec, InMemorySource(_samples(8))))
        assert shift.fit_calls == 1
        assert shift.exec_calls == 8

    def test_fit_source_scopes_fit(self):
        shift = MeanShift()
        spec = Pipeline("p").add(shift).build()
        list(
            run(
                spec,
                InMemorySource(_samples(10)),   # y = 0..9
                fit_source=InMemorySource(_samples(5)),  # y = 0..4 → mean = 2
            )
        )
        assert shift.fit_calls == 1
        assert abs(shift.mean - 2.0) < 1e-9

    def test_returns_iterator(self):
        import types
        spec = Pipeline("p").add(CountingSample()).build()
        it = run(spec, InMemorySource(_samples(3)))
        assert isinstance(it, types.GeneratorType)


# ---------------------------------------------------------------------------
# collect_task_states / load_task_states
# ---------------------------------------------------------------------------


class TestTaskStates:
    def test_collect_includes_only_dataset_tasks(self):
        spec = (
            Pipeline("p")
            .add(CountingSample("s"))
            .add(MeanShift("y"))
            .build()
        )
        list(run(spec, InMemorySource(_samples(4))))
        states = collect_task_states(spec)
        assert set(states) == {"mean_shift:y"}

    def test_load_restores_state(self):
        # Fit on one spec, dump, load into a fresh spec.
        a = MeanShift()
        spec_a = Pipeline("p").add(a).build()
        list(run(spec_a, InMemorySource(_samples(6))))    # y=0..5 → mean=2.5
        states = collect_task_states(spec_a)

        b = MeanShift()
        spec_b = Pipeline("p").add(b).build()
        load_task_states(spec_b, states)
        assert abs(b.mean - 2.5) < 1e-9
