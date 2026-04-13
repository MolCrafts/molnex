"""Tests for PipelineSpec / PipelineDSL.

Pipeline is a pure transform; it has no scheduling knobs
(``workspace`` / ``cache_dir`` / ``cache_identity``). These tests cover:

* ``transform(sample)`` — per-sample application
* ``run(source)`` — single-pass iteration semantics, DatasetTask.fit flow
* ``materialize(source, sink=...)`` — integration with the standard cache
  format (including task_states round-trip through
  :class:`MmapDataset.from_cache`)
* Pipeline public API must not expose any scheduling concept
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from molix.data.dataset import MmapDataset
from molix.data.pipeline import PipelineSpec, pipeline
from molix.data.source import InMemorySource
from molix.data.task import DatasetTask, SampleTask


# ---------------------------------------------------------------------------
# Helpers: tiny instrumented tasks
# ---------------------------------------------------------------------------


class CountingSample(SampleTask):
    """Adds a counter; counts how many times execute() was called."""

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
    """Fit the mean of a target across samples, subtract on execute."""

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
        {"Z": torch.tensor([1, 6]), "pos": torch.zeros(2, 3),
         "y": torch.tensor([float(i)])}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# transform (per-sample)
# ---------------------------------------------------------------------------


class TestTransform:
    def test_applies_prepare_tasks_in_order(self):
        counter = CountingSample("a")
        spec = pipeline("p").add(counter).build()
        out = spec.transform({"y": torch.tensor([0.0])})
        assert out["a"] is True
        assert counter.calls == 1

    def test_skips_batch_tasks(self):
        """transform must only apply prepare tasks; BatchTasks are post-collate."""
        from molix.data.task import BatchTask

        class NoopBatch(BatchTask):
            def __init__(self):
                self.calls = 0

            def execute(self, data: dict) -> dict:
                self.calls += 1
                return data

        counter = CountingSample("a")
        batch = NoopBatch()
        spec = pipeline("p").add(counter).add(batch).build()
        spec.transform({"y": torch.tensor([0.0])})
        assert counter.calls == 1
        assert batch.calls == 0


# ---------------------------------------------------------------------------
# run (single-pass iteration over source)
# ---------------------------------------------------------------------------


class TestRun:
    def test_sample_task_runs_once_per_sample(self):
        src = InMemorySource(_samples(10))
        counter = CountingSample()
        spec = pipeline("p").add(counter).build()
        out = list(spec.run(src))
        assert len(out) == 10
        assert counter.calls == 10

    def test_dataset_task_fit_called_once_on_full_source(self):
        src = InMemorySource(_samples(8))
        shift = MeanShift("y")
        spec = pipeline("p").add(shift).build()
        list(spec.run(src))
        assert shift.fit_calls == 1
        assert shift.exec_calls == 8  # one execute per sample

    def test_fit_source_scopes_fit_to_subset(self):
        src = InMemorySource(_samples(10))          # y = 0..9
        fit_src = InMemorySource(_samples(5))       # y = 0..4, mean = 2.0
        shift = MeanShift("y")
        spec = pipeline("p").add(shift).build()
        list(spec.run(src, fit_source=fit_src))
        assert shift.fit_calls == 1
        assert abs(shift.mean - 2.0) < 1e-9

    def test_returns_iterator(self):
        """run() is a generator — not eagerly materialized internally except
        where fit() forces it."""
        import types

        src = InMemorySource(_samples(3))
        spec = pipeline("p").add(CountingSample()).build()
        it = spec.run(src)
        assert isinstance(it, types.GeneratorType)


# ---------------------------------------------------------------------------
# materialize (run + write standard cache format)
# ---------------------------------------------------------------------------


class TestMaterialize:
    def test_writes_standard_cache_layout(self, tmp_path):
        src = InMemorySource(_samples(4))
        spec = pipeline("p").add(CountingSample()).build()
        sink = tmp_path / "asset"
        spec.materialize(src, sink=sink)

        assert (sink / "_READY").exists()
        assert (sink / "meta.json").exists()
        assert (sink / "samples.bin").exists()
        assert (sink / "samples.idx").exists()

    def test_meta_records_ids(self, tmp_path):
        src = InMemorySource(_samples(4), name="src-a")
        spec = pipeline("p").add(MeanShift("y")).build()
        sink = tmp_path / "asset"
        spec.materialize(src, sink=sink)

        import json
        meta = json.loads((sink / "meta.json").read_text())
        assert meta["pipeline_id"] == spec.pipeline_id
        assert meta["source_id"] == src.source_id
        assert meta["fit_source_id"] == src.source_id
        assert meta["pipeline_spec"]["name"] == "p"

    def test_roundtrip_via_from_cache(self, tmp_path):
        src = InMemorySource(_samples(6))           # y = 0..5, mean = 2.5
        spec = pipeline("p").add(MeanShift("y")).build()
        sink = tmp_path / "asset"
        spec.materialize(src, sink=sink)

        ds = MmapDataset.from_cache(sink)
        assert len(ds) == 6
        # Cached samples should have had the mean subtracted.
        assert abs(float(ds[0]["y"].item()) - (-2.5)) < 1e-6

    def test_task_states_roundtrip(self, tmp_path):
        """DatasetTask.state_dict is captured into the cache and restorable
        via MmapDataset.from_cache().get_task_state()."""
        src = InMemorySource(_samples(6))           # mean = 2.5
        spec = pipeline("p").add(MeanShift("y")).build()
        sink = tmp_path / "asset"
        spec.materialize(src, sink=sink)

        ds = MmapDataset.from_cache(sink)
        # entry name defaults to task.task_id when add() is called without name=
        state = ds.get_task_state("mean_shift:y")
        assert abs(float(state["mean"].item()) - 2.5) < 1e-9

    def test_idempotent_noop_when_ready(self, tmp_path):
        src = InMemorySource(_samples(4))
        shift1 = MeanShift("y")
        spec1 = pipeline("p").add(shift1).build()
        sink = tmp_path / "asset"
        spec1.materialize(src, sink=sink)
        assert shift1.exec_calls == 4

        # Second call on a ready sink must not invoke fit/execute.
        shift2 = MeanShift("y")
        spec2 = pipeline("p").add(shift2).build()
        spec2.materialize(src, sink=sink)
        assert shift2.fit_calls == 0
        assert shift2.exec_calls == 0

    def test_overwrite_rewrites(self, tmp_path):
        src = InMemorySource(_samples(4))
        spec = pipeline("p").add(CountingSample()).build()
        sink = tmp_path / "asset"
        spec.materialize(src, sink=sink)
        spec.materialize(src, sink=sink, overwrite=True)
        ds = MmapDataset.from_cache(sink)
        assert len(ds) == 4

    def test_fit_source_recorded_in_meta(self, tmp_path):
        src = InMemorySource(_samples(10), name="full")
        fit_src = InMemorySource(_samples(5), name="fit-only")
        spec = pipeline("p").add(MeanShift("y")).build()
        sink = tmp_path / "asset"
        spec.materialize(src, sink=sink, fit_source=fit_src)

        import json
        meta = json.loads((sink / "meta.json").read_text())
        assert meta["source_id"] == src.source_id
        assert meta["fit_source_id"] == fit_src.source_id


# ---------------------------------------------------------------------------
# pipeline_id / DSL invariants
# ---------------------------------------------------------------------------


class TestPipelineId:
    def test_stable_across_builds(self):
        a = pipeline("p").add(CountingSample("a")).add(MeanShift("y")).build()
        b = pipeline("p").add(CountingSample("a")).add(MeanShift("y")).build()
        assert a.pipeline_id == b.pipeline_id

    def test_changes_with_task_order(self):
        a = pipeline("p").add(CountingSample("a")).add(MeanShift("y")).build()
        b = pipeline("p").add(MeanShift("y")).add(CountingSample("a")).build()
        assert a.pipeline_id != b.pipeline_id

    def test_changes_with_task_config(self):
        a = pipeline("p").add(MeanShift("y")).build()
        b = pipeline("p").add(MeanShift("z")).build()
        assert a.pipeline_id != b.pipeline_id


# ---------------------------------------------------------------------------
# Scheduling purity: PipelineSpec public API must not leak scheduling concepts
# ---------------------------------------------------------------------------


class TestNoSchedulingConcepts:
    """Design-constraint regression: pipeline is pure transform.

    If any of the forbidden names appear on the public API, scheduling
    policy is bleeding into the transform layer. Redesign it.
    """

    FORBIDDEN = ("workspace", "cache_dir", "cache_identity", "name_hint")

    def test_no_forbidden_attributes(self):
        spec = pipeline("p").add(CountingSample()).build()
        for name in self.FORBIDDEN:
            assert not hasattr(spec, name), f"pipeline leaks {name!r}"

    def test_materialize_has_no_forbidden_kwargs(self):
        import inspect

        params = inspect.signature(PipelineSpec.materialize).parameters
        for name in self.FORBIDDEN:
            assert name not in params, (
                f"materialize() has forbidden kwarg {name!r}; "
                "cache placement is a caller concern"
            )

    def test_run_has_no_forbidden_kwargs(self):
        import inspect

        params = inspect.signature(PipelineSpec.run).parameters
        for name in self.FORBIDDEN:
            assert name not in params


# ---------------------------------------------------------------------------
# DDP rank guard on materialize
# ---------------------------------------------------------------------------


class TestMaterializeRankGuard:
    def test_rank_nonzero_polls_for_ready(self, tmp_path, monkeypatch):
        src = InMemorySource(_samples(3))
        spec = pipeline("p").add(CountingSample()).build()
        sink = tmp_path / "asset"
        sink.mkdir()
        (sink / "_READY").write_text("")    # pre-existing

        monkeypatch.setenv("RANK", "1")
        # Should return immediately without calling write_cache.
        spec.materialize(src, sink=sink, wait_timeout=1.0)

    def test_rank_nonzero_times_out_without_ready(self, tmp_path, monkeypatch):
        src = InMemorySource(_samples(3))
        spec = pipeline("p").add(CountingSample()).build()
        sink = tmp_path / "asset"
        monkeypatch.setenv("RANK", "1")
        with pytest.raises(TimeoutError, match="_READY"):
            spec.materialize(src, sink=sink, wait_timeout=0.5)
