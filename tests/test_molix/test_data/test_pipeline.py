"""Tests for the declarative PipelineSpec and Pipeline builder.

Pipeline is now a pure data structure — no execution, no IO, no caching.
Those concerns live in :mod:`molix.data.execute` and :mod:`molix.data.cache`
and are tested separately.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from molix.data.pipeline import Pipeline, PipelineSpec, TaskEntry
from molix.data.task import BatchTask, DatasetTask, SampleTask


# ---------------------------------------------------------------------------
# Stub tasks for identity / grouping tests
# ---------------------------------------------------------------------------


class _Sample(SampleTask):
    def __init__(self, name: str = "s") -> None:
        self._name = name

    @property
    def task_id(self) -> str:
        return f"sample:{self._name}"

    def execute(self, data: dict) -> dict:
        return {**data, self._name: True}


class _Dataset(DatasetTask):
    def __init__(self, key: str = "y") -> None:
        self.key = key

    @property
    def task_id(self) -> str:
        return f"dataset:{self.key}"

    def fit(self, samples: list[dict]) -> None: ...

    def execute(self, data: dict) -> dict:
        return data


class _Batch(BatchTask):
    def __init__(self, name: str = "b") -> None:
        self._name = name

    @property
    def task_id(self) -> str:
        return f"batch:{self._name}"

    def execute(self, data: dict) -> dict:
        return data


# ---------------------------------------------------------------------------
# Builder DSL
# ---------------------------------------------------------------------------


class TestBuilder:
    def test_add_returns_self_for_chaining(self):
        p = Pipeline("x")
        assert p.add(_Sample()) is p

    def test_add_task_instance(self):
        spec = Pipeline("p").add(_Sample("foo")).build()
        assert len(spec.tasks) == 1
        assert spec.tasks[0].task.task_id == "sample:foo"

    def test_add_bare_callable(self):
        spec = Pipeline("p").add(lambda s: s, name="noop").build()
        assert spec.tasks[0].name == "noop"

    def test_decorator_task(self):
        p = Pipeline("p")

        @p.task
        def add_tag(s: dict) -> dict:
            return {**s, "tag": True}

        spec = p.build()
        assert spec.tasks[0].name == "add_tag"

    def test_decorator_with_explicit_name(self):
        p = Pipeline("p")

        @p.task(name="custom")
        def _f(s: dict) -> dict:
            return s

        spec = p.build()
        assert spec.tasks[0].name == "custom"

    def test_rejects_non_task_non_callable(self):
        with pytest.raises(TypeError, match="Task"):
            Pipeline("p").add(42)    # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


class TestGrouping:
    def test_prepare_tasks_include_sample_and_dataset(self):
        spec = (
            Pipeline("p")
            .add(_Sample())
            .add(_Dataset())
            .add(_Batch())
            .build()
        )
        assert len(spec.prepare_tasks) == 2
        assert len(spec.batch_tasks) == 1

    def test_plain_callable_counts_as_prepare(self):
        spec = Pipeline("p").add(lambda s: s, name="noop").build()
        assert len(spec.prepare_tasks) == 1
        assert len(spec.batch_tasks) == 0

    def test_prepare_tasks_order_preserved(self):
        a, b = _Sample("a"), _Sample("b")
        spec = Pipeline("p").add(a).add(b).build()
        assert [e.task for e in spec.prepare_tasks] == [a, b]


# ---------------------------------------------------------------------------
# pipeline_id determinism
# ---------------------------------------------------------------------------


class TestPipelineId:
    def test_stable_across_builds(self):
        a = Pipeline("p").add(_Sample("a")).add(_Dataset("y")).build()
        b = Pipeline("p").add(_Sample("a")).add(_Dataset("y")).build()
        assert a.pipeline_id == b.pipeline_id

    def test_task_order_changes_id(self):
        a = Pipeline("p").add(_Sample("a")).add(_Dataset("y")).build()
        b = Pipeline("p").add(_Dataset("y")).add(_Sample("a")).build()
        assert a.pipeline_id != b.pipeline_id

    def test_task_config_changes_id(self):
        a = Pipeline("p").add(_Dataset("y")).build()
        b = Pipeline("p").add(_Dataset("z")).build()
        assert a.pipeline_id != b.pipeline_id

    def test_pipeline_name_changes_id(self):
        a = Pipeline("a").build()
        b = Pipeline("b").build()
        assert a.pipeline_id != b.pipeline_id

    def test_id_is_hex_and_short(self):
        pid = Pipeline("p").add(_Sample()).build().pipeline_id
        assert len(pid) == 16
        int(pid, 16)


# ---------------------------------------------------------------------------
# Scheduling purity: the spec must not expose any IO/cache/DDP concept.
# ---------------------------------------------------------------------------


class TestNoSchedulingConcepts:
    FORBIDDEN = (
        "materialize", "cache_identity", "is_cache_ready", "run", "transform",
        "save", "load", "write_cache", "from_cache", "from_samples",
        "collect_task_states",
    )

    def test_no_forbidden_attributes(self):
        spec = Pipeline("p").add(_Sample()).build()
        for name in self.FORBIDDEN:
            assert not hasattr(spec, name), f"PipelineSpec leaks {name!r}"

    def test_public_surface(self):
        """PipelineSpec exposes only declarative accessors."""
        public = {
            n for n in dir(PipelineSpec)
            if not n.startswith("_")
        }
        assert public == {"name", "pipeline_id", "tasks", "prepare_tasks",
                          "batch_tasks", "to_dict"}


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------


class TestToDict:
    def test_contains_task_names_and_types(self):
        spec = (
            Pipeline("p")
            .add(_Sample("a"))
            .add(_Batch("b"))
            .build()
        )
        d = spec.to_dict()
        assert d["name"] == "p"
        assert d["pipeline_id"] == spec.pipeline_id
        names = [t["name"] for t in d["tasks"]]
        types = [t["type"] for t in d["tasks"]]
        assert names == ["sample:a", "batch:b"]
        assert types == ["_Sample", "_Batch"]


# ---------------------------------------------------------------------------
# TaskEntry is immutable (catches accidental mutation of shared specs)
# ---------------------------------------------------------------------------


class TestTaskEntryImmutability:
    def test_is_frozen(self):
        e = TaskEntry(name="x", task=_Sample())
        with pytest.raises(Exception):    # FrozenInstanceError or AttributeError
            e.name = "y"                  # type: ignore[misc]
