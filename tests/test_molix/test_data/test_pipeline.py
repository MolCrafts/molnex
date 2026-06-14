"""Tests for PipelineSpec / Pipeline builder / Node / DAGCache.

PipelineSpec owns both declaration (node list, identity) and orchestration
(``.run`` / ``.transform`` / ``.cache_key`` / ``.cache`` / ``.fit``).
Node owns task dispatch (``.apply``) and per-node cache key derivation.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from molix.data.pipeline import DAGCache, Node, Pipeline
from molix.data.source import InMemorySource
from molix.data.task import BatchTask, DatasetTask, SampleTask

# ---------------------------------------------------------------------------
# Stub tasks
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


class NoopBatch(BatchTask):
    def __init__(self) -> None:
        self.calls = 0

    @property
    def task_id(self) -> str:
        return "noop:batch"

    def execute(self, data: dict) -> dict:
        self.calls += 1
        return data


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
# Builder DSL
# ---------------------------------------------------------------------------


class TestBuilder:
    def test_add_returns_self_for_chaining(self):
        p = Pipeline("x")
        assert p.add(CountingSample()) is p

    def test_add_task_instance(self):
        spec = Pipeline("p").add(CountingSample("foo")).build()
        assert len(spec.nodes) == 1
        assert spec.nodes[0].task.task_id == "counting:foo"

    def test_add_bare_callable(self):
        spec = Pipeline("p").add(lambda s: s, name="noop").build()
        assert spec.nodes[0].name == "noop"

    def test_add_prebuilt_node(self):
        node = Node(name="n", task=CountingSample())
        spec = Pipeline("p").node(node).build()
        assert spec.nodes[0] is node

    def test_rejects_non_task_non_callable(self):
        with pytest.raises(TypeError, match="Task"):
            Pipeline("p").add(42)  # type: ignore[arg-type]

    def test_rejects_duplicate_name_on_add(self):
        p = Pipeline("p").add(CountingSample(), name="shared")
        with pytest.raises(ValueError, match="already registered"):
            p.add(CountingSample(), name="shared")

    def test_rejects_duplicate_inferred_name(self):
        p = Pipeline("p").add(CountingSample())
        with pytest.raises(ValueError, match="already registered"):
            p.add(CountingSample())

    def test_add_with_reads_writes(self):
        node = (
            Pipeline("p")
            .add(CountingSample(), name="c", reads={"Z", "pos"}, writes={"edge_index"})
            .build()
            .nodes[0]
        )
        assert node.reads == frozenset({"Z", "pos"})
        assert node.writes == frozenset({"edge_index"})


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


class TestGrouping:
    def test_prepare_nodes_include_sample_and_dataset(self):
        spec = Pipeline("p").add(CountingSample()).add(MeanShift()).add(NoopBatch()).build()
        assert len(spec.prepare_nodes) == 2
        assert len(spec.batch_nodes) == 1

    def test_plain_callable_not_cacheable(self):
        spec = Pipeline("p").add(lambda s: s, name="noop").build()
        assert len(spec.prepare_nodes) == 0  # bare callable is not SampleTask/DatasetTask
        assert len(spec.batch_nodes) == 0

    def test_prepare_nodes_order_preserved(self):
        a, b = CountingSample("a"), CountingSample("b")
        spec = Pipeline("p").add(a).add(b).build()
        assert [n.task for n in spec.prepare_nodes] == [a, b]


# ---------------------------------------------------------------------------
# pipeline_id determinism
# ---------------------------------------------------------------------------


class TestPipelineId:
    def test_stable_across_builds(self):
        a = Pipeline("p").add(CountingSample("a")).add(MeanShift("y")).build()
        b = Pipeline("p").add(CountingSample("a")).add(MeanShift("y")).build()
        assert a.pipeline_id == b.pipeline_id

    def test_task_order_changes_id(self):
        a = Pipeline("p").add(CountingSample("a")).add(MeanShift("y")).build()
        b = Pipeline("p").add(MeanShift("y")).add(CountingSample("a")).build()
        assert a.pipeline_id != b.pipeline_id

    def test_task_config_changes_id(self):
        a = Pipeline("p").add(MeanShift("y")).build()
        b = Pipeline("p").add(MeanShift("z")).build()
        assert a.pipeline_id != b.pipeline_id

    def test_pipeline_name_changes_id(self):
        a = Pipeline("a").build()
        b = Pipeline("b").build()
        assert a.pipeline_id != b.pipeline_id

    def test_id_is_hex_and_short(self):
        pid = Pipeline("p").add(CountingSample()).build().pipeline_id
        assert len(pid) == 16
        int(pid, 16)


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------


class TestToDict:
    def test_contains_node_names_and_types(self):
        spec = Pipeline("p").add(CountingSample("a")).add(NoopBatch()).build()
        d = spec.to_dict()
        assert d["name"] == "p"
        assert d["pipeline_id"] == spec.pipeline_id
        names = [n["name"] for n in d["nodes"]]
        types = [n["type"] for n in d["nodes"]]
        assert names == ["counting:a", "noop:batch"]
        assert types == ["CountingSample", "NoopBatch"]
        assert d["edges"]  # linear edge derived

    def test_edges_reflect_linear_chain(self):
        spec = Pipeline("p").add(CountingSample("a")).add(CountingSample("b")).build()
        d = spec.to_dict()
        assert len(d["edges"]) == 1
        assert d["edges"][0]["from"] == "counting:a"
        assert d["edges"][0]["to"] == "counting:b"


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


class TestNode:
    def test_is_frozen(self):
        n = Node(name="x", task=CountingSample())
        with pytest.raises(Exception):
            n.name = "y"  # type: ignore[misc]

    def test_apply_dispatches_runnable(self):
        t = CountingSample()
        node = Node(name="t", task=t)
        out = node.apply({"y": torch.tensor([0.0])})
        assert out["counter"] is True
        assert t.calls == 1

    def test_apply_dispatches_bare_callable(self):
        node = Node(name="f", task=lambda s: {**s, "tag": True})
        out = node.apply({"y": 0})
        assert out["tag"] is True

    def test_apply_rejects_non_callable(self):
        node = Node(name="bad", task=42)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            node.apply({})

    def test_cache_key_depends_on_upstream_keys(self):
        node = Node(name="n", task=CountingSample())
        a = node.cache_key(("src:v1",), "", None)
        b = node.cache_key(("src:v2",), "", None)
        assert a != b

    def test_cache_key_depends_on_fit_source(self):
        node = Node(name="n", task=CountingSample())
        a = node.cache_key(("up",), "fs1", None)
        b = node.cache_key(("up",), "fs2", None)
        assert a != b

    def test_cache_key_depends_on_extra(self):
        node = Node(name="n", task=CountingSample())
        a = node.cache_key(("up",), "", {"n_train": "100"})
        b = node.cache_key(("up",), "", {"n_train": "200"})
        assert a != b

    def test_is_cacheable(self):
        assert Node(name="s", task=CountingSample()).is_cacheable is True
        assert Node(name="d", task=MeanShift()).is_cacheable is True
        assert Node(name="b", task=NoopBatch()).is_cacheable is False

    def test_needs_fit(self):
        assert Node(name="d", task=MeanShift()).needs_fit is True
        assert Node(name="s", task=CountingSample()).needs_fit is False


# ---------------------------------------------------------------------------
# transform — per-sample inference path
# ---------------------------------------------------------------------------


class TestTransform:
    def test_applies_prepare_nodes_in_order(self):
        counter = CountingSample("a")
        spec = Pipeline("p").add(counter).build()
        out = spec.transform({"y": torch.tensor([0.0])})
        assert out["a"] is True
        assert counter.calls == 1

    def test_skips_batch_nodes(self):
        counter = CountingSample("a")
        batch = NoopBatch()
        spec = Pipeline("p").add(counter).add(batch).build()
        spec.transform({"y": torch.tensor([0.0])})
        assert counter.calls == 1
        assert batch.calls == 0


# ---------------------------------------------------------------------------
# run — full-source iteration with fit()
# ---------------------------------------------------------------------------


class TestRun:
    def test_sample_task_once_per_sample(self):
        counter = CountingSample()
        spec = Pipeline("p").add(counter).build()
        out = list(spec.run(InMemorySource(_samples(10))))
        assert len(out) == 10
        assert counter.calls == 10

    def test_dataset_task_fit_on_full_source(self):
        shift = MeanShift()
        spec = Pipeline("p").add(shift).build()
        list(spec.run(InMemorySource(_samples(8))))
        assert shift.fit_calls == 1
        assert shift.exec_calls == 8

    def test_fit_source_scopes_fit(self):
        shift = MeanShift()
        spec = Pipeline("p").add(shift).build()
        list(
            spec.run(
                InMemorySource(_samples(10)),  # y = 0..9
                fit_source=InMemorySource(_samples(5)),  # y = 0..4 → mean = 2
            )
        )
        assert shift.fit_calls == 1
        assert abs(shift.mean - 2.0) < 1e-9

    def test_fit_states_skips_fit(self):
        shift = MeanShift()
        spec = Pipeline("p").add(shift).build()
        list(
            spec.run(
                InMemorySource(_samples(4)),
                fit_states={"mean_shift:y": {"mean": torch.tensor(5.0)}},
            )
        )
        assert shift.fit_calls == 0
        assert shift.mean == 5.0

    def test_returns_iterator(self):
        import types

        spec = Pipeline("p").add(CountingSample()).build()
        it = spec.run(InMemorySource(_samples(3)))
        assert isinstance(it, types.GeneratorType)


# ---------------------------------------------------------------------------
# fit — separate fit phase
# ---------------------------------------------------------------------------


class TestFit:
    def test_fit_returns_states(self):
        spec = Pipeline("p").add(MeanShift("y")).build()
        states = spec.fit(InMemorySource(_samples(6)))  # y=0..5 → mean=2.5
        assert set(states) == {"mean_shift:y"}
        assert abs(float(states["mean_shift:y"]["mean"].item()) - 2.5) < 1e-9

    def test_fit_with_upstream_sample_tasks(self):
        """Upstream SampleTasks are applied before fitting DatasetTasks."""
        counter = CountingSample("pre")
        shift = MeanShift("y")
        spec = Pipeline("p").add(counter).add(shift).build()
        states = spec.fit(InMemorySource(_samples(4)))
        assert counter.calls == 4  # applied to each fit sample
        assert abs(float(states["mean_shift:y"]["mean"].item()) - 1.5) < 1e-9

    def test_fit_only_includes_dataset_tasks(self):
        spec = Pipeline("p").add(CountingSample()).add(MeanShift()).build()
        states = spec.fit(InMemorySource(_samples(4)))
        assert set(states) == {"mean_shift:y"}


# ---------------------------------------------------------------------------
# collect_fit_states / load_fit_states
# ---------------------------------------------------------------------------


class TestFitStates:
    def test_collect_includes_only_dataset_task_nodes(self):
        spec = Pipeline("p").add(CountingSample("s")).add(MeanShift("y")).build()
        list(spec.run(InMemorySource(_samples(4))))
        states = spec.collect_fit_states()
        assert set(states) == {"mean_shift:y"}

    def test_load_restores_state(self):
        a = MeanShift()
        spec_a = Pipeline("p").add(a).build()
        list(spec_a.run(InMemorySource(_samples(6))))  # y=0..5 → mean=2.5
        states = spec_a.collect_fit_states()

        b = MeanShift()
        spec_b = Pipeline("p").add(b).build()
        spec_b.load_fit_states(states)
        assert abs(b.mean - 2.5) < 1e-9


# ---------------------------------------------------------------------------
# cache_key — PipelineSpec-level identity
# ---------------------------------------------------------------------------


class _FakeSource:
    def __init__(self, source_id: str, samples: list[dict] | None = None) -> None:
        self.source_id = source_id
        self._samples = samples or []

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        return self._samples[idx]


class TestCacheKey:
    def test_source_id_affects_key(self):
        spec = Pipeline("p").build()
        a = spec.cache_key(_FakeSource("s1"))
        b = spec.cache_key(_FakeSource("s2"))
        assert a != b

    def test_fit_source_affects_key(self):
        spec = Pipeline("p").build()
        src = _FakeSource("s")
        full_only = spec.cache_key(src)
        with_fit = spec.cache_key(src, fit_source=_FakeSource("s-subset"))
        assert full_only != with_fit

    def test_extra_affects_key(self):
        spec = Pipeline("p").build()
        a = spec.cache_key(_FakeSource("s"))
        b = spec.cache_key(_FakeSource("s"), extra={"n_train": "100"})
        assert a != b

    def test_same_tasks_same_source_same_key(self):
        """Same task chain + same source = same cache key (pipeline name doesn't matter)."""
        a = Pipeline("p1").add(CountingSample()).build().cache_key(_FakeSource("s"))
        b = Pipeline("p2").add(CountingSample()).build().cache_key(_FakeSource("s"))
        assert a == b  # identical transforms → identical data → shared cache

    def test_empty_pipeline_falls_back_to_source_based_key(self):
        spec = Pipeline("p").build()
        key = spec.cache_key(_FakeSource("s1"))
        assert len(key) == 12
        int(key, 16)
        # Different source → different key even for empty pipeline
        assert key != spec.cache_key(_FakeSource("s2"))


# ---------------------------------------------------------------------------
# cache — DDP-aware per-node materialisation
# ---------------------------------------------------------------------------


class _Tag(SampleTask):
    @property
    def task_id(self) -> str:
        return "tag"

    def execute(self, data: dict) -> dict:
        return {**data, "tagged": torch.tensor(1)}


class TestCache:
    def test_cache_returns_dag_cache(self, tmp_path):
        spec = Pipeline("p").add(_Tag()).build()
        dag = spec.cache(InMemorySource(_samples(4)), base_dir=tmp_path)
        assert isinstance(dag, DAGCache)
        assert len(dag.node_caches) == 1

    def test_dag_cache_final_dataset(self, tmp_path):
        spec = Pipeline("p").add(_Tag()).build()
        dag = spec.cache(InMemorySource(_samples(4)), base_dir=tmp_path)
        ds = dag.dataset(mmap=True)
        assert len(ds) == 4
        assert all("tagged" in ds[i] for i in range(4))

    def test_per_node_cache_files(self, tmp_path):
        spec = Pipeline("p").add(_Tag()).add(MeanShift("y")).build()
        dag = spec.cache(InMemorySource(_samples(4)), base_dir=tmp_path)
        assert len(dag.node_caches) == 2

    def test_saves_task_states(self, tmp_path):
        shift = MeanShift("y")
        spec = Pipeline("p").add(shift).build()
        dag = spec.cache(InMemorySource(_samples(4)), base_dir=tmp_path)
        payload = dag.final.load()
        assert torch.allclose(
            payload["task_states"]["mean_shift:y"]["mean"].double(),
            torch.tensor(1.5, dtype=torch.float64),
            atol=1e-9,
        )

    def test_fit_source_scopes_fit(self, tmp_path):
        shift = MeanShift("y")
        spec = Pipeline("p").add(shift).build()
        # Full source y=0..9, fit_source y=0..4 → mean should be 2.0
        dag = spec.cache(
            InMemorySource(_samples(10)),
            base_dir=tmp_path,
            fit_source=InMemorySource(_samples(5)),
        )
        payload = dag.final.load()
        state = payload["task_states"]["mean_shift:y"]
        assert abs(float(state["mean"].item()) - 2.0) < 1e-9

    def test_cache_reuse(self, tmp_path):
        spec = Pipeline("p").add(_Tag()).build()
        dag1 = spec.cache(InMemorySource(_samples(2)), base_dir=tmp_path)
        mtime = {n: dag1.node_caches[n].sink.stat().st_mtime_ns for n in dag1.node_caches}
        dag2 = spec.cache(InMemorySource(_samples(2)), base_dir=tmp_path)
        for n in dag1.node_caches:
            assert dag2.node_caches[n].sink.stat().st_mtime_ns == mtime[n]

    def test_overwrite(self, tmp_path):
        spec = Pipeline("p").add(_Tag()).build()
        spec.cache(InMemorySource(_samples(2)), base_dir=tmp_path)
        dag2 = spec.cache(InMemorySource(_samples(5)), base_dir=tmp_path, overwrite=True)
        assert len(dag2.final.load()["samples"]) == 5

    def test_node_dataset(self, tmp_path):
        spec = Pipeline("p").add(_Tag()).add(MeanShift("y")).build()
        dag = spec.cache(InMemorySource(_samples(4)), base_dir=tmp_path)
        tag_ds = dag.node_dataset("tag")
        assert len(tag_ds) == 4
        assert "tagged" in tag_ds[0]

    def test_node_dataset_unknown_node(self, tmp_path):
        spec = Pipeline("p").add(_Tag()).build()
        dag = spec.cache(InMemorySource(_samples(2)), base_dir=tmp_path)
        with pytest.raises(KeyError, match="unknown"):
            dag.node_dataset("unknown")

    def test_fit_states_skips_fit(self, tmp_path):
        shift = MeanShift("y")
        spec = Pipeline("p").add(shift).build()
        states = {"mean_shift:y": {"mean": torch.tensor(99.0)}}
        dag = spec.cache(InMemorySource(_samples(3)), base_dir=tmp_path, fit_states=states)
        payload = dag.final.load()
        assert abs(float(payload["task_states"]["mean_shift:y"]["mean"].item()) - 99.0) < 1e-9

    def test_sink_filename_embeds_cache_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RANK", "0")
        spec = Pipeline("qm9").add(_Tag()).build()
        src = _FakeSource("s", _samples(1))
        dag = spec.cache(src, base_dir=tmp_path)
        expected_key = spec.cache_key(src)
        # Last prepare node's sink uses the final cache key
        spec.prepare_nodes[-1]
        assert dag.final.sink.name == f"qm9-{expected_key}.pt"


# ---------------------------------------------------------------------------
# cache — DDP rank handling
# ---------------------------------------------------------------------------


class TestCacheDDP:
    def test_primary_rank_builds(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RANK", "0")
        spec = Pipeline("p").add(_Tag()).build()
        src = _FakeSource("s", _samples(3))
        dag = spec.cache(src, base_dir=tmp_path)
        assert dag.final.is_ready()
        assert len(dag.final.load()["samples"]) == 3

    def test_missing_rank_treated_as_primary(self, tmp_path, monkeypatch):
        monkeypatch.delenv("RANK", raising=False)
        spec = Pipeline("p").add(_Tag()).build()
        src = _FakeSource("s", _samples(2))
        dag = spec.cache(src, base_dir=tmp_path)
        assert dag.final.is_ready()

    def test_malformed_rank_treated_as_primary(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RANK", "not-a-number")
        spec = Pipeline("p").add(_Tag()).build()
        src = _FakeSource("s", _samples(2))
        dag = spec.cache(src, base_dir=tmp_path)
        assert dag.final.is_ready()

    def test_non_primary_waits_and_raises_on_timeout(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RANK", "1")
        spec = Pipeline("p").add(_Tag()).build()
        src = _FakeSource("s", _samples(1))
        import molix.data.cache as cache_mod

        orig_wait = cache_mod.PackedCache.wait_until_ready

        def fast_wait(self, *, timeout: float = 0.2, poll_interval: float = 0.05):
            return orig_wait(self, timeout=timeout, poll_interval=poll_interval)

        monkeypatch.setattr(cache_mod.PackedCache, "wait_until_ready", fast_wait)
        with pytest.raises(TimeoutError, match="Timed out"):
            spec.cache(src, base_dir=tmp_path)

    def test_different_fit_source_different_sinks(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RANK", "0")
        spec = Pipeline("p").add(MeanShift("y")).build()

        full = _FakeSource("full", _samples(10))
        fit_a = _FakeSource("fit-a", _samples(5))
        fit_b = _FakeSource("fit-b", _samples(3))

        dag_a = spec.cache(full, base_dir=tmp_path, fit_source=fit_a)
        dag_b = spec.cache(full, base_dir=tmp_path, fit_source=fit_b)
        assert dag_a.final.sink != dag_b.final.sink


# ---------------------------------------------------------------------------
# Edge derivation (key-based dependencies)
# ---------------------------------------------------------------------------


class TestEdges:
    def test_linear_chain_without_reads_writes(self):
        spec = Pipeline("p").add(CountingSample("a")).add(CountingSample("b")).build()
        assert len(spec.edges) == 1
        assert spec.edges[0].from_node == "counting:a"
        assert spec.edges[0].to_node == "counting:b"

    def test_no_edges_for_single_node(self):
        spec = Pipeline("p").add(CountingSample()).build()
        assert len(spec.edges) == 0

    def test_key_based_edge_derivation(self):
        spec = (
            Pipeline("p")
            .add(CountingSample("a"), writes={"edge_index", "bond_diff"})
            .add(CountingSample("b"), reads={"edge_index"})
            .build()
        )
        assert len(spec.edges) == 1
        assert spec.edges[0].keys == frozenset({"edge_index"})
