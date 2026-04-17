"""Tests for :mod:`molix.data.cache` — save / load / cache / cache_key / is_ready."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from molix.data.cache import cache, cache_key, is_ready, load, save
from molix.data.pipeline import Pipeline
from molix.data.source import InMemorySource
from molix.data.task import DatasetTask, SampleTask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Shift(DatasetTask):
    def __init__(self, key: str = "y") -> None:
        self.key = key
        self.mean = 0.0

    @property
    def task_id(self) -> str:
        return f"shift:{self.key}"

    def fit(self, samples: list[dict]) -> None:
        self.mean = sum(float(s[self.key].item()) for s in samples) / len(samples)

    def execute(self, data: dict) -> dict:
        return {**data, self.key: data[self.key] - self.mean}

    def state_dict(self) -> dict[str, Any]:
        return {"mean": torch.tensor(self.mean, dtype=torch.float64)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.mean = float(state["mean"].item())


class _Tag(SampleTask):
    @property
    def task_id(self) -> str:
        return "tag"

    def execute(self, data: dict) -> dict:
        return {**data, "tagged": torch.tensor(1)}


def _samples(n: int = 6) -> list[dict]:
    return [
        {
            "Z": torch.tensor([1, 6]),
            "pos": torch.zeros(2, 3),
            "y": torch.tensor([float(i)]),
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# is_ready
# ---------------------------------------------------------------------------


class TestIsReady:
    def test_false_when_missing(self, tmp_path):
        assert is_ready(tmp_path / "nope.pt") is False

    def test_false_when_empty(self, tmp_path):
        p = tmp_path / "empty.pt"
        p.write_bytes(b"")
        assert is_ready(p) is False

    def test_true_when_populated(self, tmp_path):
        p = tmp_path / "x.pt"
        p.write_bytes(b"hello")
        assert is_ready(p) is True


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_roundtrip_samples_only(self, tmp_path):
        samples = _samples(4)
        sink = tmp_path / "x.pt"
        save(sink, samples)
        data = load(sink)
        assert len(data["samples"]) == 4
        assert torch.equal(data["samples"][0]["Z"], samples[0]["Z"])

    def test_roundtrip_with_task_states(self, tmp_path):
        sink = tmp_path / "x.pt"
        save(
            sink,
            _samples(2),
            task_states={"shift:y": {"mean": torch.tensor(2.5)}},
        )
        data = load(sink)
        assert "task_states" in data
        assert torch.equal(
            data["task_states"]["shift:y"]["mean"], torch.tensor(2.5)
        )

    def test_default_is_mmap(self, tmp_path):
        """Default load uses mmap — file remains referenced via mmap views."""
        sink = tmp_path / "x.pt"
        save(sink, _samples(3))
        d1 = load(sink)
        d2 = load(sink, mmap=False)
        # Same contents either way.
        assert torch.equal(d1["samples"][0]["Z"], d2["samples"][0]["Z"])

    def test_save_is_idempotent_unless_overwrite(self, tmp_path):
        sink = tmp_path / "x.pt"
        save(sink, _samples(2))
        mtime = sink.stat().st_mtime_ns
        save(sink, _samples(99))                   # no-op
        assert sink.stat().st_mtime_ns == mtime
        assert len(load(sink)["samples"]) == 2

        save(sink, _samples(3), overwrite=True)
        assert len(load(sink)["samples"]) == 3

    def test_atomic_no_partial_on_success(self, tmp_path):
        save(tmp_path / "x.pt", _samples(2))
        siblings = {p.name for p in tmp_path.iterdir()}
        assert siblings == {"x.pt"}

    def test_atomic_cleans_partial_on_failure(self, tmp_path, monkeypatch):
        def boom(*_a, **_kw):
            raise RuntimeError("boom")
        monkeypatch.setattr("molix.data.cache.torch.save", boom)

        with pytest.raises(RuntimeError, match="boom"):
            save(tmp_path / "x.pt", _samples(1))
        assert list(tmp_path.iterdir()) == []

    def test_scalar_metadata_roundtrips(self, tmp_path):
        samples = [{"Z": torch.tensor([1]), "n_atoms": 1}]
        sink = tmp_path / "x.pt"
        save(sink, samples)
        loaded = load(sink)
        assert loaded["samples"][0]["n_atoms"] == 1


# ---------------------------------------------------------------------------
# cache (run + save)
# ---------------------------------------------------------------------------


class TestCache:
    def test_runs_and_saves(self, tmp_path):
        spec = Pipeline("p").add(_Tag()).build()
        sink = tmp_path / "x.pt"
        cache(spec, InMemorySource(_samples(4)), sink=sink)

        loaded = load(sink)
        assert len(loaded["samples"]) == 4
        assert all("tagged" in s for s in loaded["samples"])

    def test_saves_task_states(self, tmp_path):
        shift = _Shift("y")
        spec = Pipeline("p").add(shift).build()
        sink = tmp_path / "x.pt"
        cache(spec, InMemorySource(_samples(4)), sink=sink)   # y=0..3 → mean=1.5

        loaded = load(sink)
        assert torch.allclose(
            loaded["task_states"]["shift:y"]["mean"].double(),
            torch.tensor(1.5, dtype=torch.float64),
            atol=1e-9,
        )

    def test_fit_source_only_sees_subset(self, tmp_path):
        """Regression for the scientific-correctness bug in the old materialize()."""
        shift = _Shift("y")
        spec = Pipeline("p").add(shift).build()

        # Full source y=0..9, fit_source y=0..4 → mean should be 2.0 not 4.5.
        cache(
            spec,
            InMemorySource(_samples(10)),
            sink=tmp_path / "x.pt",
            fit_source=InMemorySource(_samples(5)),
        )
        state = load(tmp_path / "x.pt")["task_states"]["shift:y"]
        assert abs(float(state["mean"].item()) - 2.0) < 1e-9

    def test_is_idempotent(self, tmp_path):
        spec = Pipeline("p").add(_Tag()).build()
        sink = tmp_path / "x.pt"
        cache(spec, InMemorySource(_samples(2)), sink=sink)
        mtime = sink.stat().st_mtime_ns
        cache(spec, InMemorySource(_samples(99)), sink=sink)   # no-op
        assert sink.stat().st_mtime_ns == mtime

    def test_overwrite(self, tmp_path):
        spec = Pipeline("p").add(_Tag()).build()
        sink = tmp_path / "x.pt"
        cache(spec, InMemorySource(_samples(2)), sink=sink)
        cache(spec, InMemorySource(_samples(5)), sink=sink, overwrite=True)
        assert len(load(sink)["samples"]) == 5


# ---------------------------------------------------------------------------
# Different fit_source → different key → different cache file (motivating bug)
# ---------------------------------------------------------------------------


class TestSubsetAwareness:
    def test_different_fit_source_different_key(self):
        k_full = cache_key(pipeline_id="p", source_id="s")
        k_sub = cache_key(pipeline_id="p", source_id="s",
                          fit_source_id="s-subset")
        assert k_full != k_sub

    def test_same_sink_same_pipeline_survives_subset_change(self, tmp_path):
        """Two workflows that choose different train subsets pick different
        sinks (via cache_key), so they don't collide."""
        spec = Pipeline("p").add(_Shift("y")).build()

        full = InMemorySource(_samples(10))
        fit_a = InMemorySource(_samples(5))       # y=0..4 → mean=2
        fit_b = InMemorySource(_samples(3))       # y=0..2 → mean=1

        sink_a = tmp_path / (
            f"{spec.name}-"
            f"{cache_key(pipeline_id=spec.pipeline_id, source_id='full', fit_source_id='a')}"
            ".pt"
        )
        sink_b = tmp_path / (
            f"{spec.name}-"
            f"{cache_key(pipeline_id=spec.pipeline_id, source_id='full', fit_source_id='b')}"
            ".pt"
        )
        assert sink_a != sink_b

        cache(spec, full, sink=sink_a, fit_source=fit_a)
        cache(spec, full, sink=sink_b, fit_source=fit_b)

        mean_a = load(sink_a)["task_states"]["shift:y"]["mean"].item()
        mean_b = load(sink_b)["task_states"]["shift:y"]["mean"].item()
        assert abs(mean_a - 2.0) < 1e-9
        assert abs(mean_b - 1.0) < 1e-9
