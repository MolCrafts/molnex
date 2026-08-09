"""Tests for :class:`molix.data.cache.PackedCache` — file IO + readiness + keys."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from molix.data.cache import PackedCache
from tests.test_molix.test_data.conftest import equal_count_samples

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
        assert PackedCache(tmp_path / "nope.pt").is_ready() is False

    def test_false_when_empty(self, tmp_path):
        p = tmp_path / "empty.pt"
        p.write_bytes(b"")
        assert PackedCache(p).is_ready() is False

    def test_true_when_populated(self, tmp_path):
        p = tmp_path / "x.pt"
        p.write_bytes(b"hello")
        assert PackedCache(p).is_ready() is True


# ---------------------------------------------------------------------------
# save / load roundtrips
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_roundtrip_samples_only(self, tmp_path):
        samples = _samples(4)
        cache = PackedCache(tmp_path / "x.pt")
        cache.save(samples)
        data = cache.load()
        assert len(data["samples"]) == 4
        assert torch.equal(data["samples"][0]["Z"], samples[0]["Z"])

    def test_roundtrip_with_task_states(self, tmp_path):
        cache = PackedCache(tmp_path / "x.pt")
        cache.save(
            _samples(2),
            task_states={"shift:y": {"mean": torch.tensor(2.5)}},
        )
        data = cache.load()
        assert "task_states" in data
        assert torch.equal(data["task_states"]["shift:y"]["mean"], torch.tensor(2.5))

    def test_default_is_mmap(self, tmp_path):
        """Default load uses mmap; same contents on both paths."""
        cache = PackedCache(tmp_path / "x.pt")
        cache.save(_samples(3))
        d1 = cache.load()
        d2 = cache.load(mmap=False)
        assert torch.equal(d1["samples"][0]["Z"], d2["samples"][0]["Z"])

    def test_save_is_idempotent_unless_overwrite(self, tmp_path):
        cache = PackedCache(tmp_path / "x.pt")
        cache.save(_samples(2))
        mtime = cache.sink.stat().st_mtime_ns
        cache.save(_samples(99))  # no-op
        assert cache.sink.stat().st_mtime_ns == mtime
        assert len(cache.load()["samples"]) == 2

        cache.save(_samples(3), overwrite=True)
        assert len(cache.load()["samples"]) == 3

    def test_atomic_no_partial_on_success(self, tmp_path):
        PackedCache(tmp_path / "x.pt").save(_samples(2))
        siblings = {p.name for p in tmp_path.iterdir()}
        assert siblings == {"x.pt"}

    def test_atomic_cleans_partial_on_failure(self, tmp_path, monkeypatch):
        def boom(*_a, **_kw):
            raise RuntimeError("boom")

        monkeypatch.setattr("molix.data.cache.torch.save", boom)

        with pytest.raises(RuntimeError, match="boom"):
            PackedCache(tmp_path / "x.pt").save(_samples(1))
        assert list(tmp_path.iterdir()) == []

    def test_scalar_metadata_roundtrips(self, tmp_path):
        samples = [{"Z": torch.tensor([1]), "n_atoms": 1}]
        cache = PackedCache(tmp_path / "x.pt")
        cache.save(samples)
        loaded = cache.load()
        assert loaded["samples"][0]["n_atoms"] == 1


# ---------------------------------------------------------------------------
# Schema inference — bucket identity
# ---------------------------------------------------------------------------


class TestPackSchema:
    """Edge keys keep their bucket even when ``n_edges == n_atoms``.

    ``_infer_schema_across`` classifies a key by its leading dim. For a
    2-atom molecule with one bidirectional pair (``E = 2 = N``) that dim
    is ambiguous, and the tie must not be broken by numerology: absorbing
    ``edge_index`` / ``edge_diff`` / ``edge_dist`` into the atoms bucket
    also suppresses ``edge_ptr``, which is what every downstream edge
    consumer reads (``MmapDataset.avg_num_neighbors``,
    ``collate_packed``). The cache is the origin of that silent drop, so
    it is pinned here at the payload level.
    """

    def _payload(self, tmp_path: Path, n: int = 3) -> dict[str, Any]:
        """Save *n* equal-count samples and read the raw on-disk payload."""
        sink = tmp_path / "eq.pt"
        PackedCache(sink).save(equal_count_samples(n))
        return torch.load(sink, weights_only=True)

    def test_edge_index_schema_kind_is_edge(self, tmp_path):
        """Schema entry for ``edge_index`` is tagged ``"edge"``, not ``"atom"``."""
        payload = self._payload(tmp_path)
        assert payload["schema"]["edge_index"][0] == "edge"

    def test_edge_geometry_schema_kind_is_edge(self, tmp_path):
        """``edge_diff`` / ``edge_dist`` follow ``edge_index`` into the edge kind."""
        payload = self._payload(tmp_path)
        assert payload["schema"]["edge_diff"][0] == "edge"
        assert payload["schema"]["edge_dist"][0] == "edge"

    def test_edge_keys_packed_into_edges_bucket(self, tmp_path):
        """The three edge keys are stored under ``payload["edges"]``."""
        payload = self._payload(tmp_path)
        assert set(payload["edges"]) == {"edge_index", "edge_diff", "edge_dist"}

    def test_edge_keys_absent_from_atoms_bucket(self, tmp_path):
        """No edge key leaks into ``payload["atoms"]`` (which holds Z / pos only)."""
        payload = self._payload(tmp_path)
        assert "edge_index" not in payload["atoms"]
        assert "edge_diff" not in payload["atoms"]
        assert "edge_dist" not in payload["atoms"]
        assert set(payload["atoms"]) == {"Z", "pos"}

    def test_edge_ptr_is_written(self, tmp_path):
        """``edge_ptr`` exists — its absence is what silences edge consumers."""
        payload = self._payload(tmp_path)
        assert "edge_ptr" in payload

    def test_edge_ptr_is_per_sample_cumsum(self, tmp_path):
        """``edge_ptr`` cumsums 2 edges per sample over 3 samples."""
        payload = self._payload(tmp_path)
        assert payload["edge_ptr"].tolist() == [0, 2, 4, 6]

    def test_atom_ptr_unaffected(self, tmp_path):
        """``atom_ptr`` still cumsums 2 atoms per sample (same numbers, other axis)."""
        payload = self._payload(tmp_path)
        assert payload["atom_ptr"].tolist() == [0, 2, 4, 6]

    def test_unpack_roundtrips_edge_keys(self, tmp_path):
        """Per-sample unpack returns the hand-built edge tensors unchanged."""
        sink = tmp_path / "eq.pt"
        samples = equal_count_samples(3)
        PackedCache(sink).save(samples)
        payload = PackedCache(sink).load()
        s2 = PackedCache.unpack_sample(payload, 2)
        assert torch.equal(s2["edge_index"], samples[2]["edge_index"])
        assert torch.equal(s2["edge_diff"], samples[2]["edge_diff"])
        assert torch.equal(s2["edge_dist"], samples[2]["edge_dist"])


# ---------------------------------------------------------------------------
# unpack_sample (staticmethod)
# ---------------------------------------------------------------------------


class TestUnpackSample:
    def test_reconstructs_nested_dict(self, tmp_path):
        samples = [
            {
                "Z": torch.tensor([1, 6]),
                "pos": torch.zeros(2, 3),
                "targets": {"U0": torch.tensor([1.0])},
            },
            {
                "Z": torch.tensor([7, 8]),
                "pos": torch.ones(2, 3),
                "targets": {"U0": torch.tensor([2.0])},
            },
        ]
        cache = PackedCache(tmp_path / "x.pt")
        cache.save(samples)
        payload = cache.load()
        s1 = PackedCache.unpack_sample(payload, 1)
        assert torch.equal(s1["Z"], samples[1]["Z"])
        assert torch.allclose(s1["targets"]["U0"], samples[1]["targets"]["U0"])


# ---------------------------------------------------------------------------
# wait_until_ready (DDP polling)
# ---------------------------------------------------------------------------


class TestWaitUntilReady:
    def test_returns_immediately_when_ready(self, tmp_path):
        p = tmp_path / "ready.pt"
        p.write_bytes(b"data")
        # No raise → test passes.
        PackedCache(p).wait_until_ready(timeout=1.0, poll_interval=0.01)

    def test_raises_on_timeout(self, tmp_path):
        with pytest.raises(TimeoutError, match="Timed out"):
            PackedCache(tmp_path / "never.pt").wait_until_ready(
                timeout=0.3,
                poll_interval=0.05,
            )


# ---------------------------------------------------------------------------
# make_key removed — cache key derivation moved to Node.cache_key()
# (tested in test_pipeline.py::TestNode / TestCacheKey)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Miscellaneous
# ---------------------------------------------------------------------------


class TestMisc:
    def test_sink_property(self, tmp_path):
        p = tmp_path / "x.pt"
        cache = PackedCache(p)
        assert cache.sink == p

    def test_fspath_roundtrips_through_str(self, tmp_path):
        import os

        p = tmp_path / "x.pt"
        assert os.fspath(PackedCache(p)) == str(p)
