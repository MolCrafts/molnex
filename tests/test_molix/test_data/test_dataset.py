"""Tests for MmapDataset / CachedDataset / SubsetDataset.

These classes are now thin wrappers over a ``.pt`` cache file produced by
:func:`molix.data.cache.save` (or :func:`molix.data.cache.cache`). The tests
drive the whole read path end-to-end via save→load.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest
import torch

from molix.data.cache import save
from molix.data.dataset import (
    BaseDataset,
    CachedDataset,
    MmapDataset,
    SubsetDataset,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_samples(n: int = 8) -> list[dict]:
    """Minimal samples with variable-length tensors (like real molecules)."""
    return [
        {
            "Z": torch.arange(i + 1, dtype=torch.long),
            "pos": torch.randn(i + 1, 3),
            "targets": {"U0": torch.tensor([float(i)])},
        }
        for i in range(n)
    ]


def _make_cache(tmp_path: Path, n: int = 8, *, task_states=None) -> Path:
    sink = tmp_path / "x.pt"
    save(sink, _make_samples(n), task_states=task_states)
    return sink


# ---------------------------------------------------------------------------
# MmapDataset
# ---------------------------------------------------------------------------


class TestMmapDataset:
    def test_len_and_getitem(self, tmp_path):
        samples = _make_samples(4)
        sink = tmp_path / "x.pt"
        save(sink, samples)
        ds = MmapDataset(sink)
        assert len(ds) == 4
        assert torch.equal(ds[0]["Z"], samples[0]["Z"])

    def test_values_survive_roundtrip(self, tmp_path):
        samples = _make_samples(6)
        sink = tmp_path / "x.pt"
        save(sink, samples)
        ds = MmapDataset(sink)
        for i, orig in enumerate(samples):
            item = ds[i]
            assert torch.equal(item["Z"], orig["Z"])
            assert torch.allclose(item["pos"], orig["pos"])
            assert torch.allclose(item["targets"]["U0"], orig["targets"]["U0"])

    def test_dtypes_preserved(self, tmp_path):
        samples = [
            {
                "Z": torch.tensor([1, 6], dtype=torch.int64),
                "pos": torch.tensor(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32
                ),
                "flag": torch.tensor([True, False], dtype=torch.bool),
            }
        ]
        sink = tmp_path / "x.pt"
        save(sink, samples)
        ds = MmapDataset(sink)
        item = ds[0]
        assert item["Z"].dtype == torch.int64
        assert item["pos"].dtype == torch.float32
        assert item["flag"].dtype == torch.bool

    def test_task_states_exposed(self, tmp_path):
        sink = _make_cache(
            tmp_path,
            n=3,
            task_states={"atomic_dress": {"baseline": torch.tensor([1.0, 2.0])}},
        )
        ds = MmapDataset(sink)
        loaded = ds.get_task_state("atomic_dress")
        assert torch.equal(loaded["baseline"], torch.tensor([1.0, 2.0]))

    def test_task_states_empty_when_absent(self, tmp_path):
        ds = MmapDataset(_make_cache(tmp_path, n=2))
        assert dict(ds.task_states) == {}

    def test_task_states_keyerror_on_unknown_name(self, tmp_path):
        ds = MmapDataset(_make_cache(tmp_path, n=2))
        with pytest.raises(KeyError):
            ds.get_task_state("nope")

    def test_sink_attr(self, tmp_path):
        sink = _make_cache(tmp_path)
        ds = MmapDataset(sink)
        assert ds.sink == sink

    def test_pickle_roundtrip(self, tmp_path):
        """spawn-mode DataLoader workers pickle the dataset."""
        samples = _make_samples(4)
        sink = tmp_path / "x.pt"
        save(sink, samples)
        ds = MmapDataset(sink)
        ds2 = pickle.loads(pickle.dumps(ds))
        for i in range(len(samples)):
            assert torch.equal(ds2[i]["Z"], samples[i]["Z"])


# ---------------------------------------------------------------------------
# CachedDataset
# ---------------------------------------------------------------------------


class TestCachedDataset:
    def test_len_and_getitem(self, tmp_path):
        samples = _make_samples(5)
        sink = tmp_path / "x.pt"
        save(sink, samples)
        ds = CachedDataset(sink)
        assert len(ds) == 5
        assert torch.equal(ds[2]["Z"], samples[2]["Z"])

    def test_is_base_dataset(self):
        assert issubclass(CachedDataset, BaseDataset)

    def test_task_states_exposed(self, tmp_path):
        sink = _make_cache(
            tmp_path,
            task_states={"dress": {"baseline": torch.tensor([1.0])}},
        )
        ds = CachedDataset(sink)
        assert torch.equal(
            ds.get_task_state("dress")["baseline"], torch.tensor([1.0])
        )


# ---------------------------------------------------------------------------
# SubsetDataset
# ---------------------------------------------------------------------------


class _AttrMmap(MmapDataset):
    """Module-level subclass so pickle can find it."""

    custom_marker = "abc"


class TestSubsetDataset:
    def test_len_and_getitem_remapping(self, tmp_path):
        samples = _make_samples(8)
        sink = tmp_path / "x.pt"
        save(sink, samples)
        full = MmapDataset(sink)
        sub = SubsetDataset(full, [0, 3, 7])
        assert len(sub) == 3
        assert torch.equal(sub[0]["Z"], samples[0]["Z"])
        assert torch.equal(sub[1]["Z"], samples[3]["Z"])
        assert torch.equal(sub[2]["Z"], samples[7]["Z"])

    def test_is_base_dataset(self):
        assert issubclass(SubsetDataset, BaseDataset)

    def test_forwards_attribute_access_to_parent(self, tmp_path):
        sink = _make_cache(tmp_path, n=6)
        full = _AttrMmap(sink)
        sub = SubsetDataset(full, [0, 2, 4])
        assert sub.custom_marker == "abc"


# ---------------------------------------------------------------------------
# BaseDataset.split()
# ---------------------------------------------------------------------------


class TestSplit:
    def _full(self, tmp_path, n=20):
        sink = tmp_path / "x.pt"
        save(sink, _make_samples(n))
        return MmapDataset(sink)

    def test_split_sizes(self, tmp_path):
        ds = self._full(tmp_path)
        train, val = ds.split(ratio=0.8)
        assert len(train) + len(val) == len(ds)
        assert len(train) == int(20 * 0.8)

    def test_split_no_overlap(self, tmp_path):
        ds = self._full(tmp_path)
        train, val = ds.split(ratio=0.8)
        assert set(train._indices).isdisjoint(set(val._indices))
        assert set(train._indices) | set(val._indices) == set(range(len(ds)))

    def test_split_reproducible(self, tmp_path):
        ds = self._full(tmp_path)
        t1, v1 = ds.split(ratio=0.8, seed=42)
        t2, v2 = ds.split(ratio=0.8, seed=42)
        assert t1._indices == t2._indices
        assert v1._indices == v2._indices

    def test_split_different_seeds(self, tmp_path):
        ds = self._full(tmp_path)
        t1, _ = ds.split(ratio=0.8, seed=0)
        t2, _ = ds.split(ratio=0.8, seed=1)
        assert t1._indices != t2._indices

    def test_split_by_sizes(self, tmp_path):
        ds = self._full(tmp_path, n=10)
        train, val, test = ds.split(sizes=(6, 2, 2))
        assert len(train) == 6 and len(val) == 2 and len(test) == 2

    def test_split_sizes_mismatch_raises(self, tmp_path):
        ds = self._full(tmp_path, n=10)
        with pytest.raises(ValueError, match="sizes must sum"):
            ds.split(sizes=(5, 4))

    def test_requires_exactly_one_of_ratio_or_sizes(self, tmp_path):
        ds = self._full(tmp_path, n=10)
        with pytest.raises(ValueError, match="exactly one"):
            ds.split(ratio=0.5, sizes=(5, 5))
        with pytest.raises(ValueError, match="exactly one"):
            ds.split()

    def test_split_returns_subset_datasets(self, tmp_path):
        ds = self._full(tmp_path)
        train, val = ds.split(ratio=0.5)
        assert isinstance(train, SubsetDataset)
        assert isinstance(val, SubsetDataset)
