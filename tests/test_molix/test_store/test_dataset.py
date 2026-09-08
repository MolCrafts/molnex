"""Tests for :class:`BaseDataset` splitting and :class:`SubsetDataset` views.

Both are storage-agnostic — a dataset is anything with ``__len__`` and
``__getitem__`` — so the fixture here is a list-backed stub rather than any
on-disk format. Block-backed reading is covered by ``test_store/test_frame_store.py``
and ``test_store/test_frame_store_view.py``.
"""

from __future__ import annotations

import pickle

import pytest
import torch

from molix.store.dataset import BaseDataset, SubsetDataset


def _samples(n: int = 8) -> list[dict]:
    """Variable-length samples, the way real molecules are ragged."""
    return [
        {
            "Z": torch.arange(i + 1, dtype=torch.long),
            "pos": torch.zeros(i + 1, 3),
            "targets": {"energy": torch.tensor([float(i)])},
        }
        for i in range(n)
    ]


class _ListDataset(BaseDataset):
    """Minimal in-memory dataset; module level so pickle can find it."""

    custom_marker = "abc"

    def __init__(self, samples: list[dict]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        return self._samples[idx]


class TestSubsetDataset:
    def test_len_and_getitem_remapping(self):
        samples = _samples(8)
        sub = SubsetDataset(_ListDataset(samples), [0, 3, 7])

        assert len(sub) == 3
        assert torch.equal(sub[0]["Z"], samples[0]["Z"])
        assert torch.equal(sub[1]["Z"], samples[3]["Z"])
        assert torch.equal(sub[2]["Z"], samples[7]["Z"])

    def test_forwards_attribute_access_to_parent(self):
        sub = SubsetDataset(_ListDataset(_samples(6)), [0, 2, 4])

        assert sub.custom_marker == "abc"

    def test_pickle_roundtrip_preserves_the_view(self):
        """Workers receive the subset by pickle; ``__getattr__`` must not recurse."""
        sub = SubsetDataset(_ListDataset(_samples(6)), [1, 4])
        restored = pickle.loads(pickle.dumps(sub))

        assert len(restored) == 2
        assert torch.equal(restored[0]["Z"], _samples(6)[1]["Z"])


class TestSplit:
    def _full(self, n: int = 20) -> _ListDataset:
        return _ListDataset(_samples(n))

    def test_split_sizes(self):
        ds = self._full()
        train, val = ds.split(ratio=0.8)

        assert len(train) + len(val) == len(ds)
        assert len(train) == int(20 * 0.8)

    def test_split_no_overlap(self):
        ds = self._full()
        train, val = ds.split(ratio=0.8)

        assert set(train._indices).isdisjoint(set(val._indices))
        assert set(train._indices) | set(val._indices) == set(range(len(ds)))

    def test_split_reproducible(self):
        ds = self._full()
        t1, v1 = ds.split(ratio=0.8, seed=42)
        t2, v2 = ds.split(ratio=0.8, seed=42)

        assert t1._indices == t2._indices
        assert v1._indices == v2._indices

    def test_split_different_seeds(self):
        ds = self._full()
        t1, _ = ds.split(ratio=0.8, seed=0)
        t2, _ = ds.split(ratio=0.8, seed=1)

        assert t1._indices != t2._indices

    def test_split_by_sizes(self):
        train, val, test = self._full(10).split(sizes=(6, 2, 2))

        assert (len(train), len(val), len(test)) == (6, 2, 2)

    def test_split_sizes_mismatch_raises(self):
        with pytest.raises(ValueError, match="sizes must sum"):
            self._full(10).split(sizes=(5, 4))

    def test_requires_exactly_one_of_ratio_or_sizes(self):
        ds = self._full(10)
        with pytest.raises(ValueError, match="exactly one"):
            ds.split(ratio=0.5, sizes=(5, 5))
        with pytest.raises(ValueError, match="exactly one"):
            ds.split()
