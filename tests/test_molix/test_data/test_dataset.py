"""Tests for :class:`BaseDataset` splitting and :class:`SubsetDataset` views.

Both are storage-agnostic — a dataset is anything with ``__len__`` and
``__getitem__`` — so the fixture here is a list-backed stub rather than any
on-disk format: a dataset is an indexable space of molpy Frames and nothing more.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest
import torch
from molpy import Block, Frame

from molix.data.dataset import BaseDataset, SubsetDataset
from molix.data.frame_dataset import FrameDataset
from tests.test_molix.conftest import frames_dataset, unit_a


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

    def test_labels_are_gathered_at_the_view_indices(self):
        ds = FrameDataset(
            [
                (_geom([8], molecule_id="a"),),
                (_geom([6], molecule_id="b"),),
                (_geom([7], molecule_id="c"),),
            ],
            task_labels=("equilibrium", "torsion", "equilibrium"),
            group_ids=("a", "b", "c"),
        )
        sub = SubsetDataset(ds, [2, 0])

        assert sub.task_labels == ("equilibrium", "equilibrium")
        assert sub.group_ids == ("c", "a")

    def test_pickle_roundtrip_preserves_the_view(self):
        """Workers receive the subset by pickle; the view survives the trip."""
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


# --- BaseDataset.n_rows -------------------------------------------------
def _geom(z: list[int], *, molecule_id: str) -> Frame:
    n = len(z)
    atoms = Block()
    atoms["atomic_number"] = np.array(z, dtype=np.uint64)
    atoms["x"] = np.arange(n, dtype=np.float64)
    atoms["y"] = np.zeros(n, dtype=np.float64)
    atoms["z"] = np.zeros(n, dtype=np.float64)
    frame = Frame()
    frame["atoms"] = atoms
    frame.meta = {**frame.meta, "molecule_id": molecule_id}
    return frame


class TestAtomNrows:
    def test_frame_dataset_reads_atoms_nrows(self) -> None:
        ds = FrameDataset.from_records(
            [(_geom([8, 1, 1], molecule_id="w"),), (_geom([6, 1], molecule_id="m"),)],
            task="equilibrium",
        )
        got = BaseDataset.nrows_of(ds, "atoms")
        assert got.dtype is torch.long
        assert got.tolist() == [3, 2]

    def test_scan_sums_nrows_across_frames(self) -> None:
        frames = tuple(_geom([8, 1], molecule_id="s") for _ in range(3))
        ds = FrameDataset.from_records([frames], task="torsion")
        assert BaseDataset.nrows_of(ds, "atoms").tolist() == [6]

    def test_subset_gathers_parent_nrows(self) -> None:
        ds = FrameDataset.from_records(
            [
                (_geom([8, 1, 1], molecule_id="a"),),
                (_geom([6], molecule_id="b"),),
                (_geom([7, 1], molecule_id="c"),),
            ],
            task="equilibrium",
        )
        subset = SubsetDataset(ds, [2, 0])
        assert BaseDataset.nrows_of(subset, "atoms").tolist() == [2, 3]

    def test_dict_sample_raises_type_error(self) -> None:
        class _Dicts:
            def __len__(self) -> int:
                return 1

            def __getitem__(self, idx: int) -> dict[str, int]:
                return {"Z": 1}

        with pytest.raises(TypeError, match="Frame"):
            BaseDataset.nrows_of(_Dicts(), "atoms")


def _with_edges(frame: Frame, n_pairs: int) -> Frame:
    pairs = Block()
    pairs["i"] = np.zeros(n_pairs, dtype=np.int64)
    frame["pairs"] = pairs
    return frame


class TestEdgeNrows:
    def test_frame_dataset_reads_edges_nrows(self) -> None:
        ds = FrameDataset.from_records(
            [
                (_with_edges(_geom([8, 1, 1], molecule_id="w"), 4),),
                (_with_edges(_geom([6, 1], molecule_id="m"), 2),),
            ],
            task="equilibrium",
        )
        got = BaseDataset.nrows_of(ds, "pairs")
        assert got.dtype is torch.long
        assert got.tolist() == [4, 2]

    def test_missing_edges_table_raises(self) -> None:
        ds = FrameDataset.from_records(
            [(_geom([8, 1], molecule_id="w"),)],
            task="equilibrium",
        )
        with pytest.raises(ValueError, match="NeighborList"):
            BaseDataset.nrows_of(ds, "pairs")

    def test_subset_gathers_parent_edge_nrows(self) -> None:
        ds = FrameDataset.from_records(
            [
                (_with_edges(_geom([8], molecule_id="a"), 5),),
                (_with_edges(_geom([6], molecule_id="b"), 1),),
                (_with_edges(_geom([7], molecule_id="c"), 3),),
            ],
            task="equilibrium",
        )
        subset = SubsetDataset(ds, [2, 0])
        assert BaseDataset.nrows_of(subset, "pairs").tolist() == [3, 5]


class TestValenceNrows:
    """A valence table is answered from the Frame block, or refused by name."""

    def test_undeclared_family_raises_naming_the_table(self) -> None:
        ds = frames_dataset([unit_a()])
        with pytest.raises(ValueError, match="impropers"):
            BaseDataset.nrows_of(ds, "impropers")
