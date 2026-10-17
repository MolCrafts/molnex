"""Prepared rows concatenate to the same batch the frame collator builds."""

from __future__ import annotations

import errno

import numpy as np
import pytest
import torch

from molix.data.collate import collate_molecules
from molix.data.prepared import PreparedMmap, PreparedRow, PreparedRows
from molix.data.relation import PairRelation
from molix.neighbor import NeighborList
from tests.conftest import frame
from tests.test_molix.test_data.conftest import record


def _hpair(origin: float):
    """One bond whose partial-Hessian slot is local index 0."""
    sample = frame(
        Z=[6, 1],
        pos=[[origin, 0.0, 0.0], [origin + 1.0, 0.0, 0.0]],
        bonds=([0], [1]),
        energy=0.0,
    )
    bonds = {column: np.asarray(sample["bonds"][column]) for column in sample["bonds"].keys()}
    bonds["hpair_0_1"] = np.asarray([0], dtype=np.int64)
    sample["bonds"] = bonds
    sample["hessian_pairs"] = {
        "atomi": np.asarray([0], dtype=np.int64),
        "atomj": np.asarray([1], dtype=np.int64),
        "block": np.arange(9, dtype=np.float64).reshape(1, 9) + origin,
    }
    return (sample,)


def _tasks():
    return (
        NeighborList(cutoff=3.5, max_num_pairs=32),
        PairRelation(),
    )


def _reference(samples, tasks):
    batch = collate_molecules(samples)
    for task in tasks:
        batch = task.execute(batch)
    return batch


def _sorted_pairs(batch):
    index = batch["pairs", "pair_index"].cpu()
    order = torch.argsort(index[:, 0] * 1_000_003 + index[:, 1], stable=True)
    return (
        index[order],
        batch["pairs", "pair_diff"].cpu()[order],
        batch["pairs", "pair_dist"].cpu()[order],
        batch["pairs", "relation"].cpu()[order],
    )


def _assert_same_graph(prepared, reference) -> None:
    assert torch.equal(prepared["atoms", "Z"], reference["atoms", "Z"])
    assert torch.equal(prepared["atoms", "pos"], reference["atoms", "pos"])
    assert torch.equal(prepared["atoms", "batch"], reference["atoms", "batch"])
    assert torch.equal(prepared["graphs", "num_atoms"], reference["graphs", "num_atoms"])
    assert torch.equal(prepared["graphs", "row"], reference["graphs", "row"])
    assert torch.equal(prepared["graphs", "energy"], reference["graphs", "energy"])
    assert torch.equal(prepared["bonds", "bond_index"], reference["bonds", "bond_index"])
    assert torch.equal(prepared["angles", "atomk"], reference["angles", "atomk"])
    left = _sorted_pairs(prepared)
    right = _sorted_pairs(reference)
    for got, exp in zip(left, right, strict=True):
        assert torch.equal(got, exp)


class _Store:
    def __init__(self) -> None:
        self.values: dict[int, bytes] = {}
        self.reads: list[list[int]] = []

    def get(self, index: int) -> bytes | None:
        return self.values.get(int(index))

    def put(self, index: int, payload: bytes) -> None:
        self.values[int(index)] = bytes(payload)


class _Source:
    def __init__(self, rows) -> None:
        self.rows = list(rows)
        self.asked: list[list[int]] = []

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        self.asked.append([int(index)])
        return self.rows[int(index)]

    def __getitems__(self, indices):
        self.asked.append([int(index) for index in indices])
        return [self.rows[int(index)] for index in indices]

    source_id = "fixture"


class _Subset:
    """A subset view: public ``source`` and ``indices``, as molhub's."""

    def __init__(self, source, indices) -> None:
        self.source = source
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)


class TestPreparedRow:
    """A prepared row is the frame collator's batch, stored one row at a time."""

    def test_one_row_matches_the_frame_collator(self) -> None:
        tasks = _tasks()
        sample = record(0.0)
        prepared = PreparedRow.assemble([PreparedRow.capture(sample, prepare=None, tasks=tasks)])
        _assert_same_graph(prepared, _reference([sample], tasks))

    def test_two_rows_match_including_the_cutoff_graph(self) -> None:
        tasks = _tasks()
        records = (record(0.0), record(10.0))
        prepared = PreparedRow.assemble(
            [PreparedRow.capture(sample, prepare=None, tasks=tasks) for sample in records]
        )
        _assert_same_graph(prepared, _reference(records, tasks))
        assert torch.equal(prepared["graphs", "cell"], _reference(records, tasks)["graphs", "cell"])

    def test_frames_of_one_row_share_that_rows_index(self) -> None:
        tasks = _tasks()
        records = (record(0.0, frames=2), record(10.0))
        prepared = PreparedRow.assemble(
            [PreparedRow.capture(sample, prepare=None, tasks=tasks) for sample in records]
        )
        reference = _reference(records, tasks)
        _assert_same_graph(prepared, reference)
        assert prepared["graphs", "row"].tolist() == [0, 0, 1]
        assert prepared["graphs", "num_atoms"].tolist() == [4, 4, 4]

    def test_hessian_slots_are_rebased_by_the_pair_rows(self) -> None:
        records = (_hpair(0.0), _hpair(10.0))
        prepared = PreparedRow.assemble(
            [PreparedRow.capture(sample, prepare=None, tasks=()) for sample in records]
        )
        reference = _reference(records, ())
        assert torch.equal(prepared["bonds", "hpair_0_1"], reference["bonds", "hpair_0_1"])
        assert prepared["bonds", "hpair_0_1"].tolist() == [0, 1]
        assert torch.equal(prepared["hessian_pairs", "atomi"], reference["hessian_pairs", "atomi"])
        assert prepared["hessian_pairs", "atomi"].tolist() == [0, 2]

    def test_pack_roundtrip_assembles_the_same_batch(self) -> None:
        tasks = _tasks()
        records = (record(0.0, frames=2), record(10.0))
        rows = [PreparedRow.capture(sample, prepare=None, tasks=tasks) for sample in records]
        restored = [PreparedRow.unpack(row.pack()) for row in rows]
        _assert_same_graph(PreparedRow.assemble(restored), PreparedRow.assemble(rows))

    def test_a_foreign_payload_is_refused(self) -> None:
        with pytest.raises(ValueError, match="magic"):
            PreparedRow.unpack(b"not-a-prepared-row")

    def test_rows_that_differ_in_columns_are_refused(self) -> None:
        tasks = _tasks()
        bare = frame(Z=[6], pos=[[0.0, 0.0, 0.0]], energy=1.0)
        rows = [
            PreparedRow.capture(record(0.0), prepare=None, tasks=tasks),
            PreparedRow.capture((bare,), prepare=None, tasks=tasks),
        ]
        with pytest.raises(ValueError, match="differ in columns"):
            PreparedRow.assemble(rows)


class TestPreparedRows:
    """The second read of a row does not touch the dataset it was prepared from."""

    def test_a_hit_does_not_fetch(self) -> None:
        source = _Source([record(0.0), record(10.0)])
        store = _Store()
        rows = PreparedRows(source, prepare=None, tasks=_tasks(), store=store)
        first = rows.__getitems__([1, 0])
        assert source.asked == [[1, 0]]
        assert sorted(store.values) == [0, 1]
        source.asked.clear()
        second = rows.__getitems__([0, 1])
        assert source.asked == []
        reference = _reference([record(0.0), record(10.0)], _tasks())
        _assert_same_graph(PreparedRow.assemble(second), reference)
        assert first[0].n_atoms == 4

    def test_a_subset_is_cached_under_the_parent_index(self) -> None:
        source = _Source([record(0.0), record(5.0), record(10.0)])
        subset = _Subset(source, [2, 0])
        store = _Store()
        rows = PreparedRows(subset, prepare=None, tasks=_tasks(), store=store)
        rows.__getitems__([0, 1])
        assert source.asked == [[2, 0]]
        assert sorted(store.values) == [0, 2]
        source.asked.clear()
        again = rows[0]
        assert source.asked == []
        assert again.n_atoms == 4

    def test_an_empty_fetch_reads_nothing(self) -> None:
        source = _Source([record(0.0)])
        rows = PreparedRows(source, prepare=None, tasks=(), store=_Store())
        assert rows.__getitems__([]) == []
        assert source.asked == []

    def test_an_index_past_the_end_is_refused_before_the_store(self) -> None:
        source = _Source([record(0.0)])
        rows = PreparedRows(source, prepare=None, tasks=(), store=_Store())
        with pytest.raises(IndexError, match="outside"):
            rows[1]
        assert source.asked == []

    def test_a_full_disk_returns_the_row_and_stops_caching(self) -> None:
        class _Full(_Store):
            def __init__(self) -> None:
                super().__init__()
                self.puts = 0

            def put(self, index: int, payload: bytes) -> None:
                self.puts += 1
                raise OSError(errno.ENOSPC, "No space left on device")

        source = _Source([record(0.0), record(10.0)])
        store = _Full()
        rows = PreparedRows(source, prepare=None, tasks=_tasks(), store=store)
        got = rows.__getitems__([0, 1])
        assert [row.n_atoms for row in got] == [4, 4]
        assert store.puts == 1
        assert rows._store is None
        source.asked.clear()
        again = rows.__getitems__([0])
        assert again[0].n_atoms == 4
        assert source.asked == [[0]]
        assert store.puts == 1


class TestAssembledRows:
    """Prepared rows assemble to the batch the frame path builds, tasks included."""

    def test_assembled_rows_match_the_frame_path(self) -> None:
        tasks = _tasks()
        records = (record(0.0, frames=2), record(10.0))
        raw = _reference(records, tasks)
        rows = [PreparedRow.capture(sample, prepare=None, tasks=tasks) for sample in records]
        _assert_same_graph(PreparedRow.assemble(rows), raw)


class TestPreparedMmap:
    """A warm read is a slice of the mapped file, not a second copy of the row."""

    def test_a_mapped_row_matches_the_frame_collator(self, tmp_path) -> None:
        tasks = _tasks()
        records = (record(0.0, frames=2), record(10.0))
        rows = [PreparedRow.capture(sample, prepare=None, tasks=tasks) for sample in records]
        store = PreparedMmap(tmp_path / "rows", 4)
        store.put(2, rows[0].pack())
        store.put(0, rows[1].pack())
        store.put(2, rows[1].pack())
        restored = PreparedRow.unpack_many([store.get(2), store.get(0)])
        _assert_same_graph(
            PreparedRow.assemble(restored),
            _reference(records, tasks),
        )
        again = PreparedMmap(tmp_path / "rows", 4)
        assert again.get(1) is None
        view = again.get(2)
        assert bytes(view[:4]) == b"PRP2"
        del view, restored
        store.close()
        again.close()

    def test_a_held_view_survives_the_file_growing(self, tmp_path) -> None:
        store = PreparedMmap(tmp_path / "rows", 4)
        store.put(0, b"PRP2" + b"\x00" * 12)
        first = store.get(0)
        store.put(1, b"PRP2" + b"\x01" * 20)
        second = store.get(1)
        assert bytes(first[:4]) == b"PRP2"
        assert bytes(second[:4]) == b"PRP2"
        del first, second
        store.close()
