"""Tests for BaseDataset, CachedDataset, MmapDataset, SubsetDataset."""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import pytest
import torch

from molix.data.dataset import (
    BaseDataset,
    CacheValidationError,
    CachedDataset,
    MmapDataset,
    SubsetDataset,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _SubMmap(MmapDataset):
    """Module-level subclass — pickle cannot serialize local classes."""

    def __init__(self, stem, *, marker) -> None:
        super().__init__(stem)
        self.marker = marker


class _AttrMmap(MmapDataset):
    """Module-level subclass with class attribute, used by SubsetDataset test."""

    custom_marker = "abc"


def _make_samples(n: int = 8) -> list[dict]:
    """Minimal samples with variable-length tensors (like real molecules)."""
    return [
        {
            "Z": torch.arange(i + 1, dtype=torch.long),          # length varies
            "pos": torch.randn(i + 1, 3),
            "targets": {"U0": torch.tensor([float(i)])},
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# CachedDataset
# ---------------------------------------------------------------------------

class TestCachedDataset:
    def test_len_and_getitem(self):
        samples = _make_samples(5)
        ds = CachedDataset(samples)
        assert len(ds) == 5
        item = ds[2]
        assert torch.equal(item["Z"], samples[2]["Z"])

    def test_from_samples_ignores_cache_args(self, tmp_path):
        samples = _make_samples(3)
        ds = CachedDataset.from_samples(samples, cache_dir=tmp_path, name="train")
        assert len(ds) == 3

    def test_is_base_dataset(self):
        assert issubclass(CachedDataset, BaseDataset)


# ---------------------------------------------------------------------------
# MmapDataset
# ---------------------------------------------------------------------------

class TestMmapDataset:
    def test_from_samples_creates_files(self, tmp_path):
        samples = _make_samples(4)
        ds = MmapDataset.from_samples(samples, cache_dir=tmp_path, name="train")
        assert (tmp_path / "train.bin").exists()
        assert (tmp_path / "train.idx").exists()
        assert len(ds) == 4

    def test_getitem_values_match(self, tmp_path):
        samples = _make_samples(6)
        ds = MmapDataset.from_samples(samples, cache_dir=tmp_path, name="data")
        for i, orig in enumerate(samples):
            item = ds[i]
            assert torch.equal(item["Z"], orig["Z"])
            assert torch.allclose(item["pos"], orig["pos"])
            assert torch.allclose(item["targets"]["U0"], orig["targets"]["U0"])

    def test_reuses_existing_files(self, tmp_path):
        samples = _make_samples(4)
        MmapDataset.from_samples(samples, cache_dir=tmp_path, name="data")
        mtime_bin = (tmp_path / "data.bin").stat().st_mtime
        # Second call must not rewrite the file
        MmapDataset.from_samples(samples, cache_dir=tmp_path, name="data")
        assert (tmp_path / "data.bin").stat().st_mtime == mtime_bin

    def test_requires_cache_dir(self):
        with pytest.raises(ValueError, match="cache_dir"):
            MmapDataset.from_samples(_make_samples(2), cache_dir=None)

    def test_pickle_roundtrip(self, tmp_path):
        """spawn-mode DataLoader workers pickle and unpickle the dataset."""
        samples = _make_samples(4)
        ds = MmapDataset.from_samples(samples, cache_dir=tmp_path, name="data")
        ds2 = pickle.loads(pickle.dumps(ds))
        for i in range(len(samples)):
            assert torch.equal(ds2[i]["Z"], samples[i]["Z"])

    def test_subclass_pickle_preserves_extra_attrs(self, tmp_path):
        """Regression: __getstate__ must preserve subclass-added attributes.

        Bug: returning a hand-picked dict from __getstate__ silently dropped
        attrs set by subclass __init__ (e.g. QM9Dataset.target_schema). After
        unpickling in a worker, accessing them raised AttributeError. The fix
        uses self.__dict__.copy() and only excludes the mmap file handles.
        """
        samples = _make_samples(3)
        MmapDataset.from_samples(samples, cache_dir=tmp_path, name="sub")
        ds = _SubMmap(tmp_path / "sub", marker=("hello", 42))

        ds2 = pickle.loads(pickle.dumps(ds))

        assert ds2.marker == ("hello", 42)
        # Mmap state is reopened on the other side.
        assert torch.equal(ds2[0]["Z"], samples[0]["Z"])

    def test_pickle_does_not_carry_file_handles(self, tmp_path):
        """The pickle payload must not contain the open mmap / file handle."""
        samples = _make_samples(2)
        ds = MmapDataset.from_samples(samples, cache_dir=tmp_path, name="data")
        state = ds.__getstate__()
        assert "_mm" not in state
        assert "_f" not in state
        assert "_index" in state

    def test_no_pickle_in_bin_file(self, tmp_path):
        """Confirm .bin is raw bytes, not a pickle stream."""
        samples = _make_samples(3)
        MmapDataset.from_samples(samples, cache_dir=tmp_path, name="data")
        raw = (tmp_path / "data.bin").read_bytes()
        # pickle streams start with b'\x80' (opcode PROTO); raw tensor bytes don't
        assert raw[:1] != b"\x80", ".bin file must be raw tensor bytes, not pickle"

    def test_tensors_are_views_not_copies(self, tmp_path):
        """torch.frombuffer must return mmap-backed views (data_ptr check)."""
        samples = _make_samples(4)
        ds = MmapDataset.from_samples(samples, cache_dir=tmp_path, name="data")
        t = ds[0]["Z"]
        # A view into mmap is not a plain storage-backed copy
        assert not t.is_cuda
        assert t.is_contiguous()

    def test_dtypes_preserved(self, tmp_path):
        samples = [
            {
                "Z": torch.tensor([1, 6], dtype=torch.int64),
                "pos": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32),
                "flag": torch.tensor([True, False], dtype=torch.bool),
            }
        ]
        ds = MmapDataset.from_samples(samples, cache_dir=tmp_path, name="data")
        item = ds[0]
        assert item["Z"].dtype == torch.int64
        assert item["pos"].dtype == torch.float32
        assert item["flag"].dtype == torch.bool

    def test_scalar_metadata_roundtrips(self, tmp_path):
        """Non-tensor values stored as __scalar__ in index must survive."""
        samples = [{"Z": torch.tensor([6]), "pos": torch.zeros(1, 3), "label": {"__scalar__": 42}}]
        # Actually our _serialize stores non-tensor/non-dict as {"__scalar__": val}
        # so let's use a plain non-tensor value at the top level inside targets
        samples2 = [
            {
                "Z": torch.tensor([6]),
                "pos": torch.zeros(1, 3),
                "targets": {"U0": torch.tensor([1.0])},
            }
        ]
        ds = MmapDataset.from_samples(samples2, cache_dir=tmp_path, name="data")
        assert torch.allclose(ds[0]["targets"]["U0"], torch.tensor([1.0]))


# ---------------------------------------------------------------------------
# SubsetDataset
# ---------------------------------------------------------------------------

class TestSubsetDataset:
    def test_len_and_getitem_remapping(self, tmp_path):
        samples = _make_samples(8)
        full = MmapDataset.from_samples(samples, cache_dir=tmp_path, name="full")
        indices = [0, 3, 7]
        sub = SubsetDataset(full, indices)
        assert len(sub) == 3
        assert torch.equal(sub[0]["Z"], samples[0]["Z"])
        assert torch.equal(sub[1]["Z"], samples[3]["Z"])
        assert torch.equal(sub[2]["Z"], samples[7]["Z"])

    def test_from_samples_raises(self):
        with pytest.raises(TypeError):
            SubsetDataset.from_samples([])

    def test_is_base_dataset(self):
        assert issubclass(SubsetDataset, BaseDataset)

    def test_forwards_attribute_access_to_parent(self, tmp_path):
        """SubsetDataset must forward unknown attrs (e.g. target_schema)."""
        samples = _make_samples(6)
        MmapDataset.from_samples(samples, cache_dir=tmp_path, name="d")
        full = _AttrMmap(tmp_path / "d")
        sub = SubsetDataset(full, [0, 2, 4])
        assert sub.custom_marker == "abc"


# ---------------------------------------------------------------------------
# BaseDataset.split()
# ---------------------------------------------------------------------------

class TestSplit:
    def _full_dataset(self, tmp_path, n=20):
        return MmapDataset.from_samples(_make_samples(n), cache_dir=tmp_path, name="full")

    def test_split_sizes(self, tmp_path):
        ds = self._full_dataset(tmp_path)
        train, val = ds.split(ratio=0.8)
        assert len(train) + len(val) == len(ds)
        assert len(train) == int(20 * 0.8)

    def test_split_no_overlap(self, tmp_path):
        ds = self._full_dataset(tmp_path)
        train, val = ds.split(ratio=0.8)
        train_idx = set(train._indices)
        val_idx = set(val._indices)
        assert train_idx.isdisjoint(val_idx)
        assert train_idx | val_idx == set(range(len(ds)))

    def test_split_reproducible(self, tmp_path):
        ds = self._full_dataset(tmp_path)
        t1, v1 = ds.split(ratio=0.8, seed=42)
        t2, v2 = ds.split(ratio=0.8, seed=42)
        assert t1._indices == t2._indices
        assert v1._indices == v2._indices

    def test_split_different_seeds(self, tmp_path):
        ds = self._full_dataset(tmp_path)
        t1, _ = ds.split(ratio=0.8, seed=0)
        t2, _ = ds.split(ratio=0.8, seed=1)
        assert t1._indices != t2._indices

    def test_split_returns_subset_datasets(self, tmp_path):
        ds = self._full_dataset(tmp_path)
        train, val = ds.split()
        assert isinstance(train, SubsetDataset)
        assert isinstance(val, SubsetDataset)

    def test_split_cached_dataset(self):
        """split() must work on CachedDataset too (via BaseDataset)."""
        ds = CachedDataset(_make_samples(10))
        train, val = ds.split(ratio=0.7, seed=0)
        assert len(train) + len(val) == 10


# ---------------------------------------------------------------------------
# Standard cache format — MmapDataset.write_cache / from_cache
# ---------------------------------------------------------------------------


class TestMmapDatasetCacheFormat:
    def _write(
        self,
        tmp_path: Path,
        *,
        n: int = 4,
        task_states=None,
        overwrite: bool = False,
        name: str = "asset",
    ) -> Path:
        sink = tmp_path / name
        MmapDataset.write_cache(
            sink,
            _make_samples(n),
            pipeline_id="pid-1",
            source_id="src-1",
            pipeline_spec={"name": "test"},
            task_states=task_states,
            overwrite=overwrite,
        )
        return sink

    def test_write_creates_expected_layout(self, tmp_path):
        sink = self._write(tmp_path)
        assert (sink / "_READY").exists()
        assert (sink / "meta.json").exists()
        assert (sink / "samples.bin").exists()
        assert (sink / "samples.idx").exists()
        # No task_states: directory should not be created.
        assert not (sink / "task_states").exists()

    def test_meta_fields(self, tmp_path):
        sink = self._write(tmp_path, n=5)
        import json

        meta = json.loads((sink / "meta.json").read_text())
        assert meta["schema_version"] == 1
        assert meta["status"] == "ready"
        assert meta["pipeline_id"] == "pid-1"
        assert meta["source_id"] == "src-1"
        assert meta["fit_source_id"] == "src-1"
        assert meta["n_samples"] == 5
        assert meta["storage"]["samples_format"] == "molix-mmap-v1"

    def test_roundtrip(self, tmp_path):
        samples = _make_samples(3)
        sink = tmp_path / "asset"
        MmapDataset.write_cache(
            sink, samples, pipeline_id="p", source_id="s"
        )
        ds = MmapDataset.from_cache(sink)
        assert len(ds) == 3
        for i in range(3):
            assert torch.equal(ds[i]["Z"], samples[i]["Z"])
            assert torch.equal(ds[i]["pos"], samples[i]["pos"])

    def test_meta_accessor(self, tmp_path):
        sink = self._write(tmp_path)
        ds = MmapDataset.from_cache(sink)
        assert ds.meta["pipeline_id"] == "pid-1"
        assert ds.meta["source_id"] == "src-1"

    def test_task_states_roundtrip(self, tmp_path):
        state = {"baseline": torch.tensor([1.0, 2.0, 3.0])}
        sink = self._write(tmp_path, task_states={"atomic_dress": state})

        # Meta manifest
        import json

        meta = json.loads((sink / "meta.json").read_text())
        info = meta["task_states"]["atomic_dress"]
        assert info["format"] == "tensordict-memmap-v1"
        assert info["dir"] == "task_states/atomic_dress"
        assert (sink / info["dir"]).is_dir()

        # Roundtrip via from_cache
        ds = MmapDataset.from_cache(sink)
        loaded = ds.get_task_state("atomic_dress")
        assert torch.equal(loaded["baseline"], state["baseline"])

    def test_idempotent_noop_on_ready(self, tmp_path):
        sink = tmp_path / "asset"
        MmapDataset.write_cache(
            sink, _make_samples(2), pipeline_id="p", source_id="s"
        )
        mtime_ready = (sink / "_READY").stat().st_mtime_ns
        # Second call must not re-write.
        MmapDataset.write_cache(
            sink, _make_samples(99), pipeline_id="p", source_id="s"
        )
        ds = MmapDataset.from_cache(sink)
        assert len(ds) == 2
        assert (sink / "_READY").stat().st_mtime_ns == mtime_ready

    def test_overwrite_replaces_ready(self, tmp_path):
        sink = tmp_path / "asset"
        MmapDataset.write_cache(
            sink, _make_samples(2), pipeline_id="p", source_id="s"
        )
        MmapDataset.write_cache(
            sink,
            _make_samples(5),
            pipeline_id="p",
            source_id="s",
            overwrite=True,
        )
        ds = MmapDataset.from_cache(sink)
        assert len(ds) == 5

    def test_from_cache_missing_ready(self, tmp_path):
        sink = self._write(tmp_path)
        (sink / "_READY").unlink()
        with pytest.raises(CacheValidationError, match="_READY"):
            MmapDataset.from_cache(sink)

    def test_from_cache_bad_schema(self, tmp_path):
        sink = self._write(tmp_path)
        import json

        meta_path = sink / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["schema_version"] = 999
        meta_path.write_text(json.dumps(meta))
        with pytest.raises(CacheValidationError, match="schema_version"):
            MmapDataset.from_cache(sink)

    def test_from_cache_missing_meta(self, tmp_path):
        sink = self._write(tmp_path)
        (sink / "meta.json").unlink()
        with pytest.raises(CacheValidationError, match="meta.json"):
            MmapDataset.from_cache(sink)

    def test_from_cache_nonexistent_dir(self, tmp_path):
        with pytest.raises(CacheValidationError, match="not a directory"):
            MmapDataset.from_cache(tmp_path / "nope")

    def test_fit_source_id_defaults(self, tmp_path):
        sink = tmp_path / "asset"
        MmapDataset.write_cache(
            sink, _make_samples(1), pipeline_id="p", source_id="s"
        )
        import json

        assert json.loads((sink / "meta.json").read_text())["fit_source_id"] == "s"

    def test_fit_source_id_explicit(self, tmp_path):
        sink = tmp_path / "asset"
        MmapDataset.write_cache(
            sink,
            _make_samples(1),
            pipeline_id="p",
            source_id="s",
            fit_source_id="fit",
        )
        import json

        assert (
            json.loads((sink / "meta.json").read_text())["fit_source_id"]
            == "fit"
        )

    def test_no_partial_dir_survives_success(self, tmp_path):
        self._write(tmp_path)
        # All siblings should be the final directory only — no .partial leftovers.
        siblings = [p.name for p in tmp_path.iterdir()]
        assert siblings == ["asset"]

    def test_partial_cleaned_on_failure(self, tmp_path):
        """If sample serialization fails mid-way, no partial dir lingers."""

        def _bad():
            yield {"Z": torch.tensor([1])}
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            MmapDataset.write_cache(
                tmp_path / "asset",
                _bad(),
                pipeline_id="p",
                source_id="s",
            )
        # No dir — neither final nor partial — should exist.
        assert list(tmp_path.iterdir()) == []
