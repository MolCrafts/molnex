"""Tests for DataModule."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from molix.data.cache import PackedCache
from molix.data.datamodule import DataModule
from molix.data.dataset import CachedDataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_samples(n: int = 10) -> list[dict]:
    return [
        {
            "Z": torch.tensor([1, 6], dtype=torch.long),
            "pos": torch.randn(2, 3),
            "edge_index": torch.tensor([[0, 1]], dtype=torch.long),
            "bond_diff": torch.randn(1, 3),
            "bond_dist": torch.tensor([1.5]),
            "targets": {"U0": torch.tensor([float(i)])},
        }
        for i in range(n)
    ]


def _write_and_load_split(tmp_path: Path, train_n: int, val_n: int, dataset_cls=CachedDataset):
    samples = _make_samples(train_n + val_n)
    train_sink = tmp_path / "train.pt"
    val_sink = tmp_path / "val.pt"
    PackedCache(train_sink).save(samples[:train_n])
    PackedCache(val_sink).save(samples[train_n:])
    return dataset_cls(train_sink), dataset_cls(val_sink)


@pytest.fixture
def _dm_factory(tmp_path):
    def _make(**kwargs) -> DataModule:
        train, val = _write_and_load_split(tmp_path, 8, 2)
        # Tiny test datasets — pickling/spawning overhead outweighs parallelism.
        # The user-facing DataModule default is num_workers=4; tests pin 0.
        kwargs.setdefault("pin_memory", False)
        kwargs.setdefault("num_workers", 0)
        return DataModule(train, val, **kwargs)

    return _make


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestDataModuleConstruction:
    def test_stores_datasets(self, tmp_path):
        train, val = _write_and_load_split(tmp_path, 4, 2)
        dm = DataModule(train, val, batch_size=2, pin_memory=False)
        assert dm.train_dataset is train
        assert dm.val_dataset is val
        assert dm.batch_size == 2

    def test_setup_is_noop(self, _dm_factory):
        dm = _dm_factory()
        dm.setup("fit")
        dm.setup("test")

    def test_persistent_workers_requires_num_workers(self, tmp_path):
        train, val = _write_and_load_split(tmp_path, 4, 2)
        dm = DataModule(
            train,
            val,
            num_workers=0,
            persistent_workers=True,
            pin_memory=False,
        )
        assert dm.persistent_workers is False


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------


class TestDataLoaders:
    def test_train_dataloader_returns_dataloader(self, _dm_factory):
        dm = _dm_factory(batch_size=4)
        assert isinstance(dm.train_dataloader(), DataLoader)

    def test_val_dataloader_returns_dataloader(self, _dm_factory):
        dm = _dm_factory(batch_size=4)
        assert isinstance(dm.val_dataloader(), DataLoader)

    def test_train_dataloader_batch_size(self, _dm_factory):
        dm = _dm_factory(batch_size=4)
        dl = dm.train_dataloader()
        batch = next(iter(dl))
        assert batch["atoms", "Z"].shape[0] == 8  # 4 mols × 2 atoms

    def test_val_dataloader_no_shuffle(self, _dm_factory):
        dm = _dm_factory(batch_size=2)
        dl = dm.val_dataloader()
        assert dl.sampler is None or not getattr(dl.sampler, "shuffle", False)

    def test_multiple_epochs_consistent(self, _dm_factory):
        dm = _dm_factory(batch_size=4)
        dl1 = dm.train_dataloader()
        dl2 = dm.train_dataloader()
        assert isinstance(dl1, DataLoader)
        assert isinstance(dl2, DataLoader)


# ---------------------------------------------------------------------------
# Collation integration
# ---------------------------------------------------------------------------


class TestCollation:
    def test_batch_has_graph_batch_structure(self, _dm_factory):
        from tensordict import TensorDict

        dm = _dm_factory(batch_size=4)
        batch = next(iter(dm.train_dataloader()))
        assert isinstance(batch, TensorDict)

    def test_targets_in_graphs(self, _dm_factory):
        dm = _dm_factory(batch_size=4)
        batch = next(iter(dm.train_dataloader()))
        assert "U0" in batch["graphs"].keys()
        assert batch["graphs", "U0"].shape == (4,)

    def test_custom_target_schema(self, tmp_path):
        from molix.data.collate import TargetSchema

        train, val = _write_and_load_split(tmp_path, 4, 2)
        schema = TargetSchema(graph_level={"U0"}, atom_level=set())
        dm = DataModule(
            train, val, target_schema=schema, batch_size=2, num_workers=0, pin_memory=False
        )
        batch = next(iter(dm.train_dataloader()))
        assert "U0" in batch["graphs"].keys()

    def test_target_schema_auto_discovered_from_dataset(self, tmp_path):
        from molix.data.collate import TargetSchema

        class SchemaCarrier(CachedDataset):
            target_schema = TargetSchema(graph_level={"U0"}, atom_level=frozenset())

        samples = _make_samples(6)
        tsink, vsink = tmp_path / "t.pt", tmp_path / "v.pt"
        PackedCache(tsink).save(samples[:4])
        PackedCache(vsink).save(samples[4:])
        dm = DataModule(SchemaCarrier(tsink), SchemaCarrier(vsink), batch_size=2, pin_memory=False)
        assert dm.target_schema.graph_level == frozenset({"U0"})

    def test_explicit_target_schema_overrides_dataset(self, tmp_path):
        from molix.data.collate import TargetSchema

        class SchemaCarrier(CachedDataset):
            target_schema = TargetSchema(graph_level={"foo"}, atom_level=frozenset())

        samples = _make_samples(6)
        tsink, vsink = tmp_path / "t.pt", tmp_path / "v.pt"
        PackedCache(tsink).save(samples[:4])
        PackedCache(vsink).save(samples[4:])
        explicit = TargetSchema(graph_level={"U0"}, atom_level=frozenset())
        dm = DataModule(
            SchemaCarrier(tsink),
            SchemaCarrier(vsink),
            target_schema=explicit,
            batch_size=2,
            pin_memory=False,
        )
        assert dm.target_schema is explicit


# ---------------------------------------------------------------------------
# Pickling / forkserver compatibility (Python 3.14 default)
# ---------------------------------------------------------------------------


class TestPickling:
    def test_collate_fn_is_picklable(self, _dm_factory):
        import pickle

        dm = _dm_factory(batch_size=2)
        fn = dm._make_collate_fn()
        fn2 = pickle.loads(pickle.dumps(fn))
        samples = _make_samples(2)
        out1 = fn(samples)
        out2 = fn2(samples)
        assert torch.equal(out1["atoms", "Z"], out2["atoms", "Z"])


# ---------------------------------------------------------------------------
# Worker start method (Python 3.14 multi-threaded-fork deprecation)
# ---------------------------------------------------------------------------


class TestWorkerContext:
    def test_default_is_spawn(self, _dm_factory):
        """Default sidesteps Python 3.14's multi-threaded forkserver warning."""
        dm = _dm_factory()
        assert dm.multiprocessing_context == "spawn"

    def test_context_forwarded_when_workers_enabled(self, _dm_factory):
        dm = _dm_factory(num_workers=2, persistent_workers=False)
        dl = dm.train_dataloader()
        # torch.DataLoader resolves the string to a BaseContext object.
        ctx = dl.multiprocessing_context
        assert ctx is not None
        # Accept either the raw string (some torch versions keep it) or a
        # resolved BaseContext (others resolve eagerly).
        name = ctx if isinstance(ctx, str) else ctx.get_start_method()
        assert name == "spawn"

    def test_context_ignored_when_num_workers_zero(self, _dm_factory):
        dm = _dm_factory(num_workers=0)
        dl = dm.train_dataloader()
        # DataLoader normalises an unused context to ``None`` in sync mode.
        assert dl.multiprocessing_context is None

    def test_explicit_override(self, tmp_path):
        train, val = _write_and_load_split(tmp_path, 4, 2)
        dm = DataModule(
            train,
            val,
            batch_size=2,
            num_workers=2,
            persistent_workers=False,
            pin_memory=False,
            multiprocessing_context="forkserver",
        )
        assert dm.multiprocessing_context == "forkserver"
        dl = dm.train_dataloader()
        ctx = dl.multiprocessing_context
        name = ctx if isinstance(ctx, str) else ctx.get_start_method()
        assert name == "forkserver"


# ---------------------------------------------------------------------------
# Epoch hook
# ---------------------------------------------------------------------------


class TestEpochHook:
    def test_on_epoch_start_no_crash_without_ddp(self, _dm_factory):
        dm = _dm_factory()
        dm.train_dataloader()
        dm.on_epoch_start(0)
        dm.on_epoch_start(1)


# ---------------------------------------------------------------------------
# Non-DDP epoch reshuffle (regression: identical permutation every epoch)
# ---------------------------------------------------------------------------


def _make_shuffle_dm(tmp_path: Path, tag: str, *, seed: int = 123, n_train: int = 64) -> DataModule:
    """Build a DataModule whose samples carry a unique ``U0`` identity.

    Args:
        tmp_path: Pytest temp directory for the packed caches.
        tag: Unique filename tag so multiple modules can share ``tmp_path``.
        seed: DataModule shuffle seed.
        n_train: Number of training samples (``U0`` runs ``0..n_train-1``).

    Returns:
        A non-DDP DataModule with ``num_workers=0``.
    """
    samples = _make_samples(n_train + 2)
    train_sink = tmp_path / f"shuffle_train_{tag}.pt"
    val_sink = tmp_path / f"shuffle_val_{tag}.pt"
    PackedCache(train_sink).save(samples[:n_train])
    PackedCache(val_sink).save(samples[n_train:])
    return DataModule(
        CachedDataset(train_sink),
        CachedDataset(val_sink),
        batch_size=8,
        num_workers=0,
        pin_memory=False,
        seed=seed,
    )


def _loader_order(dl: DataLoader) -> list[float]:
    """Iterate *dl* and return per-sample ``U0`` identities in yield order."""
    return [float(u) for batch in dl for u in batch["graphs", "U0"]]


class TestEpochReshuffle:
    """Non-DDP shuffle must reseed per epoch, mirroring ``DistributedSampler.set_epoch``.

    The Trainer rebuilds the train dataloader each epoch and calls
    ``on_epoch_start(epoch)`` first; these tests simulate exactly that
    sequence. Sample identity is recovered from the unique ``U0`` target.
    """

    def test_epoch_reshuffle_differs(self, tmp_path):
        """Two consecutive epochs must yield different sample permutations."""
        dm = _make_shuffle_dm(tmp_path, "reshuffle")

        dm.on_epoch_start(0)
        order_epoch0 = _loader_order(dm.train_dataloader())
        dm.on_epoch_start(1)
        order_epoch1 = _loader_order(dm.train_dataloader())

        # Same multiset of samples either way — only the order may change.
        assert sorted(order_epoch0) == sorted(order_epoch1)
        # 64! permutations: a seeded reshuffle colliding is astronomically
        # unlikely, so equality here means the epoch seed was not advanced.
        assert order_epoch0 != order_epoch1

    def test_epoch_shuffle_resume_derivable(self, tmp_path):
        """Resume at epoch k must reproduce epoch k's permutation by derivation.

        A fresh DataModule with the same seed, told via ``on_epoch_start(1)``
        that it is resuming at epoch 1, must yield exactly the epoch-1 order
        of a continuously-trained DataModule — and that order must differ
        from epoch 0 (otherwise "reproducing" it is vacuous).
        """
        dm_continuous = _make_shuffle_dm(tmp_path, "cont", seed=777)
        dm_continuous.on_epoch_start(0)
        order_epoch0 = _loader_order(dm_continuous.train_dataloader())
        dm_continuous.on_epoch_start(1)
        order_epoch1_continuous = _loader_order(dm_continuous.train_dataloader())

        dm_fresh = _make_shuffle_dm(tmp_path, "fresh", seed=777)
        dm_fresh.on_epoch_start(1)
        order_epoch1_fresh = _loader_order(dm_fresh.train_dataloader())

        # (a) Resume reproducibility: epoch-1 order is derivable from
        #     (seed, epoch) alone, independent of prior-epoch history.
        assert order_epoch1_fresh == order_epoch1_continuous
        # (b) Guard against the degenerate pre-fix world where every epoch
        #     shares one permutation, which would make (a) pass vacuously.
        assert order_epoch1_continuous != order_epoch0

    def test_same_epoch_same_order(self, tmp_path):
        """Rebuilding the dataloader within one epoch must not change the order."""
        dm = _make_shuffle_dm(tmp_path, "same")
        dm.on_epoch_start(0)

        order_first = _loader_order(dm.train_dataloader())
        order_second = _loader_order(dm.train_dataloader())

        assert order_first == order_second


# ---------------------------------------------------------------------------
# Explicit 3-step path: cache → dataset → DataModule
# (replaces the removed from_cached_pipeline)
# ---------------------------------------------------------------------------


class TestExplicitCachedPipeline:
    def test_end_to_end(self, tmp_path):
        """Explicit 3-step: cache, split, DataModule — yields TensorDict."""
        from tensordict import TensorDict

        from molix.data import InMemorySource, NeighborList, Pipeline

        samples = _make_samples(10)
        source = InMemorySource(samples)
        pipe = Pipeline("smoke").add(NeighborList(cutoff=3.0, max_num_pairs=32, pbc=False)).build()
        dag = pipe.cache(source, base_dir=tmp_path, extra={"n_train": "6"})
        full = dag.dataset(mmap=True)
        train_ds, val_ds, test_ds = full.split(sizes=(6, 2, 2), seed=7)

        dm = DataModule(train_ds, val_ds, batch_size=2, num_workers=0, pin_memory=False)

        assert len(dm.train_dataset) == 6
        assert len(dm.val_dataset) == 2
        assert len(train_ds) == 6
        assert len(test_ds) == 2

        # Stats
        assert full.avg_num_neighbors > 0.0
        assert full.max_atoms > 0
        assert train_ds.avg_num_neighbors > 0.0
        assert train_ds.max_atoms <= full.max_atoms

        # DataLoader produces TensorDict
        batch = next(iter(dm.train_dataloader()))
        assert isinstance(batch, TensorDict)
        assert batch["atoms", "Z"].shape[0] == 4  # 2 mols × 2 atoms

    def test_two_way_split(self, tmp_path):
        from molix.data import InMemorySource, NeighborList, Pipeline

        samples = _make_samples(8)
        pipe = (
            Pipeline("two-way").add(NeighborList(cutoff=3.0, max_num_pairs=32, pbc=False)).build()
        )
        dag = pipe.cache(InMemorySource(samples), base_dir=tmp_path)
        full = dag.dataset(mmap=True)
        train_ds, val_ds = full.split(sizes=(5, 3))

        dm = DataModule(train_ds, val_ds, batch_size=2, num_workers=0, pin_memory=False)
        assert len(dm.train_dataset) == 5
        assert len(dm.val_dataset) == 3

    def test_rejects_mismatched_split(self, tmp_path):
        sink = PackedCache(tmp_path / "x.pt")
        sink.save(_make_samples(5))
        full = CachedDataset(sink)
        with pytest.raises(ValueError, match="sizes must sum to"):
            full.split(sizes=(4, 3))  # sum=7 != 5
