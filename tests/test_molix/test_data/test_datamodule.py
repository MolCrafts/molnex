"""Tests for DataModule."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from molix.data.collate import DEFAULT_TARGET_SCHEMA
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


def _make_dm(**kwargs) -> DataModule:
    samples = _make_samples(10)
    train_ds = CachedDataset(samples[:8])
    val_ds = CachedDataset(samples[8:])
    kwargs.setdefault("pin_memory", False)  # no GPU required in unit tests
    return DataModule(train_ds, val_ds, **kwargs)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestDataModuleConstruction:
    def test_stores_datasets(self):
        samples = _make_samples(6)
        train = CachedDataset(samples[:4])
        val = CachedDataset(samples[4:])
        dm = DataModule(train, val, batch_size=2)
        assert dm.train_dataset is train
        assert dm.val_dataset is val
        assert dm.batch_size == 2

    def test_setup_is_noop(self):
        dm = _make_dm()
        dm.setup("fit")   # must not raise
        dm.setup("test")  # must not raise

    def test_persistent_workers_requires_num_workers(self):
        dm = DataModule(
            CachedDataset(_make_samples(4)),
            CachedDataset(_make_samples(2)),
            num_workers=0,
            persistent_workers=True,
        )
        assert dm.persistent_workers is False


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

class TestDataLoaders:
    def test_train_dataloader_returns_dataloader(self):
        dm = _make_dm(batch_size=4)
        dl = dm.train_dataloader()
        assert isinstance(dl, DataLoader)

    def test_val_dataloader_returns_dataloader(self):
        dm = _make_dm(batch_size=4)
        dl = dm.val_dataloader()
        assert isinstance(dl, DataLoader)

    def test_train_dataloader_batch_size(self):
        dm = _make_dm(batch_size=4)
        dl = dm.train_dataloader()
        batch = next(iter(dl))
        # GraphBatch or dict — atoms should have 4*2=8 rows
        assert batch["atoms", "Z"].shape[0] == 8

    def test_val_dataloader_no_shuffle(self):
        dm = _make_dm(batch_size=2)
        dl = dm.val_dataloader()
        assert dl.sampler is None or not getattr(dl.sampler, "shuffle", False)

    def test_multiple_epochs_consistent(self):
        """Calling train_dataloader() twice must not raise."""
        dm = _make_dm(batch_size=4)
        dl1 = dm.train_dataloader()
        dl2 = dm.train_dataloader()
        assert isinstance(dl1, DataLoader)
        assert isinstance(dl2, DataLoader)


# ---------------------------------------------------------------------------
# Collation integration
# ---------------------------------------------------------------------------

class TestCollation:
    def test_batch_has_graph_batch_structure(self):
        from molix.data.types import GraphBatch
        dm = _make_dm(batch_size=4)
        batch = next(iter(dm.train_dataloader()))
        assert isinstance(batch, GraphBatch)

    def test_targets_in_graphs(self):
        dm = _make_dm(batch_size=4)
        batch = next(iter(dm.train_dataloader()))
        assert "U0" in batch["graphs"].keys()
        assert batch["graphs", "U0"].shape == (4,)

    def test_custom_target_schema(self):
        from molix.data.collate import TargetSchema
        schema = TargetSchema(graph_level={"U0"}, atom_level=set())
        dm = DataModule(
            CachedDataset(_make_samples(6)[:4]),
            CachedDataset(_make_samples(6)[4:]),
            target_schema=schema,
            batch_size=2,
            pin_memory=False,
        )
        batch = next(iter(dm.train_dataloader()))
        assert "U0" in batch["graphs"].keys()

    def test_target_schema_auto_discovered_from_dataset(self):
        """If a dataset declares `target_schema`, DataModule must pick it up."""
        from molix.data.collate import TargetSchema

        class SchemaCarrier(CachedDataset):
            target_schema = TargetSchema(graph_level={"U0"}, atom_level=frozenset())

        train = SchemaCarrier(_make_samples(6)[:4])
        val = SchemaCarrier(_make_samples(6)[4:])
        dm = DataModule(train, val, batch_size=2, pin_memory=False)
        # Auto-picked from dataset, not the global default.
        assert dm.target_schema.graph_level == frozenset({"U0"})

    def test_explicit_target_schema_overrides_dataset(self):
        from molix.data.collate import TargetSchema

        class SchemaCarrier(CachedDataset):
            target_schema = TargetSchema(graph_level={"foo"}, atom_level=frozenset())

        explicit = TargetSchema(graph_level={"U0"}, atom_level=frozenset())
        dm = DataModule(
            SchemaCarrier(_make_samples(6)[:4]),
            SchemaCarrier(_make_samples(6)[4:]),
            target_schema=explicit,
            batch_size=2,
            pin_memory=False,
        )
        assert dm.target_schema is explicit


# ---------------------------------------------------------------------------
# Pickling / forkserver compatibility (Python 3.14 default)
# ---------------------------------------------------------------------------


class TestPickling:
    def test_collate_fn_is_picklable(self):
        """forkserver workers require collate_fn to round-trip through pickle."""
        import pickle

        dm = _make_dm(batch_size=2)
        fn = dm._make_collate_fn()
        fn2 = pickle.loads(pickle.dumps(fn))
        # Same logical behavior on a tiny input
        samples = _make_samples(2)
        out1 = fn(samples)
        out2 = fn2(samples)
        assert torch.equal(out1["atoms", "Z"], out2["atoms", "Z"])


# ---------------------------------------------------------------------------
# Epoch hook
# ---------------------------------------------------------------------------

class TestEpochHook:
    def test_on_epoch_start_no_crash_without_ddp(self):
        dm = _make_dm()
        dm.train_dataloader()
        dm.on_epoch_start(0)
        dm.on_epoch_start(1)
