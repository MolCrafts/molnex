"""Tests for DataModule."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from molix.batch.datamodule import DataModule
from molix.store.cache import PackedCache
from molix.store.dataset import CachedDataset, MmapDataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


#: Atom-pair separations (A) alternated by :func:`_make_samples`. The
#: pipeline tests below build neighbor lists at ``cutoff=3.0``, so even
#: samples are in-cutoff (one bidirectional pair, 2 edges) and odd samples
#: are out-of-cutoff (0 edges). The jitter never spans the cutoff.
_IN_CUTOFF_SEP = 1.5
_OUT_OF_CUTOFF_SEP = 5.0
_JITTER = 0.01


def _make_samples(n: int = 10) -> list[dict]:
    """Build *n* two-atom samples with deterministic geometry.

    Positions come from a local :class:`torch.Generator` (seed 0), so the
    fixture is independent of global RNG state and of test execution
    order. Separations alternate across the ``3.0`` cutoff used by the
    pipeline tests, which keeps the per-sample edge count *varying* (2 and
    0) instead of landing on a constant ``E == n_atoms`` — the degenerate
    geometry pinned in ``test_cache.py::TestPackSchema``.

    Args:
        n: Number of samples. ``targets.U0`` carries the identity
            ``float(i)`` used by the shuffle-order tests.

    Returns:
        Flat sample dicts with per-atom ``Z`` / ``pos``, one pre-built
        edge (``edge_index`` ``(1, 2)``, ``edge_diff`` ``(1, 3)``,
        ``edge_dist`` ``(1,)``) and graph-level ``targets.U0`` ``(1,)``.
    """
    gen = torch.Generator().manual_seed(0)
    samples: list[dict] = []
    for i in range(n):
        sep = _IN_CUTOFF_SEP if i % 2 == 0 else _OUT_OF_CUTOFF_SEP
        pos = torch.tensor([[0.0, 0.0, 0.0], [sep, 0.0, 0.0]])
        pos = pos + _JITTER * torch.randn(2, 3, generator=gen)
        # edge_index [[0, 1]] → source 0, target 1; edge_diff follows the
        # repo convention pos[target] - pos[source].
        diff = (pos[1] - pos[0]).unsqueeze(0)
        samples.append(
            {
                "Z": torch.tensor([1, 6], dtype=torch.long),
                "pos": pos,
                "edge_index": torch.tensor([[0, 1]], dtype=torch.long),
                "edge_diff": diff,
                "edge_dist": diff.norm(dim=-1),
                "targets": {"U0": torch.tensor([float(i)])},
            }
        )
    return samples


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
        from molix.batch.collate import TargetSchema

        train, val = _write_and_load_split(tmp_path, 4, 2)
        schema = TargetSchema(graph_level={"U0"}, atom_level=set())
        dm = DataModule(
            train, val, target_schema=schema, batch_size=2, num_workers=0, pin_memory=False
        )
        batch = next(iter(dm.train_dataloader()))
        assert "U0" in batch["graphs"].keys()

    def test_target_schema_auto_discovered_from_dataset(self, tmp_path):
        from molix.batch.collate import TargetSchema

        class SchemaCarrier(CachedDataset):
            target_schema = TargetSchema(graph_level={"U0"}, atom_level=frozenset())

        samples = _make_samples(6)
        tsink, vsink = tmp_path / "t.pt", tmp_path / "v.pt"
        PackedCache(tsink).save(samples[:4])
        PackedCache(vsink).save(samples[4:])
        dm = DataModule(SchemaCarrier(tsink), SchemaCarrier(vsink), batch_size=2, pin_memory=False)
        assert dm.target_schema.graph_level == frozenset({"U0"})

    def test_explicit_target_schema_overrides_dataset(self, tmp_path):
        from molix.batch.collate import TargetSchema

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

        samples = _make_samples(10)
        sink = tmp_path / "smoke.pt"
        PackedCache(sink).save(samples)
        full = MmapDataset(sink)
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
        samples = _make_samples(8)
        sink = tmp_path / "two-way.pt"
        PackedCache(sink).save(samples)
        full = MmapDataset(sink)
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


# ---------------------------------------------------------------------------
# Batch-sampler injection seam (spec universal-class-i-09-packer)
# ---------------------------------------------------------------------------

#: Fixed plan handed to the train loader by :class:`_FixedPlanSource`.
#: Covers the 8-sample train split exactly once, in a non-sorted order so
#: the yielded ``U0`` identities cannot match by accident.
INJECTED_PLAN = [[3, 0], [7], [1, 5, 2], [6, 4]]
#: ``U0`` identities the injected plan must produce (``U0 == float(index)``).
INJECTED_PLAN_IDENTITIES = [[3.0, 0.0], [7.0], [1.0, 5.0, 2.0], [6.0, 4.0]]


class _FixedPlanSource:
    """Picklable ``BatchSamplerSource`` stub returning one fixed plan.

    Records every ``seed`` it is handed so the ``seed + epoch`` reseed
    convention can be asserted, and returns the *same* list object every
    call so ``loader.batch_sampler is source.plan`` is checkable.
    """

    def __init__(self, plan: list[list[int]]) -> None:
        self.plan = [list(group) for group in plan]
        self.seen_seeds: list[int] = []

    def batch_sampler(self, dataset: object, *, seed: int) -> list[list[int]]:
        """Record *seed* and return the fixed index plan."""
        self.seen_seeds.append(seed)
        return self.plan


class _NoBatchSamplerSource:
    """Object that does *not* satisfy the ``BatchSamplerSource`` protocol."""


def _make_injection_dm(tmp_path: Path, tag: str, *, seed: int = 42, **dm_kwargs) -> DataModule:
    """Build a DataModule over an 8-sample train split with ``U0`` identities.

    Args:
        tmp_path: Pytest temp directory for the packed caches.
        tag: Unique filename tag so multiple modules can share ``tmp_path``.
        seed: DataModule shuffle seed.
        **dm_kwargs: Extra DataModule kwargs (e.g. ``batch_sampler_source``).

    Returns:
        A non-DDP DataModule whose train samples carry ``U0 == float(index)``.
    """
    samples = _make_samples(10)
    train_sink = tmp_path / f"inject_train_{tag}.pt"
    val_sink = tmp_path / f"inject_val_{tag}.pt"
    PackedCache(train_sink).save(samples[:8])
    PackedCache(val_sink).save(samples[8:])
    dm_kwargs.setdefault("batch_size", 2)
    dm_kwargs.setdefault("num_workers", 0)
    dm_kwargs.setdefault("pin_memory", False)
    return DataModule(CachedDataset(train_sink), CachedDataset(val_sink), seed=seed, **dm_kwargs)


class TestBatchSamplerInjection:
    """``batch_sampler_source`` drives the train loader's batch composition.

    One symbol (BatchSamplerSource) is imported locally in the protocol-conformance test.
    """

    def test_source_conforms_to_protocol(self):
        """The stub satisfies the runtime-checkable ``BatchSamplerSource``."""
        from molix.batch import BatchSamplerSource

        assert isinstance(_FixedPlanSource(INJECTED_PLAN), BatchSamplerSource)

    def test_loader_batch_sampler_is_the_injected_plan(self, tmp_path):
        """The object returned by the source is handed straight to DataLoader."""
        source = _FixedPlanSource(INJECTED_PLAN)
        dm = _make_injection_dm(tmp_path, "plan", batch_sampler_source=source)
        loader = dm.train_dataloader()
        assert loader.batch_sampler is source.plan

    def test_loader_batch_size_is_none(self, tmp_path):
        """DataLoader normalises ``batch_size`` to None on the injected path."""
        dm = _make_injection_dm(
            tmp_path, "bs", batch_sampler_source=_FixedPlanSource(INJECTED_PLAN)
        )
        assert dm.train_dataloader().batch_size is None

    def test_iteration_yields_exactly_the_injected_groups(self, tmp_path):
        """Collated batches carry the injected index groups (via ``U0`` identity)."""
        dm = _make_injection_dm(
            tmp_path, "iter", batch_sampler_source=_FixedPlanSource(INJECTED_PLAN)
        )
        yielded = [[float(u) for u in batch["graphs", "U0"]] for batch in dm.train_dataloader()]
        assert yielded == INJECTED_PLAN_IDENTITIES

    def test_source_receives_seed_plus_epoch(self, tmp_path):
        """``on_epoch_start(k)`` then ``train_dataloader()`` passes ``seed + k``."""
        source = _FixedPlanSource(INJECTED_PLAN)
        dm = _make_injection_dm(tmp_path, "seed", seed=7, batch_sampler_source=source)
        dm.on_epoch_start(3)
        dm.train_dataloader()
        assert source.seen_seeds[-1] == dm.seed + 3

    def test_injected_source_under_ddp_raises(self, tmp_path, monkeypatch):
        """DDP + injected source → ValueError naming the offending kwarg."""
        monkeypatch.setattr("molix.batch.datamodule._is_distributed", lambda: True)
        dm = _make_injection_dm(
            tmp_path, "ddp", batch_sampler_source=_FixedPlanSource(INJECTED_PLAN)
        )
        with pytest.raises(ValueError, match="(?i)ddp|distributed") as excinfo:
            dm.train_dataloader()
        assert "batch_sampler_source" in str(excinfo.value)

    def test_source_with_token_budget_raises_eagerly(self, tmp_path):
        """Two competing batch policies are rejected at ``__init__``."""
        with pytest.raises(ValueError, match="batch_sampler_source"):
            _make_injection_dm(
                tmp_path,
                "clash",
                batch_sampler_source=_FixedPlanSource(INJECTED_PLAN),
                max_atoms_per_batch=16,
            )

    def test_non_conforming_source_raises_eagerly(self, tmp_path):
        """An object without ``batch_sampler`` is rejected at ``__init__``."""
        with pytest.raises(ValueError, match="batch_sampler"):
            _make_injection_dm(tmp_path, "bad", batch_sampler_source=_NoBatchSamplerSource())


# ---------------------------------------------------------------------------
# Worker start-method regression (preserved from the deleted
# tests/test_molix/test_data/test_e2e_workers.py, rebuilt on PackedCache)
# ---------------------------------------------------------------------------


def _chain_samples(n: int = 16) -> list[dict]:
    """Build *n* four-atom samples wired as a linear chain (3 edges each).

    Geometry comes from a local :class:`torch.Generator` (seed 1), so the
    fixture is independent of global RNG state and of test execution order.
    The per-sample edge count (3) never equals the atom count (4), so the
    packed schema routes per-atom and per-edge keys unambiguously.

    Args:
        n: Number of samples. ``targets.U0`` carries the identity
            ``float(i)``.

    Returns:
        Flat sample dicts with per-atom ``Z`` ``(4,)`` / ``pos`` ``(4, 3)``,
        per-edge ``edge_index`` ``(3, 2)`` / ``edge_diff`` ``(3, 3)`` /
        ``edge_dist`` ``(3,)`` and graph-level ``targets.U0`` ``(1,)``.
    """
    gen = torch.Generator().manual_seed(1)
    src = torch.arange(3)
    dst = src + 1
    edge_index = torch.stack([src, dst], dim=1).long()
    samples: list[dict] = []
    for i in range(n):
        pos = torch.randn(4, 3, generator=gen)
        diff = pos[dst] - pos[src]
        samples.append(
            {
                "Z": torch.tensor([1, 6, 1, 1], dtype=torch.long),
                "pos": pos,
                "edge_index": edge_index.clone(),
                "edge_diff": diff,
                "edge_dist": diff.norm(dim=-1),
                "targets": {"U0": torch.tensor([float(i)])},
            }
        )
    return samples


class TestDataModule:
    """Contracts the DataLoader worker path depends on.

    ``spawn`` / ``forkserver`` send every DataLoader argument to workers
    through ``pickle``; a collate callable that is a local closure — or one
    holding an unpicklable batch task — breaks a real training run at epoch
    start. These two methods pin exactly that.
    """

    def test_collate_fn_is_picklable(self, _dm_factory):
        """The collate callable — batch tasks included — survives ``pickle``."""
        import pickle

        from molix.batch import PadMolecularBatch

        dm = _dm_factory(
            batch_size=2,
            batch_tasks=[PadMolecularBatch(max_atoms=8, max_edges=8)],
        )
        fn = dm._make_collate_fn()
        fn2 = pickle.loads(pickle.dumps(fn))  # must not raise

        samples = _make_samples(2)
        out1 = fn(samples)
        out2 = fn2(samples)
        # The task travelled with the callable: both sides pad identically.
        assert out1["atoms", "Z"].shape == (8,)
        assert torch.equal(out1["atoms", "Z"], out2["atoms", "Z"])

    def test_train_dataloader_iterates_with_spawn_workers(self, tmp_path):
        """One epoch through ``num_workers=2`` spawn workers yields real batches."""
        from tensordict import TensorDict

        from molix.batch.collate import TargetSchema

        sink = tmp_path / "workers.pt"
        PackedCache(sink).save(_chain_samples(16))
        full = MmapDataset(sink)
        train, val = full.split(ratio=0.75, seed=0)
        assert len(train) == 12 and len(val) == 4

        schema = TargetSchema(graph_level=frozenset({"U0"}), atom_level=frozenset())
        dm = DataModule(
            train,
            val,
            target_schema=schema,
            batch_size=4,
            num_workers=2,
            persistent_workers=False,
            pin_memory=False,  # avoid CUDA on CI
            prefetch_factor=2,
        )

        seen_batches = 0
        for batch in dm.train_dataloader():
            assert isinstance(batch, TensorDict)
            assert batch["atoms", "Z"].shape[0] == 4 * 4  # 4 mols × 4 atoms each
            assert batch["graphs", "U0"].shape == (4,)
            assert batch["edges", "edge_index"].shape[1] == 2
            seen_batches += 1
        assert seen_batches == 3  # 12 / 4
