"""Tests for DataModule."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from molix.data import FrameCollator, MoleculeCollator, MoleculePadder, TokenBudgetBatchSampler
from molix.data import datamodule as _datamodule
from molix.data.datamodule import DataModule
from molix.data.dataset import BaseDataset
from molix.frame import TensorFrame
from tests.conftest import frame

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


def _make_samples(n: int = 10) -> list:
    """Build *n* two-atom frames with deterministic geometry.

    Positions come from a local :class:`torch.Generator` (seed 0), so the
    fixture is independent of global RNG state and of test execution order.

    Args:
        n: Number of frames. The ``energy`` meta scalar carries the identity
            ``float(i)`` used by the shuffle-order tests.

    Returns:
        molpy ``Frame`` objects — the only thing a dataset hands the collate.
    """
    gen = torch.Generator().manual_seed(0)
    frames = []
    for i in range(n):
        sep = _IN_CUTOFF_SEP if i % 2 == 0 else _OUT_OF_CUTOFF_SEP
        pos = torch.tensor([[0.0, 0.0, 0.0], [sep, 0.0, 0.0]])
        pos = pos + _JITTER * torch.randn(2, 3, generator=gen)
        frames.append(frame(Z=[1, 6], pos=pos.tolist(), energy=float(i)))
    return frames


class _SampleDataset(BaseDataset):
    """List-backed dataset — the per-sample collate route's fixture.

    Blocks are molnex's one on-disk format and carry no neighbour graph, so
    edged samples like these come from an in-memory list rather than a store.
    """

    def __init__(self, samples: list[dict]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        return self._samples[idx]


def _write_and_load_split(tmp_path: Path, train_n: int, val_n: int, dataset_cls=None):
    cls = dataset_cls or _SampleDataset
    samples = _make_samples(train_n + val_n)
    return cls(samples[:train_n]), cls(samples[train_n:])


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

        dm = _dm_factory(batch_size=4)
        batch = next(iter(dm.train_dataloader()))
        assert isinstance(batch, TensorFrame)

    def test_targets_in_graphs(self, _dm_factory):
        dm = _dm_factory(batch_size=4)
        batch = next(iter(dm.train_dataloader()))
        assert "energy" in batch["graphs"].keys()
        assert batch["graphs", "energy"].shape == (4,)


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
        tag: Unique filename tag so multiple modules can share ``tmp_path``.
        seed: DataModule shuffle seed.
        n_train: Number of training samples (``U0`` runs ``0..n_train-1``).

    Returns:
        A non-DDP DataModule with ``num_workers=0``.
    """
    samples = _make_samples(n_train + 2)
    return DataModule(
        _SampleDataset(samples[:n_train]),
        _SampleDataset(samples[n_train:]),
        batch_size=8,
        num_workers=0,
        pin_memory=False,
        seed=seed,
    )


def _loader_order(dl: DataLoader) -> list[float]:
    """Iterate *dl* and return per-sample ``U0`` identities in yield order."""
    return [float(u) for batch in dl for u in batch["graphs", "energy"]]


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

        full = _SampleDataset(_make_samples(10))
        train_ds, val_ds, test_ds = full.split(sizes=(6, 2, 2), seed=7)

        dm = DataModule(train_ds, val_ds, batch_size=2, num_workers=0, pin_memory=False)

        assert len(dm.train_dataset) == 6
        assert len(dm.val_dataset) == 2
        assert len(train_ds) == 6
        assert len(test_ds) == 2

        # DataLoader produces TensorDict
        batch = next(iter(dm.train_dataloader()))
        assert isinstance(batch, TensorFrame)
        assert batch["atoms", "Z"].shape[0] == 4  # 2 mols × 2 atoms

    def test_two_way_split(self, tmp_path):
        full = _SampleDataset(_make_samples(8))
        train_ds, val_ds = full.split(sizes=(5, 3))

        dm = DataModule(train_ds, val_ds, batch_size=2, num_workers=0, pin_memory=False)
        assert len(dm.train_dataset) == 5
        assert len(dm.val_dataset) == 3

    def test_rejects_mismatched_split(self, tmp_path):
        full = _SampleDataset(_make_samples(5))
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
        tag: Unique filename tag so multiple modules can share ``tmp_path``.
        seed: DataModule shuffle seed.
        **dm_kwargs: Extra DataModule kwargs (e.g. ``batch_sampler_source``).

    Returns:
        A non-DDP DataModule whose train samples carry ``U0 == float(index)``.
    """
    samples = _make_samples(10)
    dm_kwargs.setdefault("batch_size", 2)
    dm_kwargs.setdefault("num_workers", 0)
    dm_kwargs.setdefault("pin_memory", False)
    return DataModule(
        _SampleDataset(samples[:8]), _SampleDataset(samples[8:]), seed=seed, **dm_kwargs
    )


class TestBatchSamplerInjection:
    """``batch_sampler_source`` drives the train loader's batch composition.

    One symbol (BatchSamplerSource) is imported locally in the protocol-conformance test.
    """

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
        yielded = [[float(u) for u in batch["graphs", "energy"]] for batch in dm.train_dataloader()]
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
        monkeypatch.setattr("molix.data.datamodule._is_distributed", lambda: True)
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
# Capacity-first packing: batch_counts / val_batch_counts pack under the
# padder's own budgets, so every batch fits the static shape by construction.
# ---------------------------------------------------------------------------


def _static_padder(max_graphs: int = 3) -> MoleculePadder:
    """Two-atom frames, no valence: one ghost slot, 8 atom / 4 edge / 3 graph slots."""
    return MoleculePadder(
        max_atoms=8,
        max_pairs=4,
        cutoff=3.0,
        max_bonds=0,
        max_angles=0,
        max_propers=0,
        max_impropers=0,
        max_graphs=max_graphs,
    )


def _counts(n: int) -> dict[str, torch.Tensor]:
    """Per-sample counts of ``_make_samples``: 2 atoms, no edges (no neighbour task)."""
    return {
        "atoms": torch.full((n,), 2, dtype=torch.long),
        "pairs": torch.zeros(n, dtype=torch.long),
    }


def _make_packing_dm(tmp_path: Path, **dm_kwargs) -> DataModule:
    samples = _make_samples(10)
    dm_kwargs.setdefault("num_workers", 0)
    dm_kwargs.setdefault("pin_memory", False)
    dm_kwargs.setdefault(
        "collate_fn", FrameCollator(collator=MoleculeCollator(), padder=_static_padder())
    )
    return DataModule(_SampleDataset(samples[:8]), _SampleDataset(samples[8:]), **dm_kwargs)


class TestBatchCountsPacking:
    """``batch_counts`` drives a ``TokenBudgetBatchSampler`` over ``padder.budgets()``."""

    def test_train_loader_packs_to_the_padder_budgets(self, tmp_path):
        dm = _make_packing_dm(tmp_path, batch_counts=_counts(8))
        loader = dm.train_dataloader()
        assert isinstance(loader.batch_sampler, TokenBudgetBatchSampler)
        live = 0
        for batch in loader:
            # graphs budget = max_graphs - 1 = 2 molecules per batch, padded to 3 slots
            assert batch["graphs"].batch_size[0] == 3
            assert batch["atoms"].batch_size[0] == 8
            assert batch["pairs"].batch_size[0] == 4
            assert batch["graphs", "mask"].tolist() == [True, True, False]
            assert batch["atoms", "batch"][4:].tolist() == [2] * 4  # ghosts in the sink
            live += int(batch["graphs", "mask"].sum())
        assert live == 8

    def test_val_loader_packs_in_index_order_to_the_same_shape(self, tmp_path):
        dm = _make_packing_dm(tmp_path, batch_counts=_counts(8), val_batch_counts=_counts(2))
        batches = list(dm.val_dataloader())
        assert len(batches) == 1
        assert batches[0]["graphs"].batch_size[0] == 3
        assert [float(u) for u in batches[0]["graphs", "energy"][:2]] == [8.0, 9.0]
        assert batches[0]["graphs", "mask"].tolist() == [True, True, False]

    def test_missing_table_is_read_from_the_dataset(self, tmp_path):
        """A budgeted table absent from ``batch_counts`` comes from ``nrows``."""
        dm = _make_packing_dm(tmp_path, batch_counts={"pairs": torch.zeros(8, dtype=torch.long)})
        assert sum(len(b) for b in dm.train_dataloader().batch_sampler) == 8

    def test_epoch_reseed_keeps_one_loader_and_replans(self, tmp_path):
        dm = _make_packing_dm(tmp_path, batch_counts=_counts(8), seed=3)
        dm.on_epoch_start(0)
        loader = dm.train_dataloader()
        plan0 = list(loader.batch_sampler)
        dm.on_epoch_start(1)
        assert dm.train_dataloader() is loader
        plan1 = list(loader.batch_sampler)
        assert sorted(i for b in plan1 for i in b) == list(range(8))
        assert plan1 != plan0

    def test_requires_a_padder_on_the_collate_fn(self, tmp_path):
        with pytest.raises(ValueError, match="MoleculePadder"):
            _make_packing_dm(
                tmp_path,
                collate_fn=FrameCollator(collator=MoleculeCollator()),
                batch_counts=_counts(8),
            )

    def test_is_exclusive_with_the_budget_kwargs(self, tmp_path):
        with pytest.raises(ValueError, match="batch_counts"):
            _make_packing_dm(tmp_path, batch_counts=_counts(8), max_atoms_per_batch=16)

    def test_check_fails_closed_at_construction(self, tmp_path):
        counts = _counts(8)
        counts["atoms"][3] = 20  # no batch of 8 atom slots can carry it
        with pytest.raises(ValueError, match="atoms"):
            _make_packing_dm(tmp_path, batch_counts=counts)

    def test_under_ddp_raises_naming_the_kwarg(self, tmp_path, monkeypatch):
        monkeypatch.setattr("molix.data.datamodule._is_distributed", lambda: True)
        dm = _make_packing_dm(tmp_path, batch_counts=_counts(8), val_batch_counts=_counts(2))
        with pytest.raises(ValueError, match="batch_counts"):
            dm.train_dataloader()
        with pytest.raises(ValueError, match="val_batch_counts"):
            dm.val_dataloader()


# ---------------------------------------------------------------------------
# Worker start-method regression (preserved from the deleted
# tests/test_molix/test_data/test_e2e_workers.py)
# ---------------------------------------------------------------------------


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

        from molix.data import FrameCollator, MoleculeCollator, MoleculePadder

        dm = _dm_factory(
            batch_size=2,
            collate_fn=FrameCollator(
                collator=MoleculeCollator(),
                padder=MoleculePadder(max_atoms=8, max_pairs=8, cutoff=1.0),
            ),
        )
        fn = dm._make_collate_fn()
        fn2 = pickle.loads(pickle.dumps(fn))  # must not raise

        samples = _make_samples(2)
        out1 = fn(samples)
        out2 = fn2(samples)
        # The task travelled with the callable: both sides pad identically.
        assert out1["atoms", "Z"].shape == (8,)
        assert torch.equal(out1["atoms", "Z"], out2["atoms", "Z"])


# ---------------------------------------------------------------------------
# Block collate route
# (spec streaming-data-tiers-04-scheduler-integration-03-plan, Task 5;
#
# ---------------------------------------------------------------------------

#: Fixture blocks, deliberately tiny — a unit commit costs ~18 fsyncs
#: (``.claude/notes/notes.md`` § "[2026-08-14] Block reads stay on mmap")::
#:
#:   ordinal  block    frames  units (frames; atoms; task; local task code)
#:   0        block-p  0..3    u0 (0,1; 3; alpha; 0)   u1 (2,3; 2; beta; 1)
#:   1        block-q  4,5     u2 (4,5; 4; alpha; 0)
#:
#: Both blocks order their task dictionary the same way (``alpha`` first), so the
#: view's merged dictionary is the identity on each block's *local* codes — the
#: The stamp reads ``reader.task_codes`` and this fixture deliberately does not
#: discriminate local from global codes (see the spec note in the summary).
_BLOCK_MIXTURE: dict[str, float] = {"alpha": 0.5, "beta": 0.5}

#: Hand-computed goldens of block-p unit ``u0`` (frames 0, 1): every value is a
#: multiple of 0.5 and therefore exact in float32 — no FP tolerance is needed.
_U0_NUM_ATOMS: tuple[int, ...] = (3, 3)
_U0_ENERGY: tuple[float, ...] = (1.5, 2.0)
_U0_TASK_CODE = 0

_BLOCK_PROVENANCE: dict[str, object] = {
    "source": {"name": "molnex-unit-test", "coordinate": "n/a"},
    "generator": {"tool": "test_datamodule.py", "version": "1", "commit": "0"},
    "created_at": "2026-08-14T00:00:00+00:00",
    "units": {"pos": "angstrom", "energy": "kcal/mol", "forces": "kcal/mol/angstrom"},
}


def _block_halves(count: int, start: int) -> torch.Tensor:
    """*count* distinct multiples of 0.5 — exactly representable in float32."""
    return (torch.arange(count, dtype=torch.float32) + float(start)) * 0.5


class _PlainDataset(BaseDataset):
    """A dataset with no packed payload — the per-sample collate fallback."""

    def __init__(self, samples: list[dict]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        return self._samples[idx]


class TestCollationRoute:
    """There is one route: the per-sample collate.

    A second route used to be probed first — a block view got an index
    dataset plus a block-aware collate — and it went with the ``molix.block/1``
    reader/writer it served (spec ``data-layer-01-consolidation``). What is left
    to assert is that an ordinary dataset is handed through untouched.
    """

    def test_resolve_collation_routes_a_dataset_to_the_per_sample_collate(self, _dm_factory):
        dm = _dm_factory(batch_size=2)
        plain = _PlainDataset(_make_samples(2))

        loader_dataset, collate_fn = dm._resolve_collation(plain)

        assert loader_dataset is plain
        assert isinstance(collate_fn, _datamodule._CollateFn)
