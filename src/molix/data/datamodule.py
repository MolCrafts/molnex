"""DDP-aware DataModule (DataLoader wrapper)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from time import perf_counter
from typing import Protocol, runtime_checkable

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from molix.config import config
from molix.core.seed import make_generator, make_worker_init_fn
from molix.core.steps import batch_to
from molix.data.collate import MoleculeCollator
from molix.data.collator import FrameCollator
from molix.data.dataset import BaseDataset
from molix.data.mixture import DataMixture
from molix.data.pad import MoleculePadder
from molix.data.sampler import BatchSamplerSource, TokenBudgetBatchSampler
from molix.data.task import BatchTask

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class DataModuleProtocol(Protocol):
    """Protocol consumed by the Trainer."""

    def setup(self, stage: str = "fit") -> None:
        """Prepare datasets/samplers for *stage* (e.g. ``"fit"``), called once."""
        ...

    def train_dataloader(self) -> Iterable:
        """Return an iterable yielding collated training batches."""
        ...

    def val_dataloader(self) -> Iterable:
        """Return an iterable yielding collated validation batches."""
        ...

    def on_epoch_start(self, epoch: int) -> None:
        """Notify the module that epoch *epoch* is starting (e.g. reseed samplers)."""
        ...


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _get_rank() -> int:
    return dist.get_rank() if _is_distributed() else 0


def _get_world_size() -> int:
    return dist.get_world_size() if _is_distributed() else 1


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------


class DataModule:
    """DDP-aware DataLoader wrapper.

    Takes pre-built train/val datasets and wraps them in DataLoaders.
    Dataset construction (downloading, pipeline transforms, caching) is
    done separately — this class only concerns itself with DataLoader
    configuration.

    Usage::

        train_ds, val_ds = dataset.split(sizes=(n_train, n_val), seed=42)
        dm = DataModule(
            train_ds,
            val_ds,
            collate_fn=FrameCollator(
                collator=MoleculeCollator(),
                padder=MoleculePadder(max_atoms=256, max_pairs=4096, cutoff=5.0),
            ),
            batch_size=32,
            num_workers=4,
        )
        trainer.train(datamodule=dm, max_epochs=100)

    Args:
        train_dataset: Pre-built training dataset. A
            :class:`~molix.data.mixture.DataMixture` is a Dataset and
            also the train mix policy.
        val_dataset: Pre-built validation dataset. A
            :class:`~molix.data.mixture.DataMixture` is iterated in
            source order, not re-mixed.
        collate_fn: DataLoader ``collate_fn``. Default
            :class:`~molix.data.FrameCollator` over
            :class:`~molix.data.collate.MoleculeCollator` (ragged). Pass a
            padder on the composer for compile-static shapes. Pad is not a
            :class:`~molix.data.task.BatchTask`.
        batch_tasks: Post-collate
            :class:`~molix.data.task.BatchTask` instances, applied to every
            collated batch in the order given (neighbor list, charge — not pad).
        batch_size: Samples per batch (per rank in DDP). Ignored for the
            train loader when a token budget is set (see below).
        max_atoms_per_batch: Optional per-batch total-atom budget. Setting
            this (and/or ``max_pairs_per_batch``) switches the *train*
            dataloader to a
            :class:`~molix.data.sampler.TokenBudgetBatchSampler` so each
            batch packs as many samples as fit under the budget — keeping
            GNN compute (O(edges)) per batch approximately constant
            instead of swinging with sample sizes. Requires a
            packed-cache-backed train dataset; incompatible with DDP
            (raises at dataloader construction). The val loader keeps the
            fixed ``batch_size`` — eval has no backward pass, so memory
            headroom is ample and fixed batches keep metric accumulation
            simple.
        max_pairs_per_batch: Optional per-batch total-edge budget; same
            semantics as ``max_atoms_per_batch``, both may be combined.
        batch_sampler_source: Optional
            :class:`~molix.data.sampler.BatchSamplerSource` that supplies
            the *train* loader's ``batch_sampler`` once per epoch, called
            as ``batch_sampler(train_dataset, seed=seed + epoch)`` — the
            same reseed convention as the budget path, so epoch *k*'s
            composition stays re-derivable on resume. Mutually exclusive
            with ``max_atoms_per_batch`` / ``max_pairs_per_batch`` (two
            competing batch policies) and, like them, incompatible with
            DDP. Omit it and every existing code path is unchanged. A
            multi-block train dataset is driven by
            :class:`~molix.data.BatchSamplerSource` here, whose plan is the consumed
            order (block-pure batches, at most ``view.max_open`` blocks
            resident); the matching collate route is picked automatically
            by :meth:`_resolve_collation`.
        num_workers: DataLoader worker processes.
        pin_memory: Pin tensors for faster GPU transfer.
        persistent_workers: Keep workers alive between epochs.
        prefetch_factor: Batches prefetched per worker.
        seed: RNG seed for DDP sampler shuffling.
        multiprocessing_context: Start method for DataLoader worker
            processes. Defaults to ``"spawn"`` — see module docstring.
        atom_counts: Optional per-sample atom counts for the token-budget
            train sampler. When omitted, the sampler reads frame-table
            table nrows. A molhub loader that already called
            ``cache("graph.natoms")`` passes the vector here so this
            class never looks up a cache.
        pair_counts: Optional per-sample edge counts, same contract.
        batch_counts: Per-molecule count tables of the **train** dataset,
            ``{table: (n,) tensor}`` — switches the train loader to packing
            under the padder's own budgets
            (:meth:`~molix.data.pad.MoleculePadder.budgets` of
            ``collate_fn.padder``), so every batch fits the static shape by
            construction. Tables the padder budgets but this mapping lacks
            are read through :meth:`~molix.data.dataset.BaseDataset.nrows_of`
            (``graphs`` needs none). The padder's
            :meth:`~molix.data.pad.MoleculePadder.check` runs here, so a
            molecule no batch could carry fails at construction. Mutually
            exclusive with ``max_atoms_per_batch`` / ``max_pairs_per_batch``
            and ``batch_sampler_source``; incompatible with DDP.
        val_batch_counts: The same for the **val** dataset: its loader
            packs in index order (no shuffle) under the same budgets, so
            evaluation sees the training shape. Omit it and the val loader
            keeps the fixed ``batch_size``.

    Raises:
        ValueError: ``batch_sampler_source`` is combined with a token
            budget, or does not implement
            :class:`~molix.data.sampler.BatchSamplerSource`;
            ``batch_counts`` / ``val_batch_counts`` without a
            :class:`~molix.data.pad.MoleculePadder` on ``collate_fn``, or
            combined with another batch policy, or naming a molecule the
            padder's caps cannot hold.
    """

    def __init__(
        self,
        train_dataset: BaseDataset,
        val_dataset: BaseDataset,
        *,
        collate_fn: FrameCollator | None = None,
        batch_tasks: Sequence[BatchTask] | None = None,
        batch_size: int = 32,
        max_atoms_per_batch: int | None = None,
        max_pairs_per_batch: int | None = None,
        batch_sampler_source: BatchSamplerSource | None = None,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int | None = None,
        seed: int = 42,
        multiprocessing_context: str | None = "spawn",
        train_drop_last: bool = False,
        atom_counts: torch.Tensor | None = None,
        pair_counts: torch.Tensor | None = None,
        batch_counts: Mapping[str, torch.Tensor] | None = None,
        val_batch_counts: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        if isinstance(train_dataset, DataMixture):
            if batch_sampler_source is not None:
                raise ValueError(
                    "DataMixture already is the train mix policy; do not also "
                    "pass batch_sampler_source."
                )
            batch_sampler_source = train_dataset

        if batch_sampler_source is not None:
            if not callable(getattr(batch_sampler_source, "batch_sampler", None)):
                raise ValueError(
                    "batch_sampler_source must implement the BatchSamplerSource "
                    "protocol — a batch_sampler(dataset, *, seed: int) -> "
                    "Iterable[list[int]] method; got "
                    f"{type(batch_sampler_source).__name__}, which has none."
                )
            if max_atoms_per_batch is not None or max_pairs_per_batch is not None:
                raise ValueError(
                    "batch_sampler_source and max_atoms_per_batch / "
                    "max_pairs_per_batch are two competing batch policies for "
                    "the train loader — pass exactly one: drop the budget "
                    "kwargs, or express the budget inside the source."
                )

        padder = None if collate_fn is None else getattr(collate_fn, "padder", None)
        if batch_counts is not None or val_batch_counts is not None:
            if not isinstance(padder, MoleculePadder):
                raise ValueError(
                    "batch_counts / val_batch_counts pack batches under the padder's "
                    "budgets, so collate_fn must be a FrameCollator carrying a "
                    "MoleculePadder; got "
                    f"padder={type(padder).__name__ if padder is not None else None}."
                )
            if max_atoms_per_batch is not None or max_pairs_per_batch is not None:
                raise ValueError(
                    "batch_counts and max_atoms_per_batch / max_pairs_per_batch are two "
                    "competing batch policies — the padder's budgets already cap atoms "
                    "and edges; drop the budget kwargs."
                )
            if batch_sampler_source is not None:
                raise ValueError(
                    "batch_counts and batch_sampler_source are two competing batch "
                    "policies for the train loader — pass exactly one."
                )
            if batch_counts is not None:
                padder.check(batch_counts)
            if val_batch_counts is not None:
                padder.check(val_batch_counts)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.collate_fn = collate_fn
        self.batch_tasks: tuple[BatchTask, ...] = tuple(batch_tasks) if batch_tasks else ()
        self.batch_size = batch_size
        self.max_atoms_per_batch = max_atoms_per_batch
        self.max_pairs_per_batch = max_pairs_per_batch
        self.batch_sampler_source = batch_sampler_source
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.prefetch_factor = prefetch_factor
        self.seed = seed
        self.multiprocessing_context = multiprocessing_context
        self.atom_counts = atom_counts
        self.pair_counts = pair_counts
        self.batch_counts = None if batch_counts is None else dict(batch_counts)
        self.val_batch_counts = None if val_batch_counts is None else dict(val_batch_counts)
        # Drop the last partial train batch so every batch has the same number
        # of graphs — required (with fixed-length padding) for CUDA-graph capture.
        self.train_drop_last = train_drop_last

        self._train_sampler: DistributedSampler | None = None
        self._val_sampler: DistributedSampler | None = None
        self._epoch = 0
        # Cached loaders (non-DDP, non-budget): built once and reused across epochs
        # so persistent workers survive and the DataLoader/worker pool is not rebuilt
        # — with a spawn context the per-epoch rebuild re-spawns workers (re-importing
        # the whole stack) every epoch, ~38 s/epoch and an eventual deadlock. The
        # stored shuffle generator is reseeded per epoch so reshuffling is preserved.
        self._train_loader: DataLoader | None = None
        self._val_loader: DataLoader | None = None
        self._shuffle_generator: torch.Generator | None = None
        # Cached batch_sampler for the batch-sampler route; reseeded in place
        # by on_epoch_start so a cached loader can keep persistent workers.
        self._batch_sampler = None

    def _worker_context(self) -> str | None:
        """Start method passed to :class:`DataLoader`, or ``None`` for sync.

        ``DataLoader`` rejects ``multiprocessing_context`` when
        ``num_workers == 0``, so we only forward it for the async path.
        """
        return self.multiprocessing_context if self.num_workers > 0 else None

    def _worker_init_fn(self):
        """Per-worker seeding callable, or ``None`` for the sync path."""
        return make_worker_init_fn(self.seed) if self.num_workers > 0 else None

    # -- Lifecycle (Trainer calls these) ------------------------------------

    def setup(self, stage: str = "fit") -> None:
        """No-op — datasets are fully built at construction time.

        Args:
            stage: Lifecycle stage label (accepted for protocol
                compatibility; unused here).
        """
        pass  # datasets are ready at construction time

    def _reject_ddp_batch_sampler(self, knob: str) -> None:
        """Refuse a custom train ``batch_sampler`` while running under DDP.

        Args:
            knob: The constructor argument(s) that requested the custom
                batch sampler, named verbatim in the error so the caller
                knows what to drop.

        Raises:
            ValueError: A distributed process group is initialised.
        """
        if _is_distributed():
            raise ValueError(
                f"{knob} not supported under DDP: dynamic batching is "
                "incompatible with DistributedSampler's fixed per-rank "
                f"partitioning. Remove {knob} or run single-process; DDP "
                "support belongs to a future spec."
            )

    def _epoch_batch_sampler(self) -> Iterable[list[int]] | None:
        """Build this epoch's train ``batch_sampler``, or ``None``.

        Dispatches the three batch policies in precedence order: an
        injected :class:`~molix.data.sampler.BatchSamplerSource`, else a
        token budget, else the fixed-``batch_size`` route (``None``). Both
        custom paths are seeded ``self.seed + self._epoch``, so epoch *k*
        reshuffles yet stays re-derivable on resume, and both pass the DDP
        guard first.

        Returns:
            An iterable, sized plan of ``list[int]`` sample-index batches,
            or ``None`` when no custom batch sampler is configured.

        Raises:
            ValueError: A custom batch sampler is configured under DDP.
        """
        if self.batch_sampler_source is not None:
            self._reject_ddp_batch_sampler("batch_sampler_source")
            return self.batch_sampler_source.batch_sampler(
                self.train_dataset, seed=self.seed + self._epoch
            )
        if self.batch_counts is not None:
            self._reject_ddp_batch_sampler("batch_counts")
            return self._budget_sampler(
                self.train_dataset,
                self._padder().budgets(),
                self.batch_counts,
                seed=self.seed + self._epoch,
                shuffle=True,
            )
        if self.max_atoms_per_batch is not None or self.max_pairs_per_batch is not None:
            self._reject_ddp_batch_sampler("max_atoms_per_batch / max_pairs_per_batch")
            budgets: dict[str, int] = {}
            given: dict[str, torch.Tensor] = {}
            if self.max_atoms_per_batch is not None:
                budgets["atoms"] = self.max_atoms_per_batch
                if self.atom_counts is not None:
                    given["atoms"] = self.atom_counts
            if self.max_pairs_per_batch is not None:
                budgets["pairs"] = self.max_pairs_per_batch
                if self.pair_counts is not None:
                    given["pairs"] = self.pair_counts
            # Counts come from the real dataset; the loader may instead see an
            # _IndexDataset (fast path) — the batch_sampler yields the same
            # index lists either way.
            return self._budget_sampler(
                self.train_dataset, budgets, given, seed=self.seed + self._epoch, shuffle=True
            )
        return None

    def _padder(self) -> MoleculePadder:
        """The static-shape padder ``batch_counts`` packs for (validated at init)."""
        padder = getattr(self.collate_fn, "padder", None)
        assert isinstance(padder, MoleculePadder), "validated in __init__"
        return padder

    @staticmethod
    def _budget_sampler(
        dataset: BaseDataset,
        budgets: Mapping[str, int],
        given: Mapping[str, torch.Tensor],
        *,
        seed: int,
        shuffle: bool,
    ) -> TokenBudgetBatchSampler:
        """A packed sampler over *budgets*, completing *given* counts from the dataset.

        Args:
            dataset: The dataset being packed (train or val).
            budgets: ``{table: per-batch cap}`` — the padder's or the legacy
                atoms / edges pair.
            given: Caller-supplied ``{table: counts}``; every budgeted table
                it lacks (``graphs`` aside) is read with
                :meth:`~molix.data.dataset.BaseDataset.nrows_of`.
            seed: Shuffle seed.
            shuffle: ``False`` packs in index order (val).

        Raises:
            ValueError: A table has to be read from a dataset that exposes
                no count for it.
        """
        counts: dict[str, torch.Tensor] = {}
        for table in budgets:
            if table == "graphs":
                continue
            if table in given:
                counts[table] = given[table]
                continue
            try:
                counts[table] = BaseDataset.nrows_of(dataset, table)
            except (AttributeError, TypeError) as e:
                raise ValueError(
                    f"no per-sample {table!r} counts were given and {type(dataset).__name__} "
                    f"exposes none; pass them (batch_counts / {table}_counts) or use a "
                    "Frame dataset."
                ) from e
        return TokenBudgetBatchSampler(
            dataset, budgets=budgets, counts=counts, seed=seed, shuffle=shuffle
        )

    def train_dataloader(self) -> DataLoader:
        """Build the training :class:`~torch.utils.data.DataLoader`.

        With ``batch_counts`` set, batches come from a
        :class:`~molix.data.sampler.TokenBudgetBatchSampler` over the
        padder's budgets; with a token budget set (``max_atoms_per_batch``
        / ``max_pairs_per_batch``), over that pair. Either way the sampler
        is seeded ``seed + epoch`` and reseeded in place each epoch, so
        epochs reshuffle, epoch *k*'s composition is re-derivable on
        resume, and one loader (with its persistent workers) is kept. With a
        ``batch_sampler_source`` set, that source supplies the batch
        sampler under the same ``seed + epoch`` convention. In both modes
        ``batch_size`` / ``shuffle`` / ``sampler`` / ``drop_last`` /
        ``generator`` are not passed (PyTorch makes them mutually
        exclusive with ``batch_sampler``).

        Otherwise: under DDP, wraps the train dataset in a shuffling
        :class:`~torch.utils.data.DistributedSampler` (shuffle handled by
        the sampler, ``drop_last=True``); otherwise shuffles directly. The
        collate function casts floating-point leaves to the captured
        ``ftype`` and applies any post-collate batch tasks.

        Returns:
            A configured training ``DataLoader``.

        Raises:
            ValueError: A token budget or a ``batch_sampler_source`` is set
                while running under DDP — dynamic batching has no
                cross-rank partitioning protocol yet. Drop the batching
                kwarg or run single-process.
        """
        # Cached batch-sampler fast path: on_epoch_start reseeded the stored
        # sampler in place, so the same loader (and its persistent worker
        # pool) is reused across epochs instead of re-spawned each epoch.
        if (
            self._train_loader is not None
            and self._batch_sampler is not None
            and not _is_distributed()
        ):
            return self._train_loader

        batch_sampler = self._epoch_batch_sampler()
        if batch_sampler is not None:
            loader_dataset, collate_fn = self._resolve_collation(self.train_dataset)
            loader = DataLoader(
                loader_dataset,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                prefetch_factor=self.prefetch_factor,
                collate_fn=collate_fn,
                multiprocessing_context=self._worker_context(),
                worker_init_fn=self._worker_init_fn(),
            )
            if callable(getattr(batch_sampler, "reseed", None)) and not _is_distributed():
                self._batch_sampler = batch_sampler
                self._train_loader = loader
            return loader

        # Cached fast path: reuse the loader across epochs (persistent workers
        # survive; no per-epoch rebuild). Reseed the stored shuffle generator so the
        # next __iter__ reshuffles with (seed, epoch) — same permutation the
        # rebuild-every-epoch code produced, so resume-determinism is preserved.
        # Excluded under DDP (DistributedSampler.set_epoch handles reshuffle).
        if self._train_loader is not None and not _is_distributed():
            if self._shuffle_generator is not None:
                self._shuffle_generator.manual_seed(self.seed + self._epoch)
            return self._train_loader

        if _is_distributed():
            self._train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=_get_world_size(),
                rank=_get_rank(),
                shuffle=True,
                seed=self.seed,
            )
            shuffle = False
            generator = None
        else:
            self._train_sampler = None
            shuffle = True
            # Seed the shuffle RNG from (seed, epoch) so each epoch gets a
            # different permutation, yet epoch k's order is re-derivable on
            # resume without checkpointing generator state — mirrors
            # DistributedSampler.set_epoch semantics. Stored so the cached loader
            # can be reseeded each epoch.
            self._shuffle_generator = make_generator(self.seed + self._epoch)
            generator = self._shuffle_generator

        loader_dataset, collate_fn = self._resolve_collation(self.train_dataset)
        loader = DataLoader(
            loader_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=self._train_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            collate_fn=collate_fn,
            drop_last=self.train_drop_last or _is_distributed(),
            multiprocessing_context=self._worker_context(),
            worker_init_fn=self._worker_init_fn(),
            generator=generator,
        )
        if not _is_distributed():
            self._train_loader = loader  # cache for subsequent epochs
        return loader

    def val_dataloader(self) -> DataLoader:
        """Build the validation :class:`~torch.utils.data.DataLoader`.

        Never shuffles. With ``val_batch_counts`` set, packs the val
        dataset in index order under the padder's budgets — the training
        shape, so a compiled eval step sees one shape too. Under DDP, uses
        a non-shuffling :class:`~torch.utils.data.DistributedSampler` and
        keeps ``drop_last`` off so every validation sample is seen. Shares
        the same collate function as :meth:`train_dataloader`.

        Returns:
            A configured validation ``DataLoader``.

        Raises:
            ValueError: ``val_batch_counts`` is set under DDP.
        """
        # Cached (non-DDP): val never shuffles, so the same loader is reused every
        # eval phase — no per-eval worker re-spawn.
        if self._val_loader is not None and not _is_distributed():
            return self._val_loader

        if self.val_batch_counts is not None:
            self._reject_ddp_batch_sampler("val_batch_counts")
            loader_dataset, collate_fn = self._resolve_collation(self.val_dataset)
            loader = DataLoader(
                loader_dataset,
                batch_sampler=self._budget_sampler(
                    self.val_dataset,
                    self._padder().budgets(),
                    self.val_batch_counts,
                    seed=self.seed,
                    shuffle=False,
                ),
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                prefetch_factor=self.prefetch_factor,
                collate_fn=collate_fn,
                multiprocessing_context=self._worker_context(),
                worker_init_fn=self._worker_init_fn(),
            )
            self._val_loader = loader
            return loader

        if _is_distributed():
            self._val_sampler = DistributedSampler(
                self.val_dataset,
                num_replicas=_get_world_size(),
                rank=_get_rank(),
                shuffle=False,
            )
        else:
            self._val_sampler = None

        loader_dataset, collate_fn = self._resolve_collation(self.val_dataset)
        loader = DataLoader(
            loader_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self._val_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            collate_fn=collate_fn,
            multiprocessing_context=self._worker_context(),
            worker_init_fn=self._worker_init_fn(),
        )
        if not _is_distributed():
            self._val_loader = loader
        return loader

    def _make_collate_fn(self) -> "_CollateFn":
        frame_collator = self.collate_fn
        if frame_collator is None:
            frame_collator = FrameCollator(collator=MoleculeCollator(dtype=config["ftype"]))
        return _CollateFn(frame_collator, self.batch_tasks)

    def _resolve_collation(self, dataset: BaseDataset) -> tuple[object, object]:
        """Pick ``(loader_dataset, collate_fn)`` — one route, the per-sample collate.

        There used to be a second route here, probed on every dataloader build:
        a block view got an index dataset plus a block-aware collate. It
        was removed with the ``molix.block/1`` writer/reader it served, which had
        no producer, no consumer and no instance on disk (spec
        ``data-layer-01-consolidation``). The probe could never fire, so this is
        a deletion, not a behaviour change.

        Args:
            dataset: The train or val dataset to collate.

        Returns:
            ``(loader_dataset, collate_fn)`` to hand to ``DataLoader``.
        """
        return dataset, self._make_collate_fn()

    @property
    def ftype(self) -> torch.dtype:
        """The floating-point dtype each collated batch will be cast to.

        Captured from :data:`molix.config.config` at dataloader-construction
        time so that ``spawn``-launched workers see a stable value even after
        their module-level re-import of :mod:`molix.config`.
        """
        return config["ftype"]

    # -- Epoch hook ---------------------------------------------------------

    def on_epoch_start(self, epoch: int) -> None:
        """Reseed shuffling for *epoch* so the permutation differs each epoch.

        Records *epoch* so the next :meth:`train_dataloader` call seeds its
        shuffle generator with ``seed + epoch``, and calls
        ``set_epoch(epoch)`` on the train/val
        :class:`~torch.utils.data.DistributedSampler` instances when they
        exist (i.e. under DDP).

        Args:
            epoch: The epoch index about to start.
        """
        self._epoch = epoch
        if self._batch_sampler is not None and callable(
            getattr(self._batch_sampler, "reseed", None)
        ):
            self._batch_sampler.reseed(self.seed + epoch)
        if self._train_sampler is not None:
            self._train_sampler.set_epoch(epoch)
        if self._val_sampler is not None:
            self._val_sampler.set_epoch(epoch)


# ---------------------------------------------------------------------------
# Picklable collate wrapper (required for non-fork start methods)
# ---------------------------------------------------------------------------


class _CollateFn:
    """Picklable collate callable for DataLoader workers.

    ``spawn`` / ``forkserver`` start methods both send the collate callable
    to workers through ``pickle``, so a local closure won't survive the
    trip. A top-level class keeps it picklable on every supported Python
    and every platform (``spawn`` is already the default on macOS/Windows
    and is what we default to in :class:`DataModule` to sidestep the
    Python 3.14 multi-threaded-fork DeprecationWarning).

    Captures :data:`molix.config.config["ftype"]` at construction time
    (in the main process) and routes each emitted batch through
    :func:`molix.core.steps.batch_to` with ``dtype=self.ftype``. Compute
    floating leaves follow ``ftype``. Frame collate emits graph totals at
    :data:`~molix.schema.FLOAT64_LABEL_KEYS` as IEEE-754 binary64; the
    store route does not narrow those totals below the mmap column and
    does not upcast. This is what makes
    :meth:`molix.config.MolnexConfig.set_precision` a true
    single-source-of-truth for compute precision: workers re-import
    :mod:`molix.config` after spawn and would otherwise reset ``ftype``
    to its default ``float32``, so the value must be captured by the
    parent and pickled along with this callable.
    """

    def __init__(self, frame_collator: FrameCollator, batch_tasks: Sequence[BatchTask]) -> None:
        self.ftype = config["ftype"]
        self.frame_collator = frame_collator
        self.batch_tasks = batch_tasks
        self.stage_seconds: dict[str, float] | None = None

    def time_stages(self) -> dict[str, float]:
        """Start accumulating per-stage wall time and return the live record.

        The input path's cost is a sum over prepare / collate / one entry per
        batch task / pad, and only the total is observable from outside. A
        total says the loader leads the GPU; it does not say which stage to
        attack, and picking one without the split is guessing. Off by default
        (``None``) so the production path pays nothing; the returned dict is
        updated in place on every call and is only meaningful with
        ``num_workers=0``, since a worker's copy dies with the worker.

        Returns:
            The accumulator, keyed by stage name, in seconds.
        """
        self.stage_seconds = {}
        return self.stage_seconds

    def __call__(self, samples: list[dict]) -> dict:
        # Prepare (if any), collate, then tasks (NeighborList fills edges),
        # then pad. Pad after nlist so dead edges sit on the cutoff graph.
        # Must call prepare_samples here — wrapping FrameCollator in another
        # callable is invisible: this method uses .collator, not __call__.
        stages = self.stage_seconds
        if stages is None:
            prepare_samples = getattr(self.frame_collator, "prepare_samples", None)
            if callable(prepare_samples):
                samples = prepare_samples(samples)
            batch = self.frame_collator.collator(samples)
            for task in self.batch_tasks:
                batch = task.execute(batch)
            self.frame_collator.record_live(batch)
            padder = self.frame_collator.padder
            if padder is not None:
                batch = padder(batch)
            batch = self.frame_collator.apply_graph_schema(batch)
            self.frame_collator._last = batch
            return batch_to(batch, dtype=self.ftype)

        def _tick(name: str, start: float) -> float:
            now = perf_counter()
            stages[name] = stages.get(name, 0.0) + (now - start)
            return now

        mark = perf_counter()
        prepare_samples = getattr(self.frame_collator, "prepare_samples", None)
        if callable(prepare_samples):
            samples = prepare_samples(samples)
        mark = _tick("prepare", mark)
        batch = self.frame_collator.collator(samples)
        mark = _tick("collate", mark)
        for task in self.batch_tasks:
            batch = task.execute(batch)
            mark = _tick(f"task:{type(task).__name__}", mark)
        self.frame_collator.record_live(batch)
        mark = _tick("record_live", mark)
        padder = self.frame_collator.padder
        if padder is not None:
            batch = padder(batch)
        mark = _tick("pad", mark)
        batch = self.frame_collator.apply_graph_schema(batch)
        self.frame_collator._last = batch
        out = batch_to(batch, dtype=self.ftype)
        _tick("schema+cast", mark)
        return out
