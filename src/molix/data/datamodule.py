"""DDP-aware DataModule."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from molix.config import config
from molix.core.seed import make_generator, make_worker_init_fn
from molix.core.steps import batch_to
from molix.data.collate import (
    DEFAULT_TARGET_SCHEMA,
    TargetSchema,
    collate_molecules,
    collate_packed,
)
from molix.data.dataset import BaseDataset
from molix.data.pipeline import Node
from molix.data.sampler import TokenBudgetBatchSampler

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

        dag = pipe.cache(source, base_dir=run_dir / "cache",
                         fit_source=train_source)
        full = dag.dataset(mmap=True)
        train_ds, val_ds = full.split(sizes=(n_train, n_val), seed=42)
        dm = DataModule(train_ds, val_ds,
                        target_schema=QM9Source.TARGET_SCHEMA,
                        batch_nodes=pipe.batch_nodes,
                        batch_size=32, num_workers=4)
        trainer.train(datamodule=dm, max_epochs=100)

    Args:
        train_dataset: Pre-built training dataset.
        val_dataset: Pre-built validation dataset.
        target_schema: Which target keys are graph-level vs atom-level.
        batch_nodes: Post-collate :class:`Node` instances (from
            :attr:`PipelineSpec.batch_nodes`).
        batch_size: Samples per batch (per rank in DDP). Ignored for the
            train loader when a token budget is set (see below).
        max_atoms_per_batch: Optional per-batch total-atom budget. Setting
            this (and/or ``max_edges_per_batch``) switches the *train*
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
        max_edges_per_batch: Optional per-batch total-edge budget; same
            semantics as ``max_atoms_per_batch``, both may be combined.
        num_workers: DataLoader worker processes.
        pin_memory: Pin tensors for faster GPU transfer.
        persistent_workers: Keep workers alive between epochs.
        prefetch_factor: Batches prefetched per worker.
        seed: RNG seed for DDP sampler shuffling.
        multiprocessing_context: Start method for DataLoader worker
            processes. Defaults to ``"spawn"`` — see module docstring.
    """

    def __init__(
        self,
        train_dataset: BaseDataset,
        val_dataset: BaseDataset,
        *,
        target_schema: TargetSchema | None = None,
        batch_nodes: Sequence[Node] | None = None,
        batch_size: int = 32,
        max_atoms_per_batch: int | None = None,
        max_edges_per_batch: int | None = None,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int | None = None,
        seed: int = 42,
        multiprocessing_context: str | None = "spawn",
    ) -> None:
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        if target_schema is None:
            target_schema = getattr(train_dataset, "target_schema", DEFAULT_TARGET_SCHEMA)
        self.target_schema = target_schema
        self.batch_nodes: tuple[Node, ...] = tuple(batch_nodes) if batch_nodes else ()
        self.batch_size = batch_size
        self.max_atoms_per_batch = max_atoms_per_batch
        self.max_edges_per_batch = max_edges_per_batch
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.prefetch_factor = prefetch_factor
        self.seed = seed
        self.multiprocessing_context = multiprocessing_context

        self._train_sampler: DistributedSampler | None = None
        self._val_sampler: DistributedSampler | None = None
        self._epoch = 0

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

    def train_dataloader(self) -> DataLoader:
        """Build the training :class:`~torch.utils.data.DataLoader`.

        With a token budget set (``max_atoms_per_batch`` /
        ``max_edges_per_batch``), batches come from a
        :class:`~molix.data.sampler.TokenBudgetBatchSampler` seeded with
        ``seed + epoch`` — a fresh sampler per epoch, so epochs reshuffle
        and epoch *k*'s composition is re-derivable on resume.
        ``batch_size`` / ``shuffle`` / ``sampler`` / ``drop_last`` /
        ``generator`` are not passed in that mode (PyTorch makes them
        mutually exclusive with ``batch_sampler``).

        Otherwise: under DDP, wraps the train dataset in a shuffling
        :class:`~torch.utils.data.DistributedSampler` (shuffle handled by
        the sampler, ``drop_last=True``); otherwise shuffles directly. The
        collate function casts floating-point leaves to the captured
        ``ftype`` and applies any post-collate batch nodes.

        Returns:
            A configured training ``DataLoader``.

        Raises:
            ValueError: A token budget is set while running under DDP —
                dynamic batching has no cross-rank partitioning protocol
                yet. Drop the budget kwargs or run single-process.
        """
        if self.max_atoms_per_batch is not None or self.max_edges_per_batch is not None:
            if _is_distributed():
                raise ValueError(
                    "max_atoms_per_batch / max_edges_per_batch are not "
                    "supported under DDP: dynamic batching is incompatible "
                    "with DistributedSampler's fixed per-rank partitioning. "
                    "Remove the budget kwargs or run single-process; DDP "
                    "support belongs to a future spec."
                )
            # Counts come from the real dataset; the loader may instead see an
            # _IndexDataset (fast path) — the batch_sampler yields the same
            # index lists either way.
            batch_sampler = TokenBudgetBatchSampler(
                self.train_dataset,
                max_atoms=self.max_atoms_per_batch,
                max_edges=self.max_edges_per_batch,
                seed=self.seed + self._epoch,
            )
            loader_dataset, collate_fn = self._resolve_collation(self.train_dataset)
            return DataLoader(
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
            # DistributedSampler.set_epoch semantics.
            generator = make_generator(self.seed + self._epoch)

        loader_dataset, collate_fn = self._resolve_collation(self.train_dataset)
        return DataLoader(
            loader_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=self._train_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            collate_fn=collate_fn,
            drop_last=_is_distributed(),
            multiprocessing_context=self._worker_context(),
            worker_init_fn=self._worker_init_fn(),
            generator=generator,
        )

    def val_dataloader(self) -> DataLoader:
        """Build the validation :class:`~torch.utils.data.DataLoader`.

        Never shuffles. Under DDP, uses a non-shuffling
        :class:`~torch.utils.data.DistributedSampler` and keeps
        ``drop_last`` off so every validation sample is seen. Shares the
        same collate function as :meth:`train_dataloader`.

        Returns:
            A configured validation ``DataLoader``.
        """
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
        return DataLoader(
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

    def _make_collate_fn(self) -> "_CollateFn":
        return _CollateFn(self.target_schema, self.batch_nodes)

    def _resolve_collation(self, dataset: BaseDataset) -> tuple[object, object]:
        """Pick ``(loader_dataset, collate_fn)`` — packed fast path or fallback.

        When *dataset* is packed-cache-backed and we are not under DDP, the
        DataLoader is fed an :class:`_IndexDataset` (returning bare indices)
        plus a :class:`_PackedCollateFn` that slices the packed tensors
        directly via :func:`~molix.data.collate.collate_packed` — skipping
        per-sample unpack→repack. Otherwise the dataset is used as-is with
        the per-sample :class:`_CollateFn`. DDP is out of scope for the
        fast path, so it always falls back there.

        Args:
            dataset: The train or val dataset to collate.

        Returns:
            ``(loader_dataset, collate_fn)`` to hand to ``DataLoader``.
        """
        if not _is_distributed() and _packed_capable(dataset):
            return _IndexDataset(len(dataset)), _PackedCollateFn(
                dataset, self.target_schema, self.batch_nodes
            )
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
    :func:`molix.core.steps.batch_to` with ``dtype=self.ftype`` so the
    floating-point leaves match the model. This is what makes
    :meth:`molix.config.MolnexConfig.set_precision` a true
    single-source-of-truth for precision: workers re-import
    :mod:`molix.config` after spawn and would otherwise reset ``ftype``
    to its default ``float32``, so the value must be captured by the
    parent and pickled along with this callable.
    """

    def __init__(self, schema: TargetSchema, batch_nodes: Sequence[Node]) -> None:
        self.schema = schema
        self.batch_nodes = batch_nodes
        self.ftype = config["ftype"]

    def __call__(self, samples: list[dict]) -> dict:
        batch = collate_molecules(samples, self.schema)
        for entry in self.batch_nodes:
            batch = entry.apply(batch)
        return batch_to(batch, dtype=self.ftype)


def _packed_capable(dataset: BaseDataset) -> bool:
    """Whether *dataset* can serve the packed-aware collate fast path.

    True when the dataset exposes a working ``packed_view()`` — i.e. it is
    (or wraps) a :class:`~molix.data.cache.PackedCache`-backed dataset. A
    :class:`~molix.data.dataset.SubsetDataset` over a non-packed dataset
    has the method but raises :class:`AttributeError` when called, so we
    probe by calling it (cheap — wraps a payload reference, copies no
    tensors).
    """
    view_fn = getattr(dataset, "packed_view", None)
    if not callable(view_fn):
        return False
    try:
        view_fn()
    except AttributeError:
        return False
    return True


class _IndexDataset(Dataset):
    """Identity dataset returning the bare index for each position.

    Lets the DataLoader's sampler / ``batch_sampler`` drive batch
    composition while the real per-sample data is sliced from the packed
    cache by :class:`_PackedCollateFn`. Transparent to shuffling and the
    token-budget sampler, which depend only on ``__len__``.
    """

    def __init__(self, n: int) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> int:
        return idx


class _PackedCollateFn:
    """Picklable fast-path collate: slice packed tensors for a list of indices.

    Receives a ``list[int]`` of sample indices (from :class:`_IndexDataset`
    via the sampler), builds the batch with
    :func:`~molix.data.collate.collate_packed`, then applies the same
    post-collate contract as :class:`_CollateFn` — ``batch_nodes`` followed
    by :func:`~molix.core.steps.batch_to` with the captured ``ftype``.

    The :class:`~molix.data.dataset.PackedView` is built lazily on first
    call and excluded from the pickled state (see :meth:`__getstate__`), so
    the mmap'd payload is re-opened per worker through the dataset
    reference rather than shipped as tensors across the process boundary.
    """

    def __init__(
        self, dataset: BaseDataset, schema: TargetSchema, batch_nodes: Sequence[Node]
    ) -> None:
        self._dataset = dataset
        self.schema = schema
        self.batch_nodes = batch_nodes
        self.ftype = config["ftype"]
        self._view = None

    def _packed_view(self):
        if self._view is None:
            self._view = self._dataset.packed_view()
        return self._view

    def __call__(self, indices: list[int]) -> dict:
        batch = collate_packed(self._packed_view(), indices, self.schema)
        for entry in self.batch_nodes:
            batch = entry.apply(batch)
        return batch_to(batch, dtype=self.ftype)

    def __getstate__(self) -> dict:
        """Drop the lazily built view so pickling never captures payload tensors."""
        state = self.__dict__.copy()
        state["_view"] = None
        return state
