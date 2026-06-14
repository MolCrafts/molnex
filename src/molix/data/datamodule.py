"""DDP-aware DataModule."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from molix.config import config
from molix.core.seed import make_generator, make_worker_init_fn
from molix.core.steps import batch_to
from molix.data.collate import DEFAULT_TARGET_SCHEMA, TargetSchema, collate_molecules
from molix.data.dataset import BaseDataset
from molix.data.pipeline import Node

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
        batch_size: Samples per batch (per rank in DDP).
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
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int | None = None,
        seed: int = 42,
        multiprocessing_context: str | None = "spawn",
        train_drop_last: bool = False,
    ) -> None:
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        if target_schema is None:
            target_schema = getattr(train_dataset, "target_schema", DEFAULT_TARGET_SCHEMA)
        self.target_schema = target_schema
        self.batch_nodes: tuple[Node, ...] = tuple(batch_nodes) if batch_nodes else ()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.prefetch_factor = prefetch_factor
        self.seed = seed
        self.multiprocessing_context = multiprocessing_context
        # Drop the last partial train batch so every batch has the same number
        # of graphs — required (with fixed-length padding) for CUDA-graph capture.
        self.train_drop_last = train_drop_last

        self._train_sampler: DistributedSampler | None = None
        self._val_sampler: DistributedSampler | None = None

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

        Under DDP, wraps the train dataset in a shuffling
        :class:`~torch.utils.data.DistributedSampler` (shuffle handled by
        the sampler, ``drop_last=True``); otherwise shuffles directly. The
        collate function casts floating-point leaves to the captured
        ``ftype`` and applies any post-collate batch nodes.

        Returns:
            A configured training ``DataLoader``.
        """
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
            # Seed the shuffle RNG so epoch ordering is reproducible and
            # independent of global-RNG consumption elsewhere in the process.
            generator = make_generator(self.seed)

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=self._train_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            collate_fn=self._make_collate_fn(),
            drop_last=self.train_drop_last or _is_distributed(),
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

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self._val_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            collate_fn=self._make_collate_fn(),
            multiprocessing_context=self._worker_context(),
            worker_init_fn=self._worker_init_fn(),
        )

    def _make_collate_fn(self) -> "_CollateFn":
        return _CollateFn(self.target_schema, self.batch_nodes)

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
        """Reseed the DDP samplers for *epoch* so shuffling differs each epoch.

        Calls ``set_epoch(epoch)`` on the train/val
        :class:`~torch.utils.data.DistributedSampler` instances when they
        exist (i.e. under DDP); a no-op otherwise.

        Args:
            epoch: The epoch index about to start.
        """
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
