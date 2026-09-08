"""DDP-aware DataModule."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol, TypeGuard, runtime_checkable

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from molix.batch.collate import (
    MoleculeCollator,
)
from molix.batch.frame_store_plan import FrameStoreView
from molix.batch.mixture import DataMixture
from molix.batch.sampler import BatchSamplerSource, TokenBudgetBatchSampler
from molix.batch.task import BatchTask
from molix.config import config
from molix.core.seed import make_generator, make_worker_init_fn
from molix.core.steps import batch_to
from molix.store.dataset import BaseDataset
from molix.store.frame_store_view import _batch_from_columns

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

        full = FrameStoreDataset.load_from_disk(blocks_dir)
        train_ds, val_ds = full.split(sizes=(n_train, n_val), seed=42)
        dm = DataModule(train_ds, val_ds,
                        batch_tasks=[PadMolecularBatch(max_atoms=256,
                                                       max_edges=4096)],
                        batch_size=32, num_workers=4)
        trainer.train(datamodule=dm, max_epochs=100)

    Args:
        train_dataset: Pre-built training dataset. A
            :class:`~molix.batch.mixture.DataMixture` is a Dataset and
            also the train mix policy.
        val_dataset: Pre-built validation dataset. A
            :class:`~molix.batch.mixture.DataMixture` is iterated in
            source order, not re-mixed.
        batch_tasks: Post-collate
            :class:`~molix.batch.task.BatchTask` instances, applied to every
            collated batch in the order given.
        batch_size: Samples per batch (per rank in DDP). Ignored for the
            train loader when a token budget is set (see below).
        max_atoms_per_batch: Optional per-batch total-atom budget. Setting
            this (and/or ``max_edges_per_batch``) switches the *train*
            dataloader to a
            :class:`~molix.batch.sampler.TokenBudgetBatchSampler` so each
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
        batch_sampler_source: Optional
            :class:`~molix.batch.sampler.BatchSamplerSource` that supplies
            the *train* loader's ``batch_sampler`` once per epoch, called
            as ``batch_sampler(train_dataset, seed=seed + epoch)`` — the
            same reseed convention as the budget path, so epoch *k*'s
            composition stays re-derivable on resume. Mutually exclusive
            with ``max_atoms_per_batch`` / ``max_edges_per_batch`` (two
            competing batch policies) and, like them, incompatible with
            DDP. Omit it and every existing code path is unchanged. A
            multi-block train dataset is driven by
            :class:`~molix.batch.frame_store_plan.FrameStorePlanSource` here, whose
            :class:`~molix.batch.frame_store_plan.FrameStorePlan` is the consumed
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

    Raises:
        ValueError: ``batch_sampler_source`` is combined with a token
            budget, or does not implement
            :class:`~molix.batch.sampler.BatchSamplerSource`.
    """

    def __init__(
        self,
        train_dataset: BaseDataset,
        val_dataset: BaseDataset,
        *,
        batch_tasks: Sequence[BatchTask] | None = None,
        batch_size: int = 32,
        max_atoms_per_batch: int | None = None,
        max_edges_per_batch: int | None = None,
        batch_sampler_source: BatchSamplerSource | None = None,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int | None = None,
        seed: int = 42,
        multiprocessing_context: str | None = "spawn",
        train_drop_last: bool = False,
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
            if max_atoms_per_batch is not None or max_edges_per_batch is not None:
                raise ValueError(
                    "batch_sampler_source and max_atoms_per_batch / "
                    "max_edges_per_batch are two competing batch policies for "
                    "the train loader — pass exactly one: drop the budget "
                    "kwargs, or express the budget inside the source."
                )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_tasks: tuple[BatchTask, ...] = tuple(batch_tasks) if batch_tasks else ()
        self.batch_size = batch_size
        self.max_atoms_per_batch = max_atoms_per_batch
        self.max_edges_per_batch = max_edges_per_batch
        self.batch_sampler_source = batch_sampler_source
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
        injected :class:`~molix.batch.sampler.BatchSamplerSource`, else a
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
        if self.max_atoms_per_batch is not None or self.max_edges_per_batch is not None:
            self._reject_ddp_batch_sampler("max_atoms_per_batch / max_edges_per_batch")
            # Counts come from the real dataset; the loader may instead see an
            # _IndexDataset (fast path) — the batch_sampler yields the same
            # index lists either way.
            return TokenBudgetBatchSampler(
                self.train_dataset,
                max_atoms=self.max_atoms_per_batch,
                max_edges=self.max_edges_per_batch,
                seed=self.seed + self._epoch,
            )
        return None

    def train_dataloader(self) -> DataLoader:
        """Build the training :class:`~torch.utils.data.DataLoader`.

        With a token budget set (``max_atoms_per_batch`` /
        ``max_edges_per_batch``), batches come from a
        :class:`~molix.batch.sampler.TokenBudgetBatchSampler` seeded with
        ``seed + epoch`` — a fresh sampler per epoch, so epochs reshuffle
        and epoch *k*'s composition is re-derivable on resume. With a
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

        Never shuffles. Under DDP, uses a non-shuffling
        :class:`~torch.utils.data.DistributedSampler` and keeps
        ``drop_last`` off so every validation sample is seen. Shares the
        same collate function as :meth:`train_dataloader`.

        Returns:
            A configured validation ``DataLoader``.
        """
        # Cached (non-DDP): val never shuffles, so the same loader is reused every
        # eval phase — no per-eval worker re-spawn.
        if self._val_loader is not None and not _is_distributed():
            return self._val_loader

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
        return _CollateFn(self.batch_tasks)

    def _resolve_collation(self, dataset: BaseDataset) -> tuple[object, object]:
        """Pick ``(loader_dataset, collate_fn)`` — block, packed, or fallback.

        Three routes, probed in this order (all fast paths are skipped under
        DDP, which falls back to the per-sample collate):

        1. **Block view** — a :class:`~molix.batch.frame_store_plan.FrameStoreView`
           gets an :class:`_IndexDataset` over its global frame rows plus a
           :class:`_FrameStoreCollateFn`, which materialises one block's frames per
           batch. Probed *first* because a block view carries no single packed
           payload: the per-sample fallback is unavailable for it, and silently
           taking it would raise ``KeyError`` deep in the collate instead of an
           actionable message here.
        2. **Fallback** — the dataset is used as-is with the per-sample
           :class:`_CollateFn`.

        Args:
            dataset: The train or val dataset to collate.

        Returns:
            ``(loader_dataset, collate_fn)`` to hand to ``DataLoader``.
        """
        if not _is_distributed() and _block_capable(dataset):
            return _IndexDataset(len(dataset)), _FrameStoreCollateFn(dataset, self.batch_tasks)
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
    :func:`molix.core.steps.batch_to` with ``dtype=self.ftype`` so the
    floating-point leaves match the model. This is what makes
    :meth:`molix.config.MolnexConfig.set_precision` a true
    single-source-of-truth for precision: workers re-import
    :mod:`molix.config` after spawn and would otherwise reset ``ftype``
    to its default ``float32``, so the value must be captured by the
    parent and pickled along with this callable.
    """

    def __init__(self, batch_tasks: Sequence[BatchTask]) -> None:
        self.collator = MoleculeCollator()
        self.batch_tasks = batch_tasks
        self.ftype = config["ftype"]

    def __call__(self, samples: list[dict]) -> dict:
        batch = self.collator(samples)
        for task in self.batch_tasks:
            batch = task.execute(batch)
        return batch_to(batch, dtype=self.ftype)


def _block_capable(dataset: BaseDataset) -> TypeGuard[FrameStoreView]:
    """Whether *dataset* can serve the block-aware collate route.

    A cheap structural probe — ``isinstance`` against the
    :class:`~molix.batch.frame_store_plan.FrameStoreView` protocol, which reads member
    presence with :func:`inspect.getattr_static` (Python 3.12+, this project's
    floor), so no property is evaluated — a view's deliberately raising
    ``edge_counts`` is not tripped — and no tensor is copied. Every other
    dataset simply answers ``False``; nothing is raised here, because this is a
    route probe and not a validation.

    Args:
        dataset: The train or val dataset about to be collated.

    Returns:
        ``True`` when *dataset* exposes the multi-block view surface (``locate``
        / ``reader`` / ``block_ids`` / the ``(F,)`` code tables). Declared as a
        :class:`~typing.TypeGuard` so the route that follows sees the view
        surface it is about to consume, rather than the wider dataset type.
    """
    return isinstance(dataset, FrameStoreView)


class _IndexDataset(Dataset):
    """Identity dataset returning the bare index for each position.

    Lets the DataLoader's sampler / ``batch_sampler`` drive batch
    composition while the real per-sample data is sliced from the packed
    block by :class:`_FrameStoreCollateFn`. Transparent to shuffling and the
    token-budget sampler, which depend only on ``__len__``.
    """

    def __init__(self, n: int) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> int:
        return idx


class _FrameStoreCollateFn:
    """Picklable block-route collate: one block's frames per batch, task stamped.

    Receives a ``list[int]`` of **global frame rows** (from :class:`_IndexDataset`
    via the plan's ``batch_sampler``) and builds the batch with the composition the
    block store documents::

        view.batch(frames)

    then applies the same post-collate contract as :class:`_CollateFn` —
    ``batch_tasks`` in order, followed by
    :func:`~molix.core.steps.batch_to` with the captured ``ftype``, which casts
    floating leaves only, so the int64 task label survives untouched. Blocks carry
    no edges, so the collate's empty-edge fallback is exercised by design and the
    cutoff graph is built downstream.

    **The batch must be block-pure and single-task.** Both are guarantees of
    :class:`~molix.batch.frame_store_plan.FrameStorePlan`, so a violation means the plan was
    bypassed and is refused rather than papered over: no cross-block payload merge
    exists on this path, and
    :class:`~molix.core.steps.multitask.TaskRouter` maps one ``("graphs", "task")``
    code per batch to one :class:`~molix.core.steps.multitask.TaskSpec`.

    **The task label is stamped from the view's global codes.**
    ``FrameStoreReader.frame_payload`` skips ``meta.`` routes, so the per-frame task
    code never reaches the payload; it is stamped here as a ``targets.task`` leaf,
    which rides the existing target route and lands as a ``(B,)`` int64 leaf at
    ``("graphs", "task")``. The codes come from ``view.task_codes`` (the merged,
    global dictionary), never from a block-local ``reader.task_codes``: a block's
    local ``(0, 1)`` may be global ``(1, 2)`` after the view's merge, so a local
    stamp would route the same physical task to different losses depending on
    which block a batch came from.

    Residency is the view's: this callable keeps no reader of its own — holding one
    past the view's bounded LRU pool would make ``K + 1`` blocks resident — so
    there is no lazily built per-block state to clear, and :meth:`__getstate__`
    hands out a snapshot carrying the view
    reference, the schema, the tasks and the dtype, never a payload tensor.

    Args:
        view: The multi-block :class:`~molix.batch.frame_store_plan.FrameStoreView` the
            frame rows address.
        schema: Routes target keys to the ``atoms`` or ``graphs`` namespace.
        batch_tasks: Post-collate :class:`~molix.batch.task.BatchTask` instances,
            applied in the given order.

    Raises:
        ValueError: *schema* declares ``"task"`` atom-level, which would route the
            stamp to a per-atom leaf instead of the per-graph label ``TaskRouter``
            reads.
    """

    def __init__(self, view: FrameStoreView, batch_tasks: Sequence[BatchTask]) -> None:
        self._view = view
        self.batch_tasks = batch_tasks
        self.ftype = config["ftype"]

    def __call__(self, indices: list[int]) -> dict:
        """Collate one block-pure, single-task batch of global frame rows.

        Args:
            indices: Global frame rows of the view, ``B`` long, in batch order.

        Returns:
            The nested ``atoms`` / ``edges`` / ``graphs`` batch, its floating
            leaves cast to the captured ``ftype`` and ``("graphs", "task")``
            carrying the batch's ``(B,)`` int64 global task code.

        Raises:
            ValueError: *indices* span more than one block (the plan was
                bypassed), or more than one task code.
            FrameStoreNotReadyError: The owning block is not available — propagated
                from the view with the block named. Consumption stalls here; it
                never advances to another block's batches and never substitutes.
        """
        frames = torch.as_tensor(indices, dtype=torch.long)
        blocks, locals_ = self._view.locate(frames)
        owners = blocks.unique()
        if owners.numel() != 1:
            named = [self._view.block_ids[int(block)] for block in owners.tolist()]
            raise ValueError(
                f"a block batch holds frames from {owners.numel()} blocks ({named}), "
                "but this route materialises exactly one block per batch — there is no "
                "cross-block payload merge here, by design. Every batch a FrameStorePlan "
                "emits is block-pure, so this batch did not come from one: drive the "
                "loader with FrameStorePlan (DataModule(batch_sampler_source=...))."
            )
        block = int(owners[0])
        reader = self._view.reader(block)

        codes = self._view.task_codes[frames]
        if codes.unique().numel() != 1:
            raise ValueError(
                f"a block batch of {self._view.block_ids[block]} carries the task codes "
                f"{sorted(set(codes.tolist()))}, but TaskRouter routes one batch to one "
                'TaskSpec through the single-task label at ("graphs", "task"). Every '
                "batch a FrameStorePlan emits is single-task by construction, so this batch "
                "did not come from one."
            )

        columns = reader._columns(locals_.tolist())
        columns["graphs"]["task"] = codes.to(torch.int64)
        batch = _batch_from_columns(columns)
        for task in self.batch_tasks:
            batch = task.execute(batch)
        return batch_to(batch, dtype=self.ftype)

    def __getstate__(self) -> dict:
        """Snapshot the picklable state — a view reference, never payload tensors.

        There is no lazily built
        per-block state to clear here (readers stay in the view's bounded pool), so
        the snapshot is the whole state, copied rather than aliased.
        """
        return self.__dict__.copy()
