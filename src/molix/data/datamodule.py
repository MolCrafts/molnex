"""DDP-aware DataModule with pipeline integration."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from molix.data.collate import DEFAULT_TARGET_SCHEMA, TargetSchema, collate_molecules
from molix.data.dataset import CachedDataset
from molix.data.pipeline import PipelineSpec, _call_task
from molix.data.source import DataSource, SubsetSource

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class DataModuleProtocol(Protocol):
    """Protocol consumed by the Trainer."""

    def setup(self, stage: str = "fit") -> None: ...
    def train_dataloader(self) -> Iterable: ...
    def val_dataloader(self) -> Iterable: ...
    def on_epoch_start(self, epoch: int) -> None: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _get_rank() -> int:
    return dist.get_rank() if _is_distributed() else 0


def _get_world_size() -> int:
    return dist.get_world_size() if _is_distributed() else 1


def _split_indices(n: int, train_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen).tolist()
    split = int(n * train_ratio)
    return perm[:split], perm[split:]


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------


class DataModule:
    """DDP-aware data module that drives a :class:`PipelineSpec`.

    Lifecycle (called by Trainer)::

        setup("fit")            # split, fit, transform, cache
        train_dataloader()      # DataLoader + DistributedSampler if DDP
        on_epoch_start(epoch)   # sampler.set_epoch for DDP shuffling

    Args:
        source: Raw data provider.
        pipeline: Compiled :class:`PipelineSpec`.
        val_source: Separate validation source.  If *None*,
            ``source`` is split by ``train_val_split``.
        train_val_split: Fraction used for training.
        target_schema: How targets are collated (graph vs atom level).
        batch_size: Samples per batch (per rank in DDP).
        num_workers: DataLoader worker processes.
        pin_memory: Pin memory for GPU transfer.
        persistent_workers: Keep workers alive between epochs (requires num_workers > 0).
        prefetch_factor: Batches prefetched per worker. None uses PyTorch default.
        cache_dir: Disk cache root.
        seed: RNG seed for splitting and DDP sampler.
    """

    def __init__(
        self,
        source: DataSource,
        pipeline: PipelineSpec,
        *,
        val_source: DataSource | None = None,
        train_val_split: float = 0.8,
        target_schema: TargetSchema = DEFAULT_TARGET_SCHEMA,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        prefetch_factor: int | None = None,
        cache_dir: str | Path | None = None,
        seed: int = 42,
    ) -> None:
        self.source = source
        self.pipeline = pipeline
        self.val_source = val_source
        self.train_val_split = train_val_split
        self.target_schema = target_schema
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.prefetch_factor = prefetch_factor
        self.cache_dir = str(cache_dir) if cache_dir else None
        self.seed = seed

        self.train_dataset: CachedDataset | None = None
        self.val_dataset: CachedDataset | None = None
        self._train_sampler: DistributedSampler | None = None
        self._val_sampler: DistributedSampler | None = None

    # -- Lifecycle ----------------------------------------------------------

    def setup(self, stage: str = "fit") -> None:
        if stage != "fit":
            return

        # 1. Split
        if self.val_source is None:
            train_idx, val_idx = _split_indices(len(self.source), self.train_val_split, self.seed)
            train_src: DataSource = SubsetSource(self.source, train_idx)
            val_src: DataSource = SubsetSource(self.source, val_idx)
        else:
            train_src = self.source
            val_src = self.val_source

        # 2. Prepare (DDP-aware)
        if _is_distributed():
            if _get_rank() == 0:
                fit = [train_src[i] for i in range(len(train_src))]
                train_samples = self.pipeline.prepare(
                    train_src, fit_samples=fit, cache_dir=self.cache_dir
                )
                val_samples = self.pipeline.prepare(val_src, cache_dir=self.cache_dir)
            dist.barrier()
            if _get_rank() != 0:
                train_samples = self.pipeline.prepare(train_src, cache_dir=self.cache_dir)
                val_samples = self.pipeline.prepare(val_src, cache_dir=self.cache_dir)
        else:
            fit = [train_src[i] for i in range(len(train_src))]
            train_samples = self.pipeline.prepare(
                train_src, fit_samples=fit, cache_dir=self.cache_dir
            )
            val_samples = self.pipeline.prepare(val_src, cache_dir=self.cache_dir)

        # 3. Wrap as Dataset
        self.train_dataset = CachedDataset(train_samples)
        self.val_dataset = CachedDataset(val_samples)

    # -- DataLoaders --------------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None, "Call setup('fit') first"

        if _is_distributed():
            self._train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=_get_world_size(),
                rank=_get_rank(),
                shuffle=True,
                seed=self.seed,
            )
            shuffle = False
        else:
            self._train_sampler = None
            shuffle = True

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
            drop_last=_is_distributed(),
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None, "Call setup('fit') first"

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
        )

    def _make_collate_fn(self):
        schema = self.target_schema
        batch_tasks = self.pipeline.batch_tasks

        def collate(samples: list[dict]) -> dict:  # type: ignore[return]
            batch = collate_molecules(samples, schema)
            for entry in batch_tasks:
                batch = _call_task(entry.task, batch)
            return batch

        return collate

    # -- Epoch hook ---------------------------------------------------------

    def on_epoch_start(self, epoch: int) -> None:
        if self._train_sampler is not None:
            self._train_sampler.set_epoch(epoch)
        if self._val_sampler is not None:
            self._val_sampler.set_epoch(epoch)
