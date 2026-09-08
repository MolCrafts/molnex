"""Storage layer for MolNex: bytes on disk ↔ tensors in memory.

Surfaces:

* **Frame view** — a record is a :data:`~molix.store.frame_dataset.FrameCollection`
  (``Sequence[Frame]``). :class:`FrameDataset` is a single-task map-style view.
  Mixing named sources is :class:`~molix.batch.mixture.DataMixture`.
* **Readers** — :class:`SubsetDataset` is an index view over any dataset and
  :class:`BaseDataset` is the shared reader protocol.
* **READY-tier blocks** — :class:`FrameStoreWriter` appends molpy ``Frame``\\ s
  to one directory of raw little-endian column files, commits each with a crc32
  record and seals it with per-column sha256; :class:`FrameStoreReader` mmaps a sealed
  block and serves sampler counts plus a packed-collate payload;
  :class:`FrameStoreManifest` is the JSON contract of both.
* **Multi-block view** — :class:`FrameStoreDataset` indexes an ordered sequence of
  sealed blocks as one frame space, concatenating the sampler tables and
  serving an order-preserving cross-block payload. It refuses, by named type,
  the three ways such a view could lie: :class:`FrameStoreNotReadyError` (a planned
  block missing or unsealed), :class:`FrameStoreSplitMismatchError` (mixed manifest
  splits) and :class:`FrameStoreSchemaMismatchError` (blocks that are not
  concatenable).

Nothing here builds batches, samples indices or acquires data — those live in
their own packages.
"""

from __future__ import annotations

from molix.store.dataset import (
    BaseDataset,
    SubsetDataset,
)
from molix.store.frame_dataset import FrameCollection, FrameDataset
from molix.store.frame_store import FrameStoreManifest, FrameStoreReader, FrameStoreWriter
from molix.store.frame_store_view import (
    FrameStoreDataset,
    FrameStoreNotReadyError,
    FrameStoreSchemaMismatchError,
    FrameStoreSplitMismatchError,
)

__all__ = [
    "BaseDataset",
    "FrameStoreDataset",
    "FrameStoreManifest",
    "FrameStoreNotReadyError",
    "FrameStoreReader",
    "FrameStoreSchemaMismatchError",
    "FrameStoreSplitMismatchError",
    "FrameStoreWriter",
    "FrameCollection",
    "FrameDataset",
    "SubsetDataset",
]
