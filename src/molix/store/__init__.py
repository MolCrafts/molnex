"""Storage layer for MolNex: bytes on disk ↔ tensors in memory.

Surfaces:

* **Frame view** — a record is a :data:`~molix.store.frame_dataset.FrameCollection`
  (``Sequence[Frame]``). :class:`FrameSample` turns one ``Frame`` into the
  pre-collate dict; :class:`FrameDataset` is a single-task map-style view.
  Mixing named sources is :class:`~molix.batch.mixture.DataMixture`.
* **Readers** — :class:`SubsetDataset` is an index view over any dataset and
  :class:`BaseDataset` is the shared reader protocol. :class:`PackedView` is the
  read-only handle the packed collate fast path consumes.
* **READY-tier blocks** — :class:`BlockWriter` appends :class:`SamplingUnit`\\ s
  to one directory of raw little-endian column files, commits each with a crc32
  record and seals it with per-column sha256; :class:`BlockReader` mmaps a sealed
  block and serves sampler counts plus a packed-collate payload;
  :class:`BlockManifest` is the JSON contract of both.
* **Multi-block view** — :class:`BlockDataset` indexes an ordered sequence of
  sealed blocks as one frame space, concatenating the sampler tables and
  serving an order-preserving cross-block payload. It refuses, by named type,
  the three ways such a view could lie: :class:`BlockNotReadyError` (a planned
  block missing or unsealed), :class:`BlockSplitMismatchError` (mixed manifest
  splits) and :class:`BlockSchemaMismatchError` (blocks that are not
  concatenable).

Nothing here builds batches, samples indices or acquires data — those live in
their own packages.
"""

from __future__ import annotations

from molix.store.block import BlockManifest, BlockReader, BlockWriter, SamplingUnit
from molix.store.block_view import (
    BlockDataset,
    BlockNotReadyError,
    BlockSchemaMismatchError,
    BlockSplitMismatchError,
)
from molix.store.dataset import (
    BaseDataset,
    PackedView,
    SubsetDataset,
)
from molix.store.frame_dataset import CLASS_I_FRAME_SCHEMA, FrameCollection, FrameDataset
from molix.store.frame_sample import FrameSample

__all__ = [
    "BaseDataset",
    "BlockDataset",
    "BlockManifest",
    "BlockNotReadyError",
    "BlockReader",
    "BlockSchemaMismatchError",
    "BlockSplitMismatchError",
    "BlockWriter",
    "CLASS_I_FRAME_SCHEMA",
    "FrameCollection",
    "FrameDataset",
    "FrameSample",
    "PackedView",
    "SamplingUnit",
    "SubsetDataset",
]
