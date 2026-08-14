"""Storage layer for MolNex: bytes on disk ↔ tensors in memory.

Surfaces:

* **Packed cache** — :class:`PackedCache` writes one atomic ``.pt`` file per
  cache holding concatenated per-atom / per-edge / per-graph tensors plus
  ``atom_ptr`` / ``edge_ptr`` cumsum pointers. Run-scoped scratch, not a
  persistence format; see the module docstring for the inode-budget argument
  against ``TensorDict.memmap_()``.
* **Readers** — :class:`MmapDataset` / :class:`CachedDataset` read a
  :class:`PackedCache` file (mmap'd or fully resident), :class:`SubsetDataset`
  is an index view over either, and :class:`BaseDataset` is the shared reader
  protocol. :class:`PackedView` is the read-only handle the packed collate fast
  path consumes.

Nothing here builds batches, samples indices or acquires data — those live in
their own packages.
"""

from __future__ import annotations

from molix.store.cache import PackedCache
from molix.store.dataset import (
    BaseDataset,
    CachedDataset,
    MmapDataset,
    PackedView,
    SubsetDataset,
)

__all__ = [
    "BaseDataset",
    "CachedDataset",
    "MmapDataset",
    "PackedCache",
    "PackedView",
    "SubsetDataset",
]
