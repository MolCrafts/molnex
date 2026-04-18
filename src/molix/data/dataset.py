"""Dataset readers over :mod:`molix.data.cache` files.

Two concrete implementations, differing only in how the underlying
``torch.save`` file is loaded:

+------------------+---------+---------------------------------------------+
| Class            | mmap?   | When to use                                 |
+==================+=========+=============================================+
| :class:`MmapDataset`     | yes     | Large caches (10k+ samples) or             |
|                          |         | num_workers > 0 — tensor storages are      |
|                          |         | mmap'd, OS page cache is shared across     |
|                          |         | worker processes.                          |
+------------------+---------+---------------------------------------------+
| :class:`CachedDataset`   | no      | Small caches (<10k) or when the cache      |
|                          |         | fits in RAM and you want to avoid the      |
|                          |         | per-access mmap page-fault overhead.       |
+------------------+---------+---------------------------------------------+

:class:`SubsetDataset` wraps either with an index list for split views —
produced by :meth:`BaseDataset.split` or built explicitly from pre-computed
indices (as the workflow pattern recommends).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from molix.data.cache import load as _load_cache
from molix.data.cache import unpack_one as _unpack_one


__all__ = ["BaseDataset", "MmapDataset", "CachedDataset", "SubsetDataset"]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseDataset(Dataset[Any], ABC):
    """Abstract base for molix datasets. Subclass this, not ``Dataset``, so
    :class:`~molix.data.DataModule` can consume any implementation."""

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> dict: ...  # type: ignore[override]

    def split(
        self,
        ratio: float | None = None,
        *,
        sizes: tuple[int, ...] | None = None,
        seed: int = 42,
    ) -> tuple["SubsetDataset", ...]:
        """Shuffled N-way split as index views — no data copy."""
        if (ratio is None) == (sizes is None):
            raise ValueError("Provide exactly one of `ratio` or `sizes`.")

        n = len(self)
        gen = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=gen).tolist()

        if ratio is not None:
            cut = int(n * ratio)
            return (SubsetDataset(self, perm[:cut]), SubsetDataset(self, perm[cut:]))

        assert sizes is not None
        if sum(sizes) != n:
            raise ValueError(f"sizes must sum to len(self)={n}, got sum={sum(sizes)}")
        parts: list[SubsetDataset] = []
        offset = 0
        for sz in sizes:
            parts.append(SubsetDataset(self, perm[offset : offset + sz]))
            offset += sz
        return tuple(parts)


# ---------------------------------------------------------------------------
# Cache-backed datasets
# ---------------------------------------------------------------------------


class _CacheBacked(BaseDataset):
    """Shared implementation: load a cache file, expose samples + task_states.

    The cache is in packed layout (see :func:`molix.data.cache._pack_samples`)
    — a fixed number of big concat tensors plus cumsum pointers. ``__getitem__``
    reconstructs a sample dict on the fly by slicing into those tensors, so the
    per-item cost is O(n_keys) and uses views (no storage copy when ``mmap=True``).
    """

    def __init__(self, sink: str | Path, *, mmap: bool) -> None:
        self._sink = Path(sink)
        payload = _load_cache(self._sink, mmap=mmap)
        self._payload: dict[str, Any] = payload
        self._n_samples: int = int(payload["n_samples"])
        self._task_states: dict[str, Any] = payload.get("task_states", {}) or {}

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> dict:  # type: ignore[override]
        return _unpack_one(self._payload, idx)

    @property
    def sink(self) -> Path:
        """Path to the cache file backing this dataset."""
        return self._sink

    @property
    def task_states(self) -> Mapping[str, Any]:
        """Mapping ``{task_name: state_dict}`` restored from the cache."""
        return self._task_states

    def get_task_state(self, name: str) -> Any:
        """Return the fitted state for ``name`` (raises :class:`KeyError`)."""
        return self._task_states[name]


class MmapDataset(_CacheBacked):
    """Memory-mapped cache reader.

    Tensor storages are mmap'd via ``torch.load(mmap=True)`` — page-in on
    demand, shared OS page cache across DataLoader workers, no pickle of
    tensor data after the initial load.

    Args:
        sink: Cache file path written by :func:`molix.data.cache.cache`
            or :func:`molix.data.cache.save`.
    """

    def __init__(self, sink: str | Path) -> None:
        super().__init__(sink, mmap=True)


class CachedDataset(_CacheBacked):
    """In-memory cache reader — full copy resident in RAM.

    Fine for small datasets (≲10k samples). For larger ones, prefer
    :class:`MmapDataset`.

    Args:
        sink: Cache file path written by :func:`molix.data.cache.cache`
            or :func:`molix.data.cache.save`.
    """

    def __init__(self, sink: str | Path) -> None:
        super().__init__(sink, mmap=False)


# ---------------------------------------------------------------------------
# SubsetDataset
# ---------------------------------------------------------------------------


class SubsetDataset(BaseDataset):
    """Read-only index view into another :class:`BaseDataset`.

    Produced by :meth:`BaseDataset.split` or constructed directly from
    pre-computed indices (the workflow-driven split-first pattern).
    """

    def __init__(self, dataset: BaseDataset, indices: list[int]) -> None:
        self._dataset = dataset
        self._indices = indices

    def __getattr__(self, name: str) -> Any:
        # Forward unknown attrs (e.g. dataset-declared target_schema) to the
        # wrapped dataset so DataModule auto-discovery works on subsets.
        # Guard private/dunder names — during unpickling __setstate__ has not
        # populated __dict__ yet, so accessing self._dataset would recurse.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._dataset, name)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict:  # type: ignore[override]
        return self._dataset[self._indices[idx]]
