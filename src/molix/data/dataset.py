"""Dataset readers over :class:`~molix.data.cache.PackedCache` files.

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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from molix.data.cache import PackedCache

__all__ = [
    "BaseDataset",
    "CachedDataset",
    "MmapDataset",
    "PackedView",
    "SubsetDataset",
]


@dataclass(frozen=True)
class PackedView:
    """Read-only handle onto a :class:`~molix.data.cache.PackedCache` payload.

    Exposes exactly what the packed-aware collate fast path
    (:func:`molix.data.collate.collate_packed`) needs — the packed payload
    dict, its cumsum pointers / schema, and a local→packed index remap —
    without callers reaching into a dataset's private ``_payload``. A
    :class:`SubsetDataset`'s view carries an ``index_map`` so its local
    indices resolve to packed (global) rows; a full dataset's view uses
    identity mapping. Holds only a reference to the (possibly mmap'd)
    payload — never a tensor copy.

    Args:
        payload: The packed-cache payload mapping (``atoms`` / ``edges`` /
            ``graphs`` / ``scalars`` buckets, ``atom_ptr`` / ``edge_ptr``,
            ``schema``, ``n_samples``).
        index_map: Packed row index per local index, or ``None`` for the
            identity mapping of a full dataset.
    """

    payload: Mapping[str, Any]
    index_map: tuple[int, ...] | None = None

    def map_indices(self, local_indices: Sequence[int]) -> list[int]:
        """Resolve *local_indices* to packed (global) row indices.

        Args:
            local_indices: Indices in this view's local coordinate.

        Returns:
            The corresponding packed-cache row indices.
        """
        if self.index_map is None:
            return [int(i) for i in local_indices]
        return [self.index_map[i] for i in local_indices]

    def remap(self, local_indices: Sequence[int]) -> "PackedView":
        """Return a sub-view restricted/reordered to *local_indices*.

        Composes index maps so nested subsets resolve straight to packed
        rows in a single hop.

        Args:
            local_indices: Indices in this view's local coordinate.

        Returns:
            A new :class:`PackedView` over the same payload whose identity
            is *local_indices* expressed as packed rows.
        """
        return PackedView(self.payload, tuple(self.map_indices(local_indices)))


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseDataset(Dataset[Any], ABC):
    """Abstract base for molix datasets. Subclass this, not ``Dataset``, so
    :class:`~molix.data.DataModule` can consume any implementation."""

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> dict:  # type: ignore[override]
        """Return the ``idx``-th sample as a flat ``dict`` (raw-sample shape)."""
        ...

    def _compute_split_indices(
        self,
        sizes: Sequence[int],
        *,
        seed: int = 42,
    ) -> list[list[int]]:
        """Seeded random N-way split of ``range(len(self))`` into per-size index lists.

        The returned lists partition a prefix of a shuffled permutation —
        ``sum(sizes)`` may be less than ``n`` (trailing samples are dropped)
        but must not exceed it.
        """
        n = len(self)
        total = sum(sizes)
        if total > n:
            raise ValueError(
                f"sum(sizes)={total} exceeds n={n}; "
                "split cannot cover more samples than the source has"
            )
        gen = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=gen).tolist()
        parts: list[list[int]] = []
        offset = 0
        for sz in sizes:
            parts.append(perm[offset : offset + sz])
            offset += sz
        return parts

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
        if ratio is not None:
            cut = int(n * ratio)
            sizes = (cut, n - cut)

        assert sizes is not None
        if sum(sizes) != n:
            raise ValueError(f"sizes must sum to len(self)={n}, got sum={sum(sizes)}")
        return tuple(
            SubsetDataset(self, idx) for idx in self._compute_split_indices(sizes, seed=seed)
        )


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

    def __init__(self, sink: str | Path | PackedCache, *, mmap: bool) -> None:
        self._cache = sink if isinstance(sink, PackedCache) else PackedCache(sink)
        payload = self._cache.load(mmap=mmap)
        self._payload: dict[str, Any] = payload
        self._n_samples: int = int(payload["n_samples"])
        self._task_states: dict[str, Any] = payload.get("task_states", {}) or {}

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> dict:  # type: ignore[override]
        return PackedCache.unpack_sample(self._payload, idx)

    def packed_view(self) -> PackedView:
        """Return a read-only :class:`PackedView` over this cache's payload.

        Identity-mapped (local index == packed row). Consumed by
        :func:`molix.data.collate.collate_packed` to build batches by
        slicing the packed tensors directly.
        """
        return PackedView(self._payload)

    @property
    def sink(self) -> Path:
        """Path to the cache file backing this dataset."""
        return self._cache.sink

    @property
    def task_states(self) -> Mapping[str, Any]:
        """Mapping ``{task_name: state_dict}`` restored from the cache."""
        return self._task_states

    def get_task_state(self, name: str) -> Any:
        """Return the fitted state for ``name`` (raises :class:`KeyError`)."""
        return self._task_states[name]

    def stats(self) -> dict[str, Any]:
        """Fitted :class:`DatasetTask` state keyed by task name.

        Returns a shallow copy of ``{task_name: state_dict}`` for every
        :class:`~molix.data.task.DatasetTask` in the pipeline that
        produced this cache (e.g. ``AtomicDress``). The inner dict is the
        task's own ``state_dict`` payload.

        For dataset-wide connectivity statistics (``avg_num_neighbors``,
        ``max_atoms``, ``max_edges``) use the dedicated properties on this
        class — those are derived directly from the packed cache pointers
        and don't need a dataset task to compute.

        Also callable on :class:`SubsetDataset`: its ``__getattr__``
        forwards unknown attributes to the wrapped dataset, so
        ``subset.stats()`` returns the same mapping as the full dataset's.
        """
        return dict(self._task_states)

    @cached_property
    def avg_num_neighbors(self) -> float:
        """Dataset-wide ⟨|N(i)|⟩ = total_edges / total_atoms.

        Derived from packed cache pointers — no fit pass needed. With
        ``NeighborList(symmetry=True)`` this equals the mean number of
        neighbours per atom (Allegro/MACE normalisation constant).

        Returns ``0.0`` if the cache has no edge or atom pointers.
        """
        atom_ptr = self._payload.get("atom_ptr")
        edge_ptr = self._payload.get("edge_ptr")
        if atom_ptr is None or edge_ptr is None:
            return 0.0
        total_atoms = int(atom_ptr[-1].item())
        if total_atoms <= 0:
            return 0.0
        total_edges = int(edge_ptr[-1].item())
        return total_edges / total_atoms

    def _ptr_counts(self, ptr_key: str) -> torch.Tensor:
        """Per-sample counts ``(n_samples,)`` from the ``ptr_key`` cumsum pointer."""
        ptr = self._payload.get(ptr_key)
        if ptr is None:
            kind = "per-atom" if ptr_key == "atom_ptr" else "per-edge"
            raise ValueError(
                f"cache at {self.sink} has no '{ptr_key}' pointer — it was "
                f"built without {kind} keys. Re-run the pipeline with a task "
                f"that emits {kind} tensors (e.g. NeighborList for edges) to "
                "make these counts available."
            )
        return (ptr[1:] - ptr[:-1]).long()

    @cached_property
    def atom_counts(self) -> torch.Tensor:
        """Per-sample atom counts, shape ``(n_samples,)``, dtype ``long``.

        Derived from the packed ``atom_ptr`` cumsum pointer
        (``ptr[idx+1] - ptr[idx]``) in one vectorized pass — no sample is
        unpacked. Used by
        :class:`~molix.data.sampler.TokenBudgetBatchSampler` to enforce
        per-batch atom budgets.

        Raises:
            ValueError: The cache was built without per-atom keys.
        """
        return self._ptr_counts("atom_ptr")

    @cached_property
    def edge_counts(self) -> torch.Tensor:
        """Per-sample edge counts, shape ``(n_samples,)``, dtype ``long``.

        Derived from the packed ``edge_ptr`` cumsum pointer
        (``ptr[idx+1] - ptr[idx]``) in one vectorized pass — no sample is
        unpacked.

        Raises:
            ValueError: The cache was built without per-edge keys.
        """
        return self._ptr_counts("edge_ptr")

    @cached_property
    def max_atoms(self) -> int:
        """Largest single-sample atom count. ``0`` if no per-atom keys."""
        atom_ptr = self._payload.get("atom_ptr")
        if atom_ptr is None or atom_ptr.numel() < 2:
            return 0
        return int((atom_ptr[1:] - atom_ptr[:-1]).max().item())

    @cached_property
    def max_edges(self) -> int:
        """Largest single-sample edge count. ``0`` if no per-edge keys."""
        edge_ptr = self._payload.get("edge_ptr")
        if edge_ptr is None or edge_ptr.numel() < 2:
            return 0
        return int((edge_ptr[1:] - edge_ptr[:-1]).max().item())


class MmapDataset(_CacheBacked):
    """Memory-mapped cache reader.

    Tensor storages are mmap'd via ``torch.load(mmap=True)`` — page-in on
    demand, shared OS page cache across DataLoader workers, no pickle of
    tensor data after the initial load.

    Args:
        sink: Cache path, or an existing
            :class:`~molix.data.cache.PackedCache`, normally produced by
            :meth:`PipelineSpec.cache <molix.data.pipeline.PipelineSpec.cache>`.
    """

    def __init__(self, sink: str | Path | PackedCache) -> None:
        super().__init__(sink, mmap=True)


class CachedDataset(_CacheBacked):
    """In-memory cache reader — full copy resident in RAM.

    Fine for small datasets (≲10k samples). For larger ones, prefer
    :class:`MmapDataset`.

    Args:
        sink: Cache path, or an existing
            :class:`~molix.data.cache.PackedCache`, normally produced by
            :meth:`PipelineSpec.cache <molix.data.pipeline.PipelineSpec.cache>`.
    """

    def __init__(self, sink: str | Path | PackedCache) -> None:
        super().__init__(sink, mmap=False)


# ---------------------------------------------------------------------------
# SubsetDataset
# ---------------------------------------------------------------------------


class SubsetDataset(BaseDataset):
    """Read-only index view into another :class:`BaseDataset`.

    Produced by :meth:`BaseDataset.split` or constructed directly from
    pre-computed indices (the workflow-driven split-first pattern).

    Connectivity statistics (``avg_num_neighbors``, ``max_atoms``,
    ``max_edges``) are recomputed over ``self._indices`` — a train subset
    reports train-only stats, not the full-dataset values. This is the
    right semantics for Allegro/MACE normalisation, which must not peek at
    val/test.
    """

    def __init__(self, dataset: BaseDataset, indices: list[int]) -> None:
        self._dataset = dataset
        self._indices = indices

    def __getattr__(self, name: str) -> Any:
        """Forward unknown public attributes to the wrapped dataset.

        Lets subset views transparently expose dataset-declared attributes
        (e.g. ``target_schema``) so :class:`~molix.data.DataModule`
        auto-discovery works on splits. Private / dunder names raise
        :class:`AttributeError` instead of forwarding — during unpickling
        ``__setstate__`` has not populated ``__dict__`` yet, so reaching
        for ``self._dataset`` would recurse.

        Args:
            name: Attribute name being looked up.

        Raises:
            AttributeError: ``name`` is private/dunder, or absent on the
                wrapped dataset.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._dataset, name)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict:  # type: ignore[override]
        """Return the sample at the ``idx``-th index of this subset's view.

        Maps the local index through ``self._indices`` and defers to the
        wrapped dataset.
        """
        return self._dataset[self._indices[idx]]

    def packed_view(self) -> PackedView:
        """Return a :class:`PackedView` remapped to this subset's indices.

        Delegates to the wrapped dataset's view and composes the index
        maps, so a (possibly nested) subset's local indices resolve to
        packed rows in one hop.

        Raises:
            AttributeError: The wrapped dataset is not packed-cache-backed
                (has no ``packed_view``); the caller treats this as
                "no fast path" and falls back to per-sample collation.
        """
        return self._dataset.packed_view().remap(self._indices)

    @cached_property
    def avg_num_neighbors(self) -> float:
        """Split-local ⟨|N(i)|⟩ = Σ_{i∈subset} n_edges_i / Σ_{i∈subset} n_atoms_i."""
        payload = getattr(self._dataset, "_payload", None)
        if payload is None:
            return getattr(self._dataset, "avg_num_neighbors", 0.0)
        atom_ptr = payload.get("atom_ptr")
        edge_ptr = payload.get("edge_ptr")
        if atom_ptr is None or edge_ptr is None:
            return 0.0
        idx = torch.as_tensor(self._indices, dtype=torch.long)
        n_atoms = (atom_ptr[idx + 1] - atom_ptr[idx]).sum()
        if int(n_atoms.item()) <= 0:
            return 0.0
        n_edges = (edge_ptr[idx + 1] - edge_ptr[idx]).sum()
        return float(n_edges.item()) / float(n_atoms.item())

    @cached_property
    def atom_counts(self) -> torch.Tensor:
        """Per-sample atom counts over this subset's indices, ``(len(self),)``, dtype ``long``.

        Gathers the wrapped dataset's :attr:`atom_counts` at this view's
        packed indices — defined explicitly (not ``__getattr__``-forwarded)
        so the values are remapped instead of silently returning the
        full-dataset vector. Nested subsets compose naturally.

        Raises:
            ValueError: The wrapped cache was built without per-atom keys
                (propagated from the wrapped dataset's :attr:`atom_counts`).
        """
        idx = torch.as_tensor(self._indices, dtype=torch.long)
        return self._dataset.atom_counts[idx]

    @cached_property
    def edge_counts(self) -> torch.Tensor:
        """Per-sample edge counts over this subset's indices, ``(len(self),)``, dtype ``long``.

        Gathers the wrapped dataset's :attr:`edge_counts` at this view's
        packed indices; see :attr:`atom_counts` for why this is explicit.

        Raises:
            ValueError: The wrapped cache was built without per-edge keys
                (propagated from the wrapped dataset's :attr:`edge_counts`).
        """
        idx = torch.as_tensor(self._indices, dtype=torch.long)
        return self._dataset.edge_counts[idx]

    @cached_property
    def max_atoms(self) -> int:
        """Largest single-sample atom count over this subset's indices.

        Computed from the wrapped dataset's packed ``atom_ptr`` pointers
        restricted to ``self._indices``. Falls back to the wrapped
        dataset's ``max_atoms`` when no packed payload is reachable, and is
        ``0`` when there are no per-atom keys or the subset is empty.
        """
        payload = getattr(self._dataset, "_payload", None)
        if payload is None:
            return getattr(self._dataset, "max_atoms", 0)
        atom_ptr = payload.get("atom_ptr")
        if atom_ptr is None or not self._indices:
            return 0
        idx = torch.as_tensor(self._indices, dtype=torch.long)
        return int((atom_ptr[idx + 1] - atom_ptr[idx]).max().item())

    @cached_property
    def max_edges(self) -> int:
        """Largest single-sample edge count over this subset's indices.

        Computed from the wrapped dataset's packed ``edge_ptr`` pointers
        restricted to ``self._indices``. Falls back to the wrapped
        dataset's ``max_edges`` when no packed payload is reachable, and is
        ``0`` when there are no per-edge keys or the subset is empty.
        """
        payload = getattr(self._dataset, "_payload", None)
        if payload is None:
            return getattr(self._dataset, "max_edges", 0)
        edge_ptr = payload.get("edge_ptr")
        if edge_ptr is None or not self._indices:
            return 0
        idx = torch.as_tensor(self._indices, dtype=torch.long)
        return int((edge_ptr[idx + 1] - edge_ptr[idx]).max().item())
