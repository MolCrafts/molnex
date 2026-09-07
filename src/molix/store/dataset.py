"""Map-style datasets over a **packed payload**.

A packed payload is the shape :func:`~molix.batch.collate.collate_packed`
consumes: ``atoms`` / ``edges`` / ``graphs`` / ``scalars`` buckets of concat
tensors plus the ``atom_ptr`` / ``edge_ptr`` cumsum pointers locating each
sample's rows. :class:`~molix.store.block.BlockReader` produces one from a
``molix.block/1`` block; a caller can build one in memory.

:class:`~molix.store.block_view.BlockDataset` is the map-style reader over a
sealed block's payload; :class:`SubsetDataset` is an index view into any
:class:`BaseDataset`.
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

__all__ = [
    "BaseDataset",
    "PackedView",
    "SubsetDataset",
]


@dataclass(frozen=True)
class PackedView:
    """Read-only handle onto a packed payload.

    Exposes exactly what the packed-aware collate fast path
    (:func:`molix.batch.collate.collate_packed`) needs — the packed payload
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
    :class:`~molix.batch.DataModule` can consume any implementation."""

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        """Return the ``idx``-th record.

        Packed readers return a flat ``dict``. Frame readers return a
        :data:`~molix.store.frame_dataset.FrameCollection`.
        """
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


def _reject_misfiled_edges(payload: Mapping[str, Any], sink: Path | str) -> None:
    """Raise if *payload* has no ``edge_ptr`` because its edges were misfiled.

    A packed cache written before edge keys were routed by identity classified
    ``edge_index`` as a per-atom key whenever ``n_edges == n_atoms`` held for
    every sample, which also suppressed ``edge_ptr``. Its edges are then
    unreadable: ``collate_packed`` emits an empty edges namespace and
    ``avg_num_neighbors`` would report ``0.0``. A cache with no edge keys at
    all is a different, legitimate state and is left alone.

    Args:
        payload: The packed-cache payload mapping.
        sink: Path of the backing cache file, for the error message.

    Raises:
        ValueError: ``edge_index`` is packed under the ``atom`` schema kind.
    """
    spec = payload.get("schema", {}).get("edge_index")
    if spec is None or spec[0] != "atom":
        return
    raise ValueError(
        f"cache at {sink} packs 'edge_index' as a per-atom key and has no "
        "'edge_ptr' pointer — it was written before per-edge keys were routed "
        "by identity, so its edges are unreadable (this happens when "
        "n_edges == n_atoms for every sample). Delete the stale cache and "
        "re-run the pipeline."
    )


# ---------------------------------------------------------------------------
# Cache-backed datasets
# ---------------------------------------------------------------------------


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
        (e.g. ``target_schema``) so :class:`~molix.batch.DataModule`
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

    def __getitem__(self, idx: int) -> dict:
        """Return the sample at the ``idx``-th index of this subset's view.

        Maps the local index through ``self._indices`` and defers to the
        wrapped dataset.
        """
        return self._dataset[self._indices[idx]]

    def packed_view(self) -> PackedView:
        """Return a :class:`PackedView` whose rows are this subset's rows.

        Delegates to the wrapped dataset's view and composes the index maps, so
        a (possibly nested) subset's local indices resolve to packed rows in one
        hop. Defined explicitly rather than left to :meth:`__getattr__`, which
        would forward to the parent and hand back an **unmapped** view.

        Raises:
            AttributeError: The wrapped dataset serves no packed view; the
                caller treats that as "no fast path".
        """
        return self._dataset.packed_view().remap(self._indices)

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
