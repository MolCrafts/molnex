"""Index views over a dataset.

:class:`BaseDataset` is the reader protocol (``__len__`` / ``__getitem__`` plus
:meth:`~BaseDataset.split`); :class:`SubsetDataset` is an index view into any of
them. The storage→model conversion itself lives with the format, in
:class:`FrameDataset`, whose records hand the model
its one structure — a ``TensorDict``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Final

import numpy as np
import torch
from molpy import Frame
from torch.utils.data import Dataset

__all__ = [
    "BaseDataset",
    "SubsetDataset",
]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseDataset(Dataset[Any], ABC):
    """Abstract base for molix datasets. Subclass this, not ``Dataset``, so
    :class:`~molix.data.DataModule` can consume any implementation."""

    #: Tables this container can count **without materialising a sample**,
    #: mapped to the attribute holding the per-row vector.
    #:
    #: A container that stores its own row lengths — a packed column, a count
    #: sidecar — declares them here and is done: no method to write, no scan
    #: to avoid. Empty (the default) means every table costs one
    #: ``__getitem__`` per row, which on a four-million-row source is a
    #: quarter of an hour.
    #:
    #: A table that must be *measured* rather than read (an ASE-style LMDB
    #: has to decode a record to learn its length) does not belong here —
    #: implement :meth:`count_rows` instead, where the measurement can be
    #: cached.
    counted_tables: ClassVar[Mapping[str, str]] = {}

    def count_rows(self, table: str) -> torch.Tensor | np.ndarray:
        """Measure the per-row lengths of *table*. Override when it is cheap.

        The hook for containers that can do better than one ``__getitem__``
        per row but cannot simply read an attribute — a source that must
        decode to count, and wants to cache the result. Prefer
        :attr:`counted_tables` when the vector is already held.

        This is deliberately **a verb**. ``Block.n_rows`` is an attribute: the
        length of one block, free. This is a request over a whole dataset
        whose cost ranges from a column read to a fifteen-minute scan, and a
        caller reading the two spellings side by side should not be able to
        mistake one for the other.

        Args:
            table: Frame block name (``atoms``, ``bonds``, ``angles``, …).

        Returns:
            Per-index table lengths, ``(N,)``.

        Raises:
            NotImplementedError: This container has no cheaper route; the
                caller should fall back to materialising each sample.
        """
        raise NotImplementedError

    @staticmethod
    def nrows_of(source: Any, table: str) -> torch.Tensor:
        """Per-index length of *table* over *source*, ``(N,)`` ``torch.long``.

        The one entry point every batch budget goes through, and the reason it
        takes an arbitrary *source*: a ``DataModule`` is handed whatever
        dataset the caller owns — MolHub's, a plain map-style object — so this
        path cannot assume :class:`BaseDataset`.

        Three routes, cheapest first:

        1. :attr:`counted_tables` names an attribute already holding the
           vector — a packed column, a sidecar. A read.
        2. :meth:`count_rows` is implemented — the container measures, and may
           cache what it measured.
        3. Neither: every sample is materialised and its ``Block.n_rows`` read.
           Correct, and the slowest thing in a data path.

        Args:
            source: Any indexable dataset.
            table: Frame block name.

        Returns:
            Per-index table lengths (dimensionless).

        Raises:
            ValueError: *table* is ``pairs``, or another table this dataset
                does not declare. The cutoff graph is a function of positions
                and a radius, so it is built downstream by
                :class:`~molix.neighbor.NeighborList`, never carried here.
            TypeError: An index is not a :class:`~molpy.Frame` or a sequence
                of frames.
        """
        declared = getattr(type(source), "counted_tables", None) or {}
        if table in declared:
            return torch.as_tensor(getattr(source, declared[table]), dtype=torch.long)
        measure = getattr(source, "count_rows", None)
        if callable(measure):
            try:
                return torch.as_tensor(measure(table), dtype=torch.long)
            except NotImplementedError:
                pass
        return _counts(source, table)

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        """Return the ``idx``-th record.

        Packed readers return a flat ``dict``. Frame readers return a
        ``Sequence[Frame]``.
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


# ---------------------------------------------------------------------------
# SubsetDataset
# ---------------------------------------------------------------------------


class SubsetDataset(BaseDataset):
    """Read-only index view into another :class:`BaseDataset`.

    Produced by :meth:`BaseDataset.split` or constructed directly from
    pre-computed indices (the workflow-driven split-first pattern).

    Every per-index reading — table lengths through :meth:`count_rows`, the
    labels at :attr:`task_labels` / :attr:`group_ids` — is the wrapped
    dataset's, gathered at this view's indices: a train view reports
    train-only values, and each is ``len(self)`` long.
    """

    def __init__(self, dataset: BaseDataset, indices: list[int]) -> None:
        self._dataset = dataset
        self._indices = indices

    @property
    def dataset(self) -> BaseDataset:
        """The wrapped dataset this view indexes."""
        return self._dataset

    @property
    def indices(self) -> list[int]:
        """Parent indices this view exposes, in view order."""
        return self._indices

    @property
    def task_labels(self) -> tuple[str, ...]:
        """The wrapped dataset's per-index task labels, gathered at this view's indices.

        The wrapped dataset is a labelled one — :class:`~molix.data.FrameDataset`
        or :class:`~molix.data.DataMixture` — and the result is ``len(self)`` long.
        """
        labels = self._dataset.task_labels
        return tuple(labels[i] for i in self._indices)

    @property
    def group_ids(self) -> tuple[str, ...]:
        """The wrapped dataset's per-index sampling groups, gathered at this view's indices."""
        groups = self._dataset.group_ids
        return tuple(groups[i] for i in self._indices)

    def count_rows(self, table: str) -> torch.Tensor:
        """The parent's lengths, gathered at this subset's indices.

        Read through the parent rather than through ``self[i]`` so a parent that
        answers from a count sidecar keeps doing so — a split must not turn an
        O(1) metadata read into materialising every sample.
        """
        counts = BaseDataset.nrows_of(self.dataset, table)
        return counts[torch.as_tensor(self._indices, dtype=torch.long)]

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict:
        """Return the sample at the ``idx``-th index of this subset's view.

        Maps the local index through ``self._indices`` and defers to the
        wrapped dataset.
        """
        return self._dataset[self._indices[idx]]


#: Valence tables name their per-unit count column after the singular family
#: (``bonds`` -> ``bond_n``), which is how a count sidecar spells them.
_FAMILY_OF_TABLE: Final[dict[str, str]] = {
    "bonds": "bond",
    "angles": "angle",
    "propers": "proper",
    "impropers": "improper",
}


def _no_table(source: Any, table: str) -> ValueError:
    """Refuse a fabricated table-length vector."""
    if table == "pairs":
        return ValueError(
            f"{type(source).__name__} has no {table!r} table: the cutoff graph is a "
            "function of positions and a radius, so it is built downstream by "
            "NeighborList, never carried by a dataset. Budget batches on frame "
            "table nrows instead."
        )
    return ValueError(
        f"{type(source).__name__} has no {table!r} table: no Frame block of that name is declared."
    )


def _record_nrows(sample: Any, table: str) -> int:
    """``sample[table].n_rows`` of one Frame, or the sum over a sequence."""
    if isinstance(sample, Frame):
        try:
            return int(sample[table].n_rows)
        except KeyError as exc:
            raise _no_table(sample, table) from exc
    if isinstance(sample, (tuple, list)):
        return sum(_record_nrows(item, table) for item in sample)
    raise TypeError(
        f"{type(sample).__name__} is not a Frame or a sequence of Frames; "
        f"packing reads frame[{table!r}].n_rows"
    )


def _sidecar_counts(source: Any, column: Any, table: str) -> Any:
    """Per-unit count column of *table* from a count sidecar.

    A source that can answer ``n_<table>`` (or ``<family>_n``) without
    materialising a sample lets the sampler size every batch with **zero**
    ``__getitem__`` calls — the property
    ``TestSamplerNoUnpack::test_zero_getitem_calls_during_construction_and_iteration``
    pins. That is a dataset capability, not a property of any one storage
    backend.
    """
    names = [f"n_{table}"]
    family = _FAMILY_OF_TABLE.get(table)
    if family is not None:
        names.append(f"{family}_n")
    for name in names:
        try:
            return column(name)
        except KeyError:
            continue
    raise _no_table(source, table)


def _counts(source: Any, table: str) -> torch.Tensor:
    """Per-row lengths by materialising every sample — the fallback route.

    Reached only when a source declares no :attr:`BaseDataset.counted_tables`
    entry and implements no :meth:`BaseDataset.count_rows`.
    """
    column = getattr(source, "column", None)
    frame_mol = getattr(source, "frame_mol", None)
    if callable(column) and frame_mol is not None:
        values = torch.as_tensor(_sidecar_counts(source, column, table), dtype=torch.long)
        return values[torch.as_tensor(frame_mol, dtype=torch.long)]

    rows = torch.empty(len(source), dtype=torch.long)
    for i in range(len(source)):
        rows[i] = _record_nrows(source[i], table)
    return rows
