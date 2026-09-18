"""A Dataset of named sources mixed at the DataModule.

A :class:`DataMixture` is a map-style :class:`~molix.data.dataset.BaseDataset`
over ``sources: Mapping[str, Dataset]``. The source key **is** the task name.
Each source index is one sampling unit (one ``Sequence[Frame]``).
The mix policy lives on the same object so :class:`~molix.data.datamodule.DataModule`
can take it as ``train_dataset``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import torch

from molix.data.dataset import BaseDataset
from molix.data.sampler import TaskMixtureBatchSampler

__all__ = ["DataMixture"]


class _MapStyle(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> Any: ...


def _snapshot_nrows(source: _MapStyle, table: str) -> torch.Tensor:
    """Clone :meth:`~molix.data.dataset.BaseDataset.nrows_of` to an owned CPU long tensor."""
    snapshot = BaseDataset.nrows_of(source, table).detach().cpu().clone()
    if snapshot.dtype != torch.long:
        snapshot = snapshot.to(torch.long)
    if len(snapshot) != len(source):
        raise ValueError(
            f"{type(source).__name__} {table!r} nrows length {len(snapshot)} "
            f"does not match source length {len(source)}"
        )
    return snapshot


class DataMixture(BaseDataset):
    """Named sources as one Dataset, plus the task mixture.

    Args:
        sources: Task name → map-style dataset. Each ``source[i]`` is one
            sampling unit. The key is the task label the sampler draws.
        mixture: Task → probability, same contract as
            :class:`~molix.data.sampler.TaskMixtureBatchSampler`. Keys
            must match *sources* exactly.
        max_atoms: Per-batch atom budget.
        max_pairs: Optional per-batch edge budget.
    """

    def __init__(
        self,
        sources: Mapping[str, _MapStyle],
        *,
        mixture: Mapping[str, float],
        max_atoms: int,
        max_pairs: int | None = None,
    ) -> None:
        if not sources:
            raise ValueError("DataMixture needs at least one source")
        source_names = set(sources)
        mix_names = set(mixture)
        if source_names != mix_names:
            raise ValueError(
                "DataMixture sources and mixture must name the same tasks; "
                f"sources={sorted(source_names)}, mixture={sorted(mix_names)}"
            )
        self.sources = {str(name): sources[name] for name in sources}
        self.mixture = dict(mixture)
        self.max_atoms = int(max_atoms)
        self.max_pairs = max_pairs

        labels: list[str] = []
        groups: list[str] = []
        atom_parts: list[torch.Tensor] = []
        edge_parts: list[torch.Tensor] = []
        offsets = [0]
        for name, source in self.sources.items():
            n = len(source)
            if n == 0:
                raise ValueError(f"DataMixture source {name!r} is empty")
            labels.extend([name] * n)
            groups.extend(f"{name}:{idx}" for idx in range(n))
            atom_parts.append(_snapshot_nrows(source, "atoms"))
            if max_pairs is not None:
                edge_parts.append(_snapshot_nrows(source, "pairs"))
            offsets.append(offsets[-1] + n)
        self.task_labels: tuple[str, ...] = tuple(labels)
        self.group_ids: tuple[str, ...] = tuple(groups)
        self._table_nrows = {"atoms": torch.cat(atom_parts)}
        if edge_parts:
            self._table_nrows["pairs"] = torch.cat(edge_parts)  # noqa: E501 — kept: read by count_rows below
        self._names = tuple(self.sources)
        self._offsets = tuple(offsets)

    def count_rows(self, table: str) -> torch.Tensor:
        """Every source's lengths, concatenated in source order.

        A source need not be a :class:`~molix.data.dataset.BaseDataset` — the
        mix takes whatever map-style object the caller owns — so each is read
        through :meth:`~molix.data.dataset.BaseDataset.nrows_of`.
        """
        return torch.cat([BaseDataset.nrows_of(src, table) for src in self.sources.values()])

    def __len__(self) -> int:
        return self._offsets[-1]

    def __getitem__(self, idx: int) -> Any:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        for name, start, stop in zip(self._names, self._offsets, self._offsets[1:], strict=True):
            if start <= idx < stop:
                return self.sources[name][idx - start]
        raise IndexError(idx)

    def batch_sampler(self, dataset: BaseDataset, *, seed: int) -> TaskMixtureBatchSampler:
        """Plan this epoch over this mixture.

        *dataset* is this object when :class:`~molix.data.datamodule.DataModule`
        builds the train loader.
        """
        del dataset
        return TaskMixtureBatchSampler(
            self,
            indices=range(len(self)),
            task_labels=self.task_labels,
            group_ids=self.group_ids,
            mixture=self.mixture,
            max_atoms=self.max_atoms,
            max_pairs=self.max_pairs,
            seed=seed,
        )
