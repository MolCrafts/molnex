"""A Dataset of named sources mixed at the DataModule.

A :class:`DataMixture` is a map-style :class:`~molix.store.dataset.BaseDataset`
over ``sources: Mapping[str, Dataset]``. The source key **is** the task name.
Each source index is one sampling unit (one :data:`~molix.store.frame_dataset.FrameCollection`).
The mix policy lives on the same object so :class:`~molix.batch.datamodule.DataModule`
can take it as ``train_dataset``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import torch

from molix.batch.collate import DEFAULT_TARGET_SCHEMA, TargetSchema
from molix.batch.sampler import TaskMixtureBatchSampler
from molix.store.dataset import BaseDataset
from molix.store.frame_sample import FrameSample

__all__ = ["DataMixture"]


class _MapStyle(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> Any: ...


def _source_atom_counts(source: _MapStyle) -> torch.Tensor:
    counts = getattr(source, "atom_counts", None)
    if counts is not None:
        return torch.as_tensor(counts, dtype=torch.long)
    atoms: list[int] = []
    for idx in range(len(source)):
        record = source[idx]
        if isinstance(record, dict):
            atoms.append(int(record["Z"].shape[0]))
            continue
        n_atoms = 0
        for item in record:
            if isinstance(item, dict):
                n_atoms += int(item["Z"].shape[0])
            else:
                n_atoms += FrameSample(item).n_atoms
        atoms.append(n_atoms)
    return torch.tensor(atoms, dtype=torch.long)


def _source_edge_counts(source: _MapStyle, n: int) -> torch.Tensor:
    counts = getattr(source, "edge_counts", None)
    if counts is not None:
        return torch.as_tensor(counts, dtype=torch.long)
    return torch.zeros(n, dtype=torch.long)


class DataMixture(BaseDataset):
    """Named sources as one Dataset, plus the task mixture.

    Args:
        sources: Task name → map-style dataset. Each ``source[i]`` is one
            sampling unit. The key is the task label the sampler draws.
        mixture: Task → probability, same contract as
            :class:`~molix.batch.sampler.TaskMixtureBatchSampler`. Keys
            must match *sources* exactly.
        max_atoms: Per-batch atom budget.
        max_edges: Optional per-batch edge budget.
        target_schema: How collate routes targets. Defaults to the first
            source's ``target_schema``, else the generic default.
    """

    def __init__(
        self,
        sources: Mapping[str, _MapStyle],
        *,
        mixture: Mapping[str, float],
        max_atoms: int,
        max_edges: int | None = None,
        target_schema: TargetSchema | None = None,
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
        self.max_edges = max_edges
        if target_schema is None:
            first = next(iter(self.sources.values()))
            target_schema = getattr(first, "target_schema", DEFAULT_TARGET_SCHEMA)
        self.target_schema = target_schema

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
            atom_parts.append(_source_atom_counts(source))
            edge_parts.append(_source_edge_counts(source, n))
            offsets.append(offsets[-1] + n)
        self.task_labels: tuple[str, ...] = tuple(labels)
        self.group_ids: tuple[str, ...] = tuple(groups)
        self.atom_counts = torch.cat(atom_parts)
        self.edge_counts = torch.cat(edge_parts)
        self._names = tuple(self.sources)
        self._offsets = tuple(offsets)

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

        *dataset* is this object when :class:`~molix.batch.datamodule.DataModule`
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
            max_edges=self.max_edges,
            seed=seed,
        )
