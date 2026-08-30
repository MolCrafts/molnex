"""Map-style dataset of :data:`FrameCollection` records.

One record is always a :data:`FrameCollection` (``Sequence[Frame]``).
A single geometry is a length-1 collection; a torsion scan or relaxation is a
longer collection. This type does not fetch, split, or mix tasks — mixing
named sources is :class:`~molix.batch.mixture.DataMixture`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from molpy import Frame

from molix.batch.collate import TargetSchema
from molix.store.dataset import BaseDataset
from molix.store.frame_sample import FrameSample

__all__ = ["CLASS_I_FRAME_SCHEMA", "FrameCollection", "FrameDataset"]

#: One record: an ordered sequence of frames sharing one identity (a single
#: geometry is a length-1 collection; a scan or relaxation is longer). molpy
#: exports no such type — the alias is owned here.
FrameCollection = Sequence[Frame]

CLASS_I_FRAME_SCHEMA = TargetSchema(
    graph_level=frozenset({"energy", "mm_energy", "torsion_angle"}),
    atom_level=frozenset({"forces", "atomic_charge"}),
)

_GROUP_KEYS = ("scan_id", "molecule_id", "source_record_id")


def _group_of(frame: Frame) -> str:
    meta = getattr(frame, "meta", None) or {}
    for key in _GROUP_KEYS:
        if key in meta:
            # molrs >= 0.14 wraps meta entries in MetaValue; .value unwraps.
            raw = meta[key]
            return str(getattr(raw, "value", raw))
    return "ungrouped"


def _as_collection(record: FrameCollection) -> tuple[Frame, ...]:
    frames = tuple(record)
    if not frames:
        raise ValueError("FrameCollection must contain at least one Frame")
    return frames


class FrameDataset(BaseDataset):
    """Indexed FrameCollections plus the ``(task, group)`` labels the sampler reads.

    Each index is one collection. A scan is one index, not one index per frame.

    Args:
        records: One FrameCollection per index.
        task_labels: Per-collection task name, same length as *records*.
        group_ids: Per-collection sampling group, same length as *records*.
        target_schema: How :class:`~molix.batch.collate.collate_molecules`
            routes targets. Defaults to :data:`CLASS_I_FRAME_SCHEMA`.
    """

    def __init__(
        self,
        records: Sequence[FrameCollection],
        *,
        task_labels: Sequence[str],
        group_ids: Sequence[str],
        target_schema: TargetSchema | None = None,
    ) -> None:
        collections = [_as_collection(record) for record in records]
        if not (len(collections) == len(task_labels) == len(group_ids)):
            raise ValueError(
                f"records ({len(collections)}), task_labels ({len(task_labels)}) and "
                f"group_ids ({len(group_ids)}) must be the same length"
            )
        self._records = collections
        self.task_labels: tuple[str, ...] = tuple(task_labels)
        self.group_ids: tuple[str, ...] = tuple(str(g) for g in group_ids)
        self.target_schema = target_schema or CLASS_I_FRAME_SCHEMA
        self.atom_counts = torch.tensor(
            [
                sum(FrameSample(frame).n_atoms for frame in collection)
                for collection in self._records
            ],
            dtype=torch.long,
        )
        self.edge_counts = torch.zeros_like(self.atom_counts)

    @classmethod
    def from_records(
        cls,
        records: Sequence[FrameCollection],
        *,
        task: str,
        group_of: Callable[[Frame], str] | None = None,
        target_schema: TargetSchema | None = None,
    ) -> FrameDataset:
        """Label every collection with one *task*. Does not flatten frames."""
        key = group_of or _group_of
        collections = [_as_collection(record) for record in records]
        groups = [key(collection[0]) for collection in collections]
        return cls(
            collections,
            task_labels=(task,) * len(collections),
            group_ids=groups,
            target_schema=target_schema,
        )

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> FrameCollection:
        return self._records[idx]
