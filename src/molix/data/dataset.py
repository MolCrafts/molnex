"""Thin Dataset wrapper over pre-computed samples.

``__getitem__`` is a pure list index — O(1), zero computation.
All heavy lifting happens in :meth:`PipelineSpec.prepare`.
"""

from __future__ import annotations

from torch.utils.data import Dataset


class CachedDataset(Dataset):
    """Wraps a ``list[dict]`` as a ``torch.utils.data.Dataset``.

    This is intentionally trivial.  Samples are pre-computed and held
    in memory; ``__getitem__`` is a plain list lookup.
    """

    def __init__(self, samples: list[dict]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        return self._samples[idx]
