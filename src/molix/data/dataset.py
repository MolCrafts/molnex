"""Thin Dataset wrapper over pre-computed samples.

``__getitem__`` is a pure list index — O(1), zero computation.
All heavy lifting happens in :meth:`PipelineSpec.prepare`.
"""

from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset


class CachedDataset(Dataset[Any]):
    """Wraps a ``list[dict]`` as a ``torch.utils.data.Dataset``.

    This is intentionally trivial.  Samples are pre-computed and held
    in memory; ``__getitem__`` is a plain list lookup.
    """

    def __init__(self, samples: list[dict]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:  # type: ignore[override]
        return self._samples[idx]
