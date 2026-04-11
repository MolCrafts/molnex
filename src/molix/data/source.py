"""DataSource protocol and built-in implementations."""

from __future__ import annotations

import hashlib
from typing import Any, Protocol, runtime_checkable

Sample = dict[str, Any]


@runtime_checkable
class DataSource(Protocol):
    """Protocol for raw molecular data providers.

    A DataSource is responsible ONLY for providing raw samples.
    No preprocessing, no neighbor lists, no transforms.

    Each sample is a plain dict, typically containing at minimum
    ``Z`` (atomic numbers) and ``pos`` (positions).
    """

    @property
    def source_id(self) -> str:
        """Unique, deterministic identifier for cache key computation.

        Must be stable across runs for the same data.
        """
        ...

    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> Sample: ...


class InMemorySource:
    """DataSource wrapping an existing list of samples."""

    def __init__(self, samples: list[Sample], *, name: str = "memory") -> None:
        self._samples = samples
        self._name = name

    @property
    def source_id(self) -> str:
        return f"memory:{self._name}:{len(self._samples)}"

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Sample:
        return self._samples[idx]


class SubsetSource:
    """DataSource exposing a subset of another source by index list."""

    def __init__(self, source: DataSource, indices: list[int]) -> None:
        self._source = source
        self._indices = indices

    @property
    def source_id(self) -> str:
        idx_hash = hashlib.sha256(str(sorted(self._indices)).encode()).hexdigest()[:12]
        return f"{self._source.source_id}:subset={idx_hash}"

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Sample:
        return self._source[self._indices[idx]]
