"""Append-only training-event IO for MolNex training runs.

The package exposes two concrete classes:

* :class:`JournalWriter` — writes per-record events into a sharded
  Zarr v3 group on disk. Driven by
  :class:`molix.hooks.journal.JournalHook` during training.
* :class:`JournalReader` — reads back a run's record stream as plain
  ``dict`` instances; downstream notebooks and dashboards consume
  this surface directly.

Both classes are first-party and tightly coupled — there is no
schema dataclass and no Protocol indirection. Five reserved
``type`` strings (``"scalar"``, ``"histogram"``, ``"text"``,
``"image_ref"``, ``"json"``) are dispatched directly inside
:meth:`JournalWriter.append` via a ``match`` block.

The on-disk layout (chunks of 1024 rows, shards of 1,048,576 rows)
is HPC-inode-discipline-load-bearing — see :mod:`molix.io.writer`
docstring for full rationale.
"""

from __future__ import annotations

from molix.io.reader import JournalReader
from molix.io.writer import JournalWriter

__all__ = [
    "JournalReader",
    "JournalWriter",
]
