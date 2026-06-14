"""Checkpoint storage backend abstraction.

The default backend uses ``torch.save`` / ``torch.load`` which is suitable
for single-GPU and DDP training.  For FSDP, a future ``DCPBackend`` can
implement the same protocol using ``torch.distributed.checkpoint``.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Protocol

import torch


class CheckpointBackend(Protocol):
    """Protocol for checkpoint save/load backends."""

    def save(self, state_dict: dict[str, Any], path: str | Path) -> None:
        """Save *state_dict* to *path*."""
        ...

    def load(self, path: str | Path, *, map_location: Any = None) -> dict[str, Any]:
        """Load a state_dict from *path*."""
        ...


class TorchSaveBackend:
    """Default checkpoint backend using ``torch.save`` / ``torch.load``.

    Writes are atomic: data is first written to a temporary file in the
    same directory, then renamed to the target path.  This prevents
    half-written checkpoints if the process is killed mid-save.
    """

    def save(self, state_dict: dict[str, Any], path: str | Path) -> None:
        """Atomically write *state_dict* to *path* via ``torch.save``.

        Creates parent directories as needed, serialises to a temporary
        file in the same directory, then ``os.replace``-renames it onto
        *path* so observers never see a half-written checkpoint. The
        temporary file is removed on any failure.

        Args:
            state_dict: Object graph to serialise (model / optimizer /
                state tensors).
            path: Destination checkpoint path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: tmp file + rename
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            torch.save(state_dict, tmp)
            os.replace(tmp, path)  # atomic on POSIX
        except BaseException:
            os.close(fd)
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        else:
            os.close(fd)

    def load(self, path: str | Path, *, map_location: Any = None) -> dict[str, Any]:
        """Load a state_dict written by :meth:`save` via ``torch.load``.

        Uses ``weights_only=False`` so non-tensor state (e.g. enum stages,
        Python scalars) round-trips; only load checkpoints you trust.

        Args:
            path: Checkpoint path to read.
            map_location: ``torch.load`` device-remapping argument (e.g.
                ``"cpu"`` or a device); ``None`` keeps the saved devices.

        Returns:
            The deserialised state_dict.
        """
        return torch.load(path, map_location=map_location, weights_only=False)


# Sharded (FSDP) training will need a backend built on
# ``torch.distributed.checkpoint`` (``dcp.save`` / ``dcp.load`` with a
# ``checkpoint_id``), which handles state sharding, resharding, and
# cross-rank coordination. Not implemented yet — add when FSDP lands.
