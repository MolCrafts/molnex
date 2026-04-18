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

    def load(
        self, path: str | Path, *, map_location: Any = None
    ) -> dict[str, Any]:
        """Load a state_dict from *path*."""
        ...


class TorchSaveBackend:
    """Default checkpoint backend using ``torch.save`` / ``torch.load``.

    Writes are atomic: data is first written to a temporary file in the
    same directory, then renamed to the target path.  This prevents
    half-written checkpoints if the process is killed mid-save.
    """

    def save(self, state_dict: dict[str, Any], path: str | Path) -> None:
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

    def load(
        self, path: str | Path, *, map_location: Any = None
    ) -> dict[str, Any]:
        return torch.load(path, map_location=map_location, weights_only=False)


# Future: DCPBackend for FSDP
#
# class DCPBackend:
#     """Checkpoint backend using torch.distributed.checkpoint.
#
#     Required for FSDP training where model/optimizer state is sharded.
#     Uses torch.distributed.checkpoint.save/load which handles
#     state sharding, resharding, and cross-rank coordination.
#     """
#
#     def save(self, state_dict, path):
#         import torch.distributed.checkpoint as dcp
#         dcp.save(state_dict, checkpoint_id=str(path))
#
#     def load(self, path, *, map_location=None):
#         import torch.distributed.checkpoint as dcp
#         state_dict = {}
#         dcp.load(state_dict, checkpoint_id=str(path))
#         return state_dict
