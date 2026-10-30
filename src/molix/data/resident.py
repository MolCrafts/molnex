"""Validation batches built once and kept on the device.

A validation set does not change between evaluations: same rows, same order,
same padding. Rebuilding it every eval re-reads and re-collates every row — on
the Class-I path that was ~12 s per eval against ~26 ms of GPU work, a quarter
of a 50-minute run. This wraps any data module so its
validation batches are produced once, moved to the device, and replayed.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

__all__ = ["ResidentValidation"]


class ResidentValidation:
    """A data module whose ``val_dataloader`` replays batches held on *device*.

    Everything else — ``setup``, ``train_dataloader``, ``on_epoch_start`` — is
    the wrapped module's, unchanged. The batches are built on the first call.

    Args:
        datamodule: Any object satisfying the Trainer's data-module protocol.
        device: Where the batches live.

    Example::

        trainer.train(ResidentValidation(datamodule, "cuda"), max_epochs=...)
    """

    def __init__(self, datamodule: Any, device: str | torch.device) -> None:
        self.datamodule = datamodule
        self.device = torch.device(device)
        self._batches: list[Any] | None = None

    def setup(self, stage: str = "fit") -> None:
        self.datamodule.setup(stage)

    def on_epoch_start(self, epoch: int) -> None:
        self.datamodule.on_epoch_start(epoch)

    def train_dataloader(self) -> Iterable:
        return self.datamodule.train_dataloader()

    def val_dataloader(self) -> Iterable:
        """The validation batches, built and moved to the device on first use.

        Each pass gets a shallow copy of every batch — the key structure is new,
        the tensors are the resident ones — so what an evaluation writes onto a
        batch (the predicted columns) never lands on the resident set. Without
        it the second pass met batches with more columns than the first and
        recompiled every compiled eval body.
        """
        if self._batches is None:
            self._batches = [batch.to(self.device) for batch in self.datamodule.val_dataloader()]
        return (batch.copy() for batch in self._batches)
