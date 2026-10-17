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
        setup = getattr(self.datamodule, "setup", None)
        if callable(setup):
            setup(stage)

    def on_epoch_start(self, epoch: int) -> None:
        hook = getattr(self.datamodule, "on_epoch_start", None)
        if callable(hook):
            hook(epoch)

    def train_dataloader(self) -> Iterable:
        return self.datamodule.train_dataloader()

    def val_dataloader(self) -> Iterable:
        """The validation batches, built and moved to the device on first use."""
        if self._batches is None:
            self._batches = [batch.to(self.device) for batch in self.datamodule.val_dataloader()]
        return iter(self._batches)
