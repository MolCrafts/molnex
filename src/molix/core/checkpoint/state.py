"""Unified training state for checkpointing.

``Checkpoint`` is the serialization aggregate that knows how to produce
a complete ``state_dict`` for checkpoint save/resume.  It is **not** a
replacement for :class:`~molix.core.state.TrainState` which remains the
metrics/counter dict passed to hooks and steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from molix.core.checkpoint.rng import capture_rng_states, restore_rng_states


@dataclass
class Checkpoint:
    """Aggregate of all stateful objects needed for checkpoint resume.

    Attributes:
        model: The ``nn.Module`` being trained (required).
        optimizer: The optimizer instance (required).
        lr_scheduler: Learning rate scheduler (optional).
        scaler: AMP ``GradScaler`` (optional).
        epoch: Current epoch (synced from ``TrainState``).
        global_step: Current global step (synced from ``TrainState``).
        best_metric: Best metric value seen so far.
        best_metric_name: Name of the tracked metric.
    """

    model: nn.Module
    optimizer: torch.optim.Optimizer
    lr_scheduler: Any | None = None
    scaler: Any | None = None
    epoch: int = 0
    global_step: int = 0
    best_metric: float | None = None
    best_metric_name: str = "eval/loss"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _unwrap_model(self) -> nn.Module:
        """Return the underlying module, unwrapping DDP/FSDP if needed."""
        return self.model.module if hasattr(self.model, "module") else self.model

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Produce a complete state dict suitable for ``torch.save``.

        Returns:
            Dictionary containing all serialisable training state
            including model weights, optimizer state, scheduler state,
            AMP scaler state, scalar counters, and RNG states.
        """
        sd: dict[str, Any] = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "best_metric_name": self.best_metric_name,
            "rng_states": capture_rng_states(),
            "model_state_dict": self._unwrap_model().state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.lr_scheduler is not None:
            sd["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()
        if self.scaler is not None:
            sd["scaler_state_dict"] = self.scaler.state_dict()
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state from a previously saved *state_dict*.

        Args:
            state_dict: Dictionary produced by :meth:`state_dict`.
        """
        self.epoch = state_dict["epoch"]
        self.global_step = state_dict["global_step"]
        self.best_metric = state_dict.get("best_metric")
        self.best_metric_name = state_dict.get(
            "best_metric_name", "eval/loss"
        )

        self._unwrap_model().load_state_dict(state_dict["model_state_dict"])
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])

        if (
            self.lr_scheduler is not None
            and "lr_scheduler_state_dict" in state_dict
        ):
            self.lr_scheduler.load_state_dict(
                state_dict["lr_scheduler_state_dict"]
            )
        if self.scaler is not None and "scaler_state_dict" in state_dict:
            self.scaler.load_state_dict(state_dict["scaler_state_dict"])
        if "rng_states" in state_dict:
            restore_rng_states(state_dict["rng_states"])
