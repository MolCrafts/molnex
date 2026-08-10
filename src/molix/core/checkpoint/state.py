"""Unified training state for checkpointing.

``Checkpoint`` is the serialization aggregate that knows how to produce
a complete ``state_dict`` for checkpoint save/resume.  It is **not** a
replacement for :class:`~molix.core.state.TrainState` which remains the
metrics/counter dict passed to hooks and steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn

from molix.core.checkpoint.rng import capture_rng_states, restore_rng_states
from molix.core.state import Path


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
        best_metric_name: Path into ``state`` for the tracked metric —
            a tuple ``("eval", "loss")`` for a nested scalar or a bare
            string for a top-level key.
    """

    model: nn.Module
    optimizer: torch.optim.Optimizer
    lr_scheduler: Any | None = None
    scaler: Any | None = None
    epoch: int = 0
    global_step: int = 0
    best_metric: float | None = None
    best_metric_name: Path = ("eval", "loss")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _unwrap_model(self) -> nn.Module:
        """Return the underlying module, unwrapping torch.compile + DDP/FSDP.

        ``torch.compile`` returns an ``OptimizedModule`` that holds the original
        module as ``._orig_mod`` and whose ``state_dict()`` prefixes every key
        with ``_orig_mod.``; DDP/FSDP expose theirs as ``.module``. Unwrapping
        both — compile first, since ``torch.compile(DDP(model))`` nests in that
        order — keeps checkpoints wrapper-agnostic: a checkpoint saved from a
        compiled run loads cleanly into an uncompiled model (e.g. for eval) and
        vice versa, with no key-prefix surgery at the call site.
        """
        model = self.model
        if hasattr(model, "_orig_mod"):  # torch.compile OptimizedModule
            model = model._orig_mod
        if hasattr(model, "module"):  # DDP / FSDP
            model = model.module
        return cast(nn.Module, model)

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
        stored = state_dict.get("best_metric_name", ("eval", "loss"))
        # Tolerate slash-strings from older checkpoints by splitting on "/".
        if isinstance(stored, str) and "/" in stored:
            stored = tuple(stored.split("/"))
        self.best_metric_name = stored

        self._unwrap_model().load_state_dict(state_dict["model_state_dict"])
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])

        if self.lr_scheduler is not None and "lr_scheduler_state_dict" in state_dict:
            self.lr_scheduler.load_state_dict(state_dict["lr_scheduler_state_dict"])
        if self.scaler is not None and "scaler_state_dict" in state_dict:
            self.scaler.load_state_dict(state_dict["scaler_state_dict"])
        if "rng_states" in state_dict:
            restore_rng_states(state_dict["rng_states"])
