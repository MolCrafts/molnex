"""Early-stop hook — abort training on non-finite loss or parameters."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
import torch.nn as nn

from molix.core.hook import BaseHook

logger = logging.getLogger("molix.hooks.early_stop")


class EarlyStop(BaseHook):
    """Abort training when a stop condition is met.

    Args:
        if_nan: Abort when ``state["train"]["loss"]`` becomes non-finite
            (NaN or ±inf). On detection, saves ``<out_dir>/nan_checkpoint.pt``
            and raises ``RuntimeError`` so the Trainer unwinds cleanly. The
            caller can catch it and translate to a distinct exit code.
        param_check_every_n_steps: Also scan *every* model parameter for
            non-finiteness, but only every N steps. ``0`` (default) disables
            the parameter scan entirely — it launches one reduction kernel
            **per parameter tensor** every step, which throttles a
            launch-bound training loop; the loss check catches the same
            divergence one step later in practice. Set e.g. ``50`` to keep a
            periodic parameter audit at negligible cost.

    Example::

        EarlyStop(if_nan=True)
    """

    def __init__(
        self,
        *,
        if_nan: bool = False,
        model: nn.Module | None = None,
        out_dir: str | Path | None = None,
        param_check_every_n_steps: int = 0,
    ) -> None:
        if if_nan and model is None:
            raise ValueError("EarlyStop(if_nan=True) requires `model=` for the NaN checkpoint.")
        if param_check_every_n_steps < 0:
            raise ValueError("param_check_every_n_steps must be >= 0.")
        self._if_nan = if_nan
        self._model = model
        self._out_dir = Path(out_dir) if out_dir is not None else None
        self._param_check_every = param_check_every_n_steps
        self._batch_count = 0

    def on_train_batch_end(self, trainer, state, batch, outputs) -> None:
        """Abort training if a non-finite loss (or, optionally, parameter) is seen.

        No-op unless ``if_nan=True``. Always checks ``state["train"]["loss"]``;
        scans all parameters only on the ``param_check_every_n_steps`` cadence
        (off by default — the per-step scan is a launch-bound bottleneck).
        """
        if not self._if_nan:
            return
        self._batch_count += 1

        loss = state["train"].get("loss")
        if loss is not None and not math.isfinite(float(loss)):
            self._abort(state, reason=f"non-finite loss={float(loss)}")
            return

        if self._param_check_every and self._batch_count % self._param_check_every == 0:
            assert self._model is not None
            for name, p in self._model.named_parameters():
                if not torch.isfinite(p).all():
                    self._abort(state, reason=f"non-finite parameter {name}")
                    return

    def _abort(self, state, *, reason: str) -> None:
        step = int(state.get("global_step", 0))
        logger.error("NaN detected at step=%d — %s", step, reason)
        if self._out_dir is not None and self._model is not None:
            torch.save(self._model.state_dict(), self._out_dir / "nan_checkpoint.pt")
        raise RuntimeError(f"NaN detected — early stopping ({reason})")
