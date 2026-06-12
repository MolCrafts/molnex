"""Terminal-display hooks: ``ProgressBarHook`` and ``Log``."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from molix import logging as _logging_mod
from molix.core.hook import BaseHook, ScalarHook
from molix.core.state import Path, display, resolve


class ProgressBarHook(BaseHook):
    """Displays training progress with tqdm.

    Args:
        desc: Description for the progress bar (default: "Training").
        leave: Leave progress bar after completion (default: True).
    """

    def __init__(self, desc: str = "Training", leave: bool = True, loss_refresh_every: int = 1):
        from tqdm import tqdm

        self.tqdm = tqdm

        self.desc = desc
        self.leave = leave
        # The loss postfix requires a ``.item()`` CPU↔GPU sync. Refreshing it
        # every step serialises a launch-bound training loop, so allow
        # throttling: the bar still advances every batch, but the displayed
        # loss is sampled only every ``loss_refresh_every`` batches.
        self.loss_refresh_every = max(1, loss_refresh_every)
        self.pbar = None
        self._update_count = 0

    def on_train_start(self, trainer, state):
        """Initialize progress bar."""
        self.pbar = None
        self._update_count = 0

    def on_epoch_start(self, trainer, state):
        """Update progress bar for new epoch."""
        if self.pbar is None:
            self.pbar = self.tqdm(desc=f"{self.desc} Epoch {state.epoch}", leave=self.leave)
        else:
            self.pbar.set_description(f"{self.desc} Epoch {state.epoch}")
            self.pbar.reset()

    def on_train_batch_end(self, trainer, state, batch, outputs):
        """Update progress bar after each batch (loss postfix throttled)."""
        if self.pbar is not None:
            self._update_count += 1
            if (
                isinstance(outputs, dict)
                and "loss" in outputs
                and self._update_count % self.loss_refresh_every == 0
            ):
                loss = outputs["loss"]
                loss_value = loss.item() if hasattr(loss, "item") else loss
                self.pbar.set_postfix({"loss": f"{loss_value:.4f}"})
            self.pbar.update(1)

    def on_train_end(self, trainer, state):
        """Close progress bar."""
        if self.pbar is not None:
            self.pbar.close()


def _parse_fmt_width(fmt: str) -> int:
    """Extract the width component from a format spec like ``"{:>12.4g}"``."""
    import re

    m = re.search(r"(\d+)", fmt)
    return int(m.group(1)) if m else 12


_MISSING_CELL = "—"


def _render_cell(value: Any, fmt: str, width: int) -> str:
    """Format one ``Log`` table cell.

    * Numeric (int / non-NaN float) → ``fmt.format(value)``.
    * Real numerical NaN → ``"nan"`` (right-aligned to ``width``).
    * Anything else (``None``, missing path, non-scalar) →
      :data:`_MISSING_CELL`. Reserves ``"nan"`` for genuine numerical NaN
      so silent path-resolution failures can no longer masquerade as
      training divergence.
    """
    # 0-dim tensors are stored by hooks that defer the CPU↔GPU sync to the
    # (throttled) render path — materialise here, not per training step.
    if hasattr(value, "item") and hasattr(value, "dim"):
        try:
            if value.dim() == 0:
                value = value.item()
        except (RuntimeError, ValueError):
            return f"{_MISSING_CELL:>{width}}"
    if isinstance(value, bool):
        return fmt.format(int(value))
    if isinstance(value, int):
        return fmt.format(value)
    if isinstance(value, float):
        if value != value:
            return f"{'nan':>{width}}"
        return fmt.format(value)
    return f"{_MISSING_CELL:>{width}}"


_BUILTIN_STATE_PATHS: frozenset = frozenset(
    {
        "epoch",
        "global_step",
        "stage",
        "steps_since_last_eval",
        "best_metric",
        ("train", "loss"),
        ("eval", "loss"),
    }
)


def _normalize_key(item: Path) -> Path:
    """Coerce a user-provided key into the canonical :data:`Path` form."""
    if isinstance(item, tuple):
        return item
    if isinstance(item, str):
        return tuple(item.split("/")) if "/" in item else item
    raise TypeError(f"Log keys must be str, tuple, or ScalarHook — got {type(item).__name__}")


def _collect_keys(items: Sequence[Path | ScalarHook]) -> list[Path]:
    """Flatten a mix of paths and :class:`ScalarHook` instances."""
    seen: set[Path] = set()
    out: list[Path] = []
    for item in items:
        if isinstance(item, ScalarHook):
            paths = tuple(item.scalar_keys)
        else:
            paths = (_normalize_key(item),)
        for p in paths:
            if p not in seen:
                seen.add(p)
                out.append(p)
    return out


class Log(BaseHook):
    """Periodic LAMMPS-thermo-style stdout logger.

    Reads scalar values from ``state`` and prints a formatted row every
    ``every_n_steps`` training batches.

    Args:
        every_n_steps: Print a row every N training batches.
        keys: State paths to print. ``step`` and ``epoch`` are prepended.
        fmt: Format spec for each scalar column (default ``"{:>12.4g}"``).
        start_step: Suppress rows before this step. Lets the early-training
            loss explosion stay out of plots / logs.
    """

    def __init__(
        self,
        every_n_steps: int,
        keys: Sequence[Path | ScalarHook],
        *,
        fmt: str = "{:>12.4g}",
        header_every_n_rows: int = 50,
        epoch_separator: bool = True,
        start_step: int = 0,
    ):
        if every_n_steps <= 0:
            raise ValueError("every_n_steps must be positive")
        if header_every_n_rows <= 0:
            raise ValueError("header_every_n_rows must be positive")
        if start_step < 0:
            raise ValueError("start_step must be non-negative")
        self.every_n_steps = every_n_steps
        self.keys = _collect_keys(keys)
        self.fmt = fmt
        self.header_every_n_rows = header_every_n_rows
        self.epoch_separator = epoch_separator
        self.start_step = start_step
        self._rows_since_header = 0
        self._last_epoch: int | None = None
        self._metrics = _logging_mod.metrics_logger()

    def _paths(self) -> list[Path]:
        return ["step", "epoch", *self.keys]

    def _columns(self) -> list[str]:
        return [display(p) for p in self._paths()]

    def _table_width(self) -> int:
        col_w = _parse_fmt_width(self.fmt)
        n_cols = len(self._paths())
        return n_cols * col_w + max(0, n_cols - 1)

    def _emit_header(self) -> None:
        if _logging_mod.has_effective_handlers(self._metrics):
            self._metrics.info(
                "metrics header",
                kind="header",
                columns=self._columns(),
            )
        else:
            width = _parse_fmt_width(self.fmt)
            top_row, bot_row = _logging_mod.split_header_rows(self._columns(), width)
            print(top_row, flush=True)
            print(bot_row, flush=True)
        self._rows_since_header = 0

    def announce(self, message: str) -> None:
        """Emit a thin separator + ``message`` between table rows."""
        if self._last_epoch is None:
            return
        evt = _logging_mod.events_logger()
        width = self._table_width()
        if _logging_mod.has_effective_handlers(evt):
            evt.info(
                message,
                kind="announce",
                table_width=width,
            )
        else:
            prefix = f"─── {message} "
            pad = max(3, width - len(prefix))
            print(prefix + "─" * pad, flush=True)
        self._rows_since_header = self.header_every_n_rows

    def _validate_keys(self, trainer) -> None:
        if trainer is None:
            return
        advertised: set[Path] = set(_BUILTIN_STATE_PATHS)
        for hook in trainer.hooks:
            if isinstance(hook, ScalarHook):
                advertised.update(hook.scalar_keys)

        unknown = [p for p in self.keys if p not in advertised]
        if unknown:
            rendered = [display(p) for p in unknown]
            available = sorted({display(p) for p in advertised})
            raise ValueError(
                f"Log key(s) {rendered!r} are not advertised by any "
                f"registered ScalarHook nor known as built-in state paths. "
                f"Either register a hook that populates them or remove them "
                f"from `keys`. Available paths: {available!r}."
            )

    def on_train_start(self, trainer, state):
        """Validate the configured keys and emit the initial table header.

        Raises:
            ValueError: If any requested key is neither advertised by a
                registered :class:`ScalarHook` nor a built-in state path.
        """
        self._validate_keys(trainer)
        self._rows_since_header = 0
        self._last_epoch = int(state.get("epoch", 0))
        _logging_mod.set_table_width(self._table_width())
        self._emit_header()

    def on_epoch_end(self, trainer, state):
        """Draw an epoch-boundary separator so the table reads as sections."""
        if not self.epoch_separator:
            return
        epoch = int(state.get("epoch", 0))
        if epoch == self._last_epoch:
            return
        width = self._table_width()
        if _logging_mod.has_effective_handlers(self._metrics):
            self._metrics.info(
                "epoch separator",
                kind="epoch_sep",
                epoch=epoch,
                table_width=width,
            )
        else:
            print("─" * width, flush=True)
        self._last_epoch = epoch
        self._rows_since_header = self.header_every_n_rows

    def on_train_batch_end(self, trainer, state, batch, outputs):
        """Emit one metrics row for this step, throttled by the row schedule.

        Skips steps before ``start_step`` and any step not on the
        ``every_n_steps`` cadence; re-emits the header every
        ``header_every_n_rows`` rows.
        """
        step = int(state.get("global_step", 0)) + 1
        if step < self.start_step:
            return
        if step % self.every_n_steps != 0:
            return

        if self._rows_since_header >= self.header_every_n_rows:
            self._emit_header()

        values: dict[str, Any] = {
            "step": step,
            "epoch": int(state.get("epoch", 0)),
        }
        for path in self.keys:
            values[display(path)] = resolve(state, path)

        if _logging_mod.has_effective_handlers(self._metrics):
            self._metrics.info(
                "metrics row",
                kind="row",
                columns=self._columns(),
                values=values,
            )
        else:
            width = _parse_fmt_width(self.fmt)
            parts: list[str] = [
                f"{values['step']:>{width}d}",
                f"{values['epoch']:>{width}d}",
            ]
            for path in self.keys:
                parts.append(_render_cell(values.get(display(path)), self.fmt, width))
            print(" ".join(parts), flush=True)
        self._rows_since_header += 1
