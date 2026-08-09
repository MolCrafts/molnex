"""Hook contract layer for the Molix Trainer.

This module is the *interface* tier of the hook system:

* :class:`Hook` — the lifecycle :class:`typing.Protocol` that
  :class:`molix.core.trainer.Trainer` calls.
* :class:`BaseHook` — a concrete no-op implementation of every
  callback; users subclass it and override only what they need.
* :class:`ScalarHook` — an extension of ``BaseHook`` that advertises
  which :class:`~molix.core.state.Path` keys it writes into
  ``state``. Container hooks (e.g. ``Log``) read this attribute to
  decide which columns to render.

Concrete hook implementations (CheckpointHook, JournalHook,
GradClipHook, ...) live in :mod:`molix.hooks` — never in this
module. The contract layer must remain free of dependencies on the
side-tiers (``molix.hooks``, ``molix.io``, ``molix.recorder``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from molix.core.state import Path

if TYPE_CHECKING:
    from molix.core.state import TrainState
    from molix.core.trainer import Trainer


class Hook(Protocol):
    """Protocol for training hooks.

    Hooks receive notifications at various points in the training lifecycle.
    All methods are optional - implement only the hooks you need.

    Hook execution order:
    - By default, hooks execute in registration order
    - Use (hook, priority) tuples to override execution order
    - Lower priority values execute earlier (default priority = 100)
    - Hooks with same priority execute in registration order

    Errors are fatal, by design. If a hook raises,
    :meth:`molix.core.trainer.Trainer._call_hooks` logs the exception with a
    full traceback and then **re-raises** it, so the exception propagates out
    of the training loop and the run stops. Nothing is swallowed: swallowing
    hook errors once hid a fatal ``NaNGuardHook`` signal behind thousands of
    lines of repeated tracebacks. A hook that must tolerate its own failures
    has to catch them itself.

    Example:
        ```python
        class MyHook:
            def on_epoch_end(self, trainer, state):
                print(f"Epoch {state.epoch} completed")

        # Registration order
        trainer = Trainer(hooks=[MyHook(), OtherHook()])

        # With priority
        trainer = Trainer(hooks=[(MyHook(), 10), OtherHook()])
        ```
    """

    def on_train_start(self, trainer: "Trainer", state: "TrainState") -> None:
        """Called once at the beginning of training.

        Args:
            trainer: The trainer instance
            state: Current training state (epoch=0, global_step=0)
        """
        ...

    def on_train_end(self, trainer: "Trainer", state: "TrainState") -> None:
        """Called once at the end of training.

        Args:
            trainer: The trainer instance
            state: Final training state
        """
        ...

    def on_epoch_start(self, trainer: "Trainer", state: "TrainState") -> None:
        """Called at the start of each epoch.

        Args:
            trainer: The trainer instance
            state: Current training state
        """
        ...

    def on_epoch_end(self, trainer: "Trainer", state: "TrainState") -> None:
        """Called at the end of each epoch (after validation).

        Args:
            trainer: The trainer instance
            state: Current training state
        """
        ...

    def on_train_batch_start(self, trainer: "Trainer", state: "TrainState", batch: Any) -> None:
        """Called before processing each training batch.

        Args:
            trainer: The trainer instance
            state: Current training state
            batch: The current batch data
        """
        ...

    def on_train_batch_end(
        self, trainer: "Trainer", state: "TrainState", batch: Any, outputs: Any
    ) -> None:
        """Called after processing each training batch.

        Args:
            trainer: The trainer instance
            state: Current training state
            batch: The current batch data
            outputs: Outputs from the training step (loss, predictions, etc.)
        """
        ...

    def on_eval_phase_start(self, trainer: "Trainer", state: "TrainState") -> None:
        """Called once at the start of every eval phase, before the batch loop.

        Fires for *both* step-based (``eval_every_n_steps``) and epoch-end
        eval, so accumulating hooks (e.g. :class:`MetricsHook`) can reset
        their eval-side buffers here and make each eval phase fully
        self-contained — no reliance on ``on_epoch_start`` (which does not
        fire between mid-epoch step-based evals).

        Args:
            trainer: The trainer instance
            state: Current training state (stage already set to EVAL)
        """
        ...

    def on_eval_batch_start(self, trainer: "Trainer", state: "TrainState", batch: Any) -> None:
        """Called before processing each validation batch.

        Args:
            trainer: The trainer instance
            state: Current training state
            batch: The current batch data
        """
        ...

    def on_after_backward(self, trainer: "Trainer", state: "TrainState") -> None:
        """Called after backward pass, before optimizer step.

        Gradients are available and unscaled (if AMP is active, unscale
        has already been applied). Use this hook for gradient manipulation
        (clipping, logging, etc.).

        Args:
            trainer: The trainer instance
            state: Current training state
        """
        ...

    def on_eval_batch_end(
        self, trainer: "Trainer", state: "TrainState", batch: Any, outputs: Any
    ) -> None:
        """Called after processing each validation batch.

        Args:
            trainer: The trainer instance
            state: Current training state
            batch: The current batch data
            outputs: Outputs from the evaluation step (loss, metrics, etc.)
        """
        ...

    def on_eval_step_complete(self, trainer: "Trainer", state: "TrainState") -> None:
        """Called once at the end of every eval phase, after the batch loop.

        Fires for *both* triggers — step-based eval (``eval_every_n_steps``)
        and epoch-end eval — because :meth:`molix.core.trainer.Trainer._run_eval_phase`
        is the single code path behind both. Eval-publishing hooks
        (:class:`molix.hooks.MetricsHook`, :class:`molix.hooks.TensorBoardHook`)
        should write their ``state["eval"]`` scalars here rather than in
        ``on_epoch_end``, so the values are already published when the LR
        scheduler reads its monitored metric.

        Args:
            trainer: The trainer instance
            state: Current training state (every eval batch consumed;
                ``steps_since_last_eval`` is reset by the train loop *after*
                the eval phase returns, so it is still non-zero here)
        """
        ...


class BaseHook:
    """Base hook with no-op implementations.

    Inherit from this class and override only the methods you need.
    This provides better IDE support than implementing the Protocol directly.

    Example:
        ```python
        class MyHook(BaseHook):
            def on_epoch_end(self, trainer, state):
                print(f"Epoch {state.epoch} completed")
        ```
    """

    def on_train_start(self, trainer: "Trainer", state: "TrainState") -> None:
        """Called once at the beginning of training."""
        pass

    def on_train_end(self, trainer: "Trainer", state: "TrainState") -> None:
        """Called once at the end of training."""
        pass

    def on_epoch_start(self, trainer: "Trainer", state: "TrainState") -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(self, trainer: "Trainer", state: "TrainState") -> None:
        """Called at the end of each epoch (after validation)."""
        pass

    def on_train_batch_start(self, trainer: "Trainer", state: "TrainState", batch: Any) -> None:
        """Called before processing each training batch."""
        pass

    def on_train_batch_end(
        self, trainer: "Trainer", state: "TrainState", batch: Any, outputs: Any
    ) -> None:
        """Called after processing each training batch."""
        pass

    def on_after_backward(self, trainer: "Trainer", state: "TrainState") -> None:
        """Called after backward pass, before optimizer step."""
        pass

    def on_eval_phase_start(self, trainer: "Trainer", state: "TrainState") -> None:
        """Called once at the start of every eval phase, before the batch loop."""
        pass

    def on_eval_batch_start(self, trainer: "Trainer", state: "TrainState", batch: Any) -> None:
        """Called before processing each validation batch."""
        pass

    def on_eval_batch_end(
        self, trainer: "Trainer", state: "TrainState", batch: Any, outputs: Any
    ) -> None:
        """Called after processing each validation batch."""
        pass

    def on_eval_step_complete(self, trainer: "Trainer", state: "TrainState") -> None:
        """Called once at the end of every eval phase (step-based *and* epoch-end)."""
        pass


class ScalarHook(BaseHook):
    """Hook that writes scalar values into ``state``.

    Subclasses advertise the state paths they populate via
    ``scalar_keys``. Each path is a :data:`~molix.core.state.Path` —
    either a top-level string key (``"epoch"``) or a tuple path into
    a namespace sub-dict (``("train", "loss")``). Container hooks
    such as ``Log`` read this attribute to discover which columns to
    render.

    For hooks whose paths depend on runtime configuration (e.g.
    metric names), override ``scalar_keys`` as a ``@property``.
    """

    scalar_keys: tuple[Path, ...] = ()
