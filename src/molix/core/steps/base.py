"""Step protocol definition for Molix.

The Step protocol defines the interface for training and evaluation
computation steps, separating computation logic from control flow.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

import torch
import torch.nn as nn

from molix.autograd import Derivative, Potential
from molix.compile import unwrap
from molix.core.losses import WeightedLoss

if TYPE_CHECKING:
    from molix.core.state import TrainState
    from molix.core.trainer import Trainer


def _mark_leaves(model: nn.Module, batch: Any) -> None:
    """The eager prologue of the default steps: mark the derivative leaves.

    Reads the model under any compile wrapper (:func:`molix.compile.unwrap`)
    so ``compile(step=False)`` is served too, then finds the op the model runs
    by type — a :class:`~molix.autograd.Derivative` is the op, a
    :class:`~molix.autograd.Potential` runs its ``derivative`` — and calls
    :meth:`~molix.autograd.Derivative.mark_leaves`. Any other model needs
    nothing, and the op itself does nothing for an energy-only ``wrt=()``.
    Outside the traced body on purpose: a leaf made inside the region leaks
    through the attached derivatives and breaks the graph on stock torch; a
    leaf made here is a graph input (:mod:`molix.autograd.derivative`).

    Args:
        model: ``trainer.model``.
        batch: The batch the traced body is handed next.
    """
    model = unwrap(model)
    if isinstance(model, Derivative):
        model.mark_leaves(batch)
    elif isinstance(model, Potential):
        op = unwrap(model.derivative)
        if isinstance(op, Derivative):
            op.mark_leaves(batch)


def _combined(objective: Callable[[Any, Any], torch.Tensor] | None) -> WeightedLoss | None:
    """*objective* when it has parts to report, else ``None``.

    A :class:`~molix.core.losses.WeightedLoss` of two or more terms reports
    each by name; any other loss — a closure, a single-term combination, or
    no loss at all (``Trainer(loss_fn=None)``) — has no parts to report.
    """
    if isinstance(objective, WeightedLoss) and len(objective.names) >= 2:
        return objective
    return None


def term_keys(
    namespace: str, objective: Callable[[Any, Any], torch.Tensor] | None
) -> tuple[tuple[str, str], ...]:
    """``(namespace, "loss_<name>")`` per named term of *objective*, or ``()``."""
    combined = _combined(objective)
    if combined is None:
        return ()
    return tuple((namespace, f"loss_{name}") for name in combined.names)


def publish_terms(
    objective: Callable[[Any, Any], torch.Tensor] | None,
    predictions: Any,
    batch: Any,
    into: dict[str, Any],
) -> None:
    """Write each named term of *objective* into *into* as ``loss_<name>``.

    Reporting only, eager and under ``no_grad``: the weighted sum stays the one
    thing differentiated. A combined objective watched only as a sum cannot say
    which part is rising.

    Args:
        objective: The loss; one without named terms is skipped.
        predictions: The model output the terms read.
        batch: The batch the terms read their targets from.
        into: The state namespace to write into.
    """
    combined = _combined(objective)
    if combined is None:
        return
    with torch.no_grad():
        for name, value in combined.terms(predictions, batch).items():
            into[f"loss_{name}"] = value.detach()


class Step(Protocol):
    """Protocol for training and evaluation computation steps.

    Steps encapsulate the computation logic for processing batches during
    training and evaluation, while the Trainer handles control flow (loops,
    hooks, state management).

    Key Responsibilities:
    - Execute forward pass through model
    - Compute loss function
    - Perform backward pass and optimizer updates (training only)
    - Return outputs for logging/hooks

    The Trainer provides access to model, optimizer, loss function, and other
    resources through the `trainer` parameter.

    Gradient accumulation is **not** a Step concern — it is owned by the
    Trainer via ``Trainer(accumulate_grad_batches=N)``, which scales the loss
    and gates the optimizer / LR-scheduler / ``global_step`` advance on
    accumulation boundaries. A custom Step that re-implemented accumulation by
    skipping ``optimizer.step()`` would desync the scheduler (which the Trainer
    still advances per optimizer step). Implement a custom Step only to change
    the *computation* (e.g. an auxiliary loss term), not the step cadence.

    To change only the *objective* while keeping the default cadence, subclass
    :class:`~molix.core.steps.train.DefaultTrainStep` /
    :class:`~molix.core.steps.eval.DefaultEvalStep` and override their optional
    ``compute_loss(trainer, state, batch, predictions)`` seam instead of
    re-implementing ``on_train_batch`` / ``on_eval_batch``; the seam runs inside
    the same autocast (and eval grad) context as the forward pass, and a step
    that owns its losses there may run with ``Trainer(loss_fn=None)``. This
    Protocol deliberately requires **no** ``compute_loss`` method — it is
    duck-typed by callers, so third-party steps without it keep working.

    The same holds for ``forward_loss(trainer, state, batch) -> (loss,
    predictions)``: on the default steps it is an eager prologue that marks
    the derivative op's leaves on the batch (:func:`_mark_leaves`) followed by
    ``_traced_forward_loss`` — the forward-plus-objective region that
    :meth:`~molix.core.trainer.Trainer.compile` (``step=True``) swaps for a
    compiled wrapper — and it is what :meth:`~molix.core.trainer.Trainer.smoke`
    warms up. A step without ``_traced_forward_loss`` trains as before but
    can only be compiled model-only (``compile(step=False)``); one without
    ``forward_loss`` cannot be smoke-tested either; a step whose traced body
    host-syncs declares ``untraceable``. The multi-task label read stays in
    the eager prologue, outside that body.

    Example — a custom step that adds an L2 activation penalty::

        class PenalizedStep:
            def on_train_batch(self, trainer, state, batch):
                predictions = trainer.model(batch)
                loss = trainer.loss_fn(predictions, batch)
                loss = loss + 1e-4 * predictions["energy"].square().mean()
                trainer.optimizer.zero_grad()
                loss.backward()
                trainer._call_hooks("on_after_backward", trainer, state)
                trainer.optimizer.step()
                state["train"]["loss"] = loss.detach()
                return {"loss": loss, "predictions": predictions}

            def on_eval_batch(self, trainer, state, batch):
                predictions = trainer.model(batch)
                loss = trainer.loss_fn(predictions, batch)
                return {"loss": loss, "predictions": predictions}

    See Also:
        - DefaultTrainStep: Default training step (AMP + gradient accumulation)
        - DefaultEvalStep: Default evaluation step implementation
        - MultiTaskTrainStep: Per-batch task routing over the ``compute_loss``
          seam, train side
        - MultiTaskEvalStep: Per-batch task routing over the ``compute_loss``
          seam, eval side
    """

    #: Why the step's traced region cannot be compiled, or ``None`` when it can;
    #: :meth:`~molix.core.trainer.Trainer.compile` refuses ``step=True`` on it.
    untraceable: str | None
    #: A step with several compiled regions (one per source), or ``None``.
    traced_bodies: dict[str, Any] | None

    @property
    def scalar_keys(self) -> tuple[tuple[str, ...], ...]:
        """State paths the step writes beyond the trainer's defaults.

        Read by :class:`~molix.hooks.progress.Log` (columns it may show) and by
        :meth:`~molix.core.trainer.Trainer._run_eval_phase` (what an eval phase
        averages). Empty means only the defaults: ``("train", "loss")`` /
        ``("eval", "loss")`` and the trainer objective's terms.
        """
        ...

    def on_train_batch(
        self,
        trainer: "Trainer",
        state: "TrainState",
        batch: Any,
    ) -> dict[str, Any]:
        """Execute training batch computation.

        This method is called by the Trainer for each training batch. It should:
        1. Perform forward pass through the model
        2. Compute the loss
        3. Perform backward pass (gradients)
        4. Update model parameters via optimizer
        5. Return outputs for hooks and logging

        Args:
            trainer: Trainer instance providing access to:
                - trainer.model: Neural network model
                - trainer.optimizer: Optimizer instance
                - trainer.loss_fn: Loss function
                - trainer.state: Training state (alternative to state param)
            state: Current training state with:
                - state.global_step: Current training step
                - state.epoch: Current epoch
                - state.stage: Current stage (TRAIN, EVAL, etc.)
            batch: Input batch data (typically dict or tensor)

        Returns:
            Dictionary with outputs, must include:
                - "loss": Loss tensor or numeric value
                - "predictions": Model predictions
            Additional keys allowed (e.g., "metrics", "auxiliary_outputs")

        Example:
            ```python
            def on_train_batch(self, trainer, state, batch):
                # Forward pass
                predictions = trainer.model(batch)

                # Compute loss (loss_fn reads nested targets from the batch,
                # e.g. batch["graphs", "energy"])
                loss = trainer.loss_fn(predictions, batch)

                # Backward pass
                trainer.optimizer.zero_grad()
                loss.backward()
                trainer.optimizer.step()

                return {"loss": loss, "predictions": predictions}
            ```
        """
        ...

    def on_eval_batch(
        self,
        trainer: "Trainer",
        state: "TrainState",
        batch: Any,
    ) -> dict[str, Any]:
        """Execute evaluation batch computation.

        This method is called by the Trainer for each evaluation batch. It should:
        1. Perform forward pass through the model (without gradients)
        2. Compute the loss (for logging/metrics)
        3. Return outputs for hooks and logging

        Note: No optimizer updates should be performed during evaluation.

        Args:
            trainer: Trainer instance providing access to:
                - trainer.model: Neural network model
                - trainer.loss_fn: Loss function
                - trainer.state: Training state
            state: Current training state
            batch: Input batch data (typically dict or tensor)

        Returns:
            Dictionary with outputs, must include:
                - "loss": Loss tensor or numeric value (for logging)
                - "predictions": Model predictions
            Additional keys allowed (e.g., "metrics", "auxiliary_outputs")

        Example:
            ```python
            def on_eval_batch(self, trainer, state, batch):
                # Forward pass. Do NOT wrap force models in torch.no_grad():
                # F = -dE/dx needs a live graph (see DefaultEvalStep.no_grad).
                predictions = trainer.model(batch)
                loss = trainer.loss_fn(predictions, batch)
                return {"loss": loss, "predictions": predictions}
            ```
        """
        ...
