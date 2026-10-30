"""Default evaluation step implementation."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import torch
from tensordict import TensorDictBase

from molix.core.steps.base import _mark_leaves, publish_terms

if TYPE_CHECKING:
    from molix.core.state import TrainState
    from molix.core.trainer import Trainer


def release(obj: Any) -> None:
    """Replace every attached tensor reachable in *obj* by its detached self, in place.

    Walks a :class:`~tensordict.TensorDictBase` (nested leaves), a ``dict``
    (recursively) or leaves anything else alone. A view cannot be detached in
    place, so each attached leaf is *replaced* by ``leaf.detach()`` — the same
    storage, no graph. A marked position leaf (``requires_grad`` set by
    :meth:`~molix.autograd.Derivative.mark_leaves`) is replaced too; the next
    batch marks its own.

    Args:
        obj: The batch or the predictions of one evaluation step.
    """
    if isinstance(obj, TensorDictBase):
        for key, value in list(obj.items(include_nested=True, leaves_only=True)):
            if isinstance(value, torch.Tensor) and value.requires_grad:
                obj[key] = value.detach()
    elif isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, torch.Tensor):
                if value.requires_grad:
                    obj[key] = value.detach()
            else:
                release(value)


class DefaultEvalStep:
    """Default evaluation step with optional AMP support.

    Autocast is the Trainer's (``Trainer(autocast=…)``): when set, the forward
    pass runs under :func:`torch.amp.autocast` at that dtype. No
    ``GradScaler`` is used during evaluation.

    Args:
        no_grad: Wrap forward in ``torch.no_grad()`` (default ``False``).
            A force model derives ``F = -∂E/∂pos`` through
            :class:`~molix.autograd.Derivative`, which forces grad on for its
            own ``autograd.grad`` and so produces valid (detached) eval forces
            even under ``no_grad``; a step whose body differentiates a second
            time (a partial Hessian) needs grad on. Either way the graph does
            not outlive the step: with grad on, :meth:`on_eval_batch` detaches
            the loss and every attached column it wrote onto the batch
            (:func:`release`) once the loss is computed — an eval batch is
            often resident, and a graph left on it would stay alive through
            training. Set ``True`` for pure energy/property models where the
            graph-construction memory itself matters.
        objective: Loss evaluated on val batches instead of
            ``trainer.loss_fn`` — e.g. a plain residual when training runs on
            its saturated form, so the selected checkpoint is chosen on the
            quantity that matters rather than on a compressed one.

    :meth:`compute_loss` is the sanctioned seam for subclasses that need a
    different *objective* while keeping the grad-context / AMP semantics (see
    :class:`~molix.core.steps.multitask.MultiTaskEvalStep`).

    :meth:`forward_loss` is an eager prologue plus :meth:`_traced_forward_loss`
    — forward pass plus objective, the region
    :meth:`molix.core.trainer.Trainer.compile` traces for evaluation
    (``step=True``, its default). The prologue marks the derivative op's
    leaves on the batch as the training step does, so the eval region is one
    graph whether it runs with grad or not; the grad and autocast contexts wrap
    the call from outside, so dynamo captures one eval graph under the
    configured ``no_grad`` and AMP state. Which losses trace is listed on
    :class:`~molix.core.steps.train.DefaultTrainStep`.
    """

    #: Why :meth:`forward_loss` cannot be traced, or ``None`` when it can —
    #: see :attr:`~molix.core.steps.train.DefaultTrainStep.untraceable`.
    untraceable: str | None = None

    #: The per-source compiled regions of a step that has several
    #: (:class:`~molix.core.steps.joint.JointEvalStep`), or ``None`` for a
    #: step whose one region is ``_traced_forward_loss``.
    traced_bodies: dict[str, Any] | None = None

    def __init__(
        self, *, no_grad: bool = False, objective: Callable[[Any, Any], torch.Tensor] | None = None
    ) -> None:
        self._no_grad = no_grad
        self.objective = objective

    @property
    def scalar_keys(self) -> tuple[tuple[str, ...], ...]:
        """No paths beyond the trainer's defaults (see :class:`~molix.core.steps.base.Step`)."""
        return ()

    def compute_loss(
        self,
        trainer: "Trainer",
        state: "TrainState",
        batch: Any,
        predictions: Any,
    ) -> torch.Tensor:
        """Evaluate the objective for one batch (override point, no cadence).

        Called from inside the grad-context / autocast block of
        :meth:`on_eval_batch`, so an override sees the same AMP dtype and the
        same ``no_grad`` setting as the forward pass. The default body
        delegates to ``trainer.loss_fn``; a subclass that owns its losses (e.g.
        per-task routing) may override this and run with
        ``Trainer(loss_fn=None)``.

        Args:
            trainer: The owning :class:`~molix.core.trainer.Trainer`.
            state: The :class:`~molix.core.state.TrainState` to record into.
            batch: The collated batch handed to the model.
            predictions: The model output for ``batch``.

        Returns:
            The scalar loss tensor for this batch.
        """
        objective = self.objective if self.objective is not None else trainer.loss_fn
        assert objective is not None
        return objective(predictions, batch)

    def forward_loss(
        self,
        trainer: "Trainer",
        state: "TrainState",
        batch: Any,
    ) -> tuple[torch.Tensor, Any]:
        """The eager leaf prologue, then the traced body.

        Marks the derivative op's leaves on ``batch`` outside any compiled
        region (:func:`~molix.core.steps.base._mark_leaves`), then runs
        :meth:`_traced_forward_loss` through the instance attribute — the one
        ``Trainer.compile(step=True)`` replaces with the compiled wrapper.
        :meth:`on_eval_batch` calls this method.

        Args:
            trainer: The owning :class:`~molix.core.trainer.Trainer`
                (provides ``model`` and ``loss_fn``).
            state: The :class:`~molix.core.state.TrainState`, threaded to
                :meth:`compute_loss` unchanged.
            batch: The collated batch handed to the model.

        Returns:
            ``(loss, predictions)`` — the scalar loss for this batch and the
            model output the loss was evaluated on.
        """
        _mark_leaves(trainer.model, batch)
        return self._traced_forward_loss(trainer, state, batch)

    def _traced_forward_loss(
        self,
        trainer: "Trainer",
        state: "TrainState",
        batch: Any,
    ) -> tuple[torch.Tensor, Any]:
        """Forward pass and objective: the region ``Trainer.compile`` traces.

        ``predictions = trainer.model(batch)`` then :meth:`compute_loss`, and
        nothing else.

        Args:
            trainer: The owning :class:`~molix.core.trainer.Trainer`.
            state: The :class:`~molix.core.state.TrainState`, threaded to
                :meth:`compute_loss` unchanged.
            batch: The collated batch, its variable columns already marked.

        Returns:
            ``(loss, predictions)`` as :meth:`forward_loss`.
        """
        predictions = trainer.model(batch)
        return self.compute_loss(trainer, state, batch, predictions), predictions

    def on_train_batch(self, trainer: "Trainer", state: "TrainState", batch: Any) -> dict[str, Any]:
        """Not supported — :class:`DefaultEvalStep` handles eval batches only.

        Always raises :class:`NotImplementedError`; use
        :class:`~molix.core.steps.train.DefaultTrainStep` for training
        batches.
        """
        raise NotImplementedError(
            "DefaultEvalStep.on_train_batch() is not implemented. "
            "Use DefaultTrainStep for training batches."
        )

    def on_eval_batch(self, trainer: "Trainer", state: "TrainState", batch: Any) -> dict[str, Any]:
        """Run one evaluation batch: forward pass + loss, no parameter update.

        Runs the forward pass under ``torch.no_grad()`` (or
        ``torch.enable_grad()`` when ``no_grad=False``), optionally inside
        :func:`torch.amp.autocast` when ``trainer.autocast`` is set, then
        evaluates the loss. Writes ``state["eval"]["loss"]`` as a detached
        tensor (no per-batch GPU sync); consumers materialize via
        ``float()``.

        Args:
            trainer: The owning :class:`~molix.core.trainer.Trainer`
                (provides ``model`` and ``loss_fn``).
            state: The :class:`~molix.core.state.TrainState` to record into.
            batch: A collated batch ``TensorDict`` for the model.

        Returns:
            ``{"loss": <Tensor>, "predictions": <model output>}``.
        """
        assert trainer.model is not None

        # Resolve autocast args only when AMP is on (see DefaultTrainStep).
        grad_ctx = torch.no_grad() if self._no_grad else torch.enable_grad()
        if trainer.autocast is not None:
            device_type = next(trainer.model.parameters()).device.type
            amp_ctx = torch.amp.autocast(device_type, dtype=trainer.autocast)
        else:
            amp_ctx = nullcontext()
        with grad_ctx, amp_ctx:
            loss, predictions = self.forward_loss(trainer, state, batch)
        if not self._no_grad:
            # Evaluation keeps no graph. A step that differentiates with grad
            # on (a force or Hessian loss) writes its predictions onto the
            # batch attached to the whole forward graph, and a validation
            # batch may be resident (``molix.data.ResidentValidation``): left
            # attached, every val batch's eval graph stays alive through the
            # next eval batch and through training (a baseline eval that held
            # 90 GiB into the first training step).
            loss = loss.detach()
            release(batch)
            release(predictions)

        # Detached tensor (not .item()) so eval doesn't force a GPU→CPU sync
        # every batch — mirrors train/loss; consumers materialize via float().
        state["eval"]["loss"] = loss.detach()
        objective = self.objective if self.objective is not None else trainer.loss_fn
        publish_terms(objective, predictions, batch, state["eval"])
        return {"loss": loss, "predictions": predictions}
