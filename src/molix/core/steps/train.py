"""Default training step implementation."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import torch

from molix.core.steps.base import _mark_leaves, publish_terms

if TYPE_CHECKING:
    from molix.core.state import TrainState
    from molix.core.trainer import Trainer


class DefaultTrainStep:
    """Default training step with AMP + gradient-accumulation support.

    Autocast is the Trainer's: with ``Trainer(autocast=torch.bfloat16)`` (or
    ``torch.float16``) the forward pass runs under :func:`torch.amp.autocast`
    at that dtype, and under float16 the backward uses the
    :class:`torch.amp.GradScaler` the :class:`Trainer` owns.

    Gradient accumulation is a *Trainer* concern: this step reads
    ``trainer.accumulate_grad_batches`` and ``trainer._micro_step`` to decide
    window boundaries. The loss is scaled by ``1/accumulate_grad_batches`` so
    the accumulated gradient matches a single large batch. The optimizer is
    stepped (and ``on_after_backward`` fired, and ``zero_grad`` issued) only
    on accumulation boundaries; the Trainer advances ``global_step`` / the LR
    scheduler only on those same boundaries.

    Flow per micro-batch:
        1. (window start) ``optimizer.zero_grad()``
        2. :meth:`forward_loss` (under autocast if AMP), ``/= accum``
        3. Backward (scaled by ``trainer.scaler`` under AMP)
        4. (window end) unscale → ``on_after_backward`` → optimizer step

    ``state["train"]["loss"]`` is written as a *detached tensor* (the
    un-scaled, full-magnitude loss), not a Python float — materialising it
    here would force a CPU↔GPU sync every micro-step and serialise the
    launch-bound pipeline. Consumers (``Log``, ``ProgressBarHook``) call
    ``.item()`` on their own throttled cadence.

    :meth:`compute_loss` is the sanctioned seam for subclasses that need a
    different *objective* while keeping this cadence byte-for-byte (see
    :class:`~molix.core.steps.multitask.MultiTaskTrainStep`).

    :meth:`forward_loss` is an eager prologue plus the one region
    :meth:`molix.core.trainer.Trainer.compile` traces (``step=True``, its
    default), :meth:`_traced_forward_loss` — forward pass plus objective:
    dynamo captures ``model(batch)`` and ``compute_loss`` as one graph and
    AOTAutograd emits the backward for it, so the whole ``backward()`` below
    runs compiled too. The prologue marks the derivative op's leaves on the
    batch (:meth:`molix.autograd.Derivative.mark_leaves`) *before* the region,
    so the position leaf is a graph input and a force-supervised step is one
    graph with zero breaks on stock torch — a leaf made inside the region
    would leak through the attached forces and break there. Everything else in
    :meth:`on_train_batch` stays eager Python around that call: the
    accumulation window, ``GradScaler``, ``on_after_backward`` and the
    optimizer. The autocast context wraps the call from outside — dynamo
    guards on the autocast state and captures the ops under it — so the
    ``trainer.autocast`` read stays in eager Python and never enters the
    trace.

    Which shipped losses trace (no tensor-valued Python control flow):
    :func:`~molix.core.losses.energy_mse`,
    :func:`~molix.core.losses.energy_force_mse`,
    :class:`~molix.core.losses.MSELoss`, :class:`~molix.core.losses.MAELoss`,
    :class:`~molix.core.losses.WeightedLoss`,
    :class:`~molix.core.losses.ChargeMSELoss`,
    :class:`~molix.core.losses.ScaleNormalizedLoss` and
    :class:`~molix.core.losses.ProjectedInternalHessianLoss` (its body; the
    Hessian it consumes is produced outside the step). Which do not — they
    loop over ``group_ids.unique()`` with boolean-mask indexing, a
    data-dependent shape per group:
    :func:`~molix.core.losses.center_by_group` and everything built on it,
    :func:`~molix.core.losses.molecule_centered_energy_mse` and
    :class:`~molix.core.losses.TorsionProfileLoss`. A step running one of
    those compiles the model alone (``Trainer.compile(step=False)``).
    """

    #: Why :meth:`forward_loss` cannot be traced, or ``None`` when it can. A
    #: step whose traced body host-syncs declares the reason here, and
    #: :meth:`~molix.core.trainer.Trainer.compile` with ``step=True`` refuses
    #: it, quoting this text. :class:`~molix.core.steps.multitask.MultiTaskTrainStep`
    #: keeps its label read in the eager prologue, so it does not set this.
    untraceable: str | None = None

    #: The per-source compiled regions of a step that has several
    #: (:class:`~molix.core.steps.joint.JointTrainStep`), or ``None`` for a
    #: step whose one region is ``_traced_forward_loss``.
    traced_bodies: dict[str, Any] | None = None

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

        Called from inside the autocast block of :meth:`on_train_batch`, so an
        override sees the same AMP dtype as the forward pass. The default body
        delegates to ``trainer.loss_fn``; a subclass that owns its losses (e.g.
        per-task routing) may override this and run with
        ``Trainer(loss_fn=None)``.

        Args:
            trainer: The owning :class:`~molix.core.trainer.Trainer`.
            state: The :class:`~molix.core.state.TrainState` to record into.
            batch: The collated batch handed to the model.
            predictions: The model output for ``batch``.

        Returns:
            The scalar loss tensor to back-propagate.
        """
        assert trainer.loss_fn is not None
        return trainer.loss_fn(predictions, batch)

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
        :meth:`on_train_batch` and :meth:`~molix.core.trainer.Trainer.smoke`
        both call this method, so smoke and training run the one graph.

        Args:
            trainer: The owning :class:`~molix.core.trainer.Trainer`
                (provides ``model`` and ``loss_fn``).
            state: The :class:`~molix.core.state.TrainState`, threaded to
                :meth:`compute_loss` unchanged.
            batch: The collated batch handed to the model.

        Returns:
            ``(loss, predictions)`` — the scalar loss to back-propagate and the
            model output the loss was evaluated on (for a frame-writing model
            such as :class:`~molix.autograd.Derivative`, ``batch`` itself).
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
        nothing else — no cadence, no context, no state write, no leaf.

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
        """Run one training micro-batch; step the optimizer on window boundaries.

        Args:
            trainer: The owning :class:`~molix.core.trainer.Trainer`
                (provides ``model``, ``loss_fn``, ``optimizer``, ``scaler``,
                ``accumulate_grad_batches``, ``_micro_step``).
            state: The :class:`~molix.core.state.TrainState` to record into.
            batch: A collated batch ``TensorDict`` for the model.

        Returns:
            ``{"loss": <Tensor>, "predictions": <out>, "optimizer_applied":
            <bool>}``. ``optimizer_applied`` is ``True`` only when this
            micro-batch closed an accumulation window *and* the optimizer
            actually updated (False on AMP inf/nan-skipped steps), so the
            Trainer can suppress the LR-scheduler advance for skipped steps.
        """
        assert trainer.model is not None
        assert trainer.optimizer is not None

        accum = max(1, trainer.accumulate_grad_batches)
        is_window_start = trainer._micro_step % accum == 0
        is_window_end = (trainer._micro_step + 1) % accum == 0

        # Only resolve the model device when autocast is actually used —
        # otherwise this walked model.parameters() every micro-batch for a
        # context that is a no-op.
        if trainer.autocast is not None:
            device_type = next(trainer.model.parameters()).device.type
            ctx = torch.amp.autocast(device_type, dtype=trainer.autocast)
        else:
            ctx = nullcontext()
        with ctx:
            loss, predictions = self.forward_loss(trainer, state, batch)

        if is_window_start:
            trainer.optimizer.zero_grad()

        scaler = trainer.scaler  # None unless the Trainer autocasts at float16
        scaled_loss = loss / accum
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        optimizer_applied = False
        if is_window_end:
            if scaler is not None:
                scaler.unscale_(trainer.optimizer)
                trainer._call_hooks("on_after_backward", trainer, state)
                scale_before = scaler.get_scale()
                scaler.step(trainer.optimizer)
                scaler.update()
                # If the scaler detected inf/nan grads it skips the optimizer
                # and shrinks the scale — detect that so the scheduler doesn't
                # advance on a step that never applied.
                optimizer_applied = scaler.get_scale() >= scale_before
            else:
                trainer._call_hooks("on_after_backward", trainer, state)
                trainer.optimizer.step()
                optimizer_applied = True

        state["train"]["loss"] = loss.detach()
        publish_terms(trainer.loss_fn, predictions, batch, state["train"])
        return {
            "loss": loss,
            "predictions": predictions,
            "optimizer_applied": optimizer_applied,
        }

    def on_eval_batch(self, trainer: "Trainer", state: "TrainState", batch: Any) -> dict[str, Any]:
        """Not supported — :class:`DefaultTrainStep` handles train batches only.

        Always raises :class:`NotImplementedError`; use
        :class:`~molix.core.steps.eval.DefaultEvalStep` for evaluation
        batches.
        """
        raise NotImplementedError(
            "DefaultTrainStep.on_eval_batch() is not implemented. "
            "Use DefaultEvalStep for evaluation batches."
        )
