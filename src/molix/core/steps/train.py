"""Default training step implementation."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import torch

from molix.config import config

if TYPE_CHECKING:
    from molix.core.state import TrainState
    from molix.core.trainer import Trainer


class DefaultTrainStep:
    """Default training step with AMP + gradient-accumulation support.

    Precision is controlled globally via :meth:`molix.config.MolnexConfig.set_precision`
    which writes ``use_amp`` and ``amp_dtype`` into the global
    :data:`molix.config.config`. When ``use_amp`` is true the forward pass
    runs under :func:`torch.amp.autocast` and backward uses the
    :class:`torch.amp.GradScaler` owned by the :class:`Trainer`.

    Gradient accumulation is a *Trainer* concern: this step reads
    ``trainer.accumulate_grad_batches`` and ``trainer._micro_step`` to decide
    window boundaries. The loss is scaled by ``1/accumulate_grad_batches`` so
    the accumulated gradient matches a single large batch. The optimizer is
    stepped (and ``on_after_backward`` fired, and ``zero_grad`` issued) only
    on accumulation boundaries; the Trainer advances ``global_step`` / the LR
    scheduler only on those same boundaries.

    Flow per micro-batch:
        1. (window start) ``optimizer.zero_grad()``
        2. Forward (under autocast if AMP), loss, ``/= accum``
        3. Backward (scaled by ``trainer.scaler`` under AMP)
        4. (window end) unscale → ``on_after_backward`` → optimizer step

    ``state["train"]["loss"]`` is written as a *detached tensor* (the
    un-scaled, full-magnitude loss), not a Python float — materialising it
    here would force a CPU↔GPU sync every micro-step and serialise the
    launch-bound pipeline. Consumers (``Log``, ``ProgressBarHook``) call
    ``.item()`` on their own throttled cadence.
    """

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
        assert trainer.loss_fn is not None
        assert trainer.optimizer is not None

        accum = max(1, trainer.accumulate_grad_batches)
        is_window_start = trainer._micro_step % accum == 0
        is_window_end = (trainer._micro_step + 1) % accum == 0

        # Only resolve the model device + AMP dtype when autocast is actually
        # used — otherwise this walked model.parameters() and hit the config
        # dict every micro-batch for a context that is a no-op.
        amp_enabled = bool(config["use_amp"])
        if amp_enabled:
            device_type = next(trainer.model.parameters()).device.type
            ctx = torch.amp.autocast(device_type, dtype=config["amp_dtype"])
        else:
            ctx = nullcontext()
        with ctx:
            predictions = trainer.model(batch)
            loss = trainer.loss_fn(predictions, batch)

        if is_window_start:
            trainer.optimizer.zero_grad()

        scaler = trainer.scaler if amp_enabled else None
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
