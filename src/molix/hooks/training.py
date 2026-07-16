"""Training-loop intervention hooks: ``GradClipHook``, ``ActivationCheckpointingHook``.

Per the io-hooks-refactor scalar-vs-training split, this module
houses hooks that *modify* the training loop (gradient clipping,
activation checkpointing). Hooks whose only side-effect is writing
scalars into :class:`~molix.core.state.TrainState` live in
:mod:`molix.hooks.scalar`.
"""

from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn

from molix.core.hook import BaseHook, ScalarHook


class GradClipHook(ScalarHook):
    """Clip gradient norms after backward pass.

    Applies ``torch.nn.utils.clip_grad_norm_`` at the
    ``on_after_backward`` stage and writes the pre-clip gradient norm
    to ``state["train"]["grad_norm"]``.

    Args:
        max_norm: Maximum norm of the gradients.
        norm_type: Type of the used p-norm (default: 2.0, i.e. L2).
    """

    scalar_keys = (("train", "grad_norm"),)

    def __init__(self, max_norm: float, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type

    def on_after_backward(self, trainer, state):
        """Clip gradients in-place and record the pre-clip L2 norm.

        The norm is stored as the **0-d device tensor** returned by
        ``clip_grad_norm_`` — *not* ``float()``-ed. Materialising it here would
        force a CPU↔GPU sync on **every** optimizer step, draining the GPU queue
        and serialising the otherwise-overlapped launch pipeline (this alone
        dropped compiled PiNet training from ~19 to a few steps/s). Downstream
        consumers materialise on their own throttled cadence: the async
        :class:`~molix.hooks.JournalHook` copies it through a pinned buffer +
        CUDA event, and progress / TensorBoard hooks coerce via ``.item()`` when
        they sample. Same hot-path contract as
        :class:`~molix.hooks.scalar.MetricsHook` / ``StepSpeedHook``.
        """
        import torch

        total_norm = torch.nn.utils.clip_grad_norm_(
            trainer.model.parameters(),
            self.max_norm,
            norm_type=self.norm_type,
        )
        state["train"]["grad_norm"] = total_norm.detach()


class ActivationCheckpointingHook(BaseHook):
    """Apply activation checkpointing to model layers at training start.

    Wraps matching submodules with ``torch.utils.checkpoint.checkpoint``
    using ``use_reentrant=False`` (recommended for DDP compatibility).

    Args:
        check_fn: Predicate selecting which submodules to wrap.
            If None, wraps all direct children of the model.
    """

    def __init__(self, check_fn: Callable[[nn.Module], bool] | None = None):
        self.check_fn = check_fn

    def on_train_start(self, trainer, state):
        """Wrap matching modules with activation checkpointing."""
        _apply_activation_checkpointing(trainer.model, self.check_fn)


def _apply_activation_checkpointing(
    model: nn.Module,
    check_fn: Callable[[nn.Module], bool] | None = None,
) -> None:
    """Wrap matching submodules with activation checkpointing.

    Args:
        model: The model to apply checkpointing to.
        check_fn: Predicate that selects which submodules to wrap.
            If None, wraps all direct children.
    """
    from torch.utils.checkpoint import checkpoint

    if check_fn is None:
        targets = set(model.children())
    else:
        targets = {m for m in model.modules() if m is not model and check_fn(m)}

    for module in targets:
        original_forward = module.forward

        def _make_checkpointed(fn):
            def checkpointed_forward(*args, **kwargs):
                return checkpoint(fn, *args, use_reentrant=False, **kwargs)

            return checkpointed_forward

        module.forward = _make_checkpointed(original_forward)  # type: ignore[method-assign]
