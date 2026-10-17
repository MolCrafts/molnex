"""Training-loop intervention hooks: grad clipping, LR warmup, activation checkpointing.

Per the io-hooks-refactor scalar-vs-training split, this module
houses hooks that *modify* the training loop (gradient clipping,
activation checkpointing). Hooks whose only side-effect is writing
scalars into :class:`~molix.core.state.TrainState` live in
:mod:`molix.hooks.scalar`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
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


class LinearWarmupHook(ScalarHook):
    """Ramp every param group's learning rate linearly over the first *steps*.

    At optimizer step ``t < steps`` the rate is ``base * factor(t)`` with
    ``factor(t) = start_factor + (1 - start_factor) * t / steps``; at ``t ==
    steps`` it is back to ``base`` and the hook is a no-op from then on. The
    ramp is applied **incrementally** — each step multiplies the group's
    *current* rate by ``factor(t) / factor(t - 1)`` — so a reduction another
    scheduler makes mid-warmup (``ReduceLROnPlateau`` at an epoch end) is kept
    and scaled, not overwritten by a value the hook remembered. The same
    telescoping makes a resume exact: a checkpoint written after step ``t - 1``
    carries ``base * factor(t - 1)``, and the first call at ``t`` applies the
    next ratio; one restored past *steps* is left alone.

    ``state["train"]["lr"]`` is published on every call (a Python float read,
    no device sync), so a ``Log`` column follows the rate for the whole run.

    Args:
        steps: Optimizer steps the ramp spans. ``> 0``.
        start_factor: Fraction of the base rate at step 0, in ``(0, 1]``.

    Raises:
        ValueError: ``steps <= 0`` or ``start_factor`` outside ``(0, 1]``.
    """

    scalar_keys = (("train", "lr"),)

    def __init__(self, steps: int, *, start_factor: float = 0.01) -> None:
        if steps <= 0:
            raise ValueError(f"LinearWarmupHook: steps must be > 0, got {steps}")
        if not 0.0 < start_factor <= 1.0:
            raise ValueError(
                f"LinearWarmupHook: start_factor must be in (0, 1], got {start_factor}"
            )
        self.steps = int(steps)
        self.start_factor = float(start_factor)
        # The optimizer step whose ratio has been applied; with gradient
        # accumulation ``on_train_batch_start`` fires more than once per step.
        self._applied: int | None = None

    def factor(self, step: int) -> float:
        """The multiplier at optimizer step *step*: ``start_factor`` at 0, 1 from *steps* on."""
        if step >= self.steps:
            return 1.0
        return self.start_factor + (1.0 - self.start_factor) * step / self.steps

    def on_train_batch_start(self, trainer, state, batch) -> None:
        """Apply this step's ratio to every param group once; publish the rate."""
        del batch
        step = int(state.global_step)
        if step <= self.steps and step != self._applied:
            previous = 1.0 if step == 0 else self.factor(step - 1)
            ratio = self.factor(step) / previous
            for group in trainer.optimizer.param_groups:
                group["lr"] = group["lr"] * ratio
            self._applied = step
        state["train"]["lr"] = float(trainer.optimizer.param_groups[0]["lr"])


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

        module.forward = _make_checkpointed(original_forward)


def _group_name(group: dict, index: int) -> str:
    name = group.get("name")
    return str(name) if name else f"group_{index}"


def _group_reached(group: dict) -> bool:
    """True when some parameter carries a finite, nonzero gradient."""
    for param in group["params"]:
        grad = param.grad
        if grad is None:
            continue
        if bool(torch.isfinite(grad).all()) and bool(grad.detach().abs().sum() != 0):
            return True
    return False


class RequiredGradientHook(BaseHook):
    """Fail the run when a declared optimizer group gets no gradient.

    Checked once, from the first ``on_after_backward``. ``names`` are the
    ``"name"`` entries of ``optimizer.param_groups`` (an unnamed group is
    ``group_<index>``). A group whose every parameter has ``grad is None``,
    or only zeros, or a non-finite grad, is named in the error. A loss
    that moves is not that check: the scalar can fall while a head the
    objective claims to train is absent from the graph.

    Args:
        names: Parameter-group names this objective must reach.

    Raises:
        ValueError: ``names`` is empty.
        RuntimeError: A named group is missing from the optimizer, or it
            has no finite nonzero gradient after the first backward.
    """

    def __init__(self, names: Sequence[str]) -> None:
        declared = tuple(names)
        if not declared or any(not name for name in declared):
            raise ValueError(
                "RequiredGradientHook needs the optimizer param-group names this "
                f"objective must reach; got {list(names)}."
            )
        self.names = declared
        self._checked = False

    def on_after_backward(self, trainer, state) -> None:
        """Raise if any declared group was not reached by this backward."""
        del state
        if self._checked:
            return
        self._checked = True
        groups = {
            _group_name(group, index): _group_reached(group)
            for index, group in enumerate(trainer.optimizer.param_groups)
        }
        unknown = [name for name in self.names if name not in groups]
        if unknown:
            raise RuntimeError(
                f"RequiredGradientHook: no optimizer param group named {unknown}; "
                f"groups are {list(groups)}."
            )
        dead = [name for name in self.names if not groups[name]]
        if dead:
            raise RuntimeError(
                f"RequiredGradientHook: {dead} received no finite nonzero gradient "
                "after the first backward. Those parameters are not in this "
                "objective's graph."
            )
