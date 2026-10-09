"""Train and eval steps that stamp teacher columns before the compiled body.

:meth:`Teacher.write` stays in :meth:`forward_loss`, which is the eager
prologue. :meth:`DefaultTrainStep._traced_forward_loss` — the region
:meth:`~molix.core.trainer.Trainer.compile` replaces — still sees only the
student and the loss. Several teachers run in declaration order; an ensemble
is that tuple, not another type.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from molix.core.distill.teacher import Teacher
from molix.core.steps.eval import DefaultEvalStep
from molix.core.steps.train import DefaultTrainStep

if TYPE_CHECKING:
    from molix.core.state import TrainState
    from molix.core.trainer import Trainer


def _checked_teachers(teachers: Sequence[Teacher]) -> tuple[Teacher, ...]:
    """A non-empty tuple whose destination columns are pairwise disjoint."""
    # A bare Teacher is an nn.Module, and iterating a module yields its
    # children. Reject that before it can be read as a one-teacher sequence.
    if isinstance(teachers, nn.Module):
        raise TypeError("pass teachers as a tuple of Teacher instances, for example (teacher,)")
    group = tuple(teachers)
    if not group:
        raise ValueError("distill step needs at least one teacher")
    seen: set[tuple[str, str]] = set()
    overlaps: list[tuple[str, str]] = []
    for teacher in group:
        for key in teacher.dest_keys:
            if key in seen:
                overlaps.append(key)
            seen.add(key)
    if overlaps:
        raise ValueError(f"teacher dest keys overlap: {overlaps}")
    return group


class DistillTrainStep(DefaultTrainStep):
    """:class:`DefaultTrainStep` with a teacher stamp in the eager prologue.

    Args:
        teachers: One or more teachers, run in order on every micro-batch.
            Destination columns across the tuple must be disjoint.
    """

    def __init__(self, teachers: Sequence[Teacher]) -> None:
        self.teachers = _checked_teachers(teachers)

    def forward_loss(
        self,
        trainer: Trainer,
        state: TrainState,
        batch: Any,
    ) -> tuple[torch.Tensor, Any]:
        """Stamp every teacher, then the default eager leaf plus traced body.

        Args:
            trainer: Owning trainer. Its model is the student.
            state: Training state, threaded through to the traced body.
            batch: Student batch. Teacher columns are written onto it.

        Returns:
            ``(loss, predictions)`` from :class:`DefaultTrainStep`.
        """
        for teacher in self.teachers:
            teacher.write(batch)
        return super().forward_loss(trainer, state, batch)


class DistillEvalStep(DefaultEvalStep):
    """:class:`DefaultEvalStep` with the same teacher stamp as training.

    Args:
        teachers: Same contract as :class:`DistillTrainStep`.
        no_grad: Forwarded to :class:`DefaultEvalStep`.
        objective: Forwarded to :class:`DefaultEvalStep`.
    """

    def __init__(
        self,
        teachers: Sequence[Teacher],
        *,
        no_grad: bool = False,
        objective: Any = None,
    ) -> None:
        super().__init__(no_grad=no_grad, objective=objective)
        self.teachers = _checked_teachers(teachers)

    def forward_loss(
        self,
        trainer: Trainer,
        state: TrainState,
        batch: Any,
    ) -> tuple[torch.Tensor, Any]:
        """Stamp every teacher, then the default eval body.

        Args:
            trainer: Owning trainer. Its model is the student.
            state: Training state, threaded through to the traced body.
            batch: Student batch. Teacher columns are written onto it.

        Returns:
            ``(loss, predictions)`` from :class:`DefaultEvalStep`.
        """
        for teacher in self.teachers:
            teacher.write(batch)
        return super().forward_loss(trainer, state, batch)
