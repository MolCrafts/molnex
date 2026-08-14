"""Gradient-balance audit hook: fires a probe at fixed training-step marks.

Cadence wrapper around :class:`molix.core.audit.TaskGradientProbe`. The
primitive owns the measurement (one backward per task, per-group
``g_{t,G} = ||∂L_t/∂θ_G||₂``, full state restore); this hook owns only
*when* it runs and *where* the resulting scalar is published. See
:mod:`molix.core.audit` for the definitions, the dimensional precondition
and the references.

The same fixed ``batches`` sequence is replayed at every probe point so the
snapshots are comparable, and the probe restores every ``.grad`` buffer
byte-exactly, so firing from ``on_train_batch_end`` is safe under gradient
accumulation. Hook callbacks run outside the trainer's
``torch.amp.autocast`` block, so the recorded norms are unscaled.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from molix.core.audit import Batch, GradientBalanceReport, TaskGradientProbe
from molix.core.hook import ScalarHook

if TYPE_CHECKING:
    from molix.core.state import TrainState
    from molix.core.trainer import Trainer


class GradientBalanceHook(ScalarHook):
    """Record per-task gradient-balance probe points at fixed step marks.

    Fires the probe once from ``on_train_start`` (label ``"step_0"``) and
    once from ``on_train_batch_end`` for every other entry of ``at_steps``,
    as soon as ``state["global_step"]`` reaches it (label ``"step_<n>"``).
    After each probe point the maximum shared-group imbalance
    ``max_G I_G`` is published to ``state["train"]["grad_imbalance"]``
    through the nested write form; a run whose groups are all task-specific
    publishes nothing rather than a placeholder.

    Args:
        probe: The measurement primitive, already bound to the model under
            training and to the ``{task: loss_fn}`` mapping.
        batches: Fixed ``(task_name, batch)`` stream, replayed identically
            at every probe point so the snapshots are comparable.
        at_steps: Training-step marks to probe at. ``0`` means "at
            initialization", i.e. from ``on_train_start``.
    """

    scalar_keys = (("train", "grad_imbalance"),)

    def __init__(
        self,
        probe: TaskGradientProbe,
        batches: Sequence[tuple[str, Batch]],
        *,
        at_steps: Iterable[int] = (0, 100),
    ) -> None:
        self.probe = probe
        self.batches = list(batches)
        self.at_steps = tuple(sorted({int(mark) for mark in at_steps}))
        self._recorded: set[int] = set()
        self._report = GradientBalanceReport()

    @property
    def report(self) -> GradientBalanceReport:
        """The probe points recorded so far, in cadence order."""
        return self._report

    def on_train_start(self, trainer: "Trainer", state: "TrainState") -> None:
        """Record the initialization probe point when ``0`` is in ``at_steps``."""
        del trainer
        if 0 in self.at_steps and 0 not in self._recorded:
            self._record(0, state)

    def on_train_batch_end(
        self, trainer: "Trainer", state: "TrainState", batch: object, outputs: object
    ) -> None:
        """Record each remaining probe point once its step mark is reached."""
        del trainer, batch, outputs
        global_step = int(state["global_step"])
        for mark in self.at_steps:
            if mark != 0 and mark not in self._recorded and global_step >= mark:
                self._record(mark, state)

    def _record(self, at_step: int, state: "TrainState") -> None:
        """Run the probe once and publish the shared-group imbalance."""
        self._recorded.add(at_step)
        snapshot = self.probe.run(self.batches, label=f"step_{at_step}", at_step=at_step)
        self._report.add(snapshot)
        shared = [
            value
            for value in self._report.imbalance()[snapshot.label].values()
            if value is not None
        ]
        if shared:
            state["train"]["grad_imbalance"] = max(shared)
