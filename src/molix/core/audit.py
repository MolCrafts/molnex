"""Per-task gradient-balance audit: is a fixed unit task weight defensible?

A multi-task objective ``L = Σ_t w_t · L_t`` with ``w_t = 1`` is only
defensible when no single task dominates the gradient that reaches the
shared parameters. This module measures that, and nothing else.

Definitions
-----------
For task ``t`` and parameter group ``G`` (``θ_G`` is the parameter subset
of ``G``, by default one group per direct child module — encoder, bond
head, angle head, ... ), the **per-task, per-group gradient norm** is::

    g_{t,G} = ||∂L_t/∂θ_G||₂ = sqrt( Σ_{θ_i ∈ θ_G} (∂L_t/∂θ_i)² )

recorded as the mean over the per-batch values of a bounded probe stream
— per batch, because the per-batch norm is what an update actually sees.
The dimensionless **ratio table** and **imbalance** follow::

    r_{t,G} = g_{t,G} / max_{t'} g_{t',G}      ∈ (0, 1]
    I_G     = max_t g_{t,G} / min_t g_{t,G}    defined only when min_t g_{t,G} > 0

A group with ``min_t g_{t,G} = 0`` is **task-specific** (a head only one
task touches); ``I_G`` is reported as ``None`` there and excluded from the
verdict — a head receiving gradient from one task only is correct, not an
imbalance. The verdict is therefore taken over **shared** groups only::

    fixed unit weights (w_t = 1 ∀t) are justified
        ⟺ I_G ≤ max_ratio (default 10, one decade) for every shared group G

Dimensional precondition (binding)
----------------------------------
``g_{t,G}`` carries units ``[L_t]·[θ_G]⁻¹``. Comparing two tasks inside one
group is only dimensionally meaningful when all ``[L_t]`` agree — energy
MSE is ``kcal²·mol⁻²``, force MSE is ``(kcal·mol⁻¹·Å⁻¹)²``, charge MSE is
``e²``. ForceBalance-style normalization makes every task loss
dimensionless (``[L_t] = 1``); in this repo that is
:class:`molix.core.losses.molecular.ScaleNormalizedLoss` (``L/s²``).
:class:`TaskGradientProbe` therefore records a caller-supplied unit label
per task (default ``"1"``) and
:meth:`GradientBalanceReport.balanced` refuses a verdict when the labels
are not all identical. ``r_{t,G}`` and ``I_G`` are dimensionless.

Boundaries
----------
The probe operates **in place on the caller's model** and is
state-preserving: parameter bytes, every ``.grad`` buffer (including which
ones were ``None``), ``model.training`` and the RNG states are captured on
entry and restored in a ``finally`` block. It never constructs or advances
a training update, never touches ``global_step``, and is meant to run
outside ``torch.amp.autocast`` / ``GradScaler`` (which the
:class:`molix.core.trainer.Trainer` hook points already are) so recorded
norms are unscaled and comparable. Rendering and persistence are the
caller's job: this module has no I/O and no experiment-tracking imports.

References:
    Qiu, Y. et al. "Development and Benchmarking of Open Force Field
    v1.0.0." J. Chem. Theory Comput. 17, 6262-6280 (2021).
    https://doi.org/10.1021/acs.jctc.1c00571 — ForceBalance objective
    normalization: each residual is divided by a reference denominator so
    that unit weights are defensible.

    Takaba, K. et al. "Machine-learned molecular mechanics force fields
    from large-scale quantum chemical data." arXiv:2307.07085 —
    espaloma-0.3 sets all task weights to 1 (SI §C.2.3); this audit is the
    local reproduction of the evidence behind that choice.

    Escalation path only, **not v0**: Chen, Z. et al. "GradNorm."
    arXiv:1711.02257; Kendall, A., Gal, Y., Cipolla, R. "Multi-Task
    Learning Using Uncertainty to Weigh Losses." arXiv:1705.07115. If a
    shared group exceeds one decade these are the recorded next steps;
    nothing here implements adaptive weighting.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import torch
from torch import nn

from molix.core.checkpoint.rng import capture_rng_states, restore_rng_states

#: A task-labelled batch as consumed by the probe stream.
Batch = Mapping[str, torch.Tensor]

#: Model predictions keyed by task name.
Predictions = Mapping[str, torch.Tensor]

#: Loss callable in the molix ``loss_fn(predictions, batch)`` convention.
LossFn = Callable[[Predictions, Batch], torch.Tensor]


@dataclass(frozen=True)
class TaskGradientSnapshot:
    """One probe point: per-task loss and per-task, per-group gradient norm.

    Attributes:
        label: Probe-point label, e.g. ``"step_0"`` / ``"step_100"``.
        at_step: Number of training updates completed when the probe ran.
        loss: Task -> mean ``L_t`` over that task's probe batches.
        grad_norm: Task -> group -> ``g_{t,G}``, units ``[L_t]·[θ_G]⁻¹``.
        n_batches: Task -> number of probe batches consumed.
        units: Task -> unit label of ``L_t`` (``"1"`` when dimensionless).
    """

    label: str
    at_step: int
    loss: dict[str, float]
    grad_norm: dict[str, dict[str, float]]
    n_batches: dict[str, int]
    units: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        """Return the snapshot as plain JSON types.

        Returns:
            Mapping with ``label``, ``at_step``, ``loss``, ``grad_norm``,
            ``n_batches`` and ``units``, containing only ``str``, ``int``,
            ``float`` and ``dict`` values.
        """
        return {
            "label": self.label,
            "at_step": int(self.at_step),
            "loss": {task: float(value) for task, value in self.loss.items()},
            "grad_norm": {
                task: {group: float(value) for group, value in groups.items()}
                for task, groups in self.grad_norm.items()
            },
            "n_batches": {task: int(count) for task, count in self.n_batches.items()},
            "units": {task: str(unit) for task, unit in self.units.items()},
        }


class TaskGradientProbe:
    """Measure ``g_{t,G} = ||∂L_t/∂θ_G||₂`` without changing the caller's model.

    One backward per ``(task, batch)`` pair isolates the per-task gradient
    that a summed multi-task objective would otherwise mix. Parameter
    bytes, gradient buffers, training mode and RNG states are restored on
    exit, so the probe can be fired mid-training (including under gradient
    accumulation) without perturbing the run.

    Args:
        model: The model under audit; used in place, never deep-copied.
        losses: Task name -> ``loss_fn(predictions, batch) -> Tensor`` in
            the molix loss convention. Every task listed here must receive
            at least one batch in each probe stream.
        groups: Optional group name -> parameters. Defaults to one group
            per direct child module holding trainable parameters, falling
            back to ``{"model": <all trainable parameters>}``. Resolved
            once here, so later reparameterization is not tracked.
        units: Optional task -> unit label of ``L_t``. Defaults to ``"1"``
            (dimensionless) for every task, which is what
            ``ScaleNormalizedLoss`` delivers.

    Raises:
        ValueError: If ``losses`` is empty, or if no trainable parameter
            is reachable through ``groups`` / the default grouping.
    """

    def __init__(
        self,
        model: nn.Module,
        losses: Mapping[str, LossFn],
        *,
        groups: Mapping[str, Iterable[nn.Parameter]] | None = None,
        units: Mapping[str, str] | None = None,
    ) -> None:
        if not losses:
            raise ValueError("TaskGradientProbe needs at least one task loss")
        self._model = model
        self._losses: dict[str, LossFn] = dict(losses)
        self._groups = self._resolve_groups(model, groups)
        self._units: dict[str, str] = (
            {task: "1" for task in self._losses} if units is None else dict(units)
        )

    @staticmethod
    def _resolve_groups(
        model: nn.Module,
        groups: Mapping[str, Iterable[nn.Parameter]] | None,
    ) -> dict[str, tuple[nn.Parameter, ...]]:
        """Freeze the parameter grouping to ``name -> tuple(parameters)``."""
        if groups is not None:
            resolved = {name: tuple(members) for name, members in groups.items()}
        else:
            resolved = {}
            for name, child in model.named_children():
                trainable = tuple(p for p in child.parameters() if p.requires_grad)
                if trainable:
                    resolved[name] = trainable
            if not resolved:
                resolved = {"model": tuple(p for p in model.parameters() if p.requires_grad)}
        if not any(resolved.values()):
            raise ValueError("TaskGradientProbe found no trainable parameter to measure")
        return resolved

    def run(
        self,
        batches: Iterable[tuple[str, Batch]],
        *,
        label: str,
        at_step: int = 0,
    ) -> TaskGradientSnapshot:
        """Consume a bounded ``(task, batch)`` stream and record one probe point.

        For each pair the probe zeroes gradients, runs the forward, applies
        that task's loss, backwards it alone, and reads the L2 norm of each
        parameter group. Per-task results are means over that task's pairs.

        Args:
            batches: Bounded iterable of ``(task_name, batch)`` pairs — the
                shape a task-mixture sampler yields. Every task in
                ``losses`` must appear at least once.
            label: Probe-point label carried onto the snapshot.
            at_step: Number of completed training updates at this probe
                point, carried onto the snapshot.

        Returns:
            The :class:`TaskGradientSnapshot` for this probe point.

        Raises:
            ValueError: If the stream carries a task with no loss callable,
                if a task in ``losses`` receives no batch (a silent zero row
                would be fabricated evidence), or if a loss or group norm is
                non-finite — the message names the offending task (and
                group, for a norm).
        """
        model = self._model
        saved_grads = [
            (parameter, None if parameter.grad is None else parameter.grad.detach().clone())
            for parameter in model.parameters()
        ]
        saved_parameters = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        saved_mode = model.training
        saved_rng = capture_rng_states()
        total_loss: dict[str, float] = {}
        total_norm: dict[str, dict[str, float]] = {}
        counts: dict[str, int] = {}
        try:
            with torch.enable_grad():
                for task, batch in batches:
                    if task not in self._losses:
                        raise ValueError(
                            f"probe stream carries task {task!r}, which has no loss callable "
                            f"(known tasks: {', '.join(sorted(self._losses))})"
                        )
                    value, norms = self._measure(task, batch)
                    total_loss[task] = total_loss.get(task, 0.0) + value
                    counts[task] = counts.get(task, 0) + 1
                    accumulated = total_norm.setdefault(task, {})
                    for group, norm in norms.items():
                        accumulated[group] = accumulated.get(group, 0.0) + norm
            starved = sorted(task for task in self._losses if task not in counts)
            if starved:
                raise ValueError(
                    f"tasks received no probe batch: {', '.join(starved)} — refusing to "
                    "record a fabricated zero row"
                )
        finally:
            model.load_state_dict(saved_parameters)
            for parameter, gradient in saved_grads:
                parameter.grad = gradient
            model.train(saved_mode)
            restore_rng_states(saved_rng)
        return TaskGradientSnapshot(
            label=label,
            at_step=at_step,
            loss={task: total_loss[task] / counts[task] for task in sorted(counts)},
            grad_norm={
                task: {
                    group: total / counts[task] for group, total in sorted(total_norm[task].items())
                }
                for task in sorted(counts)
            },
            n_batches={task: counts[task] for task in sorted(counts)},
            units={task: self._units.get(task, "1") for task in sorted(counts)},
        )

    def _measure(self, task: str, batch: Batch) -> tuple[float, dict[str, float]]:
        """Return ``(L_t, {group: g_{t,G}})`` for one ``(task, batch)`` pair."""
        model = self._model
        model.zero_grad(set_to_none=True)
        predictions = model(batch)
        loss = self._losses[task](predictions, batch)
        value = float(loss.detach())
        if not math.isfinite(value):
            raise ValueError(f"non-finite loss {value} for task {task!r}")
        loss.backward()
        norms: dict[str, float] = {}
        for group, parameters in self._groups.items():
            squared = 0.0
            for parameter in parameters:
                gradient = parameter.grad
                if gradient is not None:
                    squared += float(gradient.detach().double().square().sum())
            norm = math.sqrt(squared)
            if not math.isfinite(norm):
                raise ValueError(
                    f"non-finite gradient norm {norm} for task {task!r} group {group!r}"
                )
            norms[group] = norm
        return value, norms


class GradientBalanceReport:
    """Dimensionless ratio table over one or more probe points.

    Holds :class:`TaskGradientSnapshot` instances in insertion order and
    derives ``r_{t,G}``, ``I_G`` and the unit-weight verdict from them.
    Rendering is deliberately not offered: :meth:`to_dict` is plain JSON
    and the caller decides where it goes.
    """

    def __init__(self) -> None:
        self._snapshots: list[TaskGradientSnapshot] = []

    @property
    def snapshots(self) -> tuple[TaskGradientSnapshot, ...]:
        """The recorded probe points, in the order they were added."""
        return tuple(self._snapshots)

    def add(self, snapshot: TaskGradientSnapshot) -> None:
        """Append one probe point.

        Args:
            snapshot: The probe point to record.

        Raises:
            ValueError: If a snapshot with the same label is already held —
                two probe points may not share an identity.
        """
        if any(held.label == snapshot.label for held in self._snapshots):
            raise ValueError(f"duplicate snapshot label {snapshot.label!r}")
        self._snapshots.append(snapshot)

    def ratios(self) -> dict[str, dict[str, dict[str, float]]]:
        """Return ``label -> group -> task -> r_{t,G}``.

        Returns:
            Per-probe-point ratio table, each group normalized by its own
            largest per-task gradient norm (so the dominant task reads
            ``1.0``). A group with no gradient at all reads ``0.0``.
        """
        table: dict[str, dict[str, dict[str, float]]] = {}
        for snapshot in self._snapshots:
            per_group: dict[str, dict[str, float]] = {}
            for group in self._group_names(snapshot):
                norms = self._group_norms(snapshot, group)
                largest = max(norms.values())
                per_group[group] = {
                    task: (norm / largest if largest > 0.0 else 0.0) for task, norm in norms.items()
                }
            table[snapshot.label] = per_group
        return table

    def imbalance(self) -> dict[str, dict[str, float | None]]:
        """Return ``label -> group -> I_G``, with ``None`` for task-specific groups.

        Returns:
            Per-probe-point imbalance table. ``I_G = max_t g / min_t g`` for
            shared groups; ``None`` where ``min_t g_{t,G} = 0``, i.e. a group
            only one task touches — correct, not an imbalance.
        """
        table: dict[str, dict[str, float | None]] = {}
        for snapshot in self._snapshots:
            per_group: dict[str, float | None] = {}
            for group in self._group_names(snapshot):
                norms = list(self._group_norms(snapshot, group).values())
                smallest = min(norms)
                per_group[group] = max(norms) / smallest if smallest > 0.0 else None
            table[snapshot.label] = per_group
        return table

    def balanced(self, *, max_ratio: float = 10.0) -> bool:
        """Return whether fixed unit task weights are justified.

        The verdict holds when every **shared** group stays within
        ``max_ratio`` at **every** recorded probe point.

        Args:
            max_ratio: Largest admissible ``I_G``; the default of one decade
                is the threshold recorded in the spec.

        Returns:
            ``True`` when ``I_G ≤ max_ratio`` for every shared group at every
            probe point, ``False`` otherwise.

        Raises:
            ValueError: If no probe point was recorded, if a probe point
                covers fewer than two tasks, if a probe point mixes task unit
                labels (``g_{t,G}`` would not be comparable), or if a probe
                point has no shared group. All four mean "no comparison is
                possible"; returning ``True`` there would launder a
                non-result into a GO.
        """
        if not self._snapshots:
            raise ValueError("no probe point recorded — nothing to compare")
        table = self.imbalance()
        verdict = True
        for snapshot in self._snapshots:
            if len(snapshot.grad_norm) < 2:
                raise ValueError(
                    f"probe point {snapshot.label!r} covers fewer than two tasks — "
                    "a single-task run admits no balance verdict"
                )
            labels = sorted(set(snapshot.units.values()))
            if len(labels) != 1:
                raise ValueError(
                    f"probe point {snapshot.label!r} mixes task unit labels "
                    f"({', '.join(labels)}) — gradient norms are not comparable across "
                    "unlike loss units"
                )
            shared = [value for value in table[snapshot.label].values() if value is not None]
            if not shared:
                raise ValueError(
                    f"probe point {snapshot.label!r} has no shared parameter group — "
                    "every group is task-specific, so no imbalance is defined"
                )
            verdict = verdict and max(shared) <= max_ratio
        return verdict

    def to_dict(self) -> dict[str, object]:
        """Return the report as a plain-JSON evidence dict.

        Returns:
            Mapping with ``schema`` (``"molix.grad_balance/1"``),
            ``snapshots`` (each carrying ``label``, ``at_step``, ``loss``,
            ``grad_norm``, ``n_batches``, ``units``), ``ratios`` and
            ``imbalance``. Persisting it is the caller's job.
        """
        return {
            "schema": "molix.grad_balance/1",
            "snapshots": [snapshot.to_dict() for snapshot in self._snapshots],
            "ratios": self.ratios(),
            "imbalance": self.imbalance(),
        }

    @staticmethod
    def _group_names(snapshot: TaskGradientSnapshot) -> list[str]:
        """Return every group name appearing in ``snapshot``, sorted."""
        names: set[str] = set()
        for groups in snapshot.grad_norm.values():
            names.update(groups)
        return sorted(names)

    @staticmethod
    def _group_norms(snapshot: TaskGradientSnapshot, group: str) -> dict[str, float]:
        """Return ``task -> g_{t,G}`` for one group, missing entries as ``0.0``."""
        return {task: groups.get(group, 0.0) for task, groups in snapshot.grad_norm.items()}
