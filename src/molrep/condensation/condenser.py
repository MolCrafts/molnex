"""Physics-aware greedy merge of continuous MM parameters into discrete types.

:class:`Condenser` never builds energy graphs. Optional physical gating uses an
injected ``physical_eval`` callable (torch kernel, molpy evaluator, or test
double) whose residuals feed :class:`PhysicalErrorMetrics`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from molrep.condensation.assignment import ClassAssignment
from molrep.condensation.classes import InteractionClass
from molrep.condensation.criterion import MergeCriterion, default_criterion
from molrep.condensation.metrics import PhysicalErrorMetrics, parse_physical_eval_result
from molrep.condensation.type_system import TypeSystem

__all__ = ["Condenser", "CondensationResult", "PhysicalEval"]

# physical_eval(prototype, candidate, interaction) -> residual
PhysicalEval = Callable[
    [Mapping[str, torch.Tensor], Mapping[str, torch.Tensor], InteractionClass],
    Any,
]


@dataclass(frozen=True)
class CondensationResult:
    """Outputs of one condensation run.

    Attributes:
        type_system: Discrete type table for the interaction class.
        assignment: Per-row global type ids with system provenance.
        metrics: Physical residual aggregates (empty if no ``physical_eval``).
    """

    type_system: TypeSystem
    assignment: ClassAssignment
    metrics: PhysicalErrorMetrics


def _infer_n_rows(params: Mapping[str, torch.Tensor | float] | torch.Tensor) -> int:
    if isinstance(params, torch.Tensor):
        if params.ndim == 0:
            return 1
        return int(params.shape[0])
    if not params:
        return 0
    first = next(iter(params.values()))
    if isinstance(first, torch.Tensor):
        if first.ndim == 0:
            return 1
        return int(first.shape[0])
    return 1


def _row_at(
    params: Mapping[str, torch.Tensor | float] | torch.Tensor,
    index: int,
    *,
    param_keys: Sequence[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Extract one interaction row as a float-tensor mapping."""
    if isinstance(params, torch.Tensor):
        # Bare tensor: treat as a single unnamed feature vector → "value"
        # or a stacked table (N, D) with optional key names.
        t = params.detach().to(dtype=torch.float64)
        if t.ndim == 0:
            return {"value": t.clone()}
        if t.ndim == 1:
            # Ambiguous: (D,) one row vs (N,) scalar-per-row.
            # Prefer scalar-per-row when param_keys is None and we index.
            if param_keys is not None and len(param_keys) == t.numel():
                return {
                    k: t[i].detach().to(dtype=torch.float64).clone()
                    for i, k in enumerate(param_keys)
                }
            return {"value": t[index].detach().to(dtype=torch.float64).clone()}
        # (N, D) or (N, ...)
        row = t[index]
        if param_keys is not None and row.ndim == 1 and len(param_keys) == row.numel():
            return {
                k: row[i].detach().to(dtype=torch.float64).clone() for i, k in enumerate(param_keys)
            }
        return {"value": row.clone()}

    out: dict[str, torch.Tensor] = {}
    for key, val in params.items():
        if isinstance(val, torch.Tensor):
            t = val.detach().to(dtype=torch.float64)
            if t.ndim == 0:
                out[key] = t.clone()
            else:
                out[key] = t[index].clone()
        else:
            out[key] = torch.as_tensor(val, dtype=torch.float64)
    return out


def _flatten_systems(
    params_by_system: Sequence[Mapping[str, torch.Tensor | float] | torch.Tensor],
) -> list[tuple[int, int, dict[str, torch.Tensor]]]:
    """Expand multi-system params to ``(system_id, row_index, row_params)``.

    Sort key (documented, deterministic): ``(system_id ascending,
    row_index ascending)`` — insertion order within each system, systems in
    the order provided.
    """
    rows: list[tuple[int, int, dict[str, torch.Tensor]]] = []
    for sys_id, params in enumerate(params_by_system):
        n = _infer_n_rows(params)
        for row_i in range(n):
            rows.append((sys_id, row_i, _row_at(params, row_i)))
    # Explicit sort for determinism even if caller order changes later.
    rows.sort(key=lambda item: (item[0], item[1]))
    return rows


class Condenser:
    """Physics-aware greedy merge across one or many chemical systems.

    Algorithm
    ---------
    1. Flatten ``params_by_system`` with ``system_id`` tracking.
    2. Sort by documented key ``(system_id, row_index)``.
    3. For each row: assign to the first existing type whose prototype
       passes :class:`MergeCriterion.accepts` **and** the optional physics
       gate; otherwise spawn a new type.
    4. Update the type centroid as an online mean of its members.

    Multi-system merges produce **global** type ids (not renumbered per
    molecule). No SMARTS text is emitted.

    Args:
        update_centroid: If True (default), absorb updates the prototype
            mean; if False, the first member freezes the prototype.
    """

    def __init__(self, *, update_centroid: bool = True) -> None:
        self.update_centroid = update_centroid

    def merge(
        self,
        params_by_system: Sequence[Mapping[str, torch.Tensor | float] | torch.Tensor],
        *,
        interaction: InteractionClass,
        criterion: MergeCriterion | None = None,
        physical_eval: PhysicalEval | None = None,
        energy_tol: float | None = None,
        force_tol: float | None = None,
    ) -> CondensationResult:
        """Greedy merge continuous parameters into a discrete type system.

        Args:
            params_by_system: Sequence of per-system parameter bags. Each bag
                is a mapping of Class-I tensors with leading batch dim
                ``(N_sys, ...)`` (or a bare tensor of rows).
            interaction: Interaction family for this merge.
            criterion: Merge budgets. Defaults to
                :func:`~molrep.condensation.criterion.default_criterion`.
            physical_eval: Optional injected residual evaluator
                ``(prototype, candidate, interaction) -> residual``.
                Condensation does **not** import molpot energy graphs; the
                caller supplies any physical oracle.
            energy_tol: If set with ``physical_eval``, reject merges whose
                absolute energy residual exceeds this value.
            force_tol: If set with ``physical_eval``, reject merges whose
                absolute force residual exceeds this value.

        Returns:
            :class:`CondensationResult` with type system, assignment table,
            and physical metrics.
        """
        crit = criterion if criterion is not None else default_criterion(interaction)
        if crit.interaction != interaction:
            raise ValueError(
                f"criterion.interaction={crit.interaction} does not match interaction={interaction}"
            )

        type_system = TypeSystem(interaction, criterion=crit)
        metrics = PhysicalErrorMetrics()
        rows = _flatten_systems(params_by_system)

        type_ids: list[int] = []
        system_ids: list[int] = []
        row_indices: list[int] = []

        for flat_id, (sys_id, row_i, row_params) in enumerate(rows):
            assigned = self._try_assign(
                type_system,
                row_params,
                criterion=crit,
                physical_eval=physical_eval,
                energy_tol=energy_tol,
                force_tol=force_tol,
                metrics=metrics,
                support_id=flat_id,
            )
            if assigned is None:
                assigned = type_system._spawn(row_params, support_id=flat_id)
            type_ids.append(assigned)
            system_ids.append(sys_id)
            row_indices.append(row_i)

        assignment = ClassAssignment(
            interaction=interaction,
            type_ids=torch.tensor(type_ids, dtype=torch.long),
            system_ids=torch.tensor(system_ids, dtype=torch.long),
            row_indices=torch.tensor(row_indices, dtype=torch.long),
        )
        return CondensationResult(
            type_system=type_system,
            assignment=assignment,
            metrics=metrics,
        )

    def greedy_merge(
        self,
        params_by_system: Sequence[Mapping[str, torch.Tensor | float] | torch.Tensor],
        *,
        interaction: InteractionClass,
        criterion: MergeCriterion | None = None,
        physical_eval: PhysicalEval | None = None,
        energy_tol: float | None = None,
        force_tol: float | None = None,
    ) -> CondensationResult:
        """Alias of :meth:`merge` (spec name ``Condenser.greedy_merge``)."""
        return self.merge(
            params_by_system,
            interaction=interaction,
            criterion=criterion,
            physical_eval=physical_eval,
            energy_tol=energy_tol,
            force_tol=force_tol,
        )

    def _try_assign(
        self,
        type_system: TypeSystem,
        row_params: Mapping[str, torch.Tensor],
        *,
        criterion: MergeCriterion,
        physical_eval: PhysicalEval | None,
        energy_tol: float | None,
        force_tol: float | None,
        metrics: PhysicalErrorMetrics,
        support_id: int,
    ) -> int | None:
        """Return type_id if an existing type accepts, else None."""
        for rec in type_system.records():
            proto_t = rec.prototype_tensors()
            if not criterion.accepts(proto_t, row_params):
                continue
            if physical_eval is not None and (energy_tol is not None or force_tol is not None):
                raw = physical_eval(proto_t, row_params, type_system.interaction)
                e_err, f_err = parse_physical_eval_result(raw)
                rejected = False
                if energy_tol is not None and e_err is not None and e_err > energy_tol:
                    rejected = True
                if force_tol is not None and f_err is not None and f_err > force_tol:
                    rejected = True
                metrics.record(energy_error=e_err, force_error=f_err, rejected=rejected)
                if rejected:
                    continue
            elif physical_eval is not None:
                # Evaluate for metrics only (no hard gate without tolerances).
                raw = physical_eval(proto_t, row_params, type_system.interaction)
                e_err, f_err = parse_physical_eval_result(raw)
                metrics.record(energy_error=e_err, force_error=f_err, rejected=False)

            type_system._absorb(
                rec.type_id,
                row_params,
                support_id=support_id,
                update_centroid=self.update_centroid,
            )
            return rec.type_id
        return None
