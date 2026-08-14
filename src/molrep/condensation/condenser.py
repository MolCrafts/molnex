"""Merge continuous MM parameters **within** a chemical environment.

The environment is the type key (classified first). This type never
looks at atoms, never emits SMARTS, and never builds energy graphs.
One environment → one type. If members of a bucket exceed
:class:`MergeCriterion` (or the optional physics gate), the environment
is too coarse — fail loud instead of spawning a second type with the
same key.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from molrep.condensation.assignment import ClassAssignment
from molrep.condensation.classes import InteractionClass
from molrep.condensation.criterion import MergeCriterion, default_criterion
from molrep.condensation.metrics import PhysicalErrorMetrics, parse_physical_eval_result
from molrep.condensation.type_system import TypeSystem

__all__ = ["Condenser", "CondensationResult", "PhysicalEval"]

PhysicalEval = Callable[
    [Mapping[str, torch.Tensor], Mapping[str, torch.Tensor], InteractionClass],
    Any,
]


@dataclass(frozen=True)
class CondensationResult:
    """Outputs of one condensation run.

    Attributes:
        type_system: Discrete type table (one record per environment).
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
        t = params.detach().to(dtype=torch.float64)
        if t.ndim == 0:
            return {"value": t.clone()}
        if t.ndim == 1:
            if param_keys is not None and len(param_keys) == t.numel():
                return {
                    k: t[i].detach().to(dtype=torch.float64).clone()
                    for i, k in enumerate(param_keys)
                }
            return {"value": t[index].detach().to(dtype=torch.float64).clone()}
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
    """Expand to ``(system_id, row_index, row_params)``, sorted."""
    rows: list[tuple[int, int, dict[str, torch.Tensor]]] = []
    for sys_id, params in enumerate(params_by_system):
        n = _infer_n_rows(params)
        for row_i in range(n):
            rows.append((sys_id, row_i, _row_at(params, row_i)))
    rows.sort(key=lambda item: (item[0], item[1]))
    return rows


def _env_key(item: Any) -> Hashable:
    key = getattr(item, "key", item)
    if callable(key):
        key = key()
    return key


def _label(key: Hashable) -> str | None:
    if isinstance(key, str):
        return key
    if isinstance(key, tuple) and key and all(isinstance(p, str) for p in key):
        return "-".join(key)
    return None


def _mean_params(members: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = set(members[0])
    for row in members[1:]:
        keys &= set(row)
    return {
        key: torch.stack([row[key] for row in members], dim=0).mean(dim=0) for key in sorted(keys)
    }


class Condenser:
    """One discrete type per chemical environment; prototype is the bucket mean.

    Algorithm
    ---------
    1. Flatten ``params_by_system`` with ``(system_id, row_index)``.
    2. Group rows by ``environments[s][r]`` (objects with ``.key`` or raw keys).
    3. Each group is one type. Prototype is the member mean (or the first
       member when ``update_centroid`` is False).
    4. Every member must pass :class:`MergeCriterion` against that prototype.
       Optional ``physical_eval`` with a tolerance is the same contract.
       Failure means the environment is too coarse — no second type.

    Multi-system groups share **global** type ids. No SMARTS text.
    """

    def __init__(self, *, update_centroid: bool = True) -> None:
        self.update_centroid = update_centroid

    def merge(
        self,
        params_by_system: Sequence[Mapping[str, torch.Tensor | float] | torch.Tensor],
        *,
        interaction: InteractionClass,
        environments: Sequence[Sequence[Any]],
        criterion: MergeCriterion | None = None,
        physical_eval: PhysicalEval | None = None,
        energy_tol: float | None = None,
        force_tol: float | None = None,
    ) -> CondensationResult:
        """Merge parameters inside each environment bucket.

        Args:
            params_by_system: Per-system parameter bags, leading dim
                ``(N_sys, ...)``.
            interaction: Interaction family.
            environments: ``environments[s][r]`` aligned with param rows.
                Each entry is a hashable key or an object with ``.key``.
            criterion: Merge budgets. Defaults to the class default.
            physical_eval: Optional residual evaluator.
            energy_tol / force_tol: Hard physics gate when ``physical_eval``
                is set.

        Returns:
            :class:`CondensationResult`.

        Raises:
            ValueError: Environment/param length mismatch, or a bucket's
                parameters exceed the criterion / physics gate.
        """
        crit = criterion if criterion is not None else default_criterion(interaction)
        if crit.interaction != interaction:
            raise ValueError(
                f"criterion.interaction={crit.interaction} does not match interaction={interaction}"
            )

        rows = _flatten_systems(params_by_system)
        keys = self._keys(environments, rows)
        groups: OrderedDict[Hashable, list[tuple[int, int, int, dict[str, torch.Tensor]]]] = (
            OrderedDict()
        )
        for flat_id, ((sys_id, row_i, params), key) in enumerate(zip(rows, keys, strict=True)):
            groups.setdefault(key, []).append((flat_id, sys_id, row_i, params))

        type_system = TypeSystem(interaction, criterion=crit)
        metrics = PhysicalErrorMetrics()
        assigned: dict[tuple[int, int], int] = {}

        for key, members in groups.items():
            param_rows = [m[3] for m in members]
            proto = _mean_params(param_rows) if self.update_centroid else dict(param_rows[0])
            self._check_bucket(
                proto,
                param_rows,
                criterion=crit,
                physical_eval=physical_eval,
                energy_tol=energy_tol,
                force_tol=force_tol,
                metrics=metrics,
                interaction=interaction,
            )
            type_id = type_system._spawn(proto, support_id=members[0][0], label=_label(key))
            for flat_id, sys_id, row_i, _ in members[1:]:
                type_system._absorb(
                    type_id,
                    proto,
                    support_id=flat_id,
                    update_centroid=False,
                )
            for _, sys_id, row_i, _ in members:
                assigned[(sys_id, row_i)] = type_id

        type_ids = [assigned[(s, r)] for s, r, _ in rows]
        return CondensationResult(
            type_system=type_system,
            assignment=ClassAssignment(
                interaction=interaction,
                type_ids=torch.tensor(type_ids, dtype=torch.long),
                system_ids=torch.tensor([s for s, _, _ in rows], dtype=torch.long),
                row_indices=torch.tensor([r for _, r, _ in rows], dtype=torch.long),
            ),
            metrics=metrics,
        )

    def greedy_merge(
        self,
        params_by_system: Sequence[Mapping[str, torch.Tensor | float] | torch.Tensor],
        *,
        interaction: InteractionClass,
        environments: Sequence[Sequence[Any]],
        criterion: MergeCriterion | None = None,
        physical_eval: PhysicalEval | None = None,
        energy_tol: float | None = None,
        force_tol: float | None = None,
    ) -> CondensationResult:
        """Alias of :meth:`merge`."""
        return self.merge(
            params_by_system,
            interaction=interaction,
            environments=environments,
            criterion=criterion,
            physical_eval=physical_eval,
            energy_tol=energy_tol,
            force_tol=force_tol,
        )

    @staticmethod
    def _keys(
        environments: Sequence[Sequence[Any]],
        rows: Sequence[tuple[int, int, dict[str, torch.Tensor]]],
    ) -> list[Hashable]:
        if len(environments) == 0 and len(rows) == 0:
            return []
        n_sys = (max((s for s, _, _ in rows), default=-1) + 1) if rows else 0
        if len(environments) != n_sys:
            raise ValueError(
                f"environments cover {len(environments)} system(s) but params have {n_sys}"
            )
        n_rows: dict[int, int] = {}
        for sys_id, row_i, _ in rows:
            n_rows[sys_id] = max(n_rows.get(sys_id, 0), row_i + 1)
        for sys_id, n in n_rows.items():
            if len(environments[sys_id]) != n:
                raise ValueError(
                    f"environments[{sys_id}] has {len(environments[sys_id])} rows; params have {n}"
                )
        return [_env_key(environments[s][r]) for s, r, _ in rows]

    @staticmethod
    def _check_bucket(
        proto: Mapping[str, torch.Tensor],
        members: Sequence[Mapping[str, torch.Tensor]],
        *,
        criterion: MergeCriterion,
        physical_eval: PhysicalEval | None,
        energy_tol: float | None,
        force_tol: float | None,
        metrics: PhysicalErrorMetrics,
        interaction: InteractionClass,
    ) -> None:
        gated = physical_eval is not None and (energy_tol is not None or force_tol is not None)
        for cand in members:
            if not criterion.accepts(proto, cand):
                raise ValueError(
                    "environment is too coarse: member parameters exceed MergeCriterion "
                    f"for {interaction.value}"
                )
            if physical_eval is None:
                continue
            e_err, f_err = parse_physical_eval_result(physical_eval(proto, cand, interaction))
            rejected = False
            if gated:
                if energy_tol is not None and e_err is not None and e_err > energy_tol:
                    rejected = True
                if force_tol is not None and f_err is not None and f_err > force_tol:
                    rejected = True
            metrics.record(energy_error=e_err, force_error=f_err, rejected=rejected)
            if rejected:
                raise ValueError(
                    f"environment is too coarse: physics gate failed for {interaction.value}"
                )
