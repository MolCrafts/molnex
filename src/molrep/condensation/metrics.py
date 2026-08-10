"""Physical residual metrics for physics-gated condensation.

:class:`PhysicalErrorMetrics` records energy/force residuals reported by an
**injected** ``physical_eval`` callable. Condensation never builds molpot
energy graphs itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = ["PhysicalErrorMetrics", "parse_physical_eval_result"]


@dataclass
class PhysicalErrorMetrics:
    """Aggregated residuals from optional physical evaluation during merge.

    Attributes:
        n_compared: Number of prototype–candidate pairs evaluated.
        rejected_by_physics: Pairs that passed param budgets but failed the
            physical residual gate.
        max_abs_energy_error: Max |ΔE| observed (caller units, typically
            kcal/mol), or ``None`` if never set.
        mean_abs_energy_error: Mean |ΔE| over compared pairs.
        max_abs_force_error: Max force residual norm if provided.
        mean_abs_force_error: Mean force residual if provided.
        energy_errors: Per-comparison energy residuals (append-only log).
        force_errors: Per-comparison force residuals (append-only log).
    """

    n_compared: int = 0
    rejected_by_physics: int = 0
    max_abs_energy_error: float | None = None
    mean_abs_energy_error: float | None = None
    max_abs_force_error: float | None = None
    mean_abs_force_error: float | None = None
    energy_errors: list[float] = field(default_factory=list)
    force_errors: list[float] = field(default_factory=list)

    def record(
        self,
        *,
        energy_error: float | None = None,
        force_error: float | None = None,
        rejected: bool = False,
    ) -> None:
        """Append one comparison result and refresh aggregates.

        Args:
            energy_error: Absolute energy residual for this pair.
            force_error: Absolute force residual for this pair.
            rejected: Whether the physics gate rejected the merge.
        """
        self.n_compared += 1
        if rejected:
            self.rejected_by_physics += 1
        if energy_error is not None:
            e = abs(float(energy_error))
            self.energy_errors.append(e)
            self.max_abs_energy_error = (
                e if self.max_abs_energy_error is None else max(self.max_abs_energy_error, e)
            )
            self.mean_abs_energy_error = sum(self.energy_errors) / len(self.energy_errors)
        if force_error is not None:
            f = abs(float(force_error))
            self.force_errors.append(f)
            self.max_abs_force_error = (
                f if self.max_abs_force_error is None else max(self.max_abs_force_error, f)
            )
            self.mean_abs_force_error = sum(self.force_errors) / len(self.force_errors)


def parse_physical_eval_result(
    result: Any,
) -> tuple[float | None, float | None]:
    """Normalize a ``physical_eval`` return value to ``(energy_err, force_err)``.

    Accepted shapes:

    * ``float`` / ``int`` → energy error only
    * ``Mapping`` with keys ``energy`` / ``energy_error`` / ``delta_e`` and
      optional ``force`` / ``force_error`` / ``delta_f``
    * ``(energy, force)`` 2-tuple

    Args:
        result: Raw return from the injected callable.

    Returns:
        Pair of optional absolute residuals.
    """
    if result is None:
        return None, None
    if isinstance(result, (int, float)):
        return abs(float(result)), None
    if isinstance(result, Mapping):
        e = None
        for key in ("energy_error", "energy", "delta_e", "abs_energy_error"):
            if key in result:
                e = abs(float(result[key]))
                break
        f = None
        for key in ("force_error", "force", "delta_f", "abs_force_error"):
            if key in result:
                f = abs(float(result[key]))
                break
        return e, f
    if isinstance(result, (tuple, list)) and len(result) >= 1:
        e = abs(float(result[0])) if result[0] is not None else None
        f = abs(float(result[1])) if len(result) > 1 and result[1] is not None else None
        return e, f
    raise TypeError(
        f"physical_eval must return float, mapping, or (energy, force) tuple; got {type(result)!r}"
    )
