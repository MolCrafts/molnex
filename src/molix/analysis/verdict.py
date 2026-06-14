"""Thermal-noise verdict synthesis: classify each condition cell against 8 criteria.

Consumes the static aggregation (criteria a/b + static T_eff) and the trajectory
diagnostics (criteria c-h) per condition cell, evaluates all 8 criteria against
explicit named thresholds, and classifies the cell as ``thermal-noise-approximable``
(all pass) or ``structured-non-thermal`` (any systematic violation). On failure it
reports the *named physical form* (e.g. biased + spatially-correlated ⇒ PES
distortion, not noise). Pure logic — no torch — validated on synthetic fixtures.

Failure → form priority (Domain basis): a/d (field/PES distortion) > c/g
(colored / configuration-locked) > e/f (momentum / energy injection) > b/h.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import sqrt
from typing import Any

VERDICT_THERMAL = "thermal-noise-approximable"
VERDICT_STRUCTURED = "structured-non-thermal"

_CRITERIA = ("a", "b", "c", "d", "e", "f", "g", "h")

# Failure → named physical form per criterion.
_FAILURE_FORM = {
    "a": "force-field distortion",
    "b": "non-Gaussian (Langevin mapping invalid)",
    "c": "colored / quenched noise",
    "d": "spatially-correlated PES distortion",
    "e": "COM-momentum injection",
    "f": "unbalanced injection (needs thermostat friction)",
    "g": "configuration-locked distortion",
    "h": "altered structure / dynamics",
}
# Physical priority groups (highest first).
_PRIORITY = (("a", "d"), ("c", "g"), ("e", "f"), ("b", "h"))


@dataclass(frozen=True)
class VerdictThresholds:
    """Explicit pass/fail thresholds for the 8 thermal-noise criteria (documented)."""

    bias_tol_sigma: float = 2.0  # a: |⟨ΔF⟩| within 2σ statistical error
    skew_tol: float = 0.2  # b: |skew| bound
    exkurt_tol: float = 0.5  # b: |excess kurtosis| bound
    tau_c_dt_factor: float = 3.0  # c: τ_c ≲ 3·Δt is "white"
    cov_offdiag_tol: float = 0.1  # d: normalized off-diagonal covariance bound
    net_force_tol: float = 1e-3  # e: ‖Σ ΔF‖ bound (eV/Å)
    energy_drift_tol: float = 1e-4  # f: |d⟨E⟩/dt| bound (eV/fs)
    stationarity_tol: float = 1.0  # g: block variance-spread bound
    observable_delta_tol: float = 0.05  # h: g(r)/VACF/D relative deviation bound


VERDICT_THRESHOLDS = VerdictThresholds()


def evaluate_criteria(
    cell: Mapping[str, Any], thresholds: VerdictThresholds = VERDICT_THRESHOLDS
) -> dict[str, bool]:
    """Return the 8 per-criterion pass/fail booleans for one condition cell.

    The cell merges static columns (``F_bias``/``F_std``/``F_skew``/``F_exkurt``/``n``)
    and trajectory columns (``tau_c``/``dt``/``cov_offdiag``/``mean_net_force``/
    ``energy_drift_slope``/``stationarity_var_spread``/``observable_delta``).
    """
    n = int(cell.get("n", 0))
    stderr = (float(cell["F_std"]) / sqrt(n)) if n > 1 else float("inf")
    dt = float(cell.get("dt", 1.0))
    return {
        "a": abs(float(cell["F_bias"])) <= thresholds.bias_tol_sigma * stderr,
        "b": abs(float(cell["F_skew"])) < thresholds.skew_tol
        and abs(float(cell["F_exkurt"])) < thresholds.exkurt_tol,
        "c": float(cell["tau_c"]) <= thresholds.tau_c_dt_factor * dt,
        "d": float(cell["cov_offdiag"]) < thresholds.cov_offdiag_tol,
        "e": float(cell["mean_net_force"]) <= thresholds.net_force_tol,
        "f": abs(float(cell["energy_drift_slope"])) < thresholds.energy_drift_tol,
        "g": float(cell["stationarity_var_spread"]) < thresholds.stationarity_tol,
        "h": float(cell.get("observable_delta", 0.0)) < thresholds.observable_delta_tol,
    }


def classify_cell(criteria: Mapping[str, bool]) -> str:
    """``thermal-noise-approximable`` iff all 8 criteria pass, else structured."""
    return VERDICT_THERMAL if all(criteria[c] for c in _CRITERIA) else VERDICT_STRUCTURED


def characterize_failure(criteria: Mapping[str, bool]) -> str:
    """Named physical form of the failure, combined by physical priority.

    Returns ``""`` when every criterion passes. When ``a`` and ``d`` both fail the
    combined headline ``PES distortion, not noise`` is emitted first.
    """
    failed = [c for c in _CRITERIA if not criteria[c]]
    if not failed:
        return ""
    forms: list[str] = []
    if not criteria["a"] and not criteria["d"]:
        forms.append("PES distortion, not noise (biased + spatially-correlated)")
    failed_set = set(failed)
    for group in _PRIORITY:
        for crit in group:
            if crit in failed_set and not (
                crit in ("a", "d") and "PES distortion" in " ".join(forms)
            ):
                forms.append(f"{crit}: {_FAILURE_FORM[crit]}")
    return "; ".join(forms)


_CONDITION_KEYS = ("scheme", "trained_precision", "dataset", "md_condition")


def _condition_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(k) for k in _CONDITION_KEYS)


def build_machine_table(verdicts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Stable-schema machine table: condition keys + 8 criteria + verdict/form/T_eff."""
    table: list[dict[str, Any]] = []
    for v in verdicts:
        row: dict[str, Any] = {k: v.get(k) for k in _CONDITION_KEYS}
        for c in _CRITERIA:
            row[f"crit_{c}"] = bool(v["criteria"][c])
        row["verdict"] = v["verdict"]
        row["form"] = v["form"]
        row["T_eff_ratio"] = v.get("T_eff_ratio")
        table.append(row)
    return table


def render_report(verdicts: Sequence[Mapping[str, Any]]) -> str:
    """Human-readable Markdown report grouped by condition cell."""
    lines = ["# PiNet quantization thermal-noise verdict", ""]
    for v in verdicts:
        key = ", ".join(f"{k}={v.get(k)}" for k in _CONDITION_KEYS if v.get(k) is not None)
        lines.append(f"## {key or 'cell'}")
        lines.append(f"- verdict: **{v['verdict']}**")
        failed = [c for c in _CRITERIA if not v["criteria"][c]]
        lines.append(f"- failed criteria: {', '.join(failed) if failed else 'none'}")
        if v["form"]:
            lines.append(f"- form: {v['form']}")
        if v.get("T_eff_ratio") is not None:
            lines.append(f"- T_eff(γ,Δt)/T_target: {v['T_eff_ratio']:.4g}")
        lines.append("")
    return "\n".join(lines)


def run_verdict(
    static_table: Sequence[Mapping[str, Any]],
    traj_table: Sequence[Mapping[str, Any]],
    thresholds: VerdictThresholds = VERDICT_THRESHOLDS,
) -> tuple[str, list[dict[str, Any]]]:
    """Join static + trajectory tables per condition and render report + table.

    Missing condition keys are reported as incomplete cells (verdict ``incomplete``)
    rather than silently dropped.
    """
    traj_by_key = {_condition_key(r): r for r in traj_table}
    verdicts: list[dict[str, Any]] = []
    for static in static_table:
        key = _condition_key(static)
        traj = traj_by_key.get(key)
        base = {k: static.get(k) for k in _CONDITION_KEYS}
        if traj is None:
            verdicts.append(
                {
                    **base,
                    "verdict": "incomplete",
                    "form": "missing trajectory diagnostics",
                    "criteria": {c: False for c in _CRITERIA},
                    "T_eff_ratio": None,
                }
            )
            continue
        cell = {**static, **traj}
        criteria = evaluate_criteria(cell, thresholds)
        verdicts.append(
            {
                **base,
                "criteria": criteria,
                "verdict": classify_cell(criteria),
                "form": characterize_failure(criteria),
                "T_eff_ratio": cell.get("t_eff_colored_ratio", cell.get("T_eff_ratio")),
            }
        )
    return render_report(verdicts), build_machine_table(verdicts)
