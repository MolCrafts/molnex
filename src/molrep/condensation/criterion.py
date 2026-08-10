"""Merge budgets for physics-aware chemical type condensation.

Units follow CLASS_I_CANONICAL (kcal/mol, Å, e, rad) as used by continuous
MM heads. Budgets are configuration objects — tune per library; defaults
prioritise force-field readability while bounding energetic drift.

References:
    Spec: learnable-classical-ff-06-condensation
    OpenMM User Guide §19 "Forces"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

import torch

from molrep.condensation.classes import InteractionClass

__all__ = [
    "MergeCriterion",
    "bond_default_criterion",
    "angle_default_criterion",
    "proper_default_criterion",
    "improper_default_criterion",
    "lj_default_criterion",
    "charge_default_criterion",
    "default_criterion",
]

# ---------------------------------------------------------------------------
# Default budgets (documented physical rationale)
# ---------------------------------------------------------------------------
# Bond: Δr0 ≤ 0.01 Å keeps geometry tables readable; relative k ≤ 5% (with
# absolute floor) limits force-constant drift for stiff bonds.
_BOND_ABS_R0 = 0.01  # Å
_BOND_REL_K = 0.05  # dimensionless
_BOND_ABS_K_FLOOR = 1.0  # kcal/mol/Å²

# Angle: Δθ0 ≤ 1° in rad; same relative-k spirit as bonds.
_ANGLE_ABS_THETA0 = math.radians(1.0)  # rad
_ANGLE_REL_K = 0.05
_ANGLE_ABS_K_FLOOR = 1.0  # kcal/mol/rad²

# Proper / improper: per-term barrier and phase tolerances.
_TORSION_ABS_K = 0.1  # kcal/mol
_TORSION_REL_K = 0.05
_TORSION_ABS_PHASE = math.radians(15.0)  # rad
_TORSION_ABS_CHI0 = math.radians(5.0)  # rad (harmonic improper)

# LJ: Δσ ≤ 0.01 Å; relative ε ≤ 5%.
_LJ_ABS_SIGMA = 0.01  # Å
_LJ_REL_EPSILON = 0.05

# Charge: absolute Δq (e); rarely merged aggressively.
_CHARGE_ABS_Q = 0.02  # e


def _as_tensor(
    value: torch.Tensor | float | tuple[float, ...] | list[float],
    *,
    like: torch.Tensor | None = None,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        value = list(value)
    if like is not None:
        return torch.as_tensor(value, dtype=like.dtype, device=like.device)
    return torch.as_tensor(value, dtype=torch.float64)


def _flatten_row(
    params: Mapping[str, torch.Tensor | float | tuple[float, ...] | list[float]],
) -> dict[str, torch.Tensor]:
    """Normalize a single interaction's params to tensors (no batch dim)."""
    out: dict[str, torch.Tensor] = {}
    for key, val in params.items():
        t = _as_tensor(val)
        # Squeeze a leading singleton batch dim only: (1,) → scalar, (1, T) → (T,).
        if t.ndim >= 1 and t.shape[0] == 1:
            t = t.reshape(*t.shape[1:]) if t.ndim > 1 else t.reshape(())
        out[key] = t
    return out


@dataclass(frozen=True)
class MergeCriterion:
    """Per-class merge budgets for continuous → discrete type condensation.

    All absolute length thresholds are in **Å**; angles in **radians**;
    energy-related constants in **kcal/mol** (and force-constant dimensions
    matching Class-I IR). Relative thresholds are dimensionless fractions.

    Args:
        interaction: Owning :class:`InteractionClass`.
        abs_tol: Absolute tolerances keyed by parameter name
            (e.g. ``{"r0": 0.01}`` for bonds).
        rel_tol: Relative tolerances keyed by parameter name
            (e.g. ``{"k": 0.05}``). Compared as
            ``|a - b| ≤ rel * max(|proto|, |cand|, abs_floor)``.
        abs_floor: Denominator floors for relative checks, same units as the
            parameter (prevents tiny-k blow-ups).
        required_keys: Parameter names that must be present on both sides.

    Notes:
        :meth:`accepts` is pure parameter-space. Physics gating (energy /
        force residuals) is applied by :class:`Condenser` via an injected
        ``physical_eval`` callable — this type never builds energy graphs.
    """

    interaction: InteractionClass
    abs_tol: Mapping[str, float] = field(default_factory=dict)
    rel_tol: Mapping[str, float] = field(default_factory=dict)
    abs_floor: Mapping[str, float] = field(default_factory=dict)
    required_keys: tuple[str, ...] = ()

    def accepts(
        self,
        prototype_params: Mapping[str, torch.Tensor | float | tuple[float, ...]],
        candidate_params: Mapping[str, torch.Tensor | float | tuple[float, ...]],
    ) -> bool:
        """Return True if candidate params fall inside budgets of prototype.

        Args:
            prototype_params: Type centroid / prototype parameter mapping.
            candidate_params: Candidate interaction parameters.

        Returns:
            ``True`` when every budgeted key is within absolute and/or
            relative tolerance; ``False`` otherwise.

        Raises:
            KeyError: If a required key is missing on either side.
        """
        proto = _flatten_row(prototype_params)
        cand = _flatten_row(candidate_params)

        for key in self.required_keys:
            if key not in proto or key not in cand:
                raise KeyError(
                    f"MergeCriterion({self.interaction.value}) requires key "
                    f"{key!r} on both prototype and candidate"
                )

        budgeted = set(self.abs_tol) | set(self.rel_tol)
        if not budgeted:
            keys = set(proto) & set(cand)
        else:
            # Only compare keys present on both sides (improper may emit
            # harmonic-only or periodic-only parameter families).
            keys = budgeted & set(proto) & set(cand)
            if not keys and self.required_keys:
                # Required keys already validated; nothing budgeted to compare.
                return True
            if not keys:
                return False

        for key in keys:
            a = proto[key]
            b = cand[key]
            if a.shape != b.shape:
                return False
            diff = torch.abs(a - b)

            abs_ok = True
            if key in self.abs_tol:
                abs_ok = bool(torch.all(diff <= self.abs_tol[key]).item())

            rel_ok = True
            if key in self.rel_tol:
                floor = float(self.abs_floor.get(key, 0.0))
                scale = torch.maximum(torch.abs(a), torch.abs(b))
                scale = torch.clamp(scale, min=floor)
                # When both near zero and floor is 0, require exact match via abs.
                if floor == 0.0 and bool(torch.all(scale == 0).item()):
                    rel_ok = bool(torch.all(diff == 0).item())
                else:
                    rel_ok = bool(torch.all(diff <= self.rel_tol[key] * scale).item())

            # Key passes if *any* configured budget for that key holds when both
            # are set; if only one family is set, that family decides.
            if key in self.abs_tol and key in self.rel_tol:
                if not (abs_ok or rel_ok):
                    return False
            elif key in self.abs_tol:
                if not abs_ok:
                    return False
            elif key in self.rel_tol:
                if not rel_ok:
                    return False

        return True


def bond_default_criterion() -> MergeCriterion:
    """Default bond budgets: Δr0 ≤ 0.01 Å; Δk/k ≤ 5% with abs floor 1.0."""
    return MergeCriterion(
        interaction=InteractionClass.BOND,
        abs_tol={"r0": _BOND_ABS_R0},
        rel_tol={"k": _BOND_REL_K},
        abs_floor={"k": _BOND_ABS_K_FLOOR},
        required_keys=("k", "r0"),
    )


def angle_default_criterion() -> MergeCriterion:
    """Default angle budgets: Δθ0 ≤ 1° (rad); relative k ≤ 5%."""
    return MergeCriterion(
        interaction=InteractionClass.ANGLE,
        abs_tol={"theta0": _ANGLE_ABS_THETA0},
        rel_tol={"k": _ANGLE_REL_K},
        abs_floor={"k": _ANGLE_ABS_K_FLOOR},
        required_keys=("k", "theta0"),
    )


def proper_default_criterion() -> MergeCriterion:
    """Default proper-torsion budgets: per-term |Δk| / rel k and |Δphase|."""
    return MergeCriterion(
        interaction=InteractionClass.PROPER,
        abs_tol={"k": _TORSION_ABS_K, "phase": _TORSION_ABS_PHASE},
        rel_tol={"k": _TORSION_REL_K},
        abs_floor={"k": _TORSION_ABS_K},
        required_keys=("k", "phase"),
    )


def improper_default_criterion() -> MergeCriterion:
    """Default improper budgets (harmonic χ0 and/or periodic k/phase)."""
    return MergeCriterion(
        interaction=InteractionClass.IMPROPER,
        abs_tol={
            "k": _TORSION_ABS_K,
            "phase": _TORSION_ABS_PHASE,
            "chi0": _TORSION_ABS_CHI0,
            "k_harmonic": _TORSION_ABS_K,
            "k_periodic": _TORSION_ABS_K,
        },
        rel_tol={"k": _TORSION_REL_K, "k_harmonic": _TORSION_REL_K, "k_periodic": _TORSION_REL_K},
        abs_floor={"k": _TORSION_ABS_K, "k_harmonic": _TORSION_ABS_K, "k_periodic": _TORSION_ABS_K},
        required_keys=(),
    )


def lj_default_criterion() -> MergeCriterion:
    """Default LJ budgets: Δσ ≤ 0.01 Å; relative ε ≤ 5%."""
    return MergeCriterion(
        interaction=InteractionClass.LJ,
        abs_tol={"sigma": _LJ_ABS_SIGMA},
        rel_tol={"epsilon": _LJ_REL_EPSILON},
        abs_floor={"epsilon": 1e-4},
        required_keys=("epsilon", "sigma"),
    )


def charge_default_criterion() -> MergeCriterion:
    """Default charge budget: |Δq| ≤ 0.02 e (conservative; often unused)."""
    return MergeCriterion(
        interaction=InteractionClass.CHARGE,
        abs_tol={"q": _CHARGE_ABS_Q, "charge": _CHARGE_ABS_Q},
        required_keys=(),
    )


def default_criterion(interaction: InteractionClass) -> MergeCriterion:
    """Return the documented default :class:`MergeCriterion` for ``interaction``.

    Args:
        interaction: Target interaction family.

    Returns:
        A frozen criterion with Class-I unit budgets.
    """
    table = {
        InteractionClass.BOND: bond_default_criterion,
        InteractionClass.ANGLE: angle_default_criterion,
        InteractionClass.PROPER: proper_default_criterion,
        InteractionClass.IMPROPER: improper_default_criterion,
        InteractionClass.LJ: lj_default_criterion,
        InteractionClass.CHARGE: charge_default_criterion,
    }
    return table[interaction]()
