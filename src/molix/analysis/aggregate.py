"""Phase-A cross-condition aggregation for the quantization-as-thermal-noise study.

Phase A probes the force residual ``ΔF = F_quant − F_ref`` on a *static* ensemble
(no trajectory) and answers the two statically-decidable Langevin-noise criteria:

* **(a) unbiased** — ``⟨ΔF⟩ → 0`` within statistical error (the primary hard gate);
* **(b) Gaussian** — skewness ≈ 0 and excess kurtosis ≈ 0.

plus a dimensionally-correct effective-temperature ratio ``T_eff(γ,Δt)/T_target``
(Eq8) as a magnitude indicator. The deliverable is one diagnostic row per cell of
the variable matrix ``{quant scheme × trained precision × dataset}``, with the
verdict columns ``unbiased`` / ``gaussian`` / ``T_eff_ratio`` that the
``-04-verdict`` machine table consumes directly (criteria *a* and *b*).

This is the version-controlled OOP relocation of the Phase-A aggregation that the
spec body sketched as ``examples/molzoo/aggregate_phase_a.py`` — kept in ``src``
because the capability is reused by the verdict synthesis. The per-cell moments
come from :class:`molix.quant.ForceDelta`; the temperature ratio from
:class:`molix.quant.EffectiveTemperature`. Time-domain criteria (c–h) need a
trajectory and live in :mod:`molix.analysis.trajectory` / ``-03``.

Reference:
    Bussi & Parrinello, Phys. Rev. E 75, 056707 (2007). DOI 10.1103/PhysRevE.75.056707
    Wu et al., J. Chem. Phys. (2024). arXiv:2401.11427 (ML force error as Langevin noise)
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from molix.quant import EffectiveTemperature, ForceDelta

#: Condition keys identifying one matrix cell (mirrors the -04 verdict schema).
CONDITION_KEYS: tuple[str, ...] = ("scheme", "trained_precision", "dataset")

#: Diagnostic + verdict columns emitted per row, in stable order.
ROW_COLUMNS: tuple[str, ...] = (
    *CONDITION_KEYS,
    "n",
    "F_bias",
    "F_rms",
    "F_skew",
    "F_exkurt",
    "T_eff_ratio",
    "unbiased",
    "gaussian",
)


@dataclass(frozen=True)
class PhaseAThresholds:
    """Decision tolerances for the Phase-A static criteria (a, b).

    Attributes:
        bias_tol_sigma: Unbiasedness gate. ``⟨ΔF⟩`` is "within statistical error
            of zero" iff ``|F_bias| ≤ bias_tol_sigma · SE``, where the standard
            error of the mean ``SE = F_std / √n`` (dimensionless multiple of σ;
            3.0 ≈ a 3σ two-sided bound).
        skew_tol: Gaussianity gate on |skewness| (dimensionless; 0 for a normal).
        exkurt_tol: Gaussianity gate on |excess kurtosis| (dimensionless; 0 for
            a normal, positive for heavy tails).
    """

    bias_tol_sigma: float = 3.0
    skew_tol: float = 0.2
    exkurt_tol: float = 0.5


#: Module-level default thresholds (injectable into :class:`PhaseAAggregator`).
PHASE_A_THRESHOLDS = PhaseAThresholds()


@dataclass
class PhaseACell:
    """One matrix cell's static-ensemble force residual.

    Attributes:
        scheme: Quantization scheme name (e.g. ``"int8"``, ``"int4_pc"``).
        trained_precision: Checkpoint training precision (e.g. ``"fp32"``).
        dataset: Dataset name (e.g. ``"qm9"``).
        delta_f: Force residual ``ΔF = F_quant − F_ref`` for this cell, shape
            ``(..., 3)`` or flat ``(M,)`` in eV/Å. Moments are taken over all
            components.
        dof: Total degrees of freedom ``3N`` used for the Eq8 ``T_eff`` estimate.
            Defaults to ``delta_f.numel()`` (every component is a DOF for a
            single-frame static probe).
    """

    scheme: str
    trained_precision: str
    dataset: str
    delta_f: torch.Tensor
    dof: int | None = None


class PhaseAAggregator:
    """Aggregate per-cell ΔF into a Phase-A diagnostic table with verdict columns.

    Holds the fixed Langevin system constants (Δt, γ, m) and target temperature
    that parameterize the Eq8 ``T_eff`` estimate, plus the decision thresholds.
    Each cell's moments come from :class:`~molix.quant.ForceDelta`; the
    temperature ratio from :class:`~molix.quant.EffectiveTemperature` rebuilt per
    cell with that cell's ``dof``.

    Args:
        dt: Integrator time step Δt (fs).
        gamma: Langevin friction γ (1/fs).
        mass: Particle mass m (amu), used by the Eq8 denominator.
        t_target: Target thermostat temperature (K) for the dimensionless ratio.
        thresholds: Decision tolerances; defaults to :data:`PHASE_A_THRESHOLDS`.
    """

    def __init__(
        self,
        *,
        dt: float,
        gamma: float,
        mass: float,
        t_target: float,
        thresholds: PhaseAThresholds = PHASE_A_THRESHOLDS,
    ) -> None:
        self.dt = float(dt)
        self.gamma = float(gamma)
        self.mass = float(mass)
        self.t_target = float(t_target)
        self.thresholds = thresholds

    def row(self, cell: PhaseACell) -> dict[str, Any]:
        """Build one diagnostic row (moments + Eq8 ratio + a/b verdicts) for *cell*.

        Args:
            cell: The matrix cell carrying its static-ensemble ``ΔF``.

        Returns:
            A dict keyed by :data:`ROW_COLUMNS`: the condition keys, sample count
            ``n``, the ΔF moments ``F_bias`` / ``F_rms`` / ``F_skew`` /
            ``F_exkurt`` (eV/Å), the dimensionless ``T_eff_ratio``, and the
            boolean Phase-A verdicts ``unbiased`` (criterion a) and ``gaussian``
            (criterion b).
        """
        stats = ForceDelta(cell.delta_f).summary()
        n = stats["n"]
        dof = cell.dof if cell.dof is not None else int(cell.delta_f.numel())

        t_eff = EffectiveTemperature(
            dt=self.dt, gamma=self.gamma, mass=self.mass, dof=max(1, dof)
        )
        t_eff_ratio = t_eff.ratio(stats["F_rms"] ** 2, self.t_target)

        # criterion a: |⟨ΔF⟩| within bias_tol_sigma standard errors of zero.
        se = stats["F_std"] / (n**0.5) if n > 0 else 0.0
        unbiased = abs(stats["F_bias"]) <= self.thresholds.bias_tol_sigma * se
        # criterion b: near-zero skew AND excess kurtosis.
        gaussian = (
            abs(stats["F_skew"]) < self.thresholds.skew_tol
            and abs(stats["F_exkurt"]) < self.thresholds.exkurt_tol
        )

        return {
            "scheme": cell.scheme,
            "trained_precision": cell.trained_precision,
            "dataset": cell.dataset,
            "n": n,
            "F_bias": stats["F_bias"],
            "F_rms": stats["F_rms"],
            "F_skew": stats["F_skew"],
            "F_exkurt": stats["F_exkurt"],
            "T_eff_ratio": t_eff_ratio,
            "unbiased": unbiased,
            "gaussian": gaussian,
        }

    def table(self, cells: Iterable[PhaseACell]) -> list[dict[str, Any]]:
        """Aggregate an iterable of cells into one row each (matrix order preserved).

        Args:
            cells: The variable-matrix cells (scheme × precision × dataset).

        Returns:
            One :meth:`row` per cell, in iteration order — so ``len(table) ==``
            ``|schemes|·|precisions|·|datasets|`` when given the full matrix.
        """
        return [self.row(cell) for cell in cells]

    def to_csv(self, cells: Iterable[PhaseACell], path: str | Path) -> Path:
        """Write the aggregated table to *path* as CSV with :data:`ROW_COLUMNS` header.

        Args:
            cells: The variable-matrix cells.
            path: Destination ``.csv`` path; parent dirs are created.

        Returns:
            The written :class:`~pathlib.Path`.
        """
        rows = self.table(cells)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(ROW_COLUMNS))
            writer.writeheader()
            writer.writerows(rows)
        return out
