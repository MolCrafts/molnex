"""Diagnostics for the quantization-as-thermal-noise study.

Phase-A static aggregation (criteria a/b) in :mod:`molix.analysis.aggregate`,
trajectory kernels (criteria c-h) in :mod:`molix.analysis.trajectory` +
``diagnose_trajectory``, and the cross-condition verdict synthesis in
:mod:`molix.analysis.verdict`.
"""

from molix.analysis.aggregate import (
    CONDITION_KEYS,
    PHASE_A_THRESHOLDS,
    ROW_COLUMNS,
    PhaseAAggregator,
    PhaseACell,
    PhaseAThresholds,
)
from molix.analysis.diagnose import diagnose_trajectory
from molix.analysis.trajectory import (
    autocorr_df,
    crossdof_covariance,
    diffusion_einstein,
    diffusion_green_kubo,
    energy_drift_slope,
    momentum_residual,
    rdf,
    stationarity_drift,
    t_eff_colored,
    vacf,
)
from molix.analysis.verdict import (
    VERDICT_THRESHOLDS,
    VerdictThresholds,
    build_machine_table,
    characterize_failure,
    classify_cell,
    evaluate_criteria,
    render_report,
    run_verdict,
)

__all__ = [
    "CONDITION_KEYS",
    "PHASE_A_THRESHOLDS",
    "ROW_COLUMNS",
    "VERDICT_THRESHOLDS",
    "PhaseAAggregator",
    "PhaseACell",
    "PhaseAThresholds",
    "VerdictThresholds",
    "autocorr_df",
    "build_machine_table",
    "characterize_failure",
    "classify_cell",
    "crossdof_covariance",
    "diagnose_trajectory",
    "diffusion_einstein",
    "diffusion_green_kubo",
    "energy_drift_slope",
    "evaluate_criteria",
    "momentum_residual",
    "rdf",
    "render_report",
    "run_verdict",
    "stationarity_drift",
    "t_eff_colored",
    "vacf",
]
