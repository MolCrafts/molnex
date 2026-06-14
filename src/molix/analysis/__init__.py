"""Trajectory diagnostics for the quantization-as-thermal-noise study (criteria c-h).

Pure numerical kernels over a paired-trajectory artifact plus the
``diagnose_trajectory`` aggregator. See :mod:`molix.analysis.trajectory`.
"""

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
    "VERDICT_THRESHOLDS",
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
