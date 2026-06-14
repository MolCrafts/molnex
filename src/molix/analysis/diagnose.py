"""Aggregate the trajectory diagnostic kernels into a criteria c-h scalar dict."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from molix.analysis.trajectory import (
    autocorr_df,
    crossdof_covariance,
    diffusion_einstein,
    diffusion_green_kubo,
    energy_drift_slope,
    momentum_residual,
    stationarity_drift,
    t_eff_colored,
)


def diagnose_trajectory(
    artifact: Mapping[str, Any], *, t_target: float = 298.0
) -> dict[str, float]:
    """Run all dynamical diagnostics over a paired-trajectory artifact.

    Args:
        artifact: A trajectory mapping (``TrajectoryArtifact.to_dict()`` shape) with
            ``pos`` / ``vel`` / ``energy`` / ``df`` tensors and a ``metadata`` dict
            carrying ``dt`` / ``gamma`` / ``mass`` / ``dof``.
        t_target: Target temperature for the T_eff ratio (K).

    Returns:
        Flat dict of criteria c-h diagnostic scalars (no final verdict — that is
        spec -04's job).
    """
    meta = artifact["metadata"]
    dt = float(meta["dt"])
    gamma = float(meta["gamma"])
    mass = float(meta["mass"])
    df = artifact["df"]
    dof = int(meta.get("dof", df.shape[1] * 3))

    ac = autocorr_df(df, dt)
    momentum = momentum_residual(df)
    return {
        "tau_c": ac["tau_c"],
        "cov_offdiag": crossdof_covariance(df),
        "mean_net_force": momentum["mean_net_force"],
        "energy_drift_slope": energy_drift_slope(artifact["energy"], dt)["slope"],
        "stationarity_var_spread": stationarity_drift(df)["var_spread"],
        "t_eff_colored_ratio": t_eff_colored(
            df, dt=dt, gamma=gamma, mass=mass, dof=dof, t_target=t_target
        ),
        "D_einstein": diffusion_einstein(artifact["pos"], dt),
        "D_green_kubo": diffusion_green_kubo(artifact["vel"], dt),
    }
