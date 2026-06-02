"""Synthetic-fixture tests for trajectory diagnostics (criteria c-h).

Each kernel is checked against a fixture with a known analytic answer, so the
diagnostics are provable without any real PiNet trajectory.
"""

import torch

from molix.analysis import (
    autocorr_df,
    crossdof_covariance,
    diagnose_trajectory,
    diffusion_einstein,
    diffusion_green_kubo,
    energy_drift_slope,
    momentum_residual,
    rdf,
    stationarity_drift,
    t_eff_colored,
    vacf,
)
from molzoo.quantization import t_eff_ratio

_DT = torch.float64


# (c) autocorrelation / tau_c -------------------------------------------------


def test_autocorr_white_noise_tau_c_order_dt():
    torch.manual_seed(0)
    df = torch.randn(4000, 6, 3, dtype=_DT)  # i.i.d. in time
    out = autocorr_df(df, dt=1.0)
    assert abs(out["tau_c"] - 1.0) < 0.3  # tau_c ~ dt for white noise


def test_autocorr_ar1_matches_analytic_tau_c():
    torch.manual_seed(1)
    phi, dt = 0.6, 1.0
    n = 20000
    eps = torch.randn(n, 4, 3, dtype=_DT)
    df = torch.zeros_like(eps)
    df[0] = eps[0]
    for t in range(1, n):
        df[t] = phi * df[t - 1] + eps[t]
    out = autocorr_df(df, dt=dt, max_lag=200)
    expected = dt * (1 + phi) / (1 - phi)  # = 4.0
    assert abs(out["tau_c"] - expected) / expected < 0.15


# (d) cross-DoF covariance ----------------------------------------------------


def test_covariance_independent_vs_correlated():
    torch.manual_seed(2)
    indep = torch.randn(3000, 5, 3, dtype=_DT)
    assert crossdof_covariance(indep) < 0.15
    # inject a shared component across all atoms -> strong off-diagonal
    shared = torch.randn(3000, 1, 1, dtype=_DT)
    correlated = (torch.randn(3000, 5, 3, dtype=_DT) * 0.1) + shared
    assert crossdof_covariance(correlated) > 0.5


# (e) momentum residual -------------------------------------------------------


def test_momentum_zero_sum_vs_net():
    torch.manual_seed(3)
    df = torch.randn(100, 8, 3, dtype=_DT)
    zero_sum = df - df.mean(dim=1, keepdim=True)  # Σ_i = 0 each frame
    assert momentum_residual(zero_sum)["mean_net_force"] < 1e-9
    net = df + torch.tensor([1.0, 0.0, 0.0], dtype=_DT)  # constant net per atom
    assert momentum_residual(net)["mean_net_force"] > 1.0


# (f) energy drift ------------------------------------------------------------


def test_energy_drift_flat_vs_sloped():
    torch.manual_seed(4)
    flat = torch.randn(500, dtype=_DT) * 0.01
    assert abs(energy_drift_slope(flat, dt=1.0)["slope"]) < 1e-3
    t = torch.arange(500, dtype=_DT)
    sloped = 0.05 * t + torch.randn(500, dtype=_DT) * 0.01
    assert abs(energy_drift_slope(sloped, dt=1.0)["slope"] - 0.05) < 5e-3


# (g) stationarity ------------------------------------------------------------


def test_stationarity_flags_growing_variance():
    torch.manual_seed(5)
    stationary = torch.randn(3000, 4, 3, dtype=_DT)
    ramp = torch.linspace(0.2, 3.0, 3000, dtype=_DT).reshape(-1, 1, 1)
    drifting = torch.randn(3000, 4, 3, dtype=_DT) * ramp
    assert stationarity_drift(stationary)["var_spread"] < 0.5
    assert stationarity_drift(drifting)["var_spread"] > 1.0


# (h) diffusion + colored T_eff ----------------------------------------------


def test_einstein_recovers_known_diffusion():
    torch.manual_seed(6)
    d_true, dt, n_steps, n_atoms = 0.1, 1.0, 400, 400
    steps = torch.randn(n_steps, n_atoms, 3, dtype=_DT) * (2 * d_true * dt) ** 0.5
    pos = torch.cumsum(steps, dim=0)
    d_est = diffusion_einstein(pos, dt)
    assert abs(d_est - d_true) / d_true < 0.2


def test_einstein_and_green_kubo_agree_on_ou_process():
    torch.manual_seed(7)
    dt, tau, n_steps, n_atoms = 1.0, 5.0, 600, 400
    a = torch.exp(torch.tensor(-dt / tau, dtype=_DT))
    b = (1 - a**2) ** 0.5
    vel = torch.zeros(n_steps, n_atoms, 3, dtype=_DT)
    vel[0] = torch.randn(n_atoms, 3, dtype=_DT)
    for t in range(1, n_steps):
        vel[t] = a * vel[t - 1] + b * torch.randn(n_atoms, 3, dtype=_DT)
    pos = torch.cumsum(vel * dt, dim=0)
    d_e = diffusion_einstein(pos, dt)
    d_gk = diffusion_green_kubo(vel, dt)
    assert d_e > 0 and d_gk > 0
    assert abs(d_e - d_gk) / max(d_e, d_gk) < 0.35


def test_t_eff_colored_reduces_to_white():
    torch.manual_seed(8)
    df = torch.randn(4000, 6, 3, dtype=_DT) * 0.02  # white
    out = autocorr_df(df, dt=0.5)
    white = t_eff_ratio(out["C0"], 0.5, gamma=0.01, mass=12.0, dof=18, t_target=300.0)
    colored = t_eff_colored(df, dt=0.5, gamma=0.01, mass=12.0, dof=18, t_target=300.0)
    assert abs(colored - white) / white < 0.4  # tau_c ~ dt -> colored ~ white


def test_vacf_and_rdf_shape_sanity():
    torch.manual_seed(9)
    vel = torch.randn(100, 5, 3, dtype=_DT)
    cv = vacf(vel, max_lag=20)
    assert cv.shape == (21,) and abs(float(cv[0]) - 1.0) < 1e-9
    pos = torch.randn(10, 8, 3, dtype=_DT)
    out = rdf(pos, n_bins=20, r_max=5.0)
    assert out["r"].shape == (20,) and out["g"].shape == (20,)


# aggregator ------------------------------------------------------------------


def test_diagnose_trajectory_returns_all_keys():
    torch.manual_seed(10)
    n_steps, n_atoms = 200, 4
    artifact = {
        "pos": torch.cumsum(torch.randn(n_steps, n_atoms, 3, dtype=_DT) * 0.01, dim=0),
        "vel": torch.randn(n_steps, n_atoms, 3, dtype=_DT) * 0.1,
        "energy": torch.randn(n_steps, dtype=_DT),
        "df": torch.randn(n_steps, n_atoms, 3, dtype=_DT) * 0.02,
        "metadata": {"dt": 0.5, "gamma": 0.01, "mass": 12.0, "dof": n_atoms * 3},
    }
    diag = diagnose_trajectory(artifact)
    for key in (
        "tau_c",
        "cov_offdiag",
        "mean_net_force",
        "energy_drift_slope",
        "stationarity_var_spread",
        "t_eff_colored_ratio",
        "D_einstein",
        "D_green_kubo",
    ):
        assert key in diag and isinstance(diag[key], float)
