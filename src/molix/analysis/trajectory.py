"""Trajectory diagnostics for the quantization-as-thermal-noise hypothesis.

Pure numerical kernels (double precision) consuming a paired-trajectory artifact
(pos/vel/energy/f_ref/f_quant/df + metadata, see ``molix.md.TrajectoryArtifact``).
Each kernel owns one of the dynamical thermal-noise criteria and is validated on
synthetic fixtures with known analytic answers, independent of any real PiNet run:

- (c) ``autocorr_df``    — force autocorrelation C(τ) and correlation time τ_c (Eq9)
- (d) ``crossdof_covariance`` — off-diagonal covariance fraction (δ_ij independence)
- (e) ``momentum_residual``   — per-frame Σ_i ΔF_i (Newton's third law)
- (f) ``energy_drift_slope``  — d⟨E⟩/dt least-squares slope
- (g) ``stationarity_drift``  — block-wise drift of ⟨ΔF⟩/var
- (h) ``t_eff_colored`` (Eq8 colored branch), ``vacf``, ``diffusion_einstein``
  (Eq10), ``diffusion_green_kubo`` (Eq11), ``rdf``

Units: ΔF eV/Å, D Å²/fs, T_eff K, Δt fs, γ 1/fs.
"""

from __future__ import annotations

from typing import Any

import torch

from molzoo.quantization import t_eff_ratio


def _flatten_frames(df: torch.Tensor) -> torch.Tensor:
    """``(T, N, 3)`` → ``(T, 3N)`` float64."""
    return df.detach().to(torch.float64).reshape(df.shape[0], -1)


def autocorr_df(df: torch.Tensor, dt: float, max_lag: int | None = None) -> dict[str, Any]:
    """Force-residual autocorrelation C(τ) and correlation time τ_c (Eq9, criterion c).

    ``C(τ)`` is the per-component-averaged autocovariance ⟨ΔF(t)·ΔF(t+τ)⟩; τ_c is the
    two-sided integral time τ_c = Δt·(1 + 2·Σ_{τ≥1} C(τ)/C(0)), with the sum windowed
    at the first zero-crossing of C(τ) so the noisy long-lag tail does not accumulate
    a spurious bias. White noise gives τ_c ≈ Δt; an AR(1) series with parameter φ
    gives τ_c ≈ Δt·(1+φ)/(1−φ).

    Returns:
        ``{"C": (L+1,) tensor, "tau_c": float, "C0": float}``.
    """
    x = _flatten_frames(df)
    n_steps = x.shape[0]
    lag = (n_steps - 1) if max_lag is None else min(max_lag, n_steps - 1)
    c0 = (x * x).sum(dim=1).mean()
    corr = [c0]
    for tau in range(1, lag + 1):
        corr.append((x[:-tau] * x[tau:]).sum(dim=1).mean())
    c = torch.stack(corr)
    c_norm = c / c0 if float(c0) != 0.0 else torch.zeros_like(c)
    # Window the integral at the first non-positive lag (automatic truncation).
    window = c_norm.shape[0] - 1
    for tau in range(1, c_norm.shape[0]):
        if float(c_norm[tau]) <= 0.0:
            window = tau - 1
            break
    tau_c = dt * float(1.0 + 2.0 * c_norm[1 : window + 1].sum())
    # report C per component so C0 == mean per-DoF square (consistent with Eq8)
    dof = x.shape[1]
    return {"C": c / dof, "tau_c": tau_c, "C0": float(c0 / dof)}


def crossdof_covariance(df: torch.Tensor) -> float:
    """Off-diagonal covariance fraction across atoms and x/y/z (criterion d, δ_ij).

    Returns the time-averaged covariance matrix's off-diagonal Frobenius norm over
    its total Frobenius norm. Independent degrees of freedom → ~0; spatially
    correlated residuals → significantly > 0.
    """
    x = _flatten_frames(df)
    cov = (x.unsqueeze(2) * x.unsqueeze(1)).mean(dim=0)  # (3N, 3N) time-averaged
    total = torch.linalg.norm(cov)
    if float(total) == 0.0:
        return 0.0
    offdiag = cov - torch.diag(torch.diagonal(cov))
    return float(torch.linalg.norm(offdiag) / total)


def momentum_residual(df: torch.Tensor) -> dict[str, float]:
    """Per-frame net force Σ_i ΔF_i magnitude stats (criterion e, Newton's third law)."""
    net = df.detach().to(torch.float64).sum(dim=1)  # (T, 3)
    mag = torch.linalg.norm(net, dim=1)  # (T,)
    return {"mean_net_force": float(mag.mean()), "max_net_force": float(mag.max())}


def energy_drift_slope(energy: torch.Tensor, dt: float) -> dict[str, float]:
    """Least-squares slope d⟨E⟩/dt and its standard error (criterion f, eV/fs)."""
    e = energy.detach().to(torch.float64).reshape(-1)
    n = e.numel()
    t = torch.arange(n, dtype=torch.float64) * dt
    tm, em = t.mean(), e.mean()
    sxx = ((t - tm) ** 2).sum()
    slope = ((t - tm) * (e - em)).sum() / sxx
    intercept = em - slope * tm
    resid = e - (intercept + slope * t)
    dof = max(1, n - 2)
    s_err = torch.sqrt((resid**2).sum() / dof / sxx)
    return {"slope": float(slope), "stderr": float(s_err)}


def stationarity_drift(df: torch.Tensor, n_blocks: int = 3) -> dict[str, float]:
    """Block-wise drift of ⟨ΔF⟩ and var across thirds of the trajectory (criterion g).

    Returns the relative spread of per-block variance (max-min)/mean; a stationary
    series → ~0, a series with growing variance → large.
    """
    x = _flatten_frames(df)
    blocks = torch.chunk(x, n_blocks, dim=0)
    means = torch.stack([b.mean() for b in blocks])
    variances = torch.stack([b.var(unbiased=False) for b in blocks])
    var_mean = variances.mean()
    var_spread = (
        float((variances.max() - variances.min()) / var_mean) if float(var_mean) != 0 else 0.0
    )
    mean_spread = float(means.max() - means.min())
    return {"var_spread": var_spread, "mean_spread": mean_spread}


def t_eff_colored(
    df: torch.Tensor, *, dt: float, gamma: float, mass: float, dof: int, t_target: float
) -> float:
    """Colored-noise effective-temperature ratio (Eq8 colored branch, criterion h).

    Replaces ⟨|ΔF|²⟩·Δt with the zero-frequency spectral density ∫C(τ)dτ = C0·τ_c
    and reuses the white-noise scalar :func:`molzoo.quantization.t_eff_ratio`. When
    τ_c ≈ Δt this reduces to the white-noise estimate.
    """
    ac = autocorr_df(df, dt)
    return t_eff_ratio(ac["C0"], ac["tau_c"], gamma, mass, dof, t_target)


def vacf(vel: torch.Tensor, max_lag: int | None = None) -> torch.Tensor:
    """Normalized velocity autocorrelation ⟨v(0)·v(t)⟩/⟨v(0)·v(0)⟩ (for Eq11)."""
    v = vel.detach().to(torch.float64).reshape(vel.shape[0], -1)
    n_steps = v.shape[0]
    lag = (n_steps - 1) if max_lag is None else min(max_lag, n_steps - 1)
    c0 = (v * v).sum(dim=1).mean()
    out = [torch.ones(())] if float(c0) != 0 else [torch.zeros(())]
    for tau in range(1, lag + 1):
        out.append(
            (v[:-tau] * v[tau:]).sum(dim=1).mean() / c0 if float(c0) != 0 else torch.zeros(())
        )
    return torch.stack(out)


def diffusion_einstein(pos: torch.Tensor, dt: float) -> float:
    """Einstein diffusion D from the MSD slope (Eq10): D = lim MSD/(2·d·t), d=3."""
    r = pos.detach().to(torch.float64)  # (T, N, 3)
    disp = r - r[0:1]
    msd = (disp**2).sum(dim=2).mean(dim=1)  # (T,) mean over atoms of |r(t)-r0|^2
    n = msd.numel()
    t = torch.arange(n, dtype=torch.float64) * dt
    tm = t.mean()
    slope = ((t - tm) * (msd - msd.mean())).sum() / ((t - tm) ** 2).sum()
    return float(slope / (2.0 * 3.0))


def diffusion_green_kubo(vel: torch.Tensor, dt: float) -> float:
    """Green-Kubo diffusion D = (1/3)∫⟨v(0)·v(t)⟩dt (Eq11), unnormalized VACF."""
    v = vel.detach().to(torch.float64).reshape(vel.shape[0], -1)
    n_atoms = vel.shape[1]
    n_steps = v.shape[0]
    c = []
    for tau in range(n_steps):
        c.append((v[: n_steps - tau] * v[tau:]).sum(dim=1).mean() / n_atoms)
    cv = torch.stack(c)  # ⟨v(0)·v(t)⟩ per atom (3 components summed)
    integral = torch.trapezoid(cv, dx=dt)
    return float(integral / 3.0)


def rdf(pos: torch.Tensor, *, n_bins: int = 50, r_max: float = 5.0) -> dict[str, torch.Tensor]:
    """Radial distribution g(r) histogram averaged over frames (criterion h sanity)."""
    r = pos.detach().to(torch.float64)
    if r.dim() == 2:
        r = r.unsqueeze(0)
    n_atoms = r.shape[1]
    edges = torch.linspace(0.0, r_max, n_bins + 1, dtype=torch.float64)
    counts = torch.zeros(n_bins, dtype=torch.float64)
    iu, ju = torch.triu_indices(n_atoms, n_atoms, offset=1)
    for frame in r:
        d = torch.linalg.norm(frame[iu] - frame[ju], dim=1)
        counts += torch.histogram(d, bins=edges).hist
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {"r": centers, "g": counts / r.shape[0]}
