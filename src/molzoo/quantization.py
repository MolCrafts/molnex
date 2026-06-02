"""PiNet weight quantization and paired force-delta evaluation.

Treats post-training quantization (PTQ) of PiNet weights as a perturbation and
measures the resulting force residual ``ΔF = F_quant - F_ref`` on a fixed
configuration ensemble. Provides:

- ``SCHEMES`` / ``fake_quantize_tensor`` / ``quantize_state_dict``: six
  fake-quantization schemes (quantize-dequantize back to float, so the eager
  double-backward force path is identical to true quantized inference).
- ``paired_force_delta``: ΔF over a reference vs quantized ``PiNetPotential``.
- ``summarize_delta``: single-configuration ensemble statistics covering the
  static thermal-noise criteria a (unbiased) and b (Gaussian).
- ``t_eff_estimate``: dimensionally-correct effective-temperature scalar (Eq8).

Units: forces in eV/Å, energies in eV, Δt in fs, γ in 1/fs, mass in amu,
temperature in K. ``t_eff_estimate`` returns k_B·T_eff (energy units); divide by
k_B·T_target (see ``t_eff_ratio``) for the reported dimensionless ratio.
"""

from __future__ import annotations

import torch
from torch import nn

# Six fake-quantization schemes (per-tensor and per-channel symmetric int).
SCHEMES: tuple[str, ...] = ("fp16", "bf16", "int8", "int8_pc", "int4", "int4_pc")

# Boltzmann constant in eV/K, for the T_eff ratio.
_KB_EV_PER_K = 8.617333262e-5

_COMPILE_PREFIX = "_orig_mod."


def _fake_quant_int(t: torch.Tensor, n_bits: int, *, per_channel: bool) -> torch.Tensor:
    """Symmetric fake int quantization (quantize then dequantize to float)."""
    qmax = 2 ** (n_bits - 1) - 1  # symmetric range [-qmax, qmax]; int8->127, int4->7
    if per_channel and t.dim() >= 1:
        dims = tuple(range(1, t.dim()))
        amax = t.abs().amax(dim=dims, keepdim=True) if dims else t.abs()
    else:
        amax = t.abs().max()
    scale = (amax / qmax).clamp(min=1e-12)
    q = torch.clamp(torch.round(t / scale), -qmax, qmax)
    return q * scale


def fake_quantize_tensor(t: torch.Tensor, scheme: str) -> torch.Tensor:
    """Fake-quantize a float tensor under ``scheme`` (non-float tensors pass through).

    Args:
        t: Tensor to quantize.
        scheme: One of :data:`SCHEMES`.

    Returns:
        A float tensor of the same dtype/shape with quantization error injected
        (``fp16``/``bf16`` reduce mantissa; ``int*`` round to a symmetric grid).
    """
    if not torch.is_floating_point(t):
        return t
    if scheme == "fp16":
        return t.half().to(t.dtype)
    if scheme == "bf16":
        return t.bfloat16().to(t.dtype)
    if scheme == "int8":
        return _fake_quant_int(t, 8, per_channel=False)
    if scheme == "int8_pc":
        return _fake_quant_int(t, 8, per_channel=True)
    if scheme == "int4":
        return _fake_quant_int(t, 4, per_channel=False)
    if scheme == "int4_pc":
        return _fake_quant_int(t, 4, per_channel=True)
    raise ValueError(f"unknown scheme {scheme!r}; valid schemes: {SCHEMES}")


def strip_compile_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Drop the ``_orig_mod.`` prefix torch.compile adds to checkpoint keys."""
    return {
        (k[len(_COMPILE_PREFIX) :] if k.startswith(_COMPILE_PREFIX) else k): v
        for k, v in state_dict.items()
    }


def quantize_state_dict(
    state_dict: dict[str, torch.Tensor], scheme: str
) -> dict[str, torch.Tensor]:
    """Return a copy of ``state_dict`` with every float tensor fake-quantized."""
    return {k: fake_quantize_tensor(v, scheme) for k, v in state_dict.items()}


def paired_force_delta(model_ref: nn.Module, model_quant: nn.Module, batch: object) -> torch.Tensor:
    """Compute ΔF = F_quant - F_ref on the same configuration (static ensemble).

    Each model gets a fresh clone of ``batch`` so the autograd graph and the
    leaf ``pos`` are clean and the returned forces are detached.

    Args:
        model_ref: Full-precision reference ``PiNetPotential``.
        model_quant: Fake-quantized copy.
        batch: Input TensorDict accepted by ``PiNetPotential.forward``.

    Returns:
        ΔF tensor of shape ``(N, 3)`` (eV/Å), detached.
    """
    f_ref = model_ref(batch.clone(), compute_forces=True)["forces"].detach()
    f_quant = model_quant(batch.clone(), compute_forces=True)["forces"].detach()
    return f_quant - f_ref


def summarize_delta(delta_f: torch.Tensor) -> dict[str, float]:
    """Single-configuration ensemble statistics of a force residual ΔF.

    Reductions run in float64. Covers static criterion a (``F_bias`` ≈ 0 →
    unbiased) and b (``F_skew``/``F_exkurt`` ≈ 0 → Gaussian).

    Args:
        delta_f: Force residual, any shape (flattened componentwise).

    Returns:
        Dict with ``F_bias``, ``F_std``, ``F_rms`` (eV/Å), dimensionless
        ``F_skew`` / ``F_exkurt`` (excess kurtosis), and sample count ``n``.
    """
    x = delta_f.detach().to(torch.float64).flatten()
    n = int(x.numel())
    mean = x.mean()
    std = x.std(unbiased=False)
    rms = x.pow(2).mean().sqrt()
    if float(std) > 0.0:
        z = (x - mean) / std
        skew = z.pow(3).mean()
        exkurt = z.pow(4).mean() - 3.0
    else:
        skew = torch.zeros((), dtype=torch.float64)
        exkurt = torch.zeros((), dtype=torch.float64)
    return {
        "F_bias": float(mean),
        "F_std": float(std),
        "F_rms": float(rms),
        "F_skew": float(skew),
        "F_exkurt": float(exkurt),
        "n": n,
    }


def t_eff_estimate(f_rms_sq: float, dt: float, gamma: float, mass: float, dof: int) -> float:
    """Effective temperature scalar from Eq8 (per-degree-of-freedom).

    Computes k_B·T_eff = ⟨|ΔF|²⟩·Δt / (2·γ·m·d), where d = total degrees of
    freedom (3N). Scales as 1/γ. The dimensionally-wrong legacy heuristic
    ``⟨|ΔF|²⟩/(2γ·dim)`` (units N²·s, missing Δt and m) is replaced by this.

    Args:
        f_rms_sq: ⟨|ΔF|²⟩, mean squared force residual ((eV/Å)²).
        dt: MD timestep Δt (fs).
        gamma: Langevin friction γ (1/fs).
        mass: Particle mass m (amu).
        dof: Total degrees of freedom d = 3N.

    Returns:
        k_B·T_eff in energy units. Divide by k_B·T_target for the reported ratio
        (see :func:`t_eff_ratio`).
    """
    return f_rms_sq * dt / (2.0 * gamma * mass * dof)


def t_eff_ratio(
    f_rms_sq: float, dt: float, gamma: float, mass: float, dof: int, t_target: float
) -> float:
    """Dimensionless T_eff(γ,Δt)/T_target from Eq8 (scales as 1/γ)."""
    return t_eff_estimate(f_rms_sq, dt, gamma, mass, dof) / (_KB_EV_PER_K * t_target)
