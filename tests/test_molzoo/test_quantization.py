"""Tests for molzoo.quantization (PiNet PTQ + paired force-delta + Eq8 T_eff)."""

import math

import pytest
import torch

from molzoo.pinet import PiNet, PiNetPotential
from molzoo.quantization import (
    SCHEMES,
    fake_quantize_tensor,
    paired_force_delta,
    quantize_state_dict,
    strip_compile_prefix,
    summarize_delta,
    t_eff_estimate,
    t_eff_ratio,
)
from tests.symmetry_helpers import make_graph_batch

# Compute on CPU: PiNet uses lazy params, and make_graph_batch builds some
# metadata on CPU, so a mixed CPU/CUDA batch errors. The molnex .so still loads
# on the GPU node; only the tensors stay on CPU.
_DEVICE = torch.device("cpu")


# --------------------------------------------------------------------------- #
# t_eff_estimate (Eq8) — ac-001
# --------------------------------------------------------------------------- #


def test_t_eff_estimate_matches_eq8():
    f_rms_sq, dt, gamma, mass, dof = 4.0, 0.5, 0.01, 12.0, 9
    expected = f_rms_sq * dt / (2.0 * gamma * mass * dof)
    assert t_eff_estimate(f_rms_sq, dt, gamma, mass, dof) == pytest.approx(expected)


def test_t_eff_estimate_scales_as_inverse_gamma():
    base = t_eff_estimate(4.0, 0.5, 0.01, 12.0, 9)
    doubled = t_eff_estimate(4.0, 0.5, 0.02, 12.0, 9)
    assert doubled == pytest.approx(base / 2.0)


def test_t_eff_ratio_is_dimensionless_and_inverse_gamma():
    kb = 8.617333262e-5
    r = t_eff_ratio(4.0, 0.5, 0.01, 12.0, 9, t_target=300.0)
    expected = (4.0 * 0.5 / (2.0 * 0.01 * 12.0 * 9)) / (kb * 300.0)
    assert r == pytest.approx(expected)
    # halving gamma's effect: doubling gamma halves the ratio
    assert t_eff_ratio(4.0, 0.5, 0.02, 12.0, 9, 300.0) == pytest.approx(r / 2.0)


# --------------------------------------------------------------------------- #
# summarize_delta — ac-002
# --------------------------------------------------------------------------- #


def test_summarize_delta_recovers_analytic_moments():
    torch.manual_seed(0)
    x = torch.randn(200_000, dtype=torch.float64)  # ~N(0,1)
    s = summarize_delta(x)
    assert s["F_bias"] == pytest.approx(x.mean().item(), abs=1e-9)
    assert s["F_rms"] == pytest.approx(x.pow(2).mean().sqrt().item(), abs=1e-9)
    assert abs(s["F_skew"]) < 0.05
    assert abs(s["F_exkurt"]) < 0.1
    assert s["n"] == 200_000


def test_summarize_delta_flags_bias_and_heavy_tails():
    torch.manual_seed(1)
    biased = torch.randn(50_000, dtype=torch.float64) + 3.0
    assert summarize_delta(biased)["F_bias"] == pytest.approx(3.0, abs=0.05)
    # Laplace-like heavy tails -> positive excess kurtosis
    u = torch.rand(200_000, dtype=torch.float64) - 0.5
    laplace = -u.sign() * torch.log1p(-2 * u.abs())
    assert summarize_delta(laplace)["F_exkurt"] > 1.0


def test_summarize_delta_zero_residual_has_zero_moments():
    s = summarize_delta(torch.zeros(100, dtype=torch.float64))
    assert s["F_bias"] == 0.0 and s["F_rms"] == 0.0
    assert s["F_skew"] == 0.0 and s["F_exkurt"] == 0.0


# --------------------------------------------------------------------------- #
# fake_quantize schemes + state dict
# --------------------------------------------------------------------------- #


def test_fake_quantize_all_schemes_change_high_precision_weights():
    torch.manual_seed(2)
    w = torch.randn(64, 64, dtype=torch.float32)
    for scheme in SCHEMES:
        q = fake_quantize_tensor(w, scheme)
        assert q.shape == w.shape and q.dtype == w.dtype
        assert not torch.equal(q, w), f"{scheme} left weights unchanged"


def test_int4_coarser_than_int8():
    torch.manual_seed(3)
    w = torch.randn(128, 128, dtype=torch.float32)
    err8 = (fake_quantize_tensor(w, "int8") - w).abs().mean()
    err4 = (fake_quantize_tensor(w, "int4") - w).abs().mean()
    assert err4 > err8


def test_fake_quantize_passes_through_integer_tensors():
    idx = torch.arange(10, dtype=torch.long)
    assert torch.equal(fake_quantize_tensor(idx, "int8"), idx)


def test_fake_quantize_rejects_unknown_scheme():
    with pytest.raises(ValueError, match="unknown scheme"):
        fake_quantize_tensor(torch.randn(4), "int3")


def test_quantize_state_dict_and_strip_prefix():
    sd = {"a.weight": torch.randn(8, 8), "a.idx": torch.arange(8)}
    q = quantize_state_dict(sd, "int8")
    assert not torch.equal(q["a.weight"], sd["a.weight"])
    assert torch.equal(q["a.idx"], sd["a.idx"])  # integer untouched
    prefixed = {"_orig_mod.a.weight": sd["a.weight"]}
    assert "a.weight" in strip_compile_prefix(prefixed)


# --------------------------------------------------------------------------- #
# paired force delta on PiNet — ac-003 (null control)
# --------------------------------------------------------------------------- #


def _tiny_potential() -> PiNetPotential:
    torch.manual_seed(0)
    enc = PiNet(
        atom_types=[1, 6, 7, 8],
        r_max=4.0,
        n_basis=3,
        pp_nodes=[8, 8],
        pi_nodes=[8, 8],
        ii_nodes=[8, 8],
        depth=2,
        rank=3,
    )
    return PiNetPotential(encoder=enc, hidden_dim=16).to(_DEVICE).eval()


def _tiny_batch():
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.1, 0.0], [0.3, 1.2, 0.2], [1.4, 1.1, -0.1]],
        dtype=torch.float32,
        device=_DEVICE,
    )
    z = torch.tensor([1, 6, 7, 8], dtype=torch.long, device=_DEVICE)
    edge_index = torch.tensor(
        [[0, 1], [1, 0], [0, 2], [2, 0], [1, 3], [3, 1], [2, 3], [3, 2]],
        dtype=torch.long,
        device=_DEVICE,
    )
    batch = torch.zeros(4, dtype=torch.long, device=_DEVICE)
    return make_graph_batch(pos, z, edge_index, batch)


def _warmup(model: PiNetPotential, batch: object) -> None:
    """Run one forward to materialize PiNet's lazy parameters before quantizing."""
    model(batch.clone(), compute_forces=False)


def test_null_control_identical_weights_zero_delta():
    model = _tiny_potential()
    twin = _tiny_potential()
    batch = _tiny_batch()
    _warmup(model, batch)
    _warmup(twin, batch)
    twin.load_state_dict(model.state_dict())  # identical weights
    dF = paired_force_delta(model, twin, batch)
    s = summarize_delta(dF)
    assert s["F_rms"] == pytest.approx(0.0, abs=1e-10)
    assert s["F_bias"] == pytest.approx(0.0, abs=1e-10)


def test_int4_quantization_perturbs_forces():
    model = _tiny_potential()
    quant = _tiny_potential()
    batch = _tiny_batch()
    _warmup(model, batch)
    _warmup(quant, batch)
    quant.load_state_dict(quantize_state_dict(model.state_dict(), "int4"))
    dF = paired_force_delta(model, quant, batch)
    s = summarize_delta(dF)
    assert s["F_rms"] > 0.0
    assert math.isfinite(s["F_rms"])
