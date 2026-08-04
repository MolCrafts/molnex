"""Tests for molix.quant (OOP fake-quantization infra + PTQ + Eq8 T_eff)."""

import math

import pytest
import torch

from molix.quant import (
    EffectiveTemperature,
    FakeQuantize,
    ForceDelta,
    Int4Scheme,
    Int8Scheme,
    IntScheme,
    Quantizer,
    QuantScheme,
)
from molzoo.pinet import PiNetPotential
from tests.conftest import make_graph_batch

# PiNet uses lazy params and make_graph_batch builds CPU metadata; keep tensors on CPU.
_DEVICE = torch.device("cpu")


# --------------------------------------------------------------------------- #
# QuantScheme registry + math
# --------------------------------------------------------------------------- #
def test_scheme_registry_names():
    assert set(QuantScheme.names()) == {"fp16", "bf16", "int8", "int8_pc", "int4", "int4_pc"}
    assert isinstance(QuantScheme.from_name("int8"), Int8Scheme)


@pytest.mark.parametrize("name", ["int1", "float5", "garbage", "int_pc", "int0"])
def test_from_name_rejects_unknown(name):
    # int1/int0 are below the N>=2 floor; the rest match no registered or dynamic scheme.
    with pytest.raises(ValueError, match="unknown scheme"):
        QuantScheme.from_name(name)


@pytest.mark.parametrize(
    ("name", "n_bits", "per_channel"),
    [("int3", 3, False), ("int16", 16, False), ("int5_pc", 5, True), ("int2", 2, False)],
)
def test_from_name_builds_dynamic_int(name, n_bits, per_channel):
    """Unregistered ``int<N>`` / ``int<N>_pc`` schemes are synthesized on the fly."""
    scheme = QuantScheme.from_name(name)
    assert isinstance(scheme, IntScheme)
    assert scheme.n_bits == n_bits
    assert scheme.per_channel is per_channel
    assert scheme.name == name


def test_dynamic_int_bit_width_monotonic():
    """A finer dynamic scheme (int6) must quantize less coarsely than int4."""
    torch.manual_seed(7)
    w = torch.randn(128, 128, dtype=torch.float32)
    err4 = (QuantScheme.from_name("int4").quantize(w) - w).abs().mean()
    err6 = (QuantScheme.from_name("int6").quantize(w) - w).abs().mean()
    err8 = (QuantScheme.from_name("int8").quantize(w) - w).abs().mean()
    assert err8 < err6 < err4


def test_all_schemes_change_high_precision_weights():
    torch.manual_seed(2)
    w = torch.randn(64, 64, dtype=torch.float32)
    for name in QuantScheme.names():
        q = QuantScheme.from_name(name).quantize(w)
        assert q.shape == w.shape and q.dtype == w.dtype
        assert not torch.equal(q, w), f"{name} left weights unchanged"


def test_int4_coarser_than_int8():
    torch.manual_seed(3)
    w = torch.randn(128, 128, dtype=torch.float32)
    err8 = (Int8Scheme().quantize(w) - w).abs().mean()
    err4 = (Int4Scheme().quantize(w) - w).abs().mean()
    assert err4 > err8


def test_scheme_passes_through_integer_tensors():
    idx = torch.arange(10, dtype=torch.long)
    assert torch.equal(Int8Scheme().quantize(idx), idx)


# --------------------------------------------------------------------------- #
# Quantizer: PTQ + non-invasive apply/remove + STE
# --------------------------------------------------------------------------- #
def test_quantize_state_dict_and_strip_prefix():
    q = Quantizer("int8")
    sd = {"a.weight": torch.randn(8, 8), "a.idx": torch.arange(8)}
    out = q.quantize_state_dict(sd)
    assert not torch.equal(out["a.weight"], sd["a.weight"])
    assert torch.equal(out["a.idx"], sd["a.idx"])  # integer untouched
    prefixed = {"_orig_mod.a.weight": sd["a.weight"]}
    assert "a.weight" in Quantizer.strip_compile_prefix(prefixed)


def test_none_scheme_is_baseline_noop():
    q = Quantizer(None)
    m = torch.nn.Linear(4, 4)
    w0 = m.weight.detach().clone()
    q.apply(m)
    assert torch.equal(m.weight, w0)  # untouched
    sd = q.quantize_state_dict(m.state_dict())
    assert torch.equal(sd["weight"], w0)


def test_apply_remove_is_non_invasive_and_reversible():
    torch.manual_seed(5)
    m = torch.nn.Linear(16, 16)
    w0 = m.weight.detach().clone()
    Quantizer("int4").apply(m)
    assert not torch.equal(m.weight.detach(), w0)  # quantized on access
    Quantizer("int4").remove(m)
    assert torch.allclose(m.weight.detach(), w0)  # restored exactly


def test_fake_quantize_straight_through_gradient():
    """STE: forward is quantized, backward is identity (QAT-trainable)."""
    fq = FakeQuantize(Int4Scheme())
    w = torch.randn(8, 8, requires_grad=True)
    out = fq(w)
    out.sum().backward()
    assert torch.allclose(w.grad, torch.ones_like(w))  # identity gradient


# --------------------------------------------------------------------------- #
# EffectiveTemperature (Eq8)
# --------------------------------------------------------------------------- #
def test_effective_temperature_matches_eq8():
    f_rms_sq, dt, gamma, mass, dof = 4.0, 0.5, 0.01, 12.0, 9
    teff = EffectiveTemperature(dt=dt, gamma=gamma, mass=mass, dof=dof)
    assert teff.energy(f_rms_sq) == pytest.approx(f_rms_sq * dt / (2.0 * gamma * mass * dof))


def test_effective_temperature_scales_as_inverse_gamma():
    base = EffectiveTemperature(dt=0.5, gamma=0.01, mass=12.0, dof=9).energy(4.0)
    doubled = EffectiveTemperature(dt=0.5, gamma=0.02, mass=12.0, dof=9).energy(4.0)
    assert doubled == pytest.approx(base / 2.0)


def test_effective_temperature_ratio_is_dimensionless():
    kb = EffectiveTemperature.KB_EV_PER_K
    r = EffectiveTemperature(dt=0.5, gamma=0.01, mass=12.0, dof=9).ratio(4.0, t_target=300.0)
    expected = (4.0 * 0.5 / (2.0 * 0.01 * 12.0 * 9)) / (kb * 300.0)
    assert r == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# ForceDelta
# --------------------------------------------------------------------------- #
def test_force_delta_summary_recovers_analytic_moments():
    torch.manual_seed(0)
    x = torch.randn(200_000, dtype=torch.float64)  # ~N(0,1)
    s = ForceDelta(x).summary()
    assert s["F_bias"] == pytest.approx(x.mean().item(), abs=1e-9)
    assert s["F_rms"] == pytest.approx(x.pow(2).mean().sqrt().item(), abs=1e-9)
    assert abs(s["F_skew"]) < 0.05
    assert abs(s["F_exkurt"]) < 0.1
    assert s["n"] == 200_000


def test_force_delta_summary_flags_bias_and_heavy_tails():
    torch.manual_seed(1)
    biased = torch.randn(50_000, dtype=torch.float64) + 3.0
    assert ForceDelta(biased).summary()["F_bias"] == pytest.approx(3.0, abs=0.05)
    u = torch.rand(200_000, dtype=torch.float64) - 0.5
    laplace = -u.sign() * torch.log1p(-2 * u.abs())  # heavy tails -> positive excess kurtosis
    assert ForceDelta(laplace).summary()["F_exkurt"] > 1.0


def test_force_delta_zero_residual_has_zero_moments():
    s = ForceDelta(torch.zeros(100, dtype=torch.float64)).summary()
    assert s["F_bias"] == 0.0 and s["F_rms"] == 0.0
    assert s["F_skew"] == 0.0 and s["F_exkurt"] == 0.0


# --------------------------------------------------------------------------- #
# ForceDelta.between on PiNet (PTQ on a ready-to-use model)
# --------------------------------------------------------------------------- #
def _tiny_potential() -> PiNetPotential:
    torch.manual_seed(0)
    return (
        PiNetPotential(
            atom_types=[1, 6, 7, 8],
            r_max=4.0,
            n_basis=3,
            pp_nodes=[8, 8],
            pi_nodes=[8, 8],
            ii_nodes=[8, 8],
            depth=2,
            rank=3,
            hidden_dim=16,
        )
        .to(_DEVICE)
        .eval()
    )


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


def test_null_control_identical_weights_zero_delta():
    model = _tiny_potential()
    twin = _tiny_potential()
    batch = _tiny_batch()
    model(batch.clone(), compute_forces=False)  # materialise lazy params
    twin(batch.clone(), compute_forces=False)
    twin.load_state_dict(model.state_dict())  # identical weights
    s = ForceDelta.between(model, twin, batch).summary()
    assert s["F_rms"] == pytest.approx(0.0, abs=1e-10)
    assert s["F_bias"] == pytest.approx(0.0, abs=1e-10)


def test_int4_ptq_perturbs_forces():
    model = _tiny_potential()
    quant = _tiny_potential()
    batch = _tiny_batch()
    model(batch.clone(), compute_forces=False)
    quant(batch.clone(), compute_forces=False)
    quant.load_state_dict(Quantizer("int4").quantize_state_dict(model.state_dict()))
    s = ForceDelta.between(model, quant, batch).summary()
    assert s["F_rms"] > 0.0
    assert math.isfinite(s["F_rms"])
