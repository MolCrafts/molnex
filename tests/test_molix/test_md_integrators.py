"""Tests for the BAOAB Langevin velocity-Verlet integrator component.

Decoupled from PiNet via :class:`HarmonicForceField` (F = -k x), so energy
conservation (NVE) and equipartition (Langevin) are provable analytically.
Exercises the component API: ForceField → MDState → Integrator.
"""

import pytest
import torch

from molix.md import HarmonicForceField, LangevinVerletIntegrator

_DTYPE = torch.float64


def _ig(k: float = 1.0, **kw):
    return LangevinVerletIntegrator(HarmonicForceField(k).to(_DTYPE), **kw)


def _total_energy(state, mass, k):
    return (0.5 * mass * (state.vel**2).sum() + 0.5 * k * (state.pos**2).sum()).item()


def test_nve_conserves_energy():
    torch.manual_seed(0)
    k, mass, dt = 1.0, 1.0, 0.01
    pos = torch.randn(8, 3, dtype=_DTYPE)
    vel = torch.randn(8, 3, dtype=_DTYPE)
    ig = _ig(k, dt=dt, gamma=0.0, kbt=0.0, mass=mass)
    state = ig.initial(pos, vel)
    e0 = _total_energy(state, mass, k)
    drift = 0.0
    for _ in range(2000):
        state = ig.advance(state)
        drift = max(drift, abs(_total_energy(state, mass, k) - e0) / abs(e0))
    assert drift < 1e-3, f"NVE energy drift too large: {drift}"


def test_langevin_reaches_equipartition():
    torch.manual_seed(1)
    k, mass, dt, gamma, kbt = 1.0, 1.0, 0.05, 5.0, 2.0
    n = 60
    ig = _ig(k, dt=dt, gamma=gamma, kbt=kbt, mass=mass, seed=7)
    state = ig.initial(torch.zeros(n, 3, dtype=_DTYPE), torch.zeros(n, 3, dtype=_DTYPE))
    dof = n * 3
    ke_samples = []
    for i in range(20000):
        state = ig.advance(state)
        if i >= 10000:  # discard equilibration
            ke_samples.append((0.5 * mass * (state.vel**2).sum()).item())
    measured_kbt = (sum(ke_samples) / len(ke_samples)) / (0.5 * dof)  # <KE>=0.5*dof*kbt
    assert abs(measured_kbt - kbt) / kbt < 0.05, f"equipartition off: {measured_kbt} vs {kbt}"


def test_noise_is_seed_reproducible():
    pos0 = torch.zeros(5, 3, dtype=_DTYPE)
    vel0 = torch.zeros(5, 3, dtype=_DTYPE)

    def run(seed):
        ig = _ig(1.0, dt=0.05, gamma=3.0, kbt=1.0, mass=1.0, seed=seed)
        return ig.run(pos0.clone(), vel0.clone(), 50)["pos"]

    assert torch.allclose(run(42), run(42))  # same seed -> identical
    assert not torch.allclose(run(42), run(43))  # different seed -> different


def test_run_records_trajectory_shapes():
    ig = _ig(1.0, dt=0.01, gamma=1.0, kbt=1.0, mass=1.0)
    out = ig.run(torch.zeros(4, 3, dtype=_DTYPE), torch.zeros(4, 3, dtype=_DTYPE), 10)
    assert out["pos"].shape == (10, 4, 3)
    assert out["vel"].shape == (10, 4, 3)
    assert out["energy"].shape == (10,)


def test_run_matches_manual_advance_loop():
    """``run`` records exactly the states a manual ``initial`` + ``advance`` loop visits."""
    k, mass, dt, gamma, kbt = 1.0, 1.0, 0.05, 3.0, 1.0
    pos0 = torch.randn(6, 3, dtype=_DTYPE)
    vel0 = torch.randn(6, 3, dtype=_DTYPE)

    out = _ig(k, dt=dt, gamma=gamma, kbt=kbt, mass=mass, seed=11).run(
        pos0.clone(), vel0.clone(), 30
    )

    ig = _ig(k, dt=dt, gamma=gamma, kbt=kbt, mass=mass, seed=11)
    state = ig.initial(pos0.clone(), vel0.clone())
    for i in range(30):
        state = ig.advance(state)
        assert torch.equal(out["pos"][i], state.pos)
        assert torch.equal(out["vel"][i], state.vel)


class _CountingHarmonic(HarmonicForceField):
    def __init__(self, k: float = 1.0):
        super().__init__(k)
        self.calls = 0

    def forward(self, pos):
        self.calls += 1
        return super().forward(pos)


def test_force_caching_one_eval_per_step():
    """Force caching: exactly one force-field evaluation per step (+1 to seed)."""
    ff = _CountingHarmonic(1.0).to(_DTYPE)
    ig = LangevinVerletIntegrator(ff, dt=0.05, gamma=1.0, kbt=1.0, mass=1.0, seed=1)
    state = ig.initial(torch.zeros(5, 3, dtype=_DTYPE), torch.zeros(5, 3, dtype=_DTYPE))
    for _ in range(10):
        state = ig.advance(state)
    assert ff.calls == 11  # 1 (initial) + 10 (one per step)


def test_mass_must_be_positive():
    with pytest.raises(ValueError, match="strictly positive"):
        _ig(1.0, dt=0.01, gamma=0.0, kbt=0.0, mass=-1.0)
    with pytest.raises(ValueError, match="strictly positive"):
        _ig(1.0, dt=0.01, gamma=0.0, kbt=0.0, mass=torch.tensor([1.0, -2.0, 3.0]))
