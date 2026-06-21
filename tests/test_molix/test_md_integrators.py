"""Tests for the Langevin velocity-Verlet integrator on an analytic harmonic well.

Decoupled from PiNet: the force function is F = -k x, so correctness (energy
conservation under NVE, equipartition under Langevin) is provable analytically.
"""

import torch

from molix.md import LangevinVerletIntegrator

_DTYPE = torch.float64


def _harmonic(k: float):
    def force_fn(pos: torch.Tensor):
        energy = 0.5 * k * (pos**2).sum()
        return energy, -k * pos

    return force_fn


def test_nve_conserves_energy():
    torch.manual_seed(0)
    k, mass, dt = 1.0, 1.0, 0.01
    n = 8
    pos = torch.randn(n, 3, dtype=_DTYPE)
    vel = torch.randn(n, 3, dtype=_DTYPE)
    integ = LangevinVerletIntegrator(_harmonic(k), dt=dt, gamma=0.0, kbt=0.0, mass=mass)

    def total_energy(p, v):
        return (0.5 * mass * (v**2).sum() + 0.5 * k * (p**2).sum()).item()

    e0 = total_energy(pos, vel)
    drift = 0.0
    for _ in range(2000):
        pos, vel, _, _ = integ.step(pos, vel)
        drift = max(drift, abs(total_energy(pos, vel) - e0) / abs(e0))
    assert drift < 1e-3, f"NVE energy drift too large: {drift}"


def test_langevin_reaches_equipartition():
    torch.manual_seed(1)
    k, mass, dt, gamma, kbt = 1.0, 1.0, 0.05, 5.0, 2.0
    n = 60
    pos = torch.zeros(n, 3, dtype=_DTYPE)
    vel = torch.zeros(n, 3, dtype=_DTYPE)
    integ = LangevinVerletIntegrator(_harmonic(k), dt=dt, gamma=gamma, kbt=kbt, mass=mass, seed=7)
    dof = n * 3
    ke_samples = []
    for i in range(20000):
        pos, vel, _, _ = integ.step(pos, vel)
        if i >= 10000:  # discard equilibration
            ke_samples.append((0.5 * mass * (vel**2).sum()).item())
    mean_ke = sum(ke_samples) / len(ke_samples)
    measured_kbt = mean_ke / (0.5 * dof)  # <KE> = 0.5 * dof * kbt
    assert abs(measured_kbt - kbt) / kbt < 0.05, f"equipartition off: {measured_kbt} vs {kbt}"


def test_noise_is_seed_reproducible():
    k, mass, dt, gamma, kbt = 1.0, 1.0, 0.05, 3.0, 1.0
    pos0 = torch.zeros(5, 3, dtype=_DTYPE)
    vel0 = torch.zeros(5, 3, dtype=_DTYPE)

    def run(seed):
        integ = LangevinVerletIntegrator(
            _harmonic(k), dt=dt, gamma=gamma, kbt=kbt, mass=mass, seed=seed
        )
        out = integ.run(pos0.clone(), vel0.clone(), 50)
        return out["pos"]

    assert torch.allclose(run(42), run(42))  # same seed -> identical
    assert not torch.allclose(run(42), run(43))  # different seed -> different


def test_run_records_trajectory_shapes():
    integ = LangevinVerletIntegrator(_harmonic(1.0), dt=0.01, gamma=1.0, kbt=1.0, mass=1.0)
    out = integ.run(torch.zeros(4, 3, dtype=_DTYPE), torch.zeros(4, 3, dtype=_DTYPE), 10)
    assert out["pos"].shape == (10, 4, 3)
    assert out["vel"].shape == (10, 4, 3)
    assert out["energy"].shape == (10,)


def test_force_caching_matches_naive_step_loop():
    """``run`` (one force eval/step via cache) is bit-identical to a per-step ``step`` loop."""
    k, mass, dt, gamma, kbt = 1.0, 1.0, 0.05, 3.0, 1.0
    pos0 = torch.randn(6, 3, dtype=_DTYPE)
    vel0 = torch.randn(6, 3, dtype=_DTYPE)

    cached = LangevinVerletIntegrator(_harmonic(k), dt=dt, gamma=gamma, kbt=kbt, mass=mass, seed=11)
    out = cached.run(pos0.clone(), vel0.clone(), 30)

    naive = LangevinVerletIntegrator(_harmonic(k), dt=dt, gamma=gamma, kbt=kbt, mass=mass, seed=11)
    p, v = pos0.clone(), vel0.clone()
    for i in range(30):
        p, v, _, _ = naive.step(p, v)
        assert torch.equal(out["pos"][i], p)
        assert torch.equal(out["vel"][i], v)


def test_step_cached_round_trip_matches_step():
    """``initial_force`` + ``step_cached`` reproduces ``step`` exactly (same seed)."""
    k, mass, dt, gamma, kbt = 1.0, 1.0, 0.05, 2.0, 1.5
    pos0 = torch.randn(5, 3, dtype=_DTYPE)
    vel0 = torch.randn(5, 3, dtype=_DTYPE)

    a = LangevinVerletIntegrator(_harmonic(k), dt=dt, gamma=gamma, kbt=kbt, mass=mass, seed=3)
    pa, va, _, _ = a.step(pos0.clone(), vel0.clone())

    b = LangevinVerletIntegrator(_harmonic(k), dt=dt, gamma=gamma, kbt=kbt, mass=mass, seed=3)
    force = b.initial_force(pos0.clone())
    pb, vb, _, fb = b.step_cached(pos0.clone(), vel0.clone(), force)
    assert torch.equal(pa, pb)
    assert torch.equal(va, vb)


def test_compile_matches_eager():
    """``compile=True`` integrates to the same trajectory as eager (same seed)."""
    k, mass, dt, gamma, kbt = 1.0, 1.0, 0.02, 2.0, 1.0
    pos0 = torch.randn(4, 3, dtype=_DTYPE)
    vel0 = torch.randn(4, 3, dtype=_DTYPE)

    eager = LangevinVerletIntegrator(_harmonic(k), dt=dt, gamma=gamma, kbt=kbt, mass=mass, seed=5)
    out_e = eager.run(pos0.clone(), vel0.clone(), 15)

    comp = LangevinVerletIntegrator(
        _harmonic(k), dt=dt, gamma=gamma, kbt=kbt, mass=mass, seed=5, compile=True
    )
    out_c = comp.run(pos0.clone(), vel0.clone(), 15)
    assert torch.allclose(out_e["pos"], out_c["pos"], atol=1e-10)
    assert torch.allclose(out_e["vel"], out_c["vel"], atol=1e-10)
