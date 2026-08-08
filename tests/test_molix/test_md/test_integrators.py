"""Tests for the BAOAB Langevin velocity-Verlet integrator component.

Single-function correctness only (repo rule: no e2e under ``tests/``).
Long-horizon physics validation — NVE energy conservation, equipartition —
lives in ``benchmarks/verify_md_lj_nve.py``, not here.
"""

import pytest
import torch

from molix.md import HarmonicForceField, Integrator, LangevinVerletIntegrator, MDState

_DTYPE = torch.float64


def _ig(k: float = 1.0, **kw):
    return LangevinVerletIntegrator(HarmonicForceField(k).to(_DTYPE), **kw)


def test_step_constants_match_the_closed_form():
    """The precomputed buffers are the BAOAB constants from the paper."""
    import math

    dt, gamma, kbt, mass = 0.05, 2.0, 1.5, 2.0
    ig = _ig(1.0, dt=dt, gamma=gamma, kbt=kbt, mass=mass)
    assert float(ig.c1) == pytest.approx(math.exp(-gamma * dt))
    assert float(ig.c2) == pytest.approx(math.sqrt(1.0 - math.exp(-2.0 * gamma * dt)))
    assert float(ig.sigma) == pytest.approx(math.sqrt(kbt / mass))
    assert float(ig.inv_mass) == pytest.approx(1.0 / mass)


def test_step_is_one_baoab_update():
    """``step`` applies exactly B-A-O-A-B with the precomputed constants."""
    torch.manual_seed(0)
    ig = _ig(1.0, dt=0.05, gamma=2.0, kbt=1.5, mass=2.0)
    state = ig.initial(torch.randn(4, 3, dtype=_DTYPE), torch.randn(4, 3, dtype=_DTYPE))
    noise = torch.randn(4, 3, dtype=_DTYPE)

    half = 0.5 * ig.dt
    vel = state.vel + half * state.forces * ig.inv_mass  # B (cached entry force)
    pos = state.pos + half * vel  # A
    vel = ig.c1 * vel + ig.c2 * ig.sigma * noise  # O
    pos = pos + half * vel  # A
    force = -pos  # k = 1: F = -x at the new position
    vel = vel + half * force * ig.inv_mass  # B

    out = ig.step(state, noise)
    assert torch.equal(out.pos, pos)
    assert torch.equal(out.vel, vel)
    assert torch.equal(out.forces, force)


def test_draw_noise_is_seed_reproducible():
    ref = torch.zeros(5, 3, dtype=_DTYPE)
    a = _ig(1.0, dt=0.05, gamma=3.0, kbt=1.0, mass=1.0, seed=42)
    b = _ig(1.0, dt=0.05, gamma=3.0, kbt=1.0, mass=1.0, seed=42)
    c = _ig(1.0, dt=0.05, gamma=3.0, kbt=1.0, mass=1.0, seed=43)
    assert torch.equal(a.draw_noise(ref), b.draw_noise(ref))  # same seed -> identical
    assert not torch.equal(a.draw_noise(ref), c.draw_noise(ref))  # different seed


def test_step_nve_matches_baoab_at_gamma_zero():
    """The elided O step is the float identity, so the fast path is exact."""
    torch.manual_seed(0)
    pos = torch.randn(6, 3, dtype=_DTYPE)
    vel = torch.randn(6, 3, dtype=_DTYPE)
    integ = _ig(1.0, dt=0.01, gamma=0.0, kbt=0.0, mass=1.0).cast_state(_DTYPE)
    state = integ.initial(pos, vel)
    noise = torch.randn_like(vel)  # multiplied by c2=0 — must not matter
    via_baoab = integ.step(state, noise)
    via_nve = integ.step_nve(state)
    assert torch.equal(via_baoab.pos, via_nve.pos)
    assert torch.equal(via_baoab.vel, via_nve.vel)


def test_advance_n_matches_manual_advance_loop():
    """``advance_n`` visits exactly the states a manual ``advance`` loop does."""
    k, mass, dt, gamma, kbt = 1.0, 1.0, 0.05, 3.0, 1.0
    pos0 = torch.randn(6, 3, dtype=_DTYPE)
    vel0 = torch.randn(6, 3, dtype=_DTYPE)

    chunked = _ig(k, dt=dt, gamma=gamma, kbt=kbt, mass=mass, seed=11)
    end_a = chunked.advance_n(chunked.initial(pos0.clone(), vel0.clone()), 5)

    ig = _ig(k, dt=dt, gamma=gamma, kbt=kbt, mass=mass, seed=11)
    state = ig.initial(pos0.clone(), vel0.clone())
    for _ in range(5):
        state = ig.advance(state)
    assert torch.equal(end_a.pos, state.pos)
    assert torch.equal(end_a.vel, state.vel)


class _MidpointEuler(Integrator):
    """Minimal conforming subclass: implements only what the ABC demands.

    Guards the extension seam — a subclass that writes exactly ``advance`` (+
    ``rollout``) must inherit working ``initial`` / ``advance_n`` /
    ``removed_dof`` defaults. The old ABC declared ``step``/``rollout`` but
    the runner called the undeclared ``advance_n`` — a conforming subclass
    crashed at runtime.
    """

    def __init__(self, force, dt: float):
        super().__init__(force)
        self.register_buffer("dt", torch.as_tensor(dt))

    def advance(self, state: MDState) -> MDState:
        vel = state.vel + self.dt * state.forces
        pos = state.pos + self.dt * vel
        out = self.eval_force(pos)
        return MDState(pos, vel, out.forces, out.energy)

    def rollout(self, state: MDState, n_steps: int) -> MDState:
        return self.advance_n(state, n_steps)


def test_conforming_subclass_inherits_the_abc_defaults():
    ig = _MidpointEuler(HarmonicForceField(1.0).to(_DTYPE), dt=0.01).to(_DTYPE)
    state = ig.initial(torch.randn(4, 3, dtype=_DTYPE), torch.zeros(4, 3, dtype=_DTYPE))
    out = ig.advance_n(state, 3)  # inherited default: loop over advance
    assert out.pos.shape == (4, 3)
    assert not torch.equal(out.pos, state.pos)
    assert ig.removed_dof == 3  # inherited NVE default


def test_removed_dof_follows_the_thermostat():
    assert _ig(1.0, dt=0.01, gamma=0.0, kbt=0.0, mass=1.0).removed_dof == 3
    assert _ig(1.0, dt=0.01, gamma=2.0, kbt=1.0, mass=1.0).removed_dof == 0


def test_eval_force_casts_into_the_state_dtype():
    """An fp32 force field driven by an fp64 state must not leak fp32 into it."""
    from molix.md import CallableForceField

    # Returns strictly fp32 whatever it is fed — the potential's own precision.
    ff = CallableForceField(lambda pos: ((pos.float() ** 2).sum(), -2.0 * pos.float()))
    ig = LangevinVerletIntegrator(ff, dt=0.01, gamma=0.0, kbt=0.0, mass=1.0).cast_state(_DTYPE)
    state = ig.initial(torch.zeros(3, 3, dtype=_DTYPE), torch.zeros(3, 3, dtype=_DTYPE))
    assert state.forces.dtype == _DTYPE
    assert state.energy.dtype == _DTYPE
    advanced = ig.advance(state)
    assert advanced.pos.dtype == _DTYPE
    assert advanced.vel.dtype == _DTYPE


def test_cast_state_leaves_the_force_field_alone():
    """The MD-side cast must not touch the potential's parameters."""
    ff = HarmonicForceField(1.0)
    ig = LangevinVerletIntegrator(ff, dt=0.01, gamma=0.0, kbt=0.0, mass=1.0)
    ig.cast_state(torch.float64)
    assert ig.dt.dtype == torch.float64
    assert ig.mass_col.dtype == torch.float64
    assert ff.k.dtype == torch.float32


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
