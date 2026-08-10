"""Tests for the BAOAB Langevin velocity-Verlet integrator component.

Single-function correctness only (repo rule: no e2e under ``tests/``).
Long-horizon physics validation — NVE energy conservation, equipartition —
lives in ``benchmarks/verify_md_lj_nve.py``, not here.
"""

import pytest
import torch

from molix.md import (
    HarmonicForceField,
    Integrator,
    LangevinVerletIntegrator,
    LennardJonesCutForceField,
    MDState,
    NeighborList,
)
from tests.test_molix.test_md.conftest import make_cubic_lattice

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


class _CountingNLForce(HarmonicForceField):
    """Harmonic well that records the positions passed to rebuild_neighbors."""

    def __init__(self, k: float = 1.0):
        super().__init__(k)
        self.rebuild_positions: list[torch.Tensor] = []

    def rebuild_neighbors(self, pos: torch.Tensor) -> None:
        self.rebuild_positions.append(pos.detach().clone())


def _lj_cut_force() -> tuple[LennardJonesCutForceField, torch.Tensor]:
    """lj/cut argon over a live list — a force field that owns a policy."""
    pos, cell = make_cubic_lattice(n_side=4, spacing=3.0)
    nl = NeighborList(cell=cell, cutoff=3.5, positions=pos, skin=0.5, every=1, delay=0, check=True)
    ff = LennardJonesCutForceField(epsilon=0.7, sigma=2.5, neighbors=nl).to(_DTYPE)
    return ff, pos


class TestIntegratorRebuildSwitch:
    """``Integrator.rebuild`` — a construction-time Python bool, not a counter.

    *Whether this integrator asks* the neighbour policy, derived from *whether
    the force field can answer* (``ForceField.rebuilds_neighbors``). Static so
    dynamo specialises the branch: ``rebuild=False`` leaves the body dead and
    ``torch.compile(fullgraph=True)`` still traces one graph, ``rebuild=True``
    runs the list's policy eagerly between force calls.
    """

    def test_the_switch_is_derived_from_the_force_field(self):
        """No kwarg ⇒ follow the force field, so no caller has to remember."""
        listless = Integrator(HarmonicForceField(1.0).to(_DTYPE))
        ff, _ = _lj_cut_force()
        live = LangevinVerletIntegrator(ff, dt=0.5, gamma=0.0, kbt=0.0, mass=39.95)
        assert listless.rebuild is False
        assert live.rebuild is True

    def test_the_kwarg_overrides_the_derivation_both_ways(self):
        """The frozen-list run and the forced-policy run are both reachable."""
        ff, _ = _lj_cut_force()
        frozen = LangevinVerletIntegrator(ff, dt=0.5, gamma=0.0, kbt=0.0, mass=39.95, rebuild=False)
        forced = LangevinVerletIntegrator(
            HarmonicForceField(1.0).to(_DTYPE), dt=0.01, gamma=0.0, kbt=0.0, mass=1.0, rebuild=True
        )
        assert frozen.rebuild is False
        assert forced.rebuild is True

    def test_the_seam_fires_once_per_force_evaluation_at_those_positions(self):
        """The policy must run at the positions ``F`` is evaluated at.

        Velocity-Verlet evaluates ``F`` at the *end-of-step* positions, so a
        seam wired to step-start would leave the list one displacement behind
        the positions entering ``F = -∇E`` — a systematic NVE energy leak
        (surviving assertion of the deleted ``rebuild_every`` test).
        """
        torch.manual_seed(0)
        force = _CountingNLForce(1.0).to(_DTYPE)
        ig = LangevinVerletIntegrator(
            force, dt=0.01, gamma=0.0, kbt=0.0, mass=1.0, rebuild=True
        ).cast_state(_DTYPE)
        pos0 = torch.randn(4, 3, dtype=_DTYPE)
        vel0 = torch.randn(4, 3, dtype=_DTYPE) * 0.1
        state = ig.initial(pos0, vel0)
        # initial() → one force eval at pos0
        assert len(force.rebuild_positions) == 1
        assert torch.equal(force.rebuild_positions[0], pos0)

        state = ig.step_nve(state)
        # second force eval at the *new* positions (not pos0)
        assert len(force.rebuild_positions) == 2
        assert torch.equal(force.rebuild_positions[1], state.pos)
        assert not torch.equal(force.rebuild_positions[1], pos0)

    def test_a_disabled_switch_never_touches_the_list(self):
        """``rebuild=False`` over a list-backed force field freezes the list:
        the policy clock must not even tick, or the compiled path would carry
        the host sync it exists to avoid."""
        ff, pos = _lj_cut_force()
        ig = LangevinVerletIntegrator(
            ff, dt=0.5, gamma=0.0, kbt=0.0, mass=39.95, rebuild=False
        ).cast_state(_DTYPE)
        state = ig.initial(pos, torch.zeros_like(pos))
        for _ in range(3):
            state = ig.step_nve(state)
        assert ff.neighbors.rebuild_count == 0
        assert ff.neighbors.ago == 0

    def test_the_modulo_counter_is_gone(self):
        """One owner: the list. The integrator keeps no cadence state at all."""
        ig = _ig(1.0, dt=0.01, gamma=0.0, kbt=0.0, mass=1.0)
        assert not hasattr(ig, "rebuild_every")
        assert not hasattr(ig, "_force_eval_count")

    def test_a_conforming_subclass_derives_the_switch(self):
        """``super().__init__(force)`` — the positional seam — still suffices."""
        ig = _MidpointEuler(HarmonicForceField(1.0).to(_DTYPE), dt=0.01)
        assert ig.rebuild is False
