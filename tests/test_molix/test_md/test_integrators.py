"""Tests for the BAOAB Langevin velocity-Verlet integrator component.

Single-function correctness only (repo rule: no e2e under ``tests/``).
Long-horizon physics validation — NVE energy conservation, equipartition —
lives in ``benchmarks/verify_md_lj_nve.py``, not here.
"""

import pytest
import torch

from molix.md import (
    Integrator,
    LangevinVerletIntegrator,
    MDState,
    VelocityVerletIntegrator,
)
from tests.test_molix.test_md.conftest import harmonic_potential_like

_DTYPE = torch.float64


def _harmonic_potential_like(k: float = 1.0, *, rebuilds_neighbors: bool = False):
    """Unnamed PotentialLike helper: ``E = ½k‖x‖²``, ``F = −kx``."""
    return harmonic_potential_like(k, rebuilds_neighbors=rebuilds_neighbors)


def _langevin(k: float = 1.0, *, mass=1.0, **kw):
    ig = LangevinVerletIntegrator(**kw)
    return ig.bind_force(_harmonic_potential_like(k)).bind_mass(mass)


def _nve(k: float = 1.0, *, dt: float, mass=1.0):
    ig = VelocityVerletIntegrator(dt)
    return ig.bind_force(_harmonic_potential_like(k)).bind_mass(mass)


def test_step_constants_match_the_closed_form():
    """The precomputed buffers are the BAOAB constants from the paper."""
    import math

    dt, gamma, kbt, mass = 0.05, 2.0, 1.5, 2.0
    ig = _langevin(1.0, dt=dt, gamma=gamma, kbt=kbt, mass=mass)
    assert float(ig.c1) == pytest.approx(math.exp(-gamma * dt))
    assert float(ig.c2) == pytest.approx(math.sqrt(1.0 - math.exp(-2.0 * gamma * dt)))
    assert float(ig.sigma) == pytest.approx(math.sqrt(kbt / mass))
    assert float(ig.inv_mass) == pytest.approx(1.0 / mass)


def test_step_is_one_baoab_update():
    """``step`` applies exactly B-A-O-A-B with the precomputed constants."""
    torch.manual_seed(0)
    ig = _langevin(1.0, dt=0.05, gamma=2.0, kbt=1.5, mass=2.0)
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
    a = _langevin(1.0, dt=0.05, gamma=3.0, kbt=1.0, mass=1.0, seed=42)
    b = _langevin(1.0, dt=0.05, gamma=3.0, kbt=1.0, mass=1.0, seed=42)
    c = _langevin(1.0, dt=0.05, gamma=3.0, kbt=1.0, mass=1.0, seed=43)
    assert torch.equal(a.draw_noise(ref), b.draw_noise(ref))  # same seed -> identical
    assert not torch.equal(a.draw_noise(ref), c.draw_noise(ref))  # different seed


def test_langevin_rejects_gamma_zero():
    """NVE is VelocityVerletIntegrator — Langevin is not a gamma switch."""
    with pytest.raises(ValueError, match="VelocityVerletIntegrator"):
        LangevinVerletIntegrator(0.01, gamma=0.0, kbt=0.0)


def test_advance_n_matches_manual_advance_loop():
    """``advance_n`` visits exactly the states a manual ``advance`` loop does."""
    k, mass, dt, gamma, kbt = 1.0, 1.0, 0.05, 3.0, 1.0
    pos0 = torch.randn(6, 3, dtype=_DTYPE)
    vel0 = torch.randn(6, 3, dtype=_DTYPE)

    chunked = _langevin(k, dt=dt, gamma=gamma, kbt=kbt, mass=mass, seed=11)
    end_a = chunked.advance_n(chunked.initial(pos0.clone(), vel0.clone()), 5)

    ig = _langevin(k, dt=dt, gamma=gamma, kbt=kbt, mass=mass, seed=11)
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

    def __init__(self, dt: float):
        super().__init__()
        self.register_buffer("dt", torch.as_tensor(dt))

    def advance(self, state: MDState) -> MDState:
        vel = state.vel + self.dt * state.forces
        pos = state.pos + self.dt * vel
        out = self.eval_force(pos)
        return MDState(pos, vel, out["forces"], out["energy"])

    def rollout(self, state: MDState, n_steps: int) -> MDState:
        return self.advance_n(state, n_steps)


def test_conforming_subclass_inherits_the_abc_defaults():
    ig = _MidpointEuler(dt=0.01).bind_force(_harmonic_potential_like(1.0)).to(_DTYPE)
    state = ig.initial(torch.randn(4, 3, dtype=_DTYPE), torch.zeros(4, 3, dtype=_DTYPE))
    out = ig.advance_n(state, 3)  # inherited default: loop over advance
    assert out.pos.shape == (4, 3)
    assert not torch.equal(out.pos, state.pos)
    assert ig.removed_dof == 3  # inherited NVE default


def test_removed_dof_follows_the_scheme():
    assert _nve(1.0, dt=0.01, mass=1.0).removed_dof == 3
    assert _langevin(1.0, dt=0.01, gamma=2.0, kbt=1.0, mass=1.0).removed_dof == 0


def test_eval_force_casts_into_the_state_dtype():
    """An fp32 force provider driven by an fp64 state must not leak fp32 into it."""

    class _Fp32Well:
        rebuilds_neighbors = False

        def __call__(self, pos: torch.Tensor) -> dict[str, torch.Tensor]:
            energy = (pos.float() ** 2).sum()
            return {"energy": energy, "forces": -2.0 * pos.float()}

        def rebuild_neighbors(self, pos: torch.Tensor) -> None:
            return None

    ig = VelocityVerletIntegrator(0.01).bind_force(_Fp32Well()).bind_mass(1.0).cast_state(_DTYPE)
    state = ig.initial(torch.zeros(3, 3, dtype=_DTYPE), torch.zeros(3, 3, dtype=_DTYPE))
    assert state.forces.dtype == _DTYPE
    assert state.energy.dtype == _DTYPE
    advanced = ig.advance(state)
    assert advanced.pos.dtype == _DTYPE
    assert advanced.vel.dtype == _DTYPE


def test_cast_state_leaves_the_force_provider_alone():
    """The MD-side cast must not require or recurse into the force provider."""
    helper = _harmonic_potential_like(1.0)
    ig = VelocityVerletIntegrator(0.01).bind_force(helper).bind_mass(1.0)
    ig.cast_state(torch.float64)
    assert ig.dt.dtype == torch.float64
    assert ig.mass_col.dtype == torch.float64
    assert ig.force is helper


def test_force_caching_one_eval_per_step():
    """Force caching: exactly one force-field evaluation per step (+1 to seed)."""
    helper = _harmonic_potential_like(1.0)
    ig = (
        LangevinVerletIntegrator(0.05, gamma=1.0, kbt=1.0, seed=1).bind_force(helper).bind_mass(1.0)
    )
    state = ig.initial(torch.zeros(5, 3, dtype=_DTYPE), torch.zeros(5, 3, dtype=_DTYPE))
    for _ in range(10):
        state = ig.advance(state)
    assert helper.calls == 11  # 1 (initial) + 10 (one per step)


def test_mass_must_be_positive():
    ig = VelocityVerletIntegrator(0.01)
    with pytest.raises(ValueError, match="strictly positive"):
        ig.bind_mass(-1.0)
    with pytest.raises(ValueError, match="strictly positive"):
        ig.bind_mass(torch.tensor([1.0, -2.0, 3.0]))


class TestIntegratorRebuildSwitch:
    """``Integrator.rebuild`` — a construction-time Python bool, not a counter.

    *Whether this integrator asks* the neighbour policy, derived from *whether
    the bound PotentialLike can answer* (``rebuilds_neighbors``). Static so
    dynamo specialises the branch: ``rebuild=False`` leaves the body dead and
    ``torch.compile(fullgraph=True)`` still traces one graph, ``rebuild=True``
    runs the list's policy eagerly between force calls.
    """

    def test_unnamed_harmonic_potential_like_binds_via_bind_force(self):
        """``E = ½k‖x‖²`` binds through ``Integrator.bind_force``, not a FF subclass."""
        helper = _harmonic_potential_like(k=2.0)
        ig = Integrator().bind_force(helper)
        pos = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=_DTYPE)
        out = ig.eval_force(pos)
        # ‖x‖² = 1 + 4 = 5 → E = ½·2·5 = 5; F = −2x
        assert float(out["energy"]) == pytest.approx(5.0, abs=1e-10)
        assert torch.equal(out["forces"], -2.0 * pos)

    def test_integrator_has_no_rebuild_switch(self):
        ig = Integrator().bind_force(_harmonic_potential_like(1.0))
        assert not hasattr(ig, "rebuild")
        assert not hasattr(ig, "rebuilds_neighbors")

    def test_no_class_defined_in_this_file_subclasses_forcefield(self):
        import inspect

        from molpy import ForceField as MolpyForceField

        import tests.test_molix.test_md.test_integrators as this_module
        from molix.md import ForceField

        for name, obj in inspect.getmembers(this_module, inspect.isclass):
            if obj.__module__ != this_module.__name__:
                continue
            assert not issubclass(obj, ForceField), name
            assert not issubclass(obj, MolpyForceField), name

    def test_the_modulo_counter_is_gone(self):
        """One owner: the list. The integrator keeps no cadence state at all."""
        ig = _nve(1.0, dt=0.01, mass=1.0)
        assert not hasattr(ig, "rebuild_every")
        assert not hasattr(ig, "_force_eval_count")
        assert not hasattr(ig, "rebuild")
