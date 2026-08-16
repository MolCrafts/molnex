"""Compile tests for the MD component engine.

``Integrator.step`` / ``rollout`` must ``torch.compile(fullgraph=True)`` with 0
graph breaks over an unnamed harmonic ``PotentialLike`` and match eager where
the result is deterministic (step with fixed noise; NVE rollout).
"""

import torch

from molix.md import LangevinVerletIntegrator, VelocityVerletIntegrator
from tests.test_molix.test_md.conftest import harmonic_potential_like

_DTYPE = torch.float64

# backend="aot_eager": fullgraph tracing (0 graph breaks) + AOTAutograd, which
# exercises the force path WITHOUT the Inductor CPU backend.
_BACKEND = "aot_eager"


def _harm_langevin(gamma: float):
    return (
        LangevinVerletIntegrator(0.01, gamma=gamma, kbt=1.0, seed=3)
        .bind_force(harmonic_potential_like(1.0))
        .bind_mass(1.0)
    )


def _harm_nve():
    return VelocityVerletIntegrator(0.01).bind_force(harmonic_potential_like(1.0)).bind_mass(1.0)


def test_step_compile_matches_eager_deterministic():
    """``step(state, noise)`` is pure → compiled == eager exactly (any γ)."""
    ig = _harm_langevin(gamma=2.0)
    st = ig.initial(torch.randn(5, 3, dtype=_DTYPE), torch.randn(5, 3, dtype=_DTYPE))
    noise = torch.randn(5, 3, dtype=_DTYPE)
    eager = ig.step(st, noise)
    comp = torch.compile(ig.step, fullgraph=True)(st, noise)
    assert torch.allclose(eager.pos, comp.pos, atol=1e-12)
    assert torch.allclose(eager.vel, comp.vel, atol=1e-12)


def test_rollout_compile_nve_matches_eager():
    """NVE rollout: noise weight c2=0 → compiled == eager despite RNG path."""
    pos = torch.randn(5, 3, dtype=_DTYPE)
    vel = torch.randn(5, 3, dtype=_DTYPE)
    eager = _harm_nve().rollout(_harm_nve().initial(pos.clone(), vel.clone()), 8)
    ig = _harm_nve()
    comp = torch.compile(ig.rollout, fullgraph=True)(ig.initial(pos.clone(), vel.clone()), 8)
    assert torch.allclose(eager.pos, comp.pos, atol=1e-10)
    assert torch.allclose(eager.vel, comp.vel, atol=1e-10)


def test_nve_rollout_compiles_fullgraph():
    """Integrator.rollout traces one graph over an unnamed PotentialLike."""
    helper = harmonic_potential_like(1.0)
    ig = VelocityVerletIntegrator(0.01).bind_force(helper).bind_mass(1.0)
    pos = torch.randn(5, 3, dtype=_DTYPE)
    vel = torch.zeros_like(pos)
    eager = ig.rollout(ig.initial(pos.clone(), vel.clone()), 4)
    ig2 = VelocityVerletIntegrator(0.01).bind_force(harmonic_potential_like(1.0)).bind_mass(1.0)
    comp = torch.compile(ig2.rollout, fullgraph=True, backend=_BACKEND)(
        ig2.initial(pos.clone(), vel.clone()), 4
    )
    assert torch.allclose(eager.pos, comp.pos, atol=1e-12)
    assert torch.allclose(eager.forces, comp.forces, atol=1e-12)
    assert torch.allclose(eager.energy, comp.energy, atol=1e-12)
