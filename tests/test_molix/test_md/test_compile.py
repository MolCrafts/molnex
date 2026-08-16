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


def _harm_nve(*, rebuild: bool | None = None):
    return (
        VelocityVerletIntegrator(0.01, rebuild=rebuild)
        .bind_force(harmonic_potential_like(1.0, rebuilds_neighbors=rebuild is not False))
        .bind_mass(1.0)
    )


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
    eager = _harm_nve(rebuild=False).rollout(
        _harm_nve(rebuild=False).initial(pos.clone(), vel.clone()), 8
    )
    ig = _harm_nve(rebuild=False)
    comp = torch.compile(ig.rollout, fullgraph=True)(ig.initial(pos.clone(), vel.clone()), 8)
    assert torch.allclose(eager.pos, comp.pos, atol=1e-10)
    assert torch.allclose(eager.vel, comp.vel, atol=1e-10)


def test_frozen_rebuild_rollout_compiles_fullgraph_and_skips_the_policy():
    """``rebuild=False`` is the compiled-path invariant: dynamo specialises the
    Python bool, the policy body stays dead, and ``rollout`` still traces to
    one graph and matches eager.
    """
    helper = harmonic_potential_like(1.0, rebuilds_neighbors=True)
    ig = VelocityVerletIntegrator(0.01, rebuild=False).bind_force(helper).bind_mass(1.0)
    pos = torch.randn(5, 3, dtype=_DTYPE)
    vel = torch.zeros_like(pos)
    eager = ig.rollout(ig.initial(pos.clone(), vel.clone()), 4)
    helper2 = harmonic_potential_like(1.0, rebuilds_neighbors=True)
    ig2 = VelocityVerletIntegrator(0.01, rebuild=False).bind_force(helper2).bind_mass(1.0)
    comp = torch.compile(ig2.rollout, fullgraph=True, backend=_BACKEND)(
        ig2.initial(pos.clone(), vel.clone()), 4
    )
    assert torch.allclose(eager.pos, comp.pos, atol=1e-12)
    assert torch.allclose(eager.forces, comp.forces, atol=1e-12)
    assert torch.allclose(eager.energy, comp.energy, atol=1e-12)
    assert helper.rebuild_positions == []
    assert helper2.rebuild_positions == []


def test_live_rebuild_advances_eagerly_and_drives_the_policy():
    """``rebuild=True`` keeps the loop eager and the policy actually fires."""
    helper = harmonic_potential_like(1.0, rebuilds_neighbors=True)
    ig = VelocityVerletIntegrator(0.01, rebuild=True).bind_force(helper).bind_mass(1.0)
    pos = torch.randn(5, 3, dtype=_DTYPE)
    state = ig.initial(pos, torch.zeros_like(pos))
    ig.advance_n(state, 4)
    assert len(helper.rebuild_positions) == 5  # 1 (initial) + 4 steps
