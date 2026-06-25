"""Compile + typed-contract tests for the MD component engine (spec ac-003/004/005).

- ``MDState`` / ``ForceOutput`` are pytrees (flatten → unflatten round-trip).
- ``LennardJonesForceField`` closed-form force == ``-∂E/∂pos`` (autograd ref).
- ``Integrator.step`` / ``rollout`` ``torch.compile(fullgraph=True)`` with 0 graph
  breaks — including over a real PiNet force field — and == eager where the
  result is deterministic (step with fixed noise; NVE rollout where noise has
  zero weight).
"""

import pytest
import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from molix.md import (
    ForceOutput,
    HarmonicForceField,
    LangevinVerletIntegrator,
    LennardJonesForceField,
    MDState,
    PotentialForceField,
)
from tests.test_molix.test_md_dynamics import _template, _tiny_potential

_DTYPE = torch.float64


def test_mdstate_forceoutput_are_pytrees():
    s = MDState(torch.randn(4, 3), torch.randn(4, 3), torch.randn(4, 3), torch.tensor(1.0))
    leaves, spec = tree_flatten(s)
    assert len(leaves) == 4
    rebuilt = tree_unflatten(leaves, spec)
    assert isinstance(rebuilt, MDState)
    assert torch.equal(rebuilt.pos, s.pos) and torch.equal(rebuilt.energy, s.energy)
    fo = ForceOutput(torch.tensor(1.0), torch.randn(4, 3))
    leaves2, spec2 = tree_flatten(fo)
    assert isinstance(tree_unflatten(leaves2, spec2), ForceOutput)


def test_lj_force_matches_autograd():
    torch.manual_seed(0)
    pos = torch.randn(8, 3, dtype=_DTYPE, requires_grad=True)
    ff = LennardJonesForceField(epsilon=1.0, sigma=1.0).to(_DTYPE)
    out = ff(pos)
    (ref,) = torch.autograd.grad(out.energy, pos)
    assert torch.allclose(out.forces, -ref, atol=1e-8), "LJ closed-form force != -dE/dx"


def _harm_ig(gamma: float):
    return LangevinVerletIntegrator(
        HarmonicForceField(1.0).to(_DTYPE), dt=0.01, gamma=gamma, kbt=1.0, mass=1.0, seed=3
    )


def test_step_compile_matches_eager_deterministic():
    """``step(state, noise)`` is pure → compiled == eager exactly (any γ)."""
    ig = _harm_ig(gamma=2.0)
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
    eager = _harm_ig(0.0).rollout(_harm_ig(0.0).initial(pos.clone(), vel.clone()), 8)
    ig = _harm_ig(0.0)
    comp = torch.compile(ig.rollout, fullgraph=True)(ig.initial(pos.clone(), vel.clone()), 8)
    assert torch.allclose(eager.pos, comp.pos, atol=1e-10)
    assert torch.allclose(eager.vel, comp.vel, atol=1e-10)


def _pinet_ig(dtype: torch.dtype):
    template = _template()
    model = _tiny_potential()
    model(template.clone(), compute_forces=False)  # warmup lazy params
    if dtype == torch.float64:
        model = model.to(torch.float64)
        template["atoms", "pos"] = template["atoms", "pos"].to(torch.float64)
    ff = PotentialForceField(model, template)
    ig = LangevinVerletIntegrator(ff, dt=0.001, gamma=0.0, kbt=0.0, mass=12.0, seed=1)
    pos0 = template["atoms", "pos"].clone()
    return ig, pos0, torch.zeros_like(pos0)


# backend="aot_eager": fullgraph tracing (0 graph breaks) + AOTAutograd, which
# exercises the functorch force path WITHOUT the Inductor CPU backend. Inductor's
# CPU codegen currently asserts on EnergyAggregation's scatter_add
# (scatter_mode='atomic_add'); the production path is CUDA + inductor
# (reduce-overhead), validated by benchmarks/verify_pinet_cudagraph_ef.py. The
# spec's "0 graph-break" claim is a dynamo property, independent of the backend.
_BACKEND = "aot_eager"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pinet_step_fullgraph_compiles_and_matches_eager(dtype):
    """PiNet force path compiles fullgraph (0 break, else raises) and == eager."""
    ig, pos0, vel0 = _pinet_ig(dtype)
    st = ig.initial(pos0, vel0)
    noise = torch.zeros_like(pos0)
    eager = ig.step(st, noise)
    comp = torch.compile(ig.step, fullgraph=True, backend=_BACKEND)(st, noise)
    atol = 1e-10 if dtype == torch.float64 else 1e-4
    assert torch.allclose(eager.pos, comp.pos, atol=atol)
    assert torch.allclose(eager.force, comp.force, atol=atol)
    assert torch.allclose(eager.energy, comp.energy, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pinet_rollout_fullgraph_nve(dtype):
    """PiNet NVE rollout compiles fullgraph, stays finite, and == eager."""
    ig, pos0, vel0 = _pinet_ig(dtype)
    eager = ig.rollout(ig.initial(pos0.clone(), vel0.clone()), 3)
    ig2, p2, v2 = _pinet_ig(dtype)
    comp = torch.compile(ig2.rollout, fullgraph=True, backend=_BACKEND)(
        ig2.initial(p2.clone(), v2.clone()), 3
    )
    assert torch.isfinite(comp.pos).all() and torch.isfinite(comp.energy).all()
    atol = 1e-10 if dtype == torch.float64 else 1e-3
    assert torch.allclose(eager.pos, comp.pos, atol=atol)
