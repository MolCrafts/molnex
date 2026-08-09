"""Compile tests for the MD component engine.

``Integrator.step`` / ``rollout`` must ``torch.compile(fullgraph=True)`` with 0
graph breaks — including over a real PiNet force field — and match eager where
the result is deterministic (step with fixed noise; NVE rollout where noise has
zero weight). The pytree/force-consistency contracts live in
``test_types.py`` / ``test_forcefield.py``.
"""

import pytest
import torch

from molix.md import (
    HarmonicForceField,
    LangevinVerletIntegrator,
    LennardJonesCutForceField,
    LennardJonesForceField,
    NeighborList,
    PotentialForceField,
)
from tests.test_molix.test_md.conftest import (
    make_cubic_lattice,
    make_pinet_template,
    make_tiny_potential,
)

_DTYPE = torch.float64


def test_lj_force_matches_autograd():
    torch.manual_seed(0)
    pos = torch.randn(8, 3, dtype=_DTYPE, requires_grad=True)
    ff = LennardJonesForceField(epsilon=1.0, sigma=1.0).to(_DTYPE)
    out = ff(pos)
    (ref,) = torch.autograd.grad(out.energy, pos)
    assert torch.allclose(out.forces, -ref, atol=1e-8), "LJ closed-form force != -dE/dx"


def _ljcut_ig(
    *, rebuild: bool, skin: float = 0.0
) -> tuple[LangevinVerletIntegrator, NeighborList, torch.Tensor]:
    """Argon lj/cut over the 27-atom lattice, with the rebuild switch explicit.

    ``rebuild`` is passed on purpose in both arms: the switch defaults to the
    force field's ``rebuilds_neighbors`` (``True`` here), and a live policy is
    an eager, host-syncing decision that cannot live inside a traced step —
    a compiled-path test must therefore freeze it (``rebuild=False``).
    """
    pos, cell = make_cubic_lattice(n_side=3, spacing=3.0)
    torch.manual_seed(2)
    pos = pos + 0.2 * torch.randn_like(pos)
    nl = NeighborList(cell=cell, cutoff=3.5, positions=pos, skin=skin)
    ff = LennardJonesCutForceField(epsilon=0.7, sigma=2.5, neighbors=nl).to(_DTYPE)
    ig = LangevinVerletIntegrator(
        ff, dt=0.5, gamma=0.0, kbt=0.0, mass=39.95, seed=2, rebuild=rebuild
    )
    return ig, nl, pos


def test_ljcut_step_fullgraph_compiles_and_matches_eager():
    """lj/cut over the fixed-capacity list (index_add + cutoff mask) traces fullgraph."""
    ig, _, pos = _ljcut_ig(rebuild=False)
    st = ig.initial(pos, torch.zeros_like(pos))
    noise = torch.zeros_like(pos)
    eager = ig.step(st, noise)
    comp = torch.compile(ig.step, fullgraph=True, backend=_BACKEND)(st, noise)
    assert torch.allclose(eager.pos, comp.pos, atol=1e-12)
    assert torch.allclose(eager.forces, comp.forces, atol=1e-12)
    assert torch.allclose(eager.energy, comp.energy, atol=1e-12)


def test_frozen_ljcut_rollout_compiles_fullgraph_and_leaves_the_list_alone():
    """``rebuild=False`` is the compiled-path invariant: dynamo specialises the
    Python bool, the policy body stays dead, and ``rollout`` — not just ``step``
    — still traces to one graph and matches eager. A single rebuild inside the
    loop would show up here as a graph break *and* as a nonzero count."""
    ig, nl, pos = _ljcut_ig(rebuild=False)
    vel = torch.zeros_like(pos)
    eager = ig.rollout(ig.initial(pos.clone(), vel.clone()), 4)
    comp = torch.compile(ig.rollout, fullgraph=True, backend=_BACKEND)(
        ig.initial(pos.clone(), vel.clone()), 4
    )
    assert torch.allclose(eager.pos, comp.pos, atol=1e-12)
    assert torch.allclose(eager.forces, comp.forces, atol=1e-12)
    assert torch.allclose(eager.energy, comp.energy, atol=1e-12)
    assert nl.rebuild_count == 0, "a frozen integrator rebuilt the list"


def test_live_ljcut_advances_eagerly_and_drives_the_policy():
    """The production counterpart: ``rebuild=True`` keeps the loop eager
    (``advance_n``) and the list's policy actually fires — the compiled arm
    above must not be passing because nothing ever rebuilds."""
    ig, nl, pos = _ljcut_ig(rebuild=True)
    state = ig.initial(pos, torch.zeros_like(pos))
    ig.advance_n(state, 4)
    assert nl.rebuild_count > 0


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
    template = make_pinet_template()
    model = make_tiny_potential()
    model(template.clone())  # warmup lazy params
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
    assert torch.allclose(eager.forces, comp.forces, atol=atol)
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
