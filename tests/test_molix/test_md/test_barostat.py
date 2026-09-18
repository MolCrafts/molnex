"""Tests for molix.md.barostat — isotropic Berendsen scale, not a full NPT run."""

import pytest
import torch

from molix.md import BerendsenBarostat, MDState, VelocityVerletIntegrator
from molix.md.barostat import (
    compressibility_bar_to_md,
    instantaneous_pressure,
    pressure_bar_to_md,
)

_DTYPE = torch.float64


class _ZeroPots:
    """The minimal energy the adapter's op composes: ``E = 0 · Σ pos``, zero forces."""

    def __call__(self, system):
        from tensordict import TensorDict

        pos = system["atoms", "pos"]
        if "graphs" not in system.keys():
            system["graphs"] = TensorDict({}, batch_size=[])
        system["graphs", "energy"] = (pos * 0.0).sum().reshape(system["graphs"].batch_size)
        return system


def test_pressure_bar_to_md_is_positive():
    assert pressure_bar_to_md(1.0) > 0.0


def test_scale_shrinks_a_box_under_too_low_pressure():
    """Zero virial, finite kinetic energy, P_target high → μ < 1."""
    pos = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=_DTYPE)
    vel = torch.zeros(2, 3, dtype=_DTYPE)
    cell = torch.eye(3, dtype=_DTYPE) * 10.0
    virial = torch.zeros(3, 3, dtype=_DTYPE)
    state = MDState(pos, vel, torch.zeros_like(pos), torch.tensor(0.0), cell, virial)
    baro = BerendsenBarostat(pressure=1000.0, tau=100.0)
    out = baro.scale(state, mass=1.0, dt=1.0)
    assert torch.det(out.cell).abs() < torch.det(cell).abs()
    assert out.pos.norm() < pos.norm()
    assert torch.equal(out.vel, vel)


def test_scale_is_frozen_without_compressibility():
    """κ = 0 must not move the cell; missing κ used to have the wrong dimension."""
    pos = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=_DTYPE)
    vel = torch.ones(2, 3, dtype=_DTYPE)
    cell = torch.eye(3, dtype=_DTYPE) * 10.0
    virial = torch.zeros(3, 3, dtype=_DTYPE)
    state = MDState(pos, vel, torch.zeros_like(pos), torch.tensor(0.0), cell, virial)
    frozen = BerendsenBarostat(pressure=1.0, tau=100.0, compressibility=0.0)
    out = frozen.scale(state, mass=1.0, dt=1.0)
    assert torch.allclose(out.cell, cell)
    coupled = BerendsenBarostat(pressure=1.0, tau=100.0)
    moved = coupled.scale(state, mass=1.0, dt=1.0)
    assert not torch.allclose(moved.cell, cell)


def test_compressibility_bar_to_md_keeps_kappa_p_dimensionless():
    kappa = compressibility_bar_to_md(4.5e-5)
    assert kappa * pressure_bar_to_md(1.0) == pytest.approx(4.5e-5)


def test_scale_requires_cell_and_virial():
    baro = BerendsenBarostat(pressure=1.0, tau=100.0)
    state = MDState(
        torch.zeros(2, 3, dtype=_DTYPE),
        torch.zeros(2, 3, dtype=_DTYPE),
        torch.zeros(2, 3, dtype=_DTYPE),
        torch.tensor(0.0),
    )
    with pytest.raises(RuntimeError, match="cell and virial"):
        baro.scale(state, mass=1.0, dt=0.5)


def test_eval_force_keeps_virial_on_the_dict():
    """Integrator.eval_force casts the force dict; virial is not dropped."""
    virial = torch.eye(3, dtype=torch.float32)

    class _Provider:
        def __call__(self, pos: torch.Tensor) -> dict[str, torch.Tensor]:
            return {
                "energy": pos.float().pow(2).sum(),
                "forces": -2.0 * pos.float(),
                "virial": virial,
            }

    ig = VelocityVerletIntegrator(0.01).bind_force(_Provider()).bind_mass(1.0)
    ig.cast_state(_DTYPE)
    pos = torch.ones(3, 3, dtype=_DTYPE)
    out = ig.eval_force(pos)
    assert out["virial"].dtype == _DTYPE
    state = ig.initial(pos, torch.zeros_like(pos))
    assert state.virial is not None


def test_sync_cell_keeps_batched_graphs_cell() -> None:
    """Barostat state is (3, 3); the work batch stores graphs.cell as (1, 3, 3)."""
    from tensordict import TensorDict

    from molix.md.driver import MD, _TensorDictForce

    cell = torch.eye(3, dtype=_DTYPE).unsqueeze(0)
    work = TensorDict(
        {
            "atoms": TensorDict({"pos": torch.zeros(2, 3, dtype=_DTYPE)}, batch_size=[2]),
            "graphs": TensorDict({"cell": cell}, batch_size=[1]),
        },
        batch_size=[],
    )
    md = MD(_DTYPE)
    adapter = _TensorDictForce(_ZeroPots(), 1.0)
    adapter.bind_work(work)
    md._adapter = adapter
    state = MDState(
        work["atoms", "pos"],
        torch.zeros(2, 3, dtype=_DTYPE),
        torch.zeros(2, 3, dtype=_DTYPE),
        torch.tensor(0.0, dtype=_DTYPE),
        torch.eye(3, dtype=_DTYPE) * 9.0,
        torch.zeros(3, 3, dtype=_DTYPE),
    )
    md._sync_cell(state)
    assert work["graphs", "cell"] is cell
    assert tuple(work["graphs", "cell"].shape) == (1, 3, 3)
    assert torch.allclose(work["graphs", "cell"][0], state.cell)


def test_sync_cell_follows_a_replaced_graphs_cell() -> None:
    """Virial strain used to swap graphs.cell; the barostat must still land."""
    from tensordict import TensorDict

    from molix.md.driver import MD, _TensorDictForce

    work = TensorDict(
        {
            "atoms": TensorDict({"pos": torch.zeros(2, 3, dtype=_DTYPE)}, batch_size=[2]),
            "graphs": TensorDict({"cell": torch.eye(3, dtype=_DTYPE).unsqueeze(0)}, batch_size=[1]),
        },
        batch_size=[],
    )
    md = MD(_DTYPE)
    adapter = _TensorDictForce(_ZeroPots(), 1.0)
    adapter.bind_work(work)
    md._adapter = adapter
    replaced = (torch.eye(3, dtype=_DTYPE) * 4.0).unsqueeze(0)
    work["graphs", "cell"] = replaced
    state = MDState(
        work["atoms", "pos"],
        torch.zeros(2, 3, dtype=_DTYPE),
        torch.zeros(2, 3, dtype=_DTYPE),
        torch.tensor(0.0, dtype=_DTYPE),
        torch.eye(3, dtype=_DTYPE) * 9.0,
        torch.zeros(3, 3, dtype=_DTYPE),
    )
    md._sync_cell(state)
    assert work["graphs", "cell"] is replaced
    assert torch.allclose(replaced[0], state.cell)


def test_run_does_not_require_grad_on_the_state() -> None:
    """MD trajectory tensors are inference-only — no energy graph leaks out."""
    from tensordict import TensorDict

    from molix.md import MD, VelocityVerletIntegrator
    from molpot.forcefield import ForceField
    from tests.test_molix.test_md.conftest import md_system

    pos = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=_DTYPE)
    system = md_system(pos, torch.zeros_like(pos), 1.0)
    system["bonds"] = TensorDict(
        {
            "atomi": torch.tensor([0], dtype=torch.long),
            "atomj": torch.tensor([1], dtype=torch.long),
        },
        batch_size=[1],
    )
    ff = (
        ForceField()
        .def_bondstyle("harmonic")
        .with_params(bonds={"k": torch.tensor([2.0]), "r0": torch.tensor([1.0])})
    )
    out = (
        MD(_DTYPE).set_forcefield(ff).set_integrator(VelocityVerletIntegrator(0.01)).run(system, 2)
    )
    assert not out.pos.requires_grad
    assert not out.vel.requires_grad
    assert not out.forces.requires_grad
    assert not out.energy.requires_grad


def test_instantaneous_pressure_matches_the_formula():
    vel = torch.ones(2, 3, dtype=_DTYPE)
    cell = torch.eye(3, dtype=_DTYPE) * 4.0
    virial = torch.diag(torch.tensor([1.0, 1.0, 1.0], dtype=_DTYPE))
    state = MDState(
        torch.zeros(2, 3, dtype=_DTYPE),
        vel,
        torch.zeros(2, 3, dtype=_DTYPE),
        torch.tensor(0.0),
        cell,
        virial,
    )
    mass = 2.0
    kinetic = 0.5 * mass * (vel * vel).sum()
    volume = 64.0
    expect = (2.0 * kinetic - 3.0) / (3.0 * volume)
    assert instantaneous_pressure(state, mass).item() == pytest.approx(expect)
