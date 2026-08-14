"""Tests for molix.md.driver — composer wiring and per-call contracts only."""

import pytest
import torch

from molix.md import (
    MD,
    HarmonicForceField,
    LangevinVerletIntegrator,
    LennardJonesCutForceField,
    MaxwellBoltzmann,
    MDHook,
    MDObservables,
    MDRunner,
    MDState,
    NeighborList,
    VelocityVerletIntegrator,
)
from molix.units import units
from tests.test_molix.test_md.conftest import make_cubic_lattice, md_system

_E_EV_TO_MD = (1.0 * units.eV).to("amu * angstrom ** 2 / femtosecond ** 2").magnitude


@pytest.fixture
def system():
    torch.manual_seed(0)
    pos = torch.randn(6, 3, dtype=torch.float64)
    vel = torch.zeros(6, 3, dtype=torch.float64)
    return md_system(pos, vel, 1.0)


def _lj_cut_argon(skin: float) -> tuple[LennardJonesCutForceField, torch.Tensor]:
    """Argon lj/cut over the 64-atom lattice with a policy-configured list."""
    pos, cell = make_cubic_lattice(n_side=4, spacing=3.0)
    neighbors = NeighborList(
        cell=cell,
        cutoff=3.5,
        positions=pos,
        skin=skin,
        every=1,
        delay=0,
        check=True,
        capacity_factor=2.5,
    )
    force = LennardJonesCutForceField(
        epsilon=0.0103 * _E_EV_TO_MD,
        sigma=2.5,
        neighbors=neighbors,
        cutoff=3.5,
    )
    return force, pos


def _nve(force, *, dt: float, dtype=None, hooks=None, rebuild=None) -> MD:
    return (
        MD(dtype)
        .set_forcefield(force)
        .set_integrator(VelocityVerletIntegrator(dt, rebuild=rebuild))
        .set_hooks(hooks)
    )


class _TotalEnergyHook(MDHook):
    """Sample the conserved quantity once per step — the drift observable."""

    def __init__(self) -> None:
        self.totals: list[torch.Tensor] = []

    def on_step_end(self, runner: MDRunner, step: int, obs: MDObservables) -> None:
        self.totals.append(obs.total.detach().clone())


def _argon_nve(skin: float, *, n_steps: int = 100) -> tuple[NeighborList, MDState, list[float]]:
    """100 steps of NVE argon through the public ``MD`` path at one skin."""
    force, pos = _lj_cut_argon(skin)
    sampler = _TotalEnergyHook()
    atoms = md_system(pos, torch.zeros_like(pos), 39.95)
    MaxwellBoltzmann().sample(atoms, 300.0, seed=0)
    force.to(dtype=torch.float64)
    md = _nve(force, dt=4.0, dtype=torch.float64, hooks=[sampler])
    final = md.run(atoms, n_steps, chunk=1)
    return force.neighbors, final, [float(total) for total in sampler.totals]


def _drift(totals: list[float]) -> float:
    """``max_t |E(t) − E(0)| / |E(0)|`` — the dimensionless conservation metric."""
    reference = totals[0]
    return max(abs(total - reference) for total in totals) / abs(reference)


class TestMD:
    """Test the MD composer."""

    def test_runs_and_returns_final_state(self, system):
        """The run yields the advanced typed state, not the initial one."""
        pos0 = system["atoms", "pos"].clone()
        md = _nve(HarmonicForceField(k=1.0), dt=0.01, dtype=torch.float64)
        out = md.run(system, 3)
        assert out.pos.shape == pos0.shape
        assert not torch.allclose(out.pos, pos0)
        assert torch.equal(system["atoms", "pos"], out.pos)

    def test_dtype_governs_the_md_side_only(self, system):
        """``MD(dtype)`` casts state + integrator constants, never the model."""
        force = HarmonicForceField(k=1.0)  # constructed fp32
        md = _nve(force, dt=0.01, dtype=torch.float64)
        out = md.run(system, 1)
        assert out.pos.dtype == torch.float64
        assert out.forces.dtype == torch.float64
        assert force.k.dtype == torch.float32
        assert md.integrator.dt.dtype == torch.float64

    def test_force_field_dtype_is_the_force_fields(self, system):
        """Cast the model on the force field, not through MD."""
        force = HarmonicForceField(k=1.0)
        force.to(dtype=torch.float64)
        md = _nve(force, dt=0.01, dtype=torch.float64)
        assert force.k.dtype == torch.float64
        assert md.force is force

    def test_dtype_none_leaves_precision_alone(self, system):
        """Omitting dtype must not silently downcast a float64 caller."""
        md = _nve(HarmonicForceField(k=1.0).double(), dt=0.01)
        assert md.run(system, 1).pos.dtype == torch.float64

    def test_run_requires_both_components(self, system):
        with pytest.raises(RuntimeError, match="set_forcefield and set_integrator"):
            MD(torch.float64).run(system, 1)

    def test_set_integrator_takes_a_constructed_integrator(self):
        force = HarmonicForceField(k=1.0).double()
        integrator = VelocityVerletIntegrator(0.01)
        md = MD(torch.float64).set_forcefield(force).set_integrator(integrator)
        assert md.integrator is integrator
        assert md.runner.integrator is integrator

    def test_langevin_is_a_constructed_integrator_not_an_md_kwarg(self, system):
        force = HarmonicForceField(k=1.0)
        ig = LangevinVerletIntegrator(0.01, gamma=0.1, kbt=1.0)
        md = MD(torch.float64).set_forcefield(force).set_integrator(ig)
        out = md.run(system, 2)
        assert out.pos.shape[0] == 6

    def test_facade_constructor_kwargs_are_gone(self):
        """Hard removal: no dt/gamma/force/mass/hooks/device on MD()."""
        with pytest.raises(TypeError):
            MD(HarmonicForceField(k=1.0), mass=1.0, dt=0.01)  # type: ignore[misc]


class TestMDNeighborPolicy:
    """The integrator derives the rebuild switch; MD never owns a cadence."""

    def test_the_cadence_kwarg_is_gone(self):
        with pytest.raises(TypeError):
            MD(torch.float64, rebuild_every=1)  # type: ignore[call-arg]

    def test_no_step_start_hook_is_installed(self):
        md = _nve(HarmonicForceField(k=1.0), dt=0.01)
        assert md.runner.hooks == []

    def test_the_switch_follows_the_force_field(self):
        listless = _nve(HarmonicForceField(k=1.0), dt=0.01)
        force, _ = _lj_cut_argon(skin=1.0)
        live = _nve(force, dt=4.0, dtype=torch.float64)
        assert listless.integrator.rebuild is False
        assert live.integrator.rebuild is True

    def test_a_compiled_force_field_delegates_the_capability(self):
        force, _ = _lj_cut_argon(skin=1.0)
        compiled = torch.compile(force, backend="eager")
        md = _nve(compiled, dt=4.0, dtype=torch.float64)
        assert md.integrator.rebuild is True

    def test_a_skin_costs_no_energy_conservation(self):
        _, _, gated = _argon_nve(skin=1.0)
        _, _, every_eval = _argon_nve(skin=0.0)
        baseline = _drift(every_eval)
        assert baseline > 0.0, "the no-skin baseline must have real discretisation drift"
        assert _drift(gated) <= 3.0 * baseline

    def test_the_skin_buys_rebuilds_without_moving_the_physics(self):
        arms = {skin: _argon_nve(skin=skin) for skin in (0.0, 0.5, 1.0)}
        counts = [arms[skin][0].rebuild_count for skin in (0.0, 0.5, 1.0)]
        assert counts == sorted(counts, reverse=True)
        assert counts[0] == 100
        assert counts[-1] < counts[0]
        assert arms[0.0][0].ndanger == 99
        assert arms[0.5][0].ndanger == 0
        assert arms[1.0][0].ndanger == 0
        reference = arms[0.0][1].energy
        for skin in (0.5, 1.0):
            torch.testing.assert_close(
                arms[skin][1].energy, reference, atol=1e-10, rtol=0, msg=f"skin={skin}"
            )


class TestMaxwellBoltzmann:
    """Initial-velocity sampling — reads atoms.mass, writes atoms.vel."""

    def test_removes_com_momentum(self):
        """Net momentum must vanish, which is what makes the NVE dof 3N-3."""
        pos = torch.zeros(8, 3, dtype=torch.float64)
        atoms = md_system(pos, torch.zeros_like(pos), torch.full((8,), 2.0))
        MaxwellBoltzmann().sample(atoms, 300.0, seed=1)
        vel = atoms["atoms", "vel"]
        mass = atoms["atoms", "mass"]
        assert vel.shape == (8, 3)
        assert float((mass.reshape(-1, 1).double() * vel).sum(0).abs().max()) < 1e-12

    def test_is_reproducible(self):
        pos = torch.zeros(5, 3)
        a = md_system(pos, torch.zeros_like(pos), 1.0)
        b = md_system(pos.clone(), torch.zeros_like(pos), 1.0)
        MaxwellBoltzmann().sample(a, 300.0, seed=3)
        MaxwellBoltzmann().sample(b, 300.0, seed=3)
        assert torch.equal(a["atoms", "vel"], b["atoms", "vel"])

    def test_mass_length_must_match_atoms(self):
        from tensordict import TensorDict

        atoms = TensorDict(
            {
                "atoms": TensorDict(
                    {
                        "pos": torch.zeros(7, 3),
                        "vel": torch.zeros(7, 3),
                        "mass": torch.ones(3),
                    },
                    batch_size=[],
                )
            },
            batch_size=[],
        )
        with pytest.raises(ValueError, match="n_atoms"):
            MaxwellBoltzmann().sample(atoms, 300.0, seed=0)
