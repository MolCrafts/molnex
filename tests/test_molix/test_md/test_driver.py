"""Tests for molix.md.driver — constructor wiring and per-call contracts only."""

import pytest
import torch

from molix.md import (
    MD,
    HarmonicForceField,
    LangevinVerletIntegrator,
    MaxwellBoltzmann,
)


@pytest.fixture
def system():
    torch.manual_seed(0)
    return torch.randn(6, 3, dtype=torch.float64), torch.zeros(6, 3, dtype=torch.float64)


class TestMD:
    """Test the MD component."""

    def test_runs_and_returns_final_state(self, system):
        """The run yields the advanced typed state, not the initial one."""
        pos, vel = system
        md = MD(HarmonicForceField(k=1.0), mass=1.0, dt=0.01, dtype=torch.float64)
        out = md.run(pos, vel, 3)
        assert out.pos.shape == pos.shape
        assert not torch.allclose(out.pos, pos)

    def test_dtype_governs_the_md_side_only(self, system):
        """``MD(dtype=)`` casts state + integrator constants, never the model:
        the MD process and the inference process are studied separately."""
        pos, vel = system
        force = HarmonicForceField(k=1.0)  # constructed fp32
        md = MD(force, mass=1.0, dt=0.01, dtype=torch.float64)
        out = md.run(pos.float(), vel.float(), 1)
        assert out.pos.dtype == torch.float64  # trajectory in the MD dtype
        assert out.forces.dtype == torch.float64  # boundary cast into the state
        assert force.k.dtype == torch.float32  # the potential was left alone
        assert md.integrator.dt.dtype == torch.float64  # step constants follow

    def test_set_potential_dtype_casts_the_model_explicitly(self, system):
        """The inference-side precision is its own explicit axis."""
        force = HarmonicForceField(k=1.0)
        md = MD(force, mass=1.0, dt=0.01, dtype=torch.float64)
        md.set_potential_dtype(torch.float64)
        assert force.k.dtype == torch.float64

    def test_dtype_none_leaves_precision_alone(self, system):
        """Omitting dtype must not silently downcast a float64 caller."""
        pos, vel = system
        md = MD(HarmonicForceField(k=1.0).double(), mass=1.0, dt=0.01)
        assert md.run(pos, vel, 1).pos.dtype == torch.float64

    def test_autocast_keeps_state_in_the_run_dtype(self, system):
        """bf16-mixed: the model may run reduced, the trajectory may not."""
        pos, vel = system
        md = MD(
            HarmonicForceField(k=1.0),
            mass=1.0,
            dt=0.01,
            dtype=torch.float32,
            autocast_dtype=torch.bfloat16,
        )
        out = md.run(pos, vel, 1)
        assert out.pos.dtype == torch.float32
        assert out.forces.dtype == torch.float32

    def test_accepts_a_constructed_integrator(self):
        """The integrator seam: the caller-built integrator is wired through."""
        force = HarmonicForceField(k=1.0).double()
        integrator = LangevinVerletIntegrator(force, dt=0.01, gamma=0.0, kbt=0.0, mass=1.0)
        md = MD(force, mass=1.0, integrator=integrator)
        assert md.integrator is integrator
        assert md.runner.integrator is integrator

    def test_integrator_excludes_langevin_parameters(self):
        """dt/kbt/temperature parameterise the default integrator only."""
        force = HarmonicForceField(k=1.0)
        integrator = LangevinVerletIntegrator(force, dt=0.01, gamma=0.0, kbt=0.0, mass=1.0)
        with pytest.raises(ValueError, match="mutually exclusive"):
            MD(force, mass=1.0, dt=0.5, integrator=integrator)

    def test_integrator_must_wrap_the_same_force(self):
        other = HarmonicForceField(k=2.0)
        integrator = LangevinVerletIntegrator(other, dt=0.01, gamma=0.0, kbt=0.0, mass=1.0)
        with pytest.raises(ValueError, match="must be the force field"):
            MD(HarmonicForceField(k=1.0), mass=1.0, integrator=integrator)

    def test_dt_required_without_integrator(self):
        with pytest.raises(ValueError, match="dt is required"):
            MD(HarmonicForceField(k=1.0), mass=1.0)

    def test_rebuild_every_configures_the_integrator(self):
        """``rebuild_every`` is handled in Integrator.eval_force (at force-eval
        positions), not via a step-start NeighborListHook — the latter lags
        the list by one displacement and leaks NVE energy."""
        md = MD(HarmonicForceField(k=1.0), mass=1.0, dt=0.01, rebuild_every=5)
        assert md.integrator.rebuild_every == 5
        assert md.runner.hooks == []  # no step-start NL hook

    def test_no_rebuild_when_disabled(self):
        md = MD(HarmonicForceField(k=1.0), mass=1.0, dt=0.01)
        assert md.integrator.rebuild_every is None
        assert md.runner.hooks == []

    def test_temperature_sets_kbt(self):
        """`temperature=` is the ergonomic form of `kbt=`; both cannot be given."""
        force = HarmonicForceField(k=1.0)
        MD(force, mass=1.0, dt=0.01, gamma=0.1, temperature=300.0)
        with pytest.raises(ValueError, match="not both"):
            MD(force, mass=1.0, dt=0.01, gamma=0.1, temperature=300.0, kbt=1.0)

    def test_thermostat_without_temperature_is_rejected(self):
        """gamma>0 with kbt=0 is a thermostat at 0 K — almost certainly a mistake."""
        with pytest.raises(ValueError, match="needs kbt or temperature"):
            MD(HarmonicForceField(k=1.0), mass=1.0, dt=0.01, gamma=0.1)


class TestMaxwellBoltzmann:
    """Initial-velocity sampling — its own component, off the run driver."""

    def test_removes_com_momentum(self):
        """Net momentum must vanish, which is what makes the NVE dof 3N-3."""
        mass = torch.full((8,), 2.0)
        vel = MaxwellBoltzmann(mass).sample(300.0, seed=1)
        assert vel.shape == (8, 3)
        assert float((mass.reshape(-1, 1).double() * vel).sum(0).abs().max()) < 1e-12

    def test_is_reproducible(self):
        """Same seed, same velocities — a run must be reproducible from its args."""
        sampler = MaxwellBoltzmann(torch.ones(5))
        assert torch.equal(sampler.sample(300.0, seed=3), sampler.sample(300.0, seed=3))

    def test_scalar_mass_needs_n_atoms(self):
        """A scalar mass carries no system size — silent (1, 3) output was a bug."""
        with pytest.raises(ValueError, match="n_atoms"):
            MaxwellBoltzmann(1.0)
        vel = MaxwellBoltzmann(1.0, n_atoms=7).sample(300.0, seed=0)
        assert vel.shape == (7, 3)
