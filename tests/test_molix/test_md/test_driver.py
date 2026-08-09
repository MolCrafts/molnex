"""Tests for molix.md.driver — constructor wiring and per-call contracts only."""

import pytest
import torch

from molix.md import (
    EV_PER_AMU_A2_FS2,
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
)
from tests.test_molix.test_md.conftest import make_cubic_lattice


@pytest.fixture
def system():
    torch.manual_seed(0)
    return torch.randn(6, 3, dtype=torch.float64), torch.zeros(6, 3, dtype=torch.float64)


def _lj_cut_argon(skin: float) -> tuple[LennardJonesCutForceField, torch.Tensor]:
    """Argon lj/cut over the 64-atom lattice with a policy-configured list.

    Argon in the integrator's (amu, Å, fs) system: ε = 0.0103 eV, σ = 2.5 Å,
    r_cut = 3.5 Å over a 12 Å cube (minimum perpendicular half-width 6.0 Å, so
    ``r_build = 3.5 + skin`` stays admissible up to ``skin = 2.5``).

    ``capacity_factor=2.5`` is measured, not defensive: a lattice start
    under-counts the edges a warm run reaches (600 live against the 519 rows
    the default would allocate), and the overflow guard is not what these
    tests pin.
    """
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
        epsilon=0.0103 / EV_PER_AMU_A2_FS2,  # argon well depth, eV -> amu A^2/fs^2
        sigma=2.5,
        neighbors=neighbors,
        cutoff=3.5,
    )
    return force, pos


class _TotalEnergyHook(MDHook):
    """Sample the conserved quantity once per step — the drift observable."""

    def __init__(self) -> None:
        self.totals: list[torch.Tensor] = []

    def on_step_end(self, runner: MDRunner, step: int, obs: MDObservables) -> None:
        self.totals.append(obs.total.detach().clone())


def _argon_nve(skin: float, *, n_steps: int = 100) -> tuple[NeighborList, MDState, list[float]]:
    """100 steps of NVE argon through the **public** ``MD`` path at one skin.

    Deterministic CPU float64: seeded Maxwell-Boltzmann velocities, γ = 0, no
    wall clock, no filesystem, no network. No cadence kwarg — the driver
    derives the integrator's switch from the force field, and the list owns
    when to rebuild.

    Returns:
        ``(neighbors, final_state, total_energies)`` — the list (for
        ``rebuild_count`` / ``ndanger``), the final :class:`MDState`, and the
        per-step total energy in amu·Å²/fs².
    """
    force, pos = _lj_cut_argon(skin)
    sampler = _TotalEnergyHook()
    velocities = MaxwellBoltzmann(39.95, n_atoms=64).sample(300.0, seed=0)
    md = MD(force, mass=39.95, dt=4.0, gamma=0.0, dtype=torch.float64, hooks=[sampler])
    md.set_potential_dtype(torch.float64)
    final = md.run(pos, velocities, n_steps, chunk=1)
    return force.neighbors, final, [float(total) for total in sampler.totals]


def _drift(totals: list[float]) -> float:
    """``max_t |E(t) − E(0)| / |E(0)|`` — the dimensionless conservation metric."""
    reference = totals[0]
    return max(abs(total - reference) for total in totals) / abs(reference)


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


class TestMDNeighborPolicy:
    """The driver derives the rebuild switch; it never owns a cadence.

    ``MD(rebuild_every=)`` was a second owner of a decision the neighbour list
    already makes (``skin`` / ``every`` / ``delay`` / ``check``), and the
    step-start ``NeighborListHook`` was a third at the wrong seam. Both are
    gone: ``MD`` builds the integrator and lets it read
    ``ForceField.rebuilds_neighbors``.
    """

    def test_the_cadence_kwarg_is_gone(self):
        """Hard removal (``stage: experimental``), no back-compat shim: the
        migration is ``NeighborList(skin=, every=, delay=, check=)``."""
        with pytest.raises(TypeError):
            MD(HarmonicForceField(k=1.0), mass=1.0, dt=0.01, rebuild_every=1)

    def test_no_step_start_hook_is_installed(self):
        """The wrong-seam hook is not silently replaced by another default."""
        md = MD(HarmonicForceField(k=1.0), mass=1.0, dt=0.01)
        assert md.runner.hooks == []

    def test_the_switch_follows_the_force_field(self):
        """Listless ⇒ off, list-backed ⇒ on, with no kwarg either way."""
        listless = MD(HarmonicForceField(k=1.0), mass=1.0, dt=0.01)
        force, _ = _lj_cut_argon(skin=1.0)
        live = MD(force, mass=39.95, dt=4.0, dtype=torch.float64)
        assert listless.integrator.rebuild is False
        assert live.integrator.rebuild is True

    def test_the_autocast_wrapper_delegates_the_capability(self):
        """``autocast_dtype`` wraps the force field *before* the integrator is
        built, so the wrapper must forward the flag — otherwise a bf16 run
        silently freezes its neighbour list."""
        force, _ = _lj_cut_argon(skin=1.0)
        md = MD(force, mass=39.95, dt=4.0, autocast_dtype=torch.bfloat16)
        assert md.force is not force  # the wrapper is in the way
        assert md.integrator.rebuild is True

    def test_a_compiled_force_field_delegates_the_capability(self):
        """``torch.compile(ff)`` returns an ``OptimizedModule`` that forwards
        attribute reads to the wrapped module — the assumption the GPU
        benchmark's compiled-force-field path depends on."""
        force, _ = _lj_cut_argon(skin=1.0)
        compiled = torch.compile(force, backend="eager")
        md = MD(compiled, mass=39.95, dt=4.0, dtype=torch.float64)
        assert md.integrator.rebuild is True

    def test_a_skin_costs_no_energy_conservation(self):
        """Invariant (c): a skin-gated run must conserve energy as well as one
        that runs the policy at every force evaluation.

        A pair inside ``r_cut`` but missing from the list contributes an O(1)
        force error (lj/cut shifts the *energy* continuous at the cutoff, not
        the force), which integrates into a one-signed leak — so a broken skin
        shows up as drift, not as noise. The ``skin=0`` baseline is itself
        nonzero (finite-``dt`` velocity-Verlet), which is asserted so the ratio
        cannot pass vacuously.
        """
        _, _, gated = _argon_nve(skin=1.0)
        _, _, every_eval = _argon_nve(skin=0.0)
        baseline = _drift(every_eval)
        assert baseline > 0.0, "the no-skin baseline must have real discretisation drift"
        assert _drift(gated) <= 3.0 * baseline

    def test_the_skin_buys_rebuilds_without_moving_the_physics(self):
        """Invariant (f): the observables the public path exposes.

        ``rebuild_count(skin=0) == 100`` is the anti-vacuity pin — a wiring
        that never calls the policy satisfies every equality and monotonicity
        assertion here but not that literal. One policy call per force
        evaluation over 100 steps, and the entry evaluation sits at the build
        positions (``max_d2 == 0``, strict ``>``), so it does not rebuild.
        ``ndanger(skin=0) == 99`` (not 100): the declined entry evaluation
        consumes one ``ago`` tick, so step 1's rebuild lands at ``ago == 2``
        (not dangerous) and only the remaining 99 rebuilds — each at
        ``ago == 1 == max(every, delay)`` after a reset — count. Link 04's
        degenerate-limit alarm, off by exactly the entry evaluation.
        """
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
