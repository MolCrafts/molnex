"""Tests for molix.md.driver — composer wiring and per-call contracts only."""

import pytest
import torch
from tensordict import TensorDict

from molix.md import (
    MD,
    LangevinVerletIntegrator,
    MaxwellBoltzmann,
    MDHook,
    NeighborList,
    VelocityInitHook,
    VelocityVerletIntegrator,
)
from tests.test_molix.test_md.conftest import make_cubic_lattice, md_system


@pytest.fixture
def system():
    torch.manual_seed(0)
    pos = torch.randn(6, 3, dtype=torch.float64)
    vel = torch.zeros(6, 3, dtype=torch.float64)
    return md_system(pos, vel, 1.0)


def _class_i_harmonic(*, k: float = 2.0, r0: float = 1.0):
    from molpot.forcefield import ClassIForceField

    return ClassIForceField().with_params(bonds={"k": torch.tensor([k]), "r0": torch.tensor([r0])})


def _two_atom_system(*, r: float = 1.5, dtype: torch.dtype = torch.float64) -> TensorDict:
    pos = torch.tensor([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=dtype)
    atoms = md_system(pos, torch.zeros_like(pos), 1.0)
    atoms["bonds"] = TensorDict(
        {
            "atomi": torch.tensor([0], dtype=torch.long),
            "atomj": torch.tensor([1], dtype=torch.long),
        },
        batch_size=[1],
    )
    return atoms


def _nve(force, *, dt: float, dtype=None, hooks=None, rebuild=None) -> MD:
    return (
        MD(dtype)
        .set_forcefield(force)
        .set_integrator(VelocityVerletIntegrator(dt, rebuild=rebuild))
        .set_hooks(hooks)
    )


class TestMD:
    """Test the MD composer."""

    def test_runs_and_returns_final_state(self):
        """The run yields the advanced typed state, not the initial one."""
        atoms = _two_atom_system()
        pos0 = atoms["atoms", "pos"].clone()
        md = _nve(_class_i_harmonic(), dt=0.01, dtype=torch.float64)
        out = md.run(atoms, 3)
        assert out.pos.shape == pos0.shape
        assert not torch.allclose(out.pos, pos0)
        assert torch.equal(atoms["atoms", "pos"], out.pos)

    def test_dtype_governs_the_md_side_only(self):
        """``MD(dtype)`` casts state + integrator constants, never the container."""
        force = _class_i_harmonic()
        md = _nve(force, dt=0.01, dtype=torch.float64)
        out = md.run(_two_atom_system(), 1)
        assert out.pos.dtype == torch.float64
        assert out.forces.dtype == torch.float64
        assert md.force is force
        assert md.integrator.dt.dtype == torch.float64

    def test_dtype_none_leaves_precision_alone(self):
        """Omitting dtype must not silently downcast a float64 caller."""
        md = _nve(_class_i_harmonic(), dt=0.01)
        assert md.run(_two_atom_system(), 1).pos.dtype == torch.float64

    def test_run_requires_both_components(self, system):
        with pytest.raises(RuntimeError, match="set_forcefield and set_integrator"):
            MD(torch.float64).run(system, 1)

    def test_set_integrator_takes_a_constructed_integrator(self):
        from molpy import ForceField as MolpyForceField

        force = MolpyForceField()
        integrator = VelocityVerletIntegrator(0.01)
        md = MD(torch.float64).set_forcefield(force).set_integrator(integrator)
        assert md.integrator is integrator
        assert md.runner.integrator is integrator

    def test_langevin_is_a_constructed_integrator_not_an_md_kwarg(self):
        force = _class_i_harmonic()
        ig = LangevinVerletIntegrator(0.01, gamma=0.1, kbt=1.0)
        md = MD(torch.float64).set_forcefield(force).set_integrator(ig)
        out = md.run(_two_atom_system(), 2)
        assert out.pos.shape[0] == 2

    def test_facade_constructor_kwargs_are_gone(self):
        """Hard removal: no dt/gamma/force/mass/hooks/device on MD()."""
        with pytest.raises(TypeError):
            MD(object(), mass=1.0, dt=0.01)  # type: ignore[misc]

    def test_set_forcefield_accepts_molpy_forcefield_and_returns_self(self):
        from molpy import ForceField as MolpyForceField

        ff = MolpyForceField()
        md = MD()
        assert md.set_forcefield(ff) is md
        assert md.force is ff

    def test_set_forcefield_accepts_class_i_forcefield_and_returns_self(self):
        from molpot.forcefield import ClassIForceField

        ff = ClassIForceField()
        md = MD()
        assert md.set_forcefield(ff) is md
        assert md.force is ff

    def test_set_forcefield_accepts_opls_forcefield_and_returns_self(self):
        from molpot.forcefield import OPLSForceField

        ff = OPLSForceField()
        md = MD()
        assert md.set_forcefield(ff) is md
        assert md.force is ff

    def test_set_forcefield_rejects_object(self):
        with pytest.raises(TypeError, match="ForceField"):
            MD().set_forcefield(object())  # type: ignore[arg-type]

    def test_set_forcefield_rejects_bare_module(self):
        with pytest.raises(TypeError, match="ForceField"):
            MD().set_forcefield(torch.nn.Module())  # type: ignore[arg-type]

    def test_set_forcefield_rejects_pos_callable(self):
        def _pos_callable(pos: torch.Tensor) -> torch.Tensor:
            return pos

        with pytest.raises(TypeError, match="ForceField"):
            MD().set_forcefield(_pos_callable)  # type: ignore[arg-type]

    def test_bare_molpy_to_potentials_rejects_tensordict(self):
        from molpy import ForceField as MolpyForceField

        md = (
            MD()
            .set_forcefield(MolpyForceField("bare", units="real"))
            .set_integrator(VelocityVerletIntegrator(0.1))
        )
        with pytest.raises(TypeError, match="TensorDict"):
            md.run(_two_atom_system(), 1)

    def test_set_neighbors_returns_self(self):
        pos, cell = make_cubic_lattice()
        neighbors = NeighborList(cell=cell, cutoff=3.5, positions=pos)
        md = MD()
        assert md.set_neighbors(neighbors) is md
        assert md.neighbors is neighbors

    def test_set_forcefield_set_integrator_set_neighbors_arms_rebuild(self):
        from molpy import ForceField as MolpyForceField

        pos, cell = make_cubic_lattice()
        neighbors = NeighborList(cell=cell, cutoff=3.5, positions=pos)
        md = (
            MD()
            .set_forcefield(MolpyForceField())
            .set_integrator(VelocityVerletIntegrator(0.01))
            .set_neighbors(neighbors)
        )
        assert md.integrator is not None
        assert md.integrator.rebuild is True

    def test_rebuild_is_false_without_set_neighbors(self):
        from molpy import ForceField as MolpyForceField

        md = MD().set_forcefield(MolpyForceField()).set_integrator(VelocityVerletIntegrator(0.01))
        assert md.integrator is not None
        assert md.integrator.rebuild is False

    def test_src_molix_md_has_no_molpot_import(self):
        import ast
        from pathlib import Path

        root = Path(__file__).resolve().parents[3] / "src" / "molix" / "md"
        offenders: list[str] = []
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "molpot" or alias.name.startswith("molpot."):
                            offenders.append(f"{path}:{node.lineno}: import {alias.name}")
                elif isinstance(node, ast.ImportFrom) and node.module is not None:
                    if node.module == "molpot" or node.module.startswith("molpot."):
                        offenders.append(f"{path}:{node.lineno}: from {node.module}")
        assert offenders == []


class TestMDNeighborPolicy:
    """The integrator derives the rebuild switch; MD never owns a cadence."""

    def test_the_cadence_kwarg_is_gone(self):
        with pytest.raises(TypeError):
            MD(torch.float64, rebuild_every=1)  # type: ignore[call-arg]

    def test_no_step_start_hook_is_installed(self):
        from molpy import ForceField as MolpyForceField

        md = _nve(MolpyForceField(), dt=0.01)
        assert md.runner.hooks == []

    def test_the_switch_follows_set_neighbors(self):
        from molpy import ForceField as MolpyForceField

        listless = _nve(MolpyForceField(), dt=0.01)
        pos, cell = make_cubic_lattice()
        neighbors = NeighborList(cell=cell, cutoff=3.5, positions=pos)
        live = (
            MD()
            .set_forcefield(MolpyForceField())
            .set_integrator(VelocityVerletIntegrator(0.01))
            .set_neighbors(neighbors)
        )
        assert listless.integrator.rebuild is False
        assert live.integrator.rebuild is True

    def test_rebuild_false_never_ticks_the_list(self):
        from molpy import ForceField as MolpyForceField

        pos, cell = make_cubic_lattice()
        neighbors = NeighborList(cell=cell, cutoff=3.5, positions=pos)
        md = _nve(MolpyForceField(), dt=0.01, rebuild=False)
        md.set_neighbors(neighbors)
        assert md.integrator is not None
        assert md.integrator.rebuild is False
        assert neighbors.rebuild_count == 0
        assert neighbors.ago == 0


class TestMaxwellBoltzmann:
    """Initial-velocity sampling — reads atoms.mass, writes atoms.vel."""

    def test_removes_com_momentum(self):
        """Net momentum must vanish, which is what makes the NVE dof 3N-3."""
        pos = torch.zeros(8, 3, dtype=torch.float64)
        atoms = md_system(pos, torch.zeros_like(pos), torch.full((8,), 2.0))
        vel = MaxwellBoltzmann(300.0, seed=1).velocities(
            atoms["atoms", "pos"], atoms["atoms", "mass"]
        )
        atoms["atoms", "vel"] = vel
        mass = atoms["atoms", "mass"]
        assert vel.shape == (8, 3)
        assert float((mass.reshape(-1, 1).double() * vel).sum(0).abs().max()) < 1e-12

    def test_is_reproducible(self):
        pos = torch.zeros(5, 3)
        a = md_system(pos, torch.zeros_like(pos), 1.0)
        b = md_system(pos.clone(), torch.zeros_like(pos), 1.0)
        mb = MaxwellBoltzmann(300.0, seed=3)
        a["atoms", "vel"] = mb.velocities(a["atoms", "pos"], a["atoms", "mass"])
        b["atoms", "vel"] = mb.velocities(b["atoms", "pos"], b["atoms", "mass"])
        assert torch.equal(a["atoms", "vel"], b["atoms", "vel"])

    def test_mass_length_must_match_atoms(self):
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
            MaxwellBoltzmann(300.0, seed=0).velocities(
                atoms["atoms", "pos"], atoms["atoms", "mass"]
            )

    def test_on_init_hook_writes_state_vel(self):
        """Attaching the hook to MD is what actually initializes the trajectory."""
        atoms = _two_atom_system()

        class _Capture(MDHook):
            vel = None

            def on_step_end(self, runner, step, obs):
                if self.vel is None:
                    self.vel = obs.vel.detach().clone()

        cap = _Capture()
        (
            MD(torch.float64)
            .set_forcefield(_class_i_harmonic())
            .set_integrator(VelocityVerletIntegrator(0.1))
            .set_hooks([VelocityInitHook(MaxwellBoltzmann(300.0, seed=1)), cap])
            .run(atoms, 1)
        )
        assert cap.vel is not None
        assert float(cap.vel.abs().sum()) > 0.0
        assert float((atoms["atoms", "mass"].reshape(-1, 1) * cap.vel).sum(0).abs().max()) < 1e-12
