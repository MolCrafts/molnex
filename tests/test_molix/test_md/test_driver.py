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
    from molpot.forcefield import ForceField

    return (
        ForceField()
        .def_bondstyle("harmonic")
        .with_params(bonds={"k": torch.tensor([k]), "r0": torch.tensor([r0])})
    )


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


def _nve(force, *, dt: float, dtype=None, hooks=None) -> MD:
    return (
        MD(dtype)
        .set_forcefield(force)
        .set_integrator(VelocityVerletIntegrator(dt))
        .set_hooks(hooks)
    )


class _CellRecordingPots:
    """``CellBound`` stub: records every ``bind_cell`` and evaluates to zero forces.

    Each record is ``(data_ptr, cell.clone())`` — the pointer proves the pots
    was handed the *live* storage MD copies into, and the clone freezes the
    value at call time, which a shared view would otherwise overwrite.

    Args:
        raise_after: Number of successful calls before ``bind_cell`` raises
            ``RuntimeError`` (the capacity-overflow contract). ``None`` never
            raises.
    """

    def __init__(self, *, raise_after: int | None = None) -> None:
        self.calls: list[tuple[int, torch.Tensor]] = []
        self._raise_after = raise_after

    def bind_cell(self, cell: torch.Tensor) -> None:
        self.calls.append((cell.data_ptr(), cell.detach().clone()))
        if self._raise_after is not None and len(self.calls) > self._raise_after:
            raise RuntimeError("KSpaceGrid overflow")

    def calc_energy_forces(self, work: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        pos = work["atoms", "pos"]
        return torch.zeros((), dtype=pos.dtype), torch.zeros_like(pos)

    def calc_energy_forces_virial(
        self, work: TensorDict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pos = work["atoms", "pos"]
        return (
            torch.zeros((), dtype=pos.dtype),
            torch.zeros_like(pos),
            torch.zeros(3, 3, dtype=pos.dtype),
        )


def _stub_forcefield(pots: object):
    """A minimal ``MDForceField``: molpy container with a nullary ``to_potentials``."""
    from molpy import ForceField as MolpyForceField

    class _StubForceField(MolpyForceField):
        def __new__(cls) -> "_StubForceField":
            # molrs __new__ only accepts (name, units).
            return super().__new__(cls, "stub", "metal")

        def __init__(self) -> None:
            super().__init__("stub", "metal")

        def to_potentials(self) -> object:
            return pots

    return _StubForceField()


def _expanding_barostat(factor: float = 1.01):
    """A ``BerendsenBarostat`` whose scale is a fixed expansion — no pressure physics."""
    from molix.md import BerendsenBarostat
    from molix.md.types import MDState

    class _ExpandingBarostat(BerendsenBarostat):
        def scale(self, state: MDState, *, mass: object, dt: float) -> MDState:
            return state._replace(cell=state.cell * factor)

    return _ExpandingBarostat(pressure=1.0, tau=100.0)


def _cell_system(cell: torch.Tensor) -> TensorDict:
    """Two static atoms plus ``graphs.cell`` in whichever layout is handed in."""
    pos = torch.zeros(2, 3, dtype=torch.float64)
    system = md_system(pos, torch.zeros_like(pos), 1.0)
    system["graphs"] = TensorDict({"cell": cell}, batch_size=[1] if cell.dim() == 3 else [])
    return system


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
        from molpot.forcefield import ForceField

        ff = (
            ForceField()
            .def_atomstyle("full")
            .def_bondstyle("harmonic")
            .def_anglestyle("harmonic")
            .def_dihedralstyle("fourier")
        )
        md = MD()
        assert md.set_forcefield(ff) is md
        assert md.force is ff

    def test_set_forcefield_accepts_opls_forcefield_and_returns_self(self):
        from molpot.forcefield import OPLSAAForceField

        ff = OPLSAAForceField()
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
        assert md.neighbors is neighbors

    def test_rebuild_is_false_without_set_neighbors(self):
        from molpy import ForceField as MolpyForceField

        md = MD().set_forcefield(MolpyForceField()).set_integrator(VelocityVerletIntegrator(0.01))
        assert md.integrator is not None
        assert md.neighbors is None


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
        assert listless.neighbors is None
        assert live.neighbors is neighbors

    def test_without_set_neighbors_the_list_is_not_mounted(self):
        from molpy import ForceField as MolpyForceField

        pos, cell = make_cubic_lattice()
        neighbors = NeighborList(cell=cell, cutoff=3.5, positions=pos)
        md = _nve(MolpyForceField(), dt=0.01)
        assert md.neighbors is None
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


class TestRebindCell:
    """``_TensorDictForce.rebind_cell`` — the one owner of cell resolution on the pots path."""

    def test_bind_work_dispatches_once_on_the_live_cell_storage(self):
        """In-place ``copy_`` only reaches the pots if the view shares that storage.

        Both stored layouts are checked here: ``(1, 3, 3)`` (what ``graphs.cell``
        actually carries) is sliced to ``(3, 3)`` by the caller, ``(3, 3)`` is
        passed through.
        """
        from molix.md.driver import _TensorDictForce

        batched = _cell_system((torch.eye(3, dtype=torch.float64) * 10.0).unsqueeze(0))
        batched_pots = _CellRecordingPots()
        _TensorDictForce(batched_pots, 1.0).bind_work(batched)
        assert len(batched_pots.calls) == 1
        ptr, cell = batched_pots.calls[0]
        assert tuple(cell.shape) == (3, 3)
        assert ptr == batched["graphs", "cell"][0].data_ptr()

        flat = _cell_system(torch.eye(3, dtype=torch.float64) * 8.0)
        flat_pots = _CellRecordingPots()
        _TensorDictForce(flat_pots, 1.0).bind_work(flat)
        assert len(flat_pots.calls) == 1
        assert flat_pots.calls[0][0] == flat["graphs", "cell"].data_ptr()

    def test_rebind_cell_re_reads_a_replaced_graphs_cell(self):
        """The virial strain closure swaps this tensor; a cached view copies nowhere."""
        from molix.md.driver import _TensorDictForce

        work = _cell_system((torch.eye(3, dtype=torch.float64) * 10.0).unsqueeze(0))
        pots = _CellRecordingPots()
        adapter = _TensorDictForce(pots, 1.0)
        adapter.bind_work(work)

        replacement = (torch.eye(3, dtype=torch.float64) * 4.0).unsqueeze(0)
        work["graphs", "cell"] = replacement
        new_cell = torch.eye(3, dtype=torch.float64) * 9.0
        adapter.rebind_cell(new_cell)

        assert work["graphs", "cell"] is replacement
        assert torch.allclose(replacement[0], new_cell)
        assert len(pots.calls) == 2
        assert pots.calls[1][0] == replacement[0].data_ptr()

    def test_a_non_cellbound_pots_never_dispatches_and_keeps_the_eval_resolution(self):
        """Classical fields have no ``bind_cell``; the dispatch step is a silent no-op."""
        from molix.md.driver import _TensorDictForce

        work = _cell_system((torch.eye(3, dtype=torch.float64) * 10.0).unsqueeze(0))

        classical = _TensorDictForce(object(), 1.0)
        classical.bind_work(work)
        assert classical._cell_view is not None
        assert classical._eval.__func__ is _TensorDictForce._eval_classical

        class _Mlip:
            mlip = object()

        mlip = _TensorDictForce(_Mlip(), 1.0)
        mlip.bind_work(work)
        assert mlip._eval.__func__ is _TensorDictForce._eval_mlip

        npt = _TensorDictForce(object(), 1.0)
        npt.want_virial = True
        npt.bind_work(work)
        assert npt._eval.__func__ is _TensorDictForce._eval_npt

    def test_a_missing_graphs_cell_is_a_no_op_on_both_call_paths(self):
        """No cell means no view — never hand ``None`` to a ``Tensor`` signature."""
        from molix.md.driver import _TensorDictForce

        pos = torch.zeros(2, 3, dtype=torch.float64)
        work = md_system(pos, torch.zeros_like(pos), 1.0)
        pots = _CellRecordingPots()
        adapter = _TensorDictForce(pots, 1.0)
        adapter.bind_work(work)
        adapter.rebind_cell()
        adapter.rebind_cell(torch.eye(3, dtype=torch.float64))
        assert adapter._cell_view is None
        assert pots.calls == []

    def test_bind_cell_fires_once_per_barostat_update_not_once_per_step(self):
        """``MDRunner`` syncs per chunk (runner.py:208-211) — 1 + one per chunk.

        Under NVT there is no barostat and no ``sync_state``, so the single
        ``bind_work`` call is the only one, whatever the step count.
        """
        cell = (torch.eye(3, dtype=torch.float64) * 10.0).unsqueeze(0)

        for n_steps, chunk in ((6, 2), (3, 1)):
            system = _cell_system(cell.clone())
            pots = _CellRecordingPots()
            md = (
                MD(torch.float64)
                .set_forcefield(_stub_forcefield(pots))
                .set_integrator(VelocityVerletIntegrator(0.5))
                .set_barostat(_expanding_barostat())
            )
            md.run(system, n_steps, chunk=chunk)
            live_ptr = system["graphs", "cell"][0].data_ptr()
            assert [ptr for ptr, _ in pots.calls] == [live_ptr] * 4
            expected = cell[0].clone()
            assert torch.allclose(pots.calls[0][1], expected, atol=1e-12, rtol=0.0)
            for _, recorded in pots.calls[1:]:
                expected = expected * 1.01
                assert torch.allclose(recorded, expected, atol=1e-12, rtol=0.0)

        nvt_system = _cell_system(cell.clone())
        nvt_pots = _CellRecordingPots()
        (
            MD(torch.float64)
            .set_forcefield(_stub_forcefield(nvt_pots))
            .set_integrator(VelocityVerletIntegrator(0.5))
        ).run(nvt_system, 6, chunk=2)
        assert len(nvt_pots.calls) == 1

    def test_a_bind_cell_runtimeerror_propagates_out_of_md_run(self):
        """Capacity overflow is refused loudly on both binding paths — never truncated."""
        cell = (torch.eye(3, dtype=torch.float64) * 10.0).unsqueeze(0)

        at_bind_work = _CellRecordingPots(raise_after=0)
        with pytest.raises(RuntimeError, match="KSpaceGrid overflow"):
            (
                MD(torch.float64)
                .set_forcefield(_stub_forcefield(at_bind_work))
                .set_integrator(VelocityVerletIntegrator(0.5))
                .set_barostat(_expanding_barostat())
            ).run(_cell_system(cell.clone()), 2, chunk=1)
        assert len(at_bind_work.calls) == 1

        at_sync_cell = _CellRecordingPots(raise_after=1)
        with pytest.raises(RuntimeError, match="KSpaceGrid overflow"):
            (
                MD(torch.float64)
                .set_forcefield(_stub_forcefield(at_sync_cell))
                .set_integrator(VelocityVerletIntegrator(0.5))
                .set_barostat(_expanding_barostat())
            ).run(_cell_system(cell.clone()), 2, chunk=1)
        assert len(at_sync_cell.calls) == 2


class TestCompiledMlipStep:
    """A compiled mlip core across ``_wire``'s rebuilds and the per-step energy."""

    @pytest.fixture
    def fp64(self):
        """Build the Polar mlip at fp64, then put the global precision back.

        ``molix.config`` is a process-wide singleton and cuEquivariance freezes
        its working precision at **construction**, so this test has to raise the
        global default rather than ``.double()`` the model afterwards. Leaving it
        raised would be a cross-file side effect: every module built later in the
        same session would be fp64 while its fixtures still feed fp32 tensors.
        Same save/restore as ``tests/test_molzoo/test_mace/conftest.py``.
        """
        from molix import config

        previous = config["ftype"]
        config.set_precision("fp64")
        yield
        config.set_precision("fp64" if previous == torch.float64 else "fp32")

    @pytest.mark.slow
    def test_compiled_core_survives_rebuilds_and_updates_energy_per_step(self, fp64) -> None:
        """Every ``set_*`` rebuilds ``Potentials``; the compiled core must not care.

        ``MD._wire`` calls ``force.to_potentials()`` afresh in each setter, so a
        core installed on the *container* would be silently discarded by the
        next one and the run would quietly fall back to eager. Installing it on
        the mlip — the single ``nn.Module`` the ForceField holds — is what makes
        the prescribed ``bind_cell -> one eager call -> compile_core`` sequence
        survive being handed to ``MD`` afterwards.

        The second assertion is that the step is live: ``_eval_mlip`` reads
        ``graphs.energy`` back off the work batch, so a stale value from the
        warm-up would integrate a frozen potential without any error. Two steps
        at two different geometries must give two different energies.
        """
        from molix.compile import Compiler
        from molpot.forcefield import ForceField
        from molzoo.mace.polar import MacePolar
        from molzoo.mace.spec import MACEPolarSpec
        from tests.test_molzoo.test_mace.conftest import (
            ATOMIC_ENERGIES,
            ATOMIC_NUMBERS,
            CLUSTER_POS,
            CLUSTER_Z,
            TINY_POLAR_KWARGS,
            polar_kwargs,
        )

        spec = MACEPolarSpec(
            atomic_numbers=ATOMIC_NUMBERS,
            atomic_energies=ATOMIC_ENERGIES,
            **TINY_POLAR_KWARGS,
        )
        torch.manual_seed(0)
        mlip = MacePolar(**polar_kwargs(spec, compute_forces=False)).eval()

        pos = torch.tensor(CLUSTER_POS, dtype=torch.float64)
        n_atoms = len(CLUSTER_Z)
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        system = md_system(pos.clone(), torch.full((n_atoms, 3), 0.02, dtype=torch.float64), 12.0)
        system["atoms", "Z"] = torch.tensor(CLUSTER_Z, dtype=torch.long)
        system["atoms", "batch"] = torch.zeros(n_atoms, dtype=torch.long)
        system["graphs"] = TensorDict({"cell": cell.unsqueeze(0)}, batch_size=[1])

        forcefield = ForceField("mace-polar", units="metal")
        forcefield.def_pairstyle(mlip)
        neighbors = NeighborList(cell=cell, cutoff=3.0, positions=pos, skin=0.5)
        pots = forcefield.to_potentials()
        pots.bind_cell(cell)
        warm_up = system.clone()
        neighbors.build(warm_up)
        with torch.no_grad():
            pots.calc_energy(warm_up)
            pots.compile_core(Compiler(backend="aot_eager", dynamic=True))

        energies: list[float] = []
        installed = mlip._core

        def counting(*args: object, **kwargs: object) -> torch.Tensor:
            energy = installed(*args, **kwargs)
            energies.append(float(energy.detach().sum()))
            return energy

        mlip._core = counting

        md = (
            MD(torch.float64)
            .set_forcefield(forcefield)
            .set_neighbors(neighbors)
            .set_integrator(VelocityVerletIntegrator(0.5))
        )
        state = md.run(system, 2)

        assert len(energies) >= 2
        assert energies[0] != energies[-1]
        assert torch.isfinite(state.energy).all()

    @pytest.mark.slow
    def test_an_over_capacity_cell_fails_at_kspace_before_the_neighbour_list(self, fp64) -> None:
        """ac-012: MD.run's barostat path names KSpaceGrid, not r_build.

        ``_sync_cell`` rebinds the k-grid *before* it scales the neighbour
        list, so an isotropic 2x expansion of the 10 Å Polar cube
        (``floor(2·2.94) = 5 > N_c = 4``) must raise ``RuntimeError`` naming
        both ``KSpaceGrid`` and ``capacity_factor``. The list is built with
        ``r_build`` well under half the *expanded* perpendicular width, so
        ``scale_isotropic`` would not have raised first.
        """
        from molix.compile import Compiler
        from molpot.forcefield import ForceField
        from molzoo.mace.polar import MacePolar
        from molzoo.mace.spec import MACEPolarSpec
        from tests.test_molzoo.test_mace.conftest import (
            ATOMIC_ENERGIES,
            ATOMIC_NUMBERS,
            CLUSTER_POS,
            CLUSTER_Z,
            TINY_POLAR_KWARGS,
            polar_kwargs,
        )

        spec = MACEPolarSpec(
            atomic_numbers=ATOMIC_NUMBERS,
            atomic_energies=ATOMIC_ENERGIES,
            **TINY_POLAR_KWARGS,
        )
        torch.manual_seed(0)
        mlip = MacePolar(**polar_kwargs(spec, compute_forces=False)).eval()
        pos = torch.tensor(CLUSTER_POS, dtype=torch.float64)
        n_atoms = len(CLUSTER_Z)
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        system = md_system(pos.clone(), torch.zeros(n_atoms, 3, dtype=torch.float64), 12.0)
        system["atoms", "Z"] = torch.tensor(CLUSTER_Z, dtype=torch.long)
        system["atoms", "batch"] = torch.zeros(n_atoms, dtype=torch.long)
        system["graphs"] = TensorDict({"cell": cell.unsqueeze(0)}, batch_size=[1])

        forcefield = ForceField("mace-polar", units="metal")
        forcefield.def_pairstyle(mlip)
        # r_build = 1.0 Å; half-width of the 2x cell is 10 Å, so the list
        # cannot raise ValueError about r_build on this expansion.
        neighbors = NeighborList(
            cell=cell, cutoff=1.0, positions=pos, skin=0.0, every=1, delay=0, check=False
        )
        pots = forcefield.to_potentials()
        pots.bind_cell(cell)
        warm_up = system.clone()
        neighbors.build(warm_up)
        with torch.no_grad():
            pots.calc_energy(warm_up)
            pots.compile_core(Compiler(backend="aot_eager", dynamic=True))

        md = (
            MD(torch.float64)
            .set_forcefield(forcefield)
            .set_neighbors(neighbors)
            .set_integrator(VelocityVerletIntegrator(0.5))
            .set_barostat(_expanding_barostat(2.0))
        )
        with pytest.raises(RuntimeError) as excinfo:
            md.run(system, 1, chunk=1)
        message = str(excinfo.value)
        assert "KSpaceGrid" in message
        assert "capacity_factor" in message
        assert "r_build" not in message
        assert "ValueError" not in message
