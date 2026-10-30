"""Tests for molix.md.driver — composer wiring and per-call contracts only."""

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from molix.autograd import Derivative, Potential
from molix.compile import Compiler
from molix.data import collate_molecules
from molix.frame import TensorFrame
from molix.md import (
    MD,
    LangevinVerletIntegrator,
    MaxwellBoltzmann,
    MDHook,
    NeighborList,
    VelocityInitHook,
    VelocityVerletIntegrator,
)
from molix.schema import ENERGY_KEY, FORCES_KEY, POS_KEY, STRAIN_KEY
from tests.conftest import frame
from tests.test_molix.test_md.conftest import make_cubic_lattice, md_system

FORCES = (POS_KEY,)
FORCES_AND_VIRIAL = (POS_KEY, STRAIN_KEY)

#: How close the split-precision run has to sit to a *fully* fp32 evaluation,
#: as a fraction of the fp32-vs-fp64 gap the same fixture shows (``max|ΔF| =
#: 1.7e-4`` kcal mol⁻¹ Å⁻¹, 1.9e-5 relative — 7.1e-8 once converted to MD
#: units, which is why this is a fraction and not an absolute: the ratio is
#: scale-free). A run that evaluated the potential at fp32 lands a long way
#: inside the gap, differing from the reference only by reduction order; one
#: that evaluated at fp64 and cast the answer down at the end lands *at* the
#: gap, and so does one that casts only half the inputs and lets torch promote
#: the rest back to fp64. A quarter of the gap separates the two.
FP32_MATCH_FRACTION = 0.25


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


def _two_atom_system(
    *, r: float = 1.5, dtype: torch.dtype = torch.float64, box: float | None = None
) -> TensorDict:
    """One bonded pair at rest, ``r`` Å apart — a molpy ``Frame`` through the collate.

    ``mass``, a zero ``vel`` and (with *box*) the cubic lattice are the frame's
    own; the batch is built at *dtype*.
    """
    built = frame(
        Z=(1, 1),
        pos=((0.0, 0.0, 0.0), (r, 0.0, 0.0)),
        vel=np.zeros((2, 3)),
        box=None if box is None else np.eye(3) * box,
        bonds=([0], [1]),
    )
    built["atoms"]["mass"] = np.ones(2, dtype=np.float64)
    return collate_molecules([built], dtype=dtype)


def _nve(force, *, dt: float, dtype=None, hooks=None) -> MD:
    return (
        MD(dtype)
        .set_forcefield(force)
        .set_integrator(VelocityVerletIntegrator(dt))
        .set_hooks(hooks)
    )


class _ZeroPots(Potential):
    """The minimal energy the adapter's op composes: ``E = 0 · Σ pos``, zero forces.

    The energy has to hang off the positions — a constant would give the op
    nothing to differentiate — so it is a zero *function* of them.
    """

    def __init__(self) -> None:
        super().__init__()
        self._derivative = Derivative(self.energy)

    def energy(self, work: TensorDict) -> TensorDict:
        pos = work["atoms", "pos"]
        if "graphs" not in work.keys():
            work["graphs"] = TensorDict({}, batch_size=[])
        work[ENERGY_KEY] = (pos * 0.0).sum().reshape(work["graphs"].batch_size)
        return work


class _CellRecordingPots(_ZeroPots):
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
        super().__init__()
        self.calls: list[tuple[int, torch.Tensor]] = []
        self._raise_after = raise_after

    def bind_cell(self, cell: torch.Tensor) -> None:
        self.calls.append((cell.data_ptr(), cell.detach().clone()))
        if self._raise_after is not None and len(self.calls) > self._raise_after:
            raise RuntimeError("KSpaceGrid overflow")


def _stub_forcefield(pots: Potential):
    """A minimal ``MDForceField``: molpy container with a nullary ``to_potentials``."""
    from molpy.ff.forcefield import ForceField as MolpyForceField

    class _StubForceField(MolpyForceField):
        def __new__(cls) -> "_StubForceField":
            # The molrs base is built in __new__ (name, units) and has no __init__.
            return super().__new__(cls, "stub", "metal")

        def to_potentials(self) -> Potential:
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

    def test_set_compiler_compiles_one_composed_op_per_run(self):
        """One ``Derivative`` over the assembled potential per run, handed to the compiler.

        Every ``set_*`` rebuilds the potential (``force.to_potentials()`` runs
        afresh in each setter), so the op is composed at ``run`` over the
        container that is actually about to run — forces alone for NVE, forces
        and virial under a barostat — and the compiler held on ``MD`` is applied
        to exactly that op. The recorder is the identity, so it changes
        nothing; what it records is *what* was compiled.
        """
        from molpot.forcefield import Potentials

        ops = []

        def recording(op):
            ops.append(op)
            return op

        md = (
            MD(torch.float64)
            .set_compiler(recording)
            .set_integrator(VelocityVerletIntegrator(0.01))
            .set_forcefield(_class_i_harmonic())
        )
        assert ops == []
        md.run(_two_atom_system(), 0)
        md.run(_two_atom_system(), 0)
        assert len(ops) == 2
        assert all(isinstance(op, Derivative) for op in ops)
        assert all(op.wrt == FORCES for op in ops)
        assert isinstance(ops[0].energy.__self__, Potentials)
        assert ops[0] is not ops[1]
        assert ops[0].energy.__self__ is ops[1].energy.__self__

        md.set_integrator(VelocityVerletIntegrator(0.02))
        md.run(_two_atom_system(), 0)
        assert len(ops) == 3
        assert ops[2].energy.__self__ is not ops[0].energy.__self__

        md.set_barostat(_expanding_barostat())
        md.run(_two_atom_system(box=10.0), 0)
        assert ops[3].wrt == FORCES_AND_VIRIAL

    def test_an_installed_compiler_changes_no_force(self):
        """The seam is a build switch, not a second physics path.

        ``MDState.forces`` here is ``MD.compute_force``'s own output — the
        integrator's entry evaluation, taken at ``n_steps=0`` so nothing has
        moved and both runs are asked about the same geometry. With the
        identity compiler installed the two builds are the same function, so
        the forces have to agree to the bit-level tolerance of a reduction
        reorder; anything larger means the compiled route is not evaluating the
        field the eager one evaluates.
        """
        eager = _nve(_class_i_harmonic(), dt=0.01, dtype=torch.float64)
        compiled = (
            MD(torch.float64)
            .set_compiler(lambda core: core)
            .set_forcefield(_class_i_harmonic())
            .set_integrator(VelocityVerletIntegrator(0.01))
        )
        reference = eager.run(_two_atom_system(), 0).forces
        assert torch.allclose(
            compiled.run(_two_atom_system(), 0).forces, reference, atol=1e-12, rtol=0.0
        )

    def test_potential_dtype_evaluates_the_potential_at_the_second_precision(self):
        """fp64 trajectory over an fp32 potential — and no cast at all without it.

        The quantization-as-thermal-noise operating point (``molix.quant``):
        the potential's precision is the perturbation under study, so it has to
        be reducible *without* the integrator becoming a second noise source.
        Two things therefore have to hold at once. The state stays fp64 —
        positions, velocities and the forces handed back — while the numbers in
        those forces are the fp32 field's, not the fp64 one's.

        The fp32 reference is built by hand rather than read off the seam: the
        same ForceField with every float parameter table cast, evaluated on a
        cast copy of the same batch. ``.to(dtype)`` on the assembled
        ``Potentials`` reaches its bonded buffers only — the per-atom ε / σ / q
        live in the molpy container — so a field whose *pair* tables are still
        fp64 is exactly the failure this reference can see.

        The tolerance is the fixture's own: fp32 and fp64 disagree by
        ``max|ΔF| = 1.7e-4`` kcal mol⁻¹ Å⁻¹ on this box (1.9e-5 relative), and
        a run that agrees with fp32 to a quarter of that gap
        (:data:`FP32_MATCH_FRACTION`) cannot be an fp64 evaluation with a cast
        bolted on the end. Below it, the default path is pinned the other way — with
        no ``potential_dtype`` there is no cast anywhere, so its forces are
        *bitwise* the fp64 evaluation's.
        """
        from molix.units import preset_energy_to_md
        from tests.conftest import butane_pair_batch, butane_pair_class_i_forcefield

        def forces_at(dtype: torch.dtype) -> torch.Tensor:
            """The whole field and the whole batch at ``dtype``, in MD units."""
            forcefield = butane_pair_class_i_forcefield()
            tables = {
                namespace: {
                    name: value.to(dtype) if value.is_floating_point() else value
                    for name, value in (forcefield.params(namespace) or {}).items()
                }
                for namespace in ("atoms", "bonds", "angles", "propers")
            }
            potentials = forcefield.with_params(**tables).to_potentials().to(dtype)
            batch = butane_pair_batch().apply(
                lambda leaf: leaf.to(dtype) if leaf.is_floating_point() else leaf
            )
            forces = Derivative(potentials.energy, wrt=FORCES)(batch)[FORCES_KEY]
            return forces.double() * preset_energy_to_md(forcefield.units)

        reference32 = forces_at(torch.float32)
        reference64 = forces_at(torch.float64)
        gap = float((reference64 - reference32).abs().max())
        assert gap > 0.0, "the fixture must be able to tell the two precisions apart"

        split = (
            MD(torch.float64, potential_dtype=torch.float32)
            .set_forcefield(butane_pair_class_i_forcefield())
            .set_integrator(VelocityVerletIntegrator(0.5))
        ).run(butane_pair_batch(), 0)
        assert (split.pos.dtype, split.vel.dtype, split.forces.dtype) == (
            torch.float64,
            torch.float64,
            torch.float64,
        )
        assert float((split.forces - reference32).abs().max()) <= FP32_MATCH_FRACTION * gap

        plain = (
            MD(torch.float64)
            .set_forcefield(butane_pair_class_i_forcefield())
            .set_integrator(VelocityVerletIntegrator(0.5))
        ).run(butane_pair_batch(), 0)
        assert torch.equal(plain.forces, reference64)

    @pytest.mark.parametrize("compiler_first", [True, False])
    def test_set_compiler_is_honoured_under_a_barostat(self, compiler_first: bool):
        """The compiled op is the one the NPT force path calls, and it derives the virial.

        An NPT run with a compiler installed used to be an eager run that every
        log line described as compiled — the worst kind of silence, because
        the only visible symptom is a step time nobody compares against a
        baseline. The op is compiled as a whole; this records the compiled
        build being reached on every step, with ``graphs.strain`` among its
        variables.
        """
        calls: list[object] = []

        def recording(core):
            def wrapper(*args, **kwargs):
                calls.append(core)
                return core(*args, **kwargs)

            return wrapper

        # 10 Å cube around a 1.5 Å bond: well inside the F29 half-width, so the
        # only thing under test here is which build evaluates the field.
        system = _two_atom_system(box=10.0)

        md = (
            MD(torch.float64)
            .set_forcefield(_class_i_harmonic())
            .set_integrator(VelocityVerletIntegrator(0.5))
        )
        if compiler_first:
            md.set_compiler(recording).set_barostat(_expanding_barostat())
        else:
            md.set_barostat(_expanding_barostat()).set_compiler(recording)
        md.run(system, 2, chunk=1)
        assert calls
        assert all(isinstance(core, Derivative) for core in calls)
        assert {core.wrt for core in calls} == {FORCES_AND_VIRIAL}

    @pytest.mark.slow
    def test_classical_npt_runs_through_aot_eager_fullgraph(self):
        """A real compiled energy and its AOT backward match eager NPT."""
        from molix.compile import Compiler

        eager = (
            MD(torch.float64)
            .set_forcefield(_class_i_harmonic())
            .set_integrator(VelocityVerletIntegrator(0.5))
            .set_barostat(_expanding_barostat())
        ).run(_two_atom_system(box=10.0), 2, chunk=1)
        state = (
            MD(torch.float64)
            .set_compiler(Compiler(backend="aot_eager", fullgraph=True, dynamic=True))
            .set_forcefield(_class_i_harmonic())
            .set_integrator(VelocityVerletIntegrator(0.5))
            .set_barostat(_expanding_barostat())
        ).run(_two_atom_system(box=10.0), 2, chunk=1)

        assert bool(torch.isfinite(state.pos).all())
        assert bool(torch.isfinite(state.forces).all())
        assert bool(torch.isfinite(state.energy).all())
        assert state.virial is not None
        assert tuple(state.virial.shape) == (3, 3)
        assert eager.cell is not None and eager.virial is not None
        torch.testing.assert_close(state.pos, eager.pos, atol=1e-12, rtol=0.0)
        torch.testing.assert_close(state.forces, eager.forces, atol=1e-12, rtol=0.0)
        torch.testing.assert_close(state.energy, eager.energy, atol=1e-12, rtol=0.0)
        torch.testing.assert_close(state.cell, eager.cell, atol=1e-12, rtol=0.0)
        torch.testing.assert_close(state.virial, eager.virial, atol=1e-12, rtol=0.0)

    def test_potential_dtype_leaves_the_callers_cell_at_the_md_dtype(self):
        """The precision seam is the potential's alone — the caller's lattice is not cast.

        ``potential_dtype`` says at which precision the *field* is evaluated;
        ``dtype`` says at which the trajectory is carried. The adapter casts
        the geometry on the way in, and for positions that cast is a copy — but
        the cell was cast **in place of** the batch's own tensor, so a run left
        the caller's ``graphs.cell`` at fp32 after asking for an fp64
        trajectory. Everything downstream of the run then reads a lattice at
        the potential's precision: the volume in a reported pressure, a
        neighbour list rebuilt against it, a second run over the same system.
        The state's ``cell`` is the same lattice by another name and has to
        agree.
        """
        from tests.conftest import butane_pair_batch, butane_pair_class_i_forcefield

        system = butane_pair_batch()
        assert system["graphs", "cell"].dtype == torch.float64

        state = (
            MD(torch.float64, potential_dtype=torch.float32)
            .set_forcefield(butane_pair_class_i_forcefield())
            .set_integrator(VelocityVerletIntegrator(0.5))
        ).run(system, 1)

        assert system["graphs", "cell"].dtype == torch.float64
        assert state.cell is not None
        assert state.cell.dtype == torch.float64

    def test_run_on_a_typed_view_matches_the_plain_run(self):
        """``MD.run`` on a ``TensorFrame`` is the same trajectory, bit for bit.

        The driver casts and writes back ``atoms.pos`` / ``vel`` / ``mass`` on
        whatever it is handed, and the typed view of the batch (CLAUDE.md
        § TensorDict Contract) holds exactly those leaves. So a view must be
        integrable: the same state out, *and* the caller's own batch advanced,
        since the view owns no storage of its own and every write lands in the
        batch it wraps.
        """
        plain_system = _two_atom_system()
        viewed_system = _two_atom_system()
        view = TensorFrame.from_tensordict(viewed_system)

        plain = _nve(_class_i_harmonic(), dt=0.5).run(plain_system, 4)
        viewed = _nve(_class_i_harmonic(), dt=0.5).run(view, 4)

        assert torch.equal(viewed.pos, plain.pos)
        assert torch.equal(viewed.vel, plain.vel)
        assert torch.equal(viewed.forces, plain.forces)
        assert torch.equal(viewed_system["atoms", "pos"], plain_system["atoms", "pos"])

    def test_run_requires_both_components(self, system):
        with pytest.raises(RuntimeError, match="set_forcefield and set_integrator"):
            MD(torch.float64).run(system, 1)

    def test_set_integrator_takes_a_constructed_integrator(self):
        from molpot.forcefield import ForceField

        force = ForceField()
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

    def test_set_forcefield_accepts_an_mdforcefield_and_returns_self(self):
        from molpot.forcefield import ForceField

        ff = ForceField()
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

    def test_a_bare_molpy_forcefield_is_refused_at_set_forcefield(self):
        """It has no to_potentials (molrs 0.15); refused where it is attached,
        not deep inside the first wire."""
        from molpy.ff.forcefield import ForceField as MolpyForceField

        with pytest.raises(TypeError, match="to_potentials"):
            MD().set_forcefield(MolpyForceField("bare", units="real"))

    def test_set_neighbors_returns_self(self):
        pos, cell = make_cubic_lattice()
        neighbors = NeighborList(cell=cell, cutoff=3.5, positions=pos)
        md = MD()
        assert md.set_neighbors(neighbors) is md
        assert md.neighbors is neighbors

    def test_set_forcefield_set_integrator_set_neighbors_arms_rebuild(self):
        from molpot.forcefield import ForceField

        pos, cell = make_cubic_lattice()
        neighbors = NeighborList(cell=cell, cutoff=3.5, positions=pos)
        md = (
            MD()
            .set_forcefield(ForceField())
            .set_integrator(VelocityVerletIntegrator(0.01))
            .set_neighbors(neighbors)
        )
        assert md.integrator is not None
        assert md.neighbors is neighbors

    def test_rebuild_is_false_without_set_neighbors(self):
        from molpot.forcefield import ForceField

        md = MD().set_forcefield(ForceField()).set_integrator(VelocityVerletIntegrator(0.01))
        assert md.integrator is not None
        assert md.neighbors is None


class TestMDNeighborPolicy:
    """The integrator derives the rebuild switch; MD never owns a cadence."""

    def test_the_cadence_kwarg_is_gone(self):
        with pytest.raises(TypeError):
            MD(torch.float64, rebuild_every=1)  # type: ignore[call-arg]

    def test_no_step_start_hook_is_installed(self):
        from molpot.forcefield import ForceField

        md = _nve(ForceField(), dt=0.01)
        assert md.runner.hooks == []

    def test_the_switch_follows_set_neighbors(self):
        from molpot.forcefield import ForceField

        listless = _nve(ForceField(), dt=0.01)
        pos, cell = make_cubic_lattice()
        neighbors = NeighborList(cell=cell, cutoff=3.5, positions=pos)
        live = (
            MD()
            .set_forcefield(ForceField())
            .set_integrator(VelocityVerletIntegrator(0.01))
            .set_neighbors(neighbors)
        )
        assert listless.neighbors is None
        assert live.neighbors is neighbors

    def test_without_set_neighbors_the_list_is_not_mounted(self):
        from molpot.forcefield import ForceField

        pos, cell = make_cubic_lattice()
        neighbors = NeighborList(cell=cell, cutoff=3.5, positions=pos)
        md = _nve(ForceField(), dt=0.01)
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

    def test_a_non_cellbound_pots_never_dispatches_and_the_op_follows_the_barostat(self):
        """No ``bind_cell`` means no dispatch; the op's variables follow ``want_virial``."""
        from molix.md.driver import _TensorDictForce

        work = _cell_system((torch.eye(3, dtype=torch.float64) * 10.0).unsqueeze(0))

        nve = _TensorDictForce(_ZeroPots(), 1.0)
        nve.bind_work(work)
        assert nve._cell_view is not None
        assert isinstance(nve._op, Derivative)
        assert nve._op.wrt == FORCES

        npt = _TensorDictForce(_ZeroPots(), 1.0)
        npt.want_virial = True
        npt.bind_work(work)
        assert isinstance(npt._op, Derivative)
        assert npt._op.wrt == FORCES_AND_VIRIAL
        out = npt(work[POS_KEY])
        assert tuple(out["virial"].shape) == (3, 3)

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

    def test_rebind_cell_inside_an_evaluation_is_refused(self):
        """The potential-dtype swap window has no correct rebind, so none is allowed.

        *Between* evaluations ``graphs.cell`` is the trajectory's own fp64
        storage, and ``rebind_cell`` copies a barostat update straight into it
        — which is why both in-tree callers sit there. *During* one,
        ``_cast_in`` has the fp32 copy standing under ``graphs.cell`` and the
        real tensor parked on ``_cell_md``, so the same call would resolve the
        copy as the live lattice and write the update into a buffer
        ``_restore_cell`` is about to discard: the barostat's step silently
        lost, the potential left on the pre-barostat cell, and a trajectory
        that keeps integrating.

        No in-tree path reaches the adapter from inside an evaluation, which is
        precisely why the window is worth a test — an unreachable guard is the
        one nobody notices deleting. The re-entry is driven the honest way,
        through a potential that calls back during its energy, rather than by
        setting the flag by hand.
        """
        from molix.md.driver import _TensorDictForce

        holder: list[_TensorDictForce] = []

        class _ReentrantPots(_CellRecordingPots):
            def energy(self, work: TensorDict) -> TensorDict:
                holder[0].rebind_cell(torch.eye(3, dtype=torch.float64) * 4.0)
                return super().energy(work)

        work = _cell_system((torch.eye(3, dtype=torch.float64) * 10.0).unsqueeze(0))
        adapter = _TensorDictForce(
            _ReentrantPots(), 1.0, md_dtype=torch.float64, pot_dtype=torch.float32
        )
        holder.append(adapter)
        adapter.bind_work(work)

        with pytest.raises(AssertionError, match="in-flight"):
            adapter(torch.zeros(2, 3, dtype=torch.float64))

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
    """The composed op over an mlip field, compiled by ``MD`` for the run."""

    @staticmethod
    def _polar_setup(*, cutoff: float, skin: float, **list_options: object):
        """A tiny periodic Polar mlip field, its list and the system to run."""
        from molpot.forcefield import ForceField
        from molzoo.mace.polar import MacePolar
        from molzoo.mace.spec import MACEPolarSpec
        from tests.test_molzoo.test_mace.conftest import (
            ATOMIC_ENERGIES,
            ATOMIC_NUMBERS,
            CLUSTER_POS,
            CLUSTER_Z,
            TINY_POLAR_KWARGS,
        )

        spec = MACEPolarSpec(
            atomic_numbers=ATOMIC_NUMBERS,
            atomic_energies=ATOMIC_ENERGIES,
            **TINY_POLAR_KWARGS,
        )
        torch.manual_seed(0)
        mlip = MacePolar(spec, dtype=torch.float64).eval()
        pos = torch.tensor(CLUSTER_POS, dtype=torch.float64)
        n_atoms = len(CLUSTER_Z)
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        system = md_system(pos.clone(), torch.full((n_atoms, 3), 0.02, dtype=torch.float64), 12.0)
        system["atoms", "Z"] = torch.tensor(CLUSTER_Z, dtype=torch.long)
        system["atoms", "batch"] = torch.zeros(n_atoms, dtype=torch.long)
        system["graphs"] = TensorDict({"cell": cell.unsqueeze(0)}, batch_size=[1])
        forcefield = ForceField("mace-polar", units="metal")
        forcefield.def_pairstyle(mlip)
        neighbors = NeighborList(cell=cell, cutoff=cutoff, positions=pos, skin=skin, **list_options)
        return forcefield, neighbors, system

    @pytest.mark.slow
    def test_a_compiled_mlip_run_matches_eager(self) -> None:
        """``set_compiler`` on an mlip field: one eager warm-up, then the compiled op.

        The warm-up ``bind_work`` performs is what spends the element gate and
        sizes the k-space grids outside the trace; the compiled op then drives
        every step, and the whole two-step trajectory — positions, forces and
        energy at two geometries — must match the eager run to reduction-order
        precision.
        """
        from molix.compile import Compiler

        forcefield, neighbors, system = self._polar_setup(cutoff=3.0, skin=0.5)
        eager = (
            MD(torch.float64)
            .set_forcefield(forcefield)
            .set_neighbors(neighbors)
            .set_integrator(VelocityVerletIntegrator(0.5))
        ).run(system.clone(), 2)

        forcefield, neighbors, system = self._polar_setup(cutoff=3.0, skin=0.5)
        compiled = (
            MD(torch.float64)
            .set_forcefield(forcefield)
            .set_neighbors(neighbors)
            .set_integrator(VelocityVerletIntegrator(0.5))
            .set_compiler(Compiler(backend="aot_eager", dynamic=True))
        ).run(system, 2)

        assert torch.isfinite(compiled.energy).all()
        torch.testing.assert_close(compiled.pos, eager.pos, atol=1e-10, rtol=0.0)
        torch.testing.assert_close(compiled.forces, eager.forces, atol=1e-10, rtol=0.0)
        torch.testing.assert_close(compiled.energy, eager.energy, atol=1e-10, rtol=0.0)

    @pytest.mark.slow
    def test_an_over_capacity_cell_fails_at_kspace_before_the_neighbour_list(self) -> None:
        """MD.run's barostat path names KSpaceGrid, not r_build.

        ``_sync_cell`` rebinds the k-grid *before* it scales the neighbour
        list, so an isotropic 2x expansion of the 10 Å Polar cube
        (``floor(2·2.94) = 5 > N_c = 4``) must raise ``RuntimeError`` naming
        both ``KSpaceGrid`` and ``capacity_factor``. The list is built with
        ``r_build`` well under half the *expanded* perpendicular width, so
        ``scale_isotropic`` would not have raised first.
        """
        from molix.compile import Compiler

        # r_build = 1.0 Å; half-width of the 2x cell is 10 Å, so the list
        # cannot raise ValueError about r_build on this expansion.
        forcefield, neighbors, system = self._polar_setup(
            cutoff=1.0, skin=0.0, every=1, delay=0, check=False
        )
        md = (
            MD(torch.float64)
            .set_forcefield(forcefield)
            .set_neighbors(neighbors)
            .set_integrator(VelocityVerletIntegrator(0.5))
            .set_barostat(_expanding_barostat(2.0))
            .set_compiler(Compiler(backend="aot_eager", dynamic=True))
        )
        with pytest.raises(RuntimeError) as excinfo:
            md.run(system, 1, chunk=1)
        message = str(excinfo.value)
        assert "KSpaceGrid" in message
        assert "capacity_factor" in message
        assert "r_build" not in message
        assert "ValueError" not in message


#: Argon ``lj/cut`` in eV / Å for the static-capacity cases below; the lattice
#: is jittered so no force vanishes by symmetry.
_ARGON_EPSILON_EV = 0.0104
_ARGON_SIGMA = 3.4
_ARGON_CUTOFF = 4.0
_ARGON_SKIN = 0.5
_ARGON_SPACING = 3.8
_ARGON_JITTER = 0.15


def _argon_system() -> tuple[TensorDict, torch.Tensor]:
    """A jittered 27-atom argon cube (11.4 Å) as an MD working batch, plus its cell."""
    pos, cell = make_cubic_lattice(n_side=3, spacing=_ARGON_SPACING)
    generator = torch.Generator().manual_seed(7)
    pos = pos + _ARGON_JITTER * torch.randn(pos.shape, generator=generator, dtype=torch.float64)
    n = int(pos.shape[0])
    system = md_system(pos, torch.zeros_like(pos), 39.948)
    system["atoms", "Z"] = torch.full((n,), 18, dtype=torch.long)
    system["atoms", "batch"] = torch.zeros(n, dtype=torch.long)
    system["graphs"] = TensorDict({"cell": cell.unsqueeze(0)}, batch_size=[1])
    return system, cell


def _argon_forcefield(n_atoms: int):
    from molpot.forcefield import ForceField

    return (
        ForceField("argon", units="metal")
        .def_pairstyle("lj/cut", cutoff=_ARGON_CUTOFF)
        .with_params(
            atoms={
                "epsilon": torch.full((n_atoms,), _ARGON_EPSILON_EV, dtype=torch.float64),
                "sigma": torch.full((n_atoms,), _ARGON_SIGMA, dtype=torch.float64),
            }
        )
    )


def _argon_list(system: TensorDict, cell: torch.Tensor, **options: object) -> NeighborList:
    return NeighborList(
        cell=cell, cutoff=_ARGON_CUTOFF, positions=system[POS_KEY], skin=_ARGON_SKIN, **options
    )


class _EdgeRecordingPots(_ZeroPots):
    """Records the ``pairs`` batch size the op hands the potential on every call."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[torch.Size] = []

    def energy(self, work: TensorDict) -> TensorDict:
        self.seen.append(work["pairs"].batch_size)
        return super().energy(work)


class TestStaticCapacity:
    """The op is handed the list's whole capacity buffer, every step.

    The list pads its dead rows as ``(0, 0)`` self-loops displaced by the
    dead-edge shift and recomputes the geometry over the whole capacity, so
    every op call sees a static ``E = capacity`` — the precondition for
    ``Compiler(cuda_graphs=True)`` on the composed op. There is no live-prefix
    slice on the way to the potential any more (it changed ``E`` on every
    rebuild and forbade CUDA-graph capture).
    """

    def test_the_op_is_handed_the_full_capacity_edges_every_step(self):
        system, cell = _argon_system()
        neighbors = _argon_list(system, cell)
        pots = _EdgeRecordingPots()
        (
            MD(torch.float64)
            .set_forcefield(_stub_forcefield(pots))
            .set_neighbors(neighbors)
            .set_integrator(VelocityVerletIntegrator(0.5))
            .run(system, 2)
        )
        assert neighbors.capacity > neighbors.num_pairs  # there *are* dead rows to hand over
        assert len(pots.seen) >= 3
        assert all(size == torch.Size([neighbors.capacity]) for size in pots.seen)

    def test_capacity_padded_forces_equal_the_live_only_forces(self):
        """Dead rows price exactly nothing: MD forces match a live-prefix evaluation to 1e-12."""
        from molix.units import preset_energy_to_md

        system, cell = _argon_system()
        n_atoms = int(system[POS_KEY].shape[0])
        state = (
            MD(torch.float64)
            .set_forcefield(_argon_forcefield(n_atoms))
            .set_neighbors(_argon_list(system, cell))
            .set_integrator(VelocityVerletIntegrator(0.5))
            .run(system, 0)
        )

        live, _ = _argon_system()
        reference_list = _argon_list(live, cell)
        reference_list.build(live)
        n_live = reference_list.num_pairs
        live["pairs"] = TensorDict(
            {key: column[:n_live] for key, column in live["pairs"].items()}, batch_size=[n_live]
        )
        forces = Derivative(_argon_forcefield(n_atoms).to_potentials().energy, wrt=FORCES)(live)[
            FORCES_KEY
        ]
        expected = forces * preset_energy_to_md("metal")
        assert float(expected.abs().max()) > 0.0
        assert torch.allclose(state.forces, expected, atol=1e-12, rtol=0.0)

    def test_potential_dtype_geometry_comes_from_the_list_and_is_never_swapped(self):
        """Under a ``potential_dtype`` the list writes the geometry at that precision.

        The op is handed the list-bound ``pairs`` namespace itself on every
        call — one object for the whole run, never a per-step cast copy — with
        its float columns at the potential's dtype, while the list's own buffers
        and the trajectory stay at the MD dtype.
        """
        seen: list[tuple[int, torch.dtype, torch.dtype, torch.dtype, torch.dtype]] = []

        def recording(op):
            def compiled(work):
                pairs = work["pairs"]
                seen.append(
                    (
                        id(pairs),
                        pairs["shifts"].dtype,
                        pairs["pair_diff"].dtype,
                        pairs["pair_dist"].dtype,
                        work[POS_KEY].dtype,
                    )
                )
                return op(work)

            return compiled

        system, cell = _argon_system()
        n_atoms = int(system[POS_KEY].shape[0])
        neighbors = _argon_list(system, cell)
        state = (
            MD(torch.float64, potential_dtype=torch.float32)
            .set_forcefield(_argon_forcefield(n_atoms))
            .set_neighbors(neighbors)
            .set_integrator(VelocityVerletIntegrator(0.5))
            .set_compiler(recording)
            .run(system, 2)
        )
        assert len(seen) >= 3
        assert len({entry[0] for entry in seen}) == 1
        assert all(entry[1:] == (torch.float32,) * 4 for entry in seen)
        assert system["pairs", "shifts"].dtype == torch.float32
        assert neighbors.shifts.dtype == torch.float64
        assert state.pos.dtype == torch.float64 and state.forces.dtype == torch.float64

    @pytest.mark.slow
    def test_the_composed_op_traces_to_one_graph_on_the_full_capacity_buffer(self):
        """The compiled region is the whole op over ``E = capacity``; the rebuild sits outside."""
        from molix.compile import Compiler

        ops = []

        def recording(op):
            ops.append(op)
            return op

        system, cell = _argon_system()
        n_atoms = int(system[POS_KEY].shape[0])
        (
            MD(torch.float64)
            .set_forcefield(_argon_forcefield(n_atoms))
            .set_neighbors(_argon_list(system, cell))
            .set_integrator(VelocityVerletIntegrator(0.5))
            .set_compiler(recording)
            .run(system, 0)
        )
        with torch.no_grad():
            explanation = Compiler.explain(ops[-1], system)
        assert (explanation.graph_count, explanation.graph_break_count) == (1, 0), (
            explanation.break_sites
        )


class _GraphCapturingCompiler(Compiler):
    """Stands in for ``Compiler(cuda_graphs=True)``: an identity build that says so.

    Records the force tensor the op wrote on each compiled call, so the test can
    tell whether the adapter left that (graph-pool) tensor on the batch or
    replaced it with its own copy.
    """

    def __init__(self, *, cuda_graphs: bool) -> None:
        self._captures = cuda_graphs
        self.raw_forces: list[torch.Tensor] = []

    @property
    def cuda_graphs(self) -> bool:
        return self._captures

    def __call__(self, op):
        def compiled(work):
            out = op(work)
            self.raw_forces.append(out[FORCES_KEY])
            return out

        return compiled


class TestCudaGraphStepContract:
    """What a CUDA-graph build needs from the caller, applied only when it is one.

    ``reduce-overhead`` replays a recorded graph whose outputs live in a static
    memory pool overwritten by the next replay, and it needs
    ``torch.compiler.cudagraph_mark_step_begin()`` before each replay to know
    the previous outputs are dead. The adapter reads ``compiler.cuda_graphs``
    (``molix.compile.Compiler`` exposes it): under a CUDA-graph build it marks
    the step before the op and clones the outputs it leaves on the batch; under
    any other build it does neither, so the default path is byte-identical to
    having no seam.
    """

    @pytest.mark.parametrize("cuda_graphs", [True, False])
    def test_mark_step_and_cloned_outputs_follow_the_compiler(self, monkeypatch, cuda_graphs: bool):
        marks: list[int] = []
        monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", lambda: marks.append(1))
        compiler = _GraphCapturingCompiler(cuda_graphs=cuda_graphs)
        system = _two_atom_system()
        state = (
            MD(torch.float64)
            .set_compiler(compiler)
            .set_forcefield(_class_i_harmonic())
            .set_integrator(VelocityVerletIntegrator(0.01))
            .run(system, 1)
        )
        assert len(compiler.raw_forces) == 2  # the step-0 evaluation and one step
        raw = compiler.raw_forces[-1]
        if cuda_graphs:
            # one mark per op call: the eager warm-up in bind_work and every compiled call
            assert len(marks) == 3
            assert system[FORCES_KEY] is not raw
            assert torch.equal(system[FORCES_KEY], raw)
        else:
            assert marks == []
            assert system[FORCES_KEY] is raw
        assert torch.isfinite(state.forces).all()
