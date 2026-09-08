"""Tests for the hook-driven MD runner (molix.md.runner).

Single-function correctness only. ``MDRunner.run`` is exercised with an
analytic harmonic well just far enough to observe its own contract (hook
dispatch order, step accounting, typed return); every hook class is driven
**directly** with synthetic :class:`MDObservables` / :class:`MDState` — no
trajectory loop stands between a hook test and the method it tests.

:class:`EnergyDriftHook` is the one exception, and only in its ``slow`` arm:
the NVE conservation criterion it owns is a statement about a *trajectory*, so
its release gate runs one real rollout. Every other property of the hook —
the least-squares slope, the constant-series floor, the refusal below two
samples, the timestep it reads — is pinned on the primitive with synthetic
series, where the arithmetic has an exact answer.
"""

import math

import pytest
import torch

from molix.md import (
    MD,
    EnergyDrift,
    EnergyDriftHook,
    LangevinVerletIntegrator,
    MDCheckpointHook,
    MDHook,
    MDObservables,
    MDRunner,
    MDState,
    NeighborList,
    TrajectoryHook,
    VelocityVerletIntegrator,
)
from tests.conftest import (
    BUTANE_PAIR_BOX,
    BUTANE_PAIR_LJ_CUTOFF,
    BUTANE_PAIR_LR_WAVELENGTH_MD,
    butane_pair_batch,
    butane_pair_class_i_forcefield,
    butane_pair_positions,
)
from tests.test_molix.test_md.conftest import harmonic_potential_like, md_system

_DTYPE = torch.float64


def _integrator(seed: int = 0):
    return LangevinVerletIntegrator(0.01, gamma=1.0, kbt=1.0, seed=seed).bind_force(
        harmonic_potential_like(1.0)
    )


def _sys(n: int = 4):
    pos = torch.zeros(n, 3, dtype=_DTYPE)
    return md_system(pos, torch.zeros_like(pos), 1.0)


def _obs(n_atoms: int = 4, fill: float = 1.0) -> MDObservables:
    """A synthetic observation — hooks only read fields, so values are free."""
    return MDObservables(
        pos=torch.full((n_atoms, 3), fill, dtype=_DTYPE),
        vel=torch.full((n_atoms, 3), 2.0 * fill, dtype=_DTYPE),
        forces=torch.full((n_atoms, 3), -fill, dtype=_DTYPE),
        potential=torch.tensor(fill, dtype=_DTYPE),
        kinetic=torch.tensor(2.0 * fill, dtype=_DTYPE),
        total=torch.tensor(3.0 * fill, dtype=_DTYPE),
        temperature=torch.tensor(300.0, dtype=_DTYPE),
    )


class _RecordingHook(MDHook):
    """Records lifecycle call order and the per-step observables."""

    def __init__(self):
        self.events: list[str] = []
        self.steps: list[int] = []
        self.observed: list = []

    def on_run_start(self, runner):
        self.events.append("start")

    def on_step_end(self, runner, step, obs):
        self.events.append("step")
        self.steps.append(step)
        self.observed.append(obs)

    def on_run_end(self, runner):
        self.events.append("end")


class TestMDRunner:
    def test_fires_lifecycle_and_advances_step(self):
        rec = _RecordingHook()
        runner = MDRunner(_integrator(), hooks=[rec])
        out = runner.run(_sys(4), 3)

        assert rec.events == ["start", "step", "step", "step", "end"]
        assert rec.steps == [1, 2, 3]
        assert isinstance(out, MDState)
        assert out.pos.shape == (4, 3)
        assert out.forces.shape == (4, 3)

    def test_observables_are_typed_and_consistent(self):
        rec = _RecordingHook()
        runner = MDRunner(_integrator(), hooks=[rec])
        runner.run(
            md_system(torch.randn(4, 3, dtype=_DTYPE), torch.randn(4, 3, dtype=_DTYPE), 1.0),
            2,
        )

        last = rec.observed[-1]
        assert torch.allclose(last.total, last.potential + last.kinetic)
        assert last.forces.shape == (4, 3)

    def test_dof_follows_the_integrator_convention(self):
        """The DoF convention lives on ``Integrator.removed_dof`` — the runner
        must not duck-type thermostat internals (the old ``getattr(integrator,
        "gamma")`` sniffing silently mis-reported temperature for custom
        integrators)."""
        assert _integrator().removed_dof == 0  # gamma > 0
        nve = VelocityVerletIntegrator(0.01).bind_force(harmonic_potential_like(1.0))
        assert nve.removed_dof == 3

    def test_step_start_precedes_each_advance(self):
        """``on_step_start`` fires before the chunk it precedes — the ordering
        a hook needs to observe or stage per-step state ahead of the force
        evaluations inside the advance. (Neighbour rebuilds are **not** such a
        user: the list's policy runs inside ``Integrator.eval_force``, at the
        positions being evaluated.)"""
        events: list[tuple[str, int]] = []

        class _Interleaved(MDHook):
            def on_step_start(self, runner, step, state):
                events.append(("start", step))

            def on_step_end(self, runner, step, obs):
                events.append(("end", step))

        runner = MDRunner(_integrator(), hooks=[_Interleaved()])
        runner.run(_sys(2), 4, chunk=2)
        assert events == [("start", 0), ("end", 2), ("start", 2), ("end", 4)]

    def test_misaligned_hook_cadence_is_rejected(self):
        """A declared cadence that chunking would silently skip must raise."""
        import pytest

        hook = TrajectoryHook("/tmp/_chunk_misalign.pt", stride=3, write_xyz=False)
        runner = MDRunner(_integrator(), hooks=[hook])
        with pytest.raises(ValueError, match="multiple of chunk"):
            runner.run(_sys(2), 6, chunk=2)

    def test_hook_priority_orders_firing(self):
        """Lower priority fires earlier; ties keep registration order."""
        order: list[str] = []

        class _Tagged(MDHook):
            def __init__(self, tag):
                self.tag = tag

            def on_run_start(self, runner):
                order.append(self.tag)

        runner = MDRunner(
            _integrator(),
            hooks=[(_Tagged("late"), 100), (_Tagged("first"), 0), _Tagged("default")],
        )
        runner.run(_sys(2), 1)
        assert order == ["first", "late", "default"]


class TestTrajectoryHook:
    """Driven directly: synthetic observables in, files out."""

    @pytest.mark.parametrize("dtype", [None, torch.float32, torch.float64])
    def test_storage_precision_is_explicit(self, tmp_path, dtype):
        out = tmp_path / "precision.pt"
        obs = _obs(fill=1.0 + 1e-10)
        hook = TrajectoryHook(out, dtype=dtype, write_xyz=False)
        hook.on_step_end(None, 1, obs)
        hook.on_run_end(None)
        saved = torch.load(out, weights_only=True)
        for key, source in (("pos", obs.pos), ("vel", obs.vel), ("forces", obs.forces)):
            expected = source.to(dtype=dtype or source.dtype)
            torch.testing.assert_close(saved[key][0], expected, atol=0, rtol=0)
        torch.testing.assert_close(saved["etot"][0], obs.total, atol=0, rtol=0)

    def test_cpu_frames_do_not_alias_live_state(self, tmp_path):
        out = tmp_path / "snapshot.pt"
        obs = _obs()
        hook = TrajectoryHook(out, write_xyz=False)
        hook.on_step_end(None, 1, obs)
        for field in (obs.pos, obs.vel, obs.forces, obs.total, obs.temperature):
            field.add_(10.0)
        hook.on_run_end(None)
        saved = torch.load(out, weights_only=True)
        assert saved["pos"][0, 0, 0].item() == 1.0
        assert saved["vel"][0, 0, 0].item() == 2.0
        assert saved["forces"][0, 0, 0].item() == -1.0
        assert saved["etot"][0].item() == 3.0
        assert saved["temp"][0].item() == 300.0

    def test_rejects_half_precision_storage(self, tmp_path):
        with pytest.raises(ValueError, match="trajectory dtype"):
            TrajectoryHook(tmp_path / "bad.pt", dtype=torch.float16)

    def _drive(self, hook: TrajectoryHook, n_steps: int) -> None:
        for step in range(1, n_steps + 1):
            hook.on_step_end(None, step, _obs(fill=float(step)))
        hook.on_run_end(None)

    def test_persists_pt_and_xyz(self, tmp_path):
        numbers = torch.tensor([1, 6, 8, 1])
        out_pt = tmp_path / "traj.pt"
        self._drive(TrajectoryHook(out_pt, stride=2, numbers=numbers, write_xyz=True), 10)

        payload = torch.load(out_pt, weights_only=False)
        # stride=2 over steps 1..10: kept at 2,4,6,8,10 -> 5 frames.
        assert payload["pos"].shape == (5, 4, 3)
        assert payload["forces"].shape == (5, 4, 3)
        assert payload["temp"].shape == (5,)
        assert payload["stride"] == 2
        assert torch.equal(payload["Z"], numbers)
        assert float(payload["pos"][0, 0, 0]) == 2.0  # first kept step is 2

        lines = out_pt.with_suffix(".xyz").read_text().splitlines()
        # Each extended-XYZ frame: 1 count line + 1 comment + N atom lines.
        assert lines[0] == "4"
        assert "energy=" in lines[1] and "temperature=" in lines[1]
        assert lines[2].split()[0] == "H"  # Z=1 -> H

    def test_shard_flush_matches_single_buffer(self, tmp_path):
        """Shard-flushing (bounded host memory) yields the same file as one
        in-memory buffer, and cleans up its shard files."""
        big = tmp_path / "big.pt"  # one buffer (flush_every >> frames)
        self._drive(TrajectoryHook(big, flush_every=1000), 10)
        small = tmp_path / "small.pt"  # flush every 2 frames -> 5 shards
        self._drive(TrajectoryHook(small, flush_every=2), 10)

        a = torch.load(big, weights_only=True)
        b = torch.load(small, weights_only=True)
        assert b["pos"].shape == (10, 4, 3)
        for key in ("pos", "vel", "forces", "pe", "temp"):
            assert torch.equal(a[key], b[key]), key
        assert not list(tmp_path.glob("small.part*.pt"))  # shards removed

    def test_without_numbers_skips_xyz(self, tmp_path):
        out_pt = tmp_path / "traj.pt"
        self._drive(TrajectoryHook(out_pt, stride=1, write_xyz=True), 4)  # no numbers
        assert out_pt.exists()
        assert not out_pt.with_suffix(".xyz").exists()

    def test_declares_its_stride_as_cadence(self):
        assert TrajectoryHook("x.pt", stride=3).cadence == 3


#: Synthetic drift series of :class:`TestEnergyDriftHook`: ``E(t) = E0 + a t``
#: sampled every ``dt`` fs for ``n`` samples, i.e. over ``(n - 1) dt`` fs.
_DRIFT_E0 = -12.5
_DRIFT_SLOPE = 3e-4
_DRIFT_DT_FS = 0.5
_DRIFT_SAMPLES = 200
#: The least-squares fit of an exactly linear series is exact in fp64; ``1e-12``
#: is round-off on the centred sums, not a fitting margin.
_DRIFT_EXACT = 1e-12

#: The NVE gate: 200 steps of velocity-Verlet at 0.5 fs on the OPLS butane pair
#: in its 20 Å periodic box, against the criterion
#: ``benchmarks/verify_md_ljcut_nve.py`` declares.
_NVE_STEPS = 200
_NVE_DT_FS = 0.5
#: Neighbour-list skin (Å). ``r_build = cutoff + skin = 9.0`` stays under the
#: minimum-image limit ``L/2 = 10``, which the list refuses to cross.
_NVE_SKIN = 1.0
_NVE_DRIFT_BOUND = 1e-3


class TestEnergyDriftHook:
    """Driven directly: synthetic ``(t, E_tot)`` series in, one drift record out."""

    def test_recovers_a_synthetic_linear_slope_exactly(self) -> None:
        """A series that drifts linearly is reported with its own slope.

        The whole point of the hook is that ``rel_drift`` is a *fitted* number,
        not ``|E_last − E_first| / |E_first|``: a trajectory's total energy
        oscillates at the integrator's own frequency, and an endpoint
        difference reads that oscillation as drift. Feeding the primitive
        ``E(t) = E₀ + a t`` — the one series whose least-squares slope has a
        closed form — pins that the fit is a fit and that ``rel_drift`` is
        ``|a T| / |E₀|`` with ``T`` the sampled duration, both to fp64.
        """
        hook = EnergyDriftHook()
        for sample in range(_DRIFT_SAMPLES):
            time_fs = sample * _DRIFT_DT_FS
            hook.record(time_fs, _DRIFT_E0 + _DRIFT_SLOPE * time_fs)

        result: EnergyDrift = hook.drift()
        duration = (_DRIFT_SAMPLES - 1) * _DRIFT_DT_FS
        assert result.slope == pytest.approx(_DRIFT_SLOPE, abs=_DRIFT_EXACT, rel=0.0)
        assert result.rel_drift == pytest.approx(
            abs(_DRIFT_SLOPE * duration) / abs(_DRIFT_E0), abs=_DRIFT_EXACT, rel=0.0
        )

    def test_constant_series_has_zero_fluctuation(self) -> None:
        """A perfectly conserved series reports no drift and no fluctuation.

        ``rms_rel`` is the second half of the criterion — the bounded
        oscillation a symplectic integrator carries around the conserved value
        — and it has to read zero when there is nothing to bound. Exactly
        zero, not "small": every deviation from the mean is identically ``0``,
        so any non-zero answer here is a bug in the reduction rather than
        round-off.
        """
        hook = EnergyDriftHook()
        for sample in range(_DRIFT_SAMPLES):
            hook.record(sample * _DRIFT_DT_FS, _DRIFT_E0)

        result = hook.drift()
        assert result.rms_rel == 0.0
        assert result.rel_drift == 0.0

    def test_fewer_than_two_samples_is_refused(self) -> None:
        """One sample is not a slope — the hook raises rather than inventing one.

        A single observation makes the least-squares denominator
        ``Σ (t − t̄)²`` exactly zero, and dividing by it would hand back a
        ``nan`` (or worse, a ``0/0`` that some reduction turns into ``0.0``)
        that reads as "no drift". A run too short to have a slope has to say
        so.
        """
        hook = EnergyDriftHook()
        hook.record(0.0, _DRIFT_E0)
        with pytest.raises(ValueError):
            hook.drift()

    def test_on_step_end_reads_the_integrator_timestep(self) -> None:
        """The time axis comes from the runner's integrator, not from the step count.

        ``on_step_end`` is handed a step *index*; the drift criterion is per
        unit time, so the hook has to convert with the timestep the run is
        actually using. Reading it off ``runner.integrator.dt`` is what keeps
        the reported duration in femtoseconds when the same 200 steps are run
        at another ``dt`` — a hook that recorded the raw step index would
        report the same drift for a 0.5 fs run and a 2 fs one.
        """
        runner = MDRunner(VelocityVerletIntegrator(_DRIFT_DT_FS))
        hook = EnergyDriftHook()
        n_calls = 5
        hook.on_run_start(runner)
        for step in range(1, n_calls + 1):
            hook.on_step_end(runner, step, _obs(fill=float(step)))
        hook.on_run_end(runner)

        result = hook.result
        assert result is not None, "on_run_end must publish the drift record"
        assert result.duration_fs == pytest.approx(
            _DRIFT_DT_FS * (n_calls - 1), abs=_DRIFT_EXACT, rel=0.0
        )

    def test_a_run_too_short_to_fit_leaves_result_none(self) -> None:
        """A one-observation run ends quietly with no record, and does not raise.

        The refusal of :meth:`EnergyDriftHook.drift` is for a *caller* who asks
        for a slope that does not exist. ``on_run_end`` is not that caller: it
        is a lifecycle callback the runner fires unconditionally, so raising
        there turns "this run was too short to measure" into a failed run, and
        takes down a rollout whose only fault is that it was one chunk long —
        including one whose real result the user was about to read off some
        *other* hook. The two are different questions with different answers:
        no record (``result is None``) versus no slope (``ValueError``, which
        :meth:`test_fewer_than_two_samples_is_refused` owns), and this pins the
        lifecycle half of that contrast.
        """
        runner = MDRunner(VelocityVerletIntegrator(_DRIFT_DT_FS))
        hook = EnergyDriftHook()
        hook.on_run_start(runner)
        hook.on_step_end(runner, 1, _obs())
        hook.on_run_end(runner)

        assert hook.result is None

    @pytest.mark.slow
    def test_nve_drift_on_the_butane_pair_box_is_within_the_criterion(self) -> None:
        """200 fp64 velocity-Verlet steps on the butane pair conserve energy.

        The release gate the hook exists for, on the same OPLS butane pair in
        the same 20 Å periodic Ewald box the assembled-energy symmetry gates
        run on
        (``tests/test_molpot/test_forcefield/test_potentials.py``): a
        conservative force field integrated by a symplectic integrator has no
        secular energy drift, only a bounded oscillation. The criterion is the
        one ``benchmarks/verify_md_ljcut_nve.py`` declares — ``|slope · T| /
        |E_tot(0)| < 1e-3`` — and it is asserted here, on the hook that now
        owns it, rather than in a script nothing runs.

        ``rel_drift`` also carries the finiteness half of that criterion: it is
        a reduction over **every** sample, so a single ``nan`` or ``inf`` total
        propagates into it and fails the comparison rather than being averaged
        away. ``rms_rel`` is asserted finite for the same reason on the
        fluctuation side.

        The rollout starts from rest, which makes it deterministic to the last
        bit: no thermostat, no sampled velocities, one trajectory. It runs the
        k-sum at :data:`~tests.conftest.BUTANE_PAIR_LR_WAVELENGTH_MD` rather
        than at the single-point gates' finer wavelength: the explicit sum
        costs ``(L / λ)³`` vectors on **every** one of the 200 force
        evaluations, and both wavelengths are converged some twenty orders
        below the ``1e-3`` this asserts.
        """
        batch = butane_pair_batch()
        hook = EnergyDriftHook()
        neighbors = NeighborList(
            cell=torch.eye(3, dtype=torch.float64) * BUTANE_PAIR_BOX,
            cutoff=BUTANE_PAIR_LJ_CUTOFF,
            positions=butane_pair_positions(),
            skin=_NVE_SKIN,
        )
        (
            MD(torch.float64)
            .set_forcefield(
                butane_pair_class_i_forcefield(lr_wavelength=BUTANE_PAIR_LR_WAVELENGTH_MD)
            )
            .set_neighbors(neighbors)
            .set_integrator(VelocityVerletIntegrator(_NVE_DT_FS))
            .set_hooks([hook])
            .run(batch, _NVE_STEPS)
        )

        result = hook.result
        assert result is not None, "on_run_end must publish the drift record"
        assert result.duration_fs == pytest.approx(
            _NVE_DT_FS * (_NVE_STEPS - 1), abs=_DRIFT_EXACT, rel=0.0
        )
        assert math.isfinite(result.rms_rel)
        assert result.rel_drift < _NVE_DRIFT_BOUND, (
            f"NVE energy drift {result.rel_drift:.3e} exceeds {_NVE_DRIFT_BOUND:g} "
            f"(slope {result.slope:.3e} per fs over {result.duration_fs:g} fs, "
            f"E_tot(0) = {result.e0:.6g}, rms {result.rms_rel:.3e})"
        )


class TestMDCheckpointHook:
    def test_writes_a_restartable_payload_on_cadence(self, tmp_path):
        ck = tmp_path / "state.pt"
        hook = MDCheckpointHook(ck, every=5)
        hook.on_step_end(None, 3, _obs())  # off-cadence: no write
        assert not ck.exists()
        hook.on_step_end(None, 5, _obs(fill=7.0))
        saved = torch.load(ck, weights_only=True)
        assert saved["step"] == 5
        assert torch.equal(saved["pos"], torch.full((4, 3), 7.0, dtype=_DTYPE))
        assert torch.equal(saved["vel"], torch.full((4, 3), 14.0, dtype=_DTYPE))

    def test_step_offset_keeps_one_step_axis(self, tmp_path):
        """A resumed segment reports absolute steps, not segment-local ones."""
        ck = tmp_path / "state.pt"
        MDCheckpointHook(ck, every=5, step_offset=100).on_step_end(None, 5, _obs())
        assert torch.load(ck, weights_only=True)["step"] == 105

    def test_declares_its_interval_as_cadence(self, tmp_path):
        assert MDCheckpointHook(tmp_path / "s.pt", every=7).cadence == 7
