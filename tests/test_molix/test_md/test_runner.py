"""Tests for the hook-driven MD runner (molix.md.runner).

Single-function correctness only. ``MDRunner.run`` is exercised with an
analytic harmonic well just far enough to observe its own contract (hook
dispatch order, step accounting, typed return); every hook class is driven
**directly** with synthetic :class:`MDObservables` / :class:`MDState` — no
trajectory loop stands between a hook test and the method it tests.
"""

import torch

from molix.md import (
    ForceOutput,
    HarmonicForceField,
    LangevinVerletIntegrator,
    MDCheckpointHook,
    MDHook,
    MDObservables,
    MDRunner,
    MDState,
    NeighborListHook,
    TrajectoryHook,
)
from molix.md.forcefield import ForceField

_DTYPE = torch.float64


def _integrator(seed: int = 0):
    return LangevinVerletIntegrator(
        HarmonicForceField(1.0).to(_DTYPE), dt=0.01, gamma=1.0, kbt=1.0, mass=1.0, seed=seed
    )


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
        runner = MDRunner(_integrator(), mass=1.0, hooks=[rec])
        out = runner.run(torch.zeros(4, 3, dtype=_DTYPE), torch.zeros(4, 3, dtype=_DTYPE), 3)

        assert rec.events == ["start", "step", "step", "step", "end"]
        assert rec.steps == [1, 2, 3]
        assert isinstance(out, MDState)
        assert out.pos.shape == (4, 3)
        assert out.forces.shape == (4, 3)

    def test_observables_are_typed_and_consistent(self):
        rec = _RecordingHook()
        runner = MDRunner(_integrator(), mass=1.0, hooks=[rec])
        runner.run(torch.randn(4, 3, dtype=_DTYPE), torch.randn(4, 3, dtype=_DTYPE), 2)

        last = rec.observed[-1]
        assert torch.allclose(last.total, last.potential + last.kinetic)
        assert last.forces.shape == (4, 3)

    def test_dof_follows_the_integrator_convention(self):
        """The DoF convention lives on ``Integrator.removed_dof`` — the runner
        must not duck-type thermostat internals (the old ``getattr(integrator,
        "gamma")`` sniffing silently mis-reported temperature for custom
        integrators)."""
        assert _integrator().removed_dof == 0  # gamma > 0
        nve = LangevinVerletIntegrator(
            HarmonicForceField(1.0).to(_DTYPE), dt=0.01, gamma=0.0, kbt=0.0, mass=1.0
        )
        assert nve.removed_dof == 3

    def test_step_start_precedes_each_advance(self):
        """``on_step_start`` fires before the chunk it precedes — the ordering
        that lets NeighborListHook refresh position-derived state ahead of the
        force evaluations inside the advance."""
        events: list[tuple[str, int]] = []

        class _Interleaved(MDHook):
            def on_step_start(self, runner, step, state):
                events.append(("start", step))

            def on_step_end(self, runner, step, obs):
                events.append(("end", step))

        runner = MDRunner(_integrator(), mass=1.0, hooks=[_Interleaved()])
        runner.run(torch.zeros(2, 3, dtype=_DTYPE), torch.zeros(2, 3, dtype=_DTYPE), 4, chunk=2)
        assert events == [("start", 0), ("end", 2), ("start", 2), ("end", 4)]

    def test_misaligned_hook_cadence_is_rejected(self):
        """A declared cadence that chunking would silently skip must raise."""
        import pytest

        hook = TrajectoryHook("/tmp/_chunk_misalign.pt", stride=3, write_xyz=False)
        runner = MDRunner(_integrator(), mass=1.0, hooks=[hook])
        with pytest.raises(ValueError, match="multiple of chunk"):
            runner.run(torch.zeros(2, 3, dtype=_DTYPE), torch.zeros(2, 3, dtype=_DTYPE), 6, chunk=2)

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
            mass=1.0,
            hooks=[(_Tagged("late"), 100), (_Tagged("first"), 0), _Tagged("default")],
        )
        runner.run(torch.zeros(2, 3, dtype=_DTYPE), torch.zeros(2, 3, dtype=_DTYPE), 1)
        assert order == ["first", "late", "default"]


class TestTrajectoryHook:
    """Driven directly: synthetic observables in, files out."""

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


class TestNeighborListHook:
    class _Recorder(ForceField):
        def __init__(self):
            super().__init__()
            self.rebuilt_at: list[int] = []

        def rebuild_neighbors(self, pos):
            self.rebuilt_at.append(int(pos[0, 0]))

        def forward(self, pos):
            return ForceOutput(pos.sum(), pos)

    def test_rebuilds_on_cadence_with_the_live_positions(self):
        force = self._Recorder()
        hook = NeighborListHook(force, every=3)
        for step in range(9):
            state = MDState(
                torch.full((2, 3), float(step)),
                torch.zeros(2, 3),
                torch.zeros(2, 3),
                torch.tensor(0.0),
            )
            hook.on_step_start(None, step, state)
        assert force.rebuilt_at == [0, 3, 6]
        assert hook.cadence == 3

    def test_rejects_a_nonpositive_cadence(self):
        import pytest

        with pytest.raises(ValueError, match=">= 1"):
            NeighborListHook(self._Recorder(), every=0)


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
