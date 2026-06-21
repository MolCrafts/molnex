"""Tests for the hook-driven MD runner (molix.md.runner).

The integrator is an analytic harmonic well (F = -k x) so the runner can be
exercised without PiNet. Coverage: hook lifecycle ordering + global_step
advance, physics handed via the ``outputs`` channel (never the reserved state
namespaces), and the reference :class:`TrajectoryHook` capture / persistence.
"""

import torch

from molix.core.hook import BaseHook
from molix.md import LangevinVerletIntegrator, MDRunner, TrajectoryHook

_DTYPE = torch.float64


def _harmonic(k: float):
    def force_fn(pos: torch.Tensor):
        energy = 0.5 * k * (pos**2).sum()
        return energy, -k * pos

    return force_fn


def _integrator(seed: int = 0):
    return LangevinVerletIntegrator(
        _harmonic(1.0), dt=0.01, gamma=1.0, kbt=1.0, mass=1.0, seed=seed
    )


class _RecordingHook(BaseHook):
    """Records lifecycle call order and the per-step outputs payload."""

    def __init__(self):
        self.events: list[str] = []
        self.steps: list[int] = []
        self.outputs: list[dict] = []

    def on_train_start(self, trainer, state):
        self.events.append("start")

    def on_train_batch_end(self, trainer, state, batch, outputs):
        self.events.append("batch")
        self.steps.append(state["global_step"])
        self.outputs.append(outputs)

    def on_train_end(self, trainer, state):
        self.events.append("end")


def test_runner_fires_lifecycle_and_advances_step():
    rec = _RecordingHook()
    runner = MDRunner(_integrator(), mass=1.0, hooks=[rec])
    out = runner.run(torch.zeros(4, 3, dtype=_DTYPE), torch.zeros(4, 3, dtype=_DTYPE), 5)

    assert rec.events == ["start", "batch", "batch", "batch", "batch", "batch", "end"]
    assert rec.steps == [1, 2, 3, 4, 5]
    assert out["state"]["global_step"] == 5
    assert out["pos"].shape == (4, 3)
    assert out["force"].shape == (4, 3)


def test_runner_outputs_carry_physics_not_state_namespaces():
    rec = _RecordingHook()
    runner = MDRunner(_integrator(), mass=1.0, hooks=[rec])
    runner.run(torch.randn(4, 3, dtype=_DTYPE), torch.randn(4, 3, dtype=_DTYPE), 3)

    last = rec.outputs[-1]
    for key in ("pos", "vel", "forces", "potential", "kinetic", "total", "temperature"):
        assert key in last
    assert torch.allclose(last["total"], last["potential"] + last["kinetic"])
    # Physics must NOT leak into the reserved TrainState namespaces.
    for ns in ("train", "eval", "performance", "gpu"):
        assert not runner.state.get(ns)


def test_runner_temperature_is_positive_under_thermostat():
    rec = _RecordingHook()
    runner = MDRunner(_integrator(seed=2), mass=1.0, hooks=[rec])
    runner.run(torch.zeros(20, 3, dtype=_DTYPE), torch.zeros(20, 3, dtype=_DTYPE), 50)
    temps = torch.stack([o["temperature"] for o in rec.outputs])
    assert (temps > 0).all()


def test_trajectory_hook_persists_pt_and_xyz(tmp_path):
    numbers = torch.tensor([1, 6, 8, 1])
    out_pt = tmp_path / "traj.pt"
    hook = TrajectoryHook(out_pt, stride=2, numbers=numbers, write_xyz=True)
    runner = MDRunner(_integrator(), mass=1.0, hooks=[hook])
    runner.run(torch.zeros(4, 3, dtype=_DTYPE), torch.zeros(4, 3, dtype=_DTYPE), 10)

    assert out_pt.exists()
    payload = torch.load(out_pt, weights_only=False)
    # stride=2 over 10 steps (global_step 1..10): kept at steps 2,4,6,8,10 -> 5 frames.
    assert payload["pos"].shape == (5, 4, 3)
    assert payload["forces"].shape == (5, 4, 3)
    assert payload["temp"].shape == (5,)
    assert payload["stride"] == 2
    assert torch.equal(payload["Z"], numbers)

    xyz = out_pt.with_suffix(".xyz")
    assert xyz.exists()
    lines = xyz.read_text().splitlines()
    # Each extended-XYZ frame: 1 count line + 1 comment + N atom lines.
    assert lines[0] == "4"
    assert "Etot=" in lines[1] and "T=" in lines[1]
    assert lines[2].split()[0] == "H"  # Z=1 -> H


def test_trajectory_hook_without_numbers_skips_xyz(tmp_path):
    out_pt = tmp_path / "traj.pt"
    hook = TrajectoryHook(out_pt, stride=1, write_xyz=True)  # no numbers -> no xyz
    runner = MDRunner(_integrator(), mass=1.0, hooks=[hook])
    runner.run(torch.zeros(3, 3, dtype=_DTYPE), torch.zeros(3, 3, dtype=_DTYPE), 4)

    assert out_pt.exists()
    assert not out_pt.with_suffix(".xyz").exists()
    payload = torch.load(out_pt, weights_only=False)
    assert payload["pos"].shape == (4, 3, 3)
