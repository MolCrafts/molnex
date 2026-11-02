"""Regression: the skin-gated neighbour list over a real NVE trajectory.

Spec: `md-neighborlist-skin-04-policy` / `md-neighborlist-skin-07-wire`.

Moved out of the unit suite (formerly
``tests/test_molix/test_md/test_neighbors.py::TestNeighborListPolicy`` and
``tests/test_molix/test_md/test_driver.py::TestMD``): each arm integrates
100 NVE steps, which is a physics trajectory, not a unit test. The gate's
single-decision arms stay unit-tested; ``07-wire`` pins the rebuild
accounting literals. This file pins what only a trajectory can show:

* **Completeness.** At every step of the ``skin=0.5`` run, the exact O(N^2)
  minimum-image pair set within the 3.5 A cutoff is a subset of the live list,
  and each pair's list-reconstructed distance matches the oracle to 1e-9.
  The distance half catches a stale shift on a surviving index pair.
* **Same physics.** The ``skin=0.5`` run and a ``skin=0`` run that rebuilds at
  every force evaluation agree per step in total energy and forces to 1e-10.
* **No dangerous builds.** ``skin=0.5`` gives ``ndanger == 0`` with
  ``rebuild_count > 0`` (the gate is alive).
* **The skin buys rebuilds.** ``rebuild_count`` is non-increasing over
  ``skin = 0, 0.25, 0.5, 1.0`` and strictly lower at 1.0 than at 0.
* **Energy conservation.** The ``skin=1.0`` drift
  ``max_t |E(t) - E(0)| / |E(0)|`` is at most 3x the ``skin=0`` drift, which
  is itself nonzero (finite-dt velocity Verlet), so the ratio is not vacuous.

Capture:
    command: PYTHONPATH=src python regressions/md-neighborlist-skin-08-trajectory.py
    commit:  e89a92c (tests moved verbatim; no new literals)
    torch:   2.14.1+cpu (python 3.12)
    date:    2026-10-08
    device:  cpu, float64
    oracle:  none — relations between arms of the same seeded run, plus the
             in-file O(N^2) minimum-image pair set.

System: 64 argon atoms on a 4x4x4 lattice (3 A spacing, 12 A cube),
eps = 0.0103 eV, sigma = 2.5 A, cutoff = 3.5 A, m = 39.95 amu, dt = 4 fs,
Maxwell-Boltzmann velocities at 300 K with seed 0, gamma = 0.
"""

from __future__ import annotations

import sys
from typing import NamedTuple

import torch

from molix.md import (
    EV_PER_AMU_A2_FS2,
    MD,
    LennardJonesCutForceField,
    MaxwellBoltzmann,
    MDHook,
    MDObservables,
    MDRunner,
)
from molix.md.neighbors import NeighborList

N_ATOMS = 64
N_STEPS = 100
CUTOFF = 3.5


class Frame(NamedTuple):
    pos: torch.Tensor
    edge_index: torch.Tensor
    shifts: torch.Tensor
    total: torch.Tensor
    forces: torch.Tensor


class FrameRecorder(MDHook):
    """Capture ``obs.pos`` together with the live list it was evaluated against."""

    def __init__(self, neighbors: NeighborList) -> None:
        self._neighbors = neighbors
        self.frames: list[Frame] = []

    def on_step_end(self, runner: MDRunner, step: int, obs: MDObservables) -> None:
        n = self._neighbors.num_edges
        self.frames.append(
            Frame(
                pos=obs.pos.detach().clone(),
                edge_index=self._neighbors.edge_index[:n].clone(),
                shifts=self._neighbors.shifts[:n].clone(),
                total=obs.total.detach().clone(),
                forces=obs.forces.detach().clone(),
            )
        )


def lattice() -> tuple[torch.Tensor, torch.Tensor]:
    grid = torch.arange(4, dtype=torch.float64) * 3.0
    pos = torch.stack(torch.meshgrid(grid, grid, grid, indexing="ij"), dim=-1)
    return pos.reshape(-1, 3), torch.eye(3, dtype=torch.float64) * 12.0


def argon_nve(skin: float) -> tuple[NeighborList, list[Frame]]:
    """100 NVE steps through the public ``MD`` path at one skin."""
    pos, cell = lattice()
    # capacity_factor=2.5: at skin=0.5 the run reaches 600 live edges against
    # the 519 rows the default 1.35 would allocate from the initial 384.
    neighbors = NeighborList(
        cell=cell, cutoff=CUTOFF, positions=pos, skin=skin, capacity_factor=2.5
    )
    force = LennardJonesCutForceField(
        epsilon=0.0103 / EV_PER_AMU_A2_FS2, sigma=2.5, neighbors=neighbors, cutoff=CUTOFF
    )
    recorder = FrameRecorder(neighbors)
    velocities = MaxwellBoltzmann(39.95, n_atoms=N_ATOMS).sample(300.0, seed=0)
    md = MD(force, mass=39.95, dt=4.0, gamma=0.0, dtype=torch.float64, hooks=[recorder])
    md.set_potential_dtype(torch.float64)
    md.run(pos, velocities, N_STEPS, chunk=1)
    return neighbors, recorder.frames


def reference_pairs(pos: torch.Tensor, cell: torch.Tensor, cutoff: float):
    """Exact O(N^2) minimum-image ordered pairs within ``cutoff`` (cubic cell)."""
    fractional = pos @ torch.linalg.inv(cell)
    delta = fractional.unsqueeze(0) - fractional.unsqueeze(1)
    delta = delta - torch.round(delta)
    distance = torch.linalg.norm(delta @ cell, dim=-1)
    inside = (distance < cutoff) & ~torch.eye(pos.shape[0], dtype=torch.bool)
    source, target = torch.nonzero(inside, as_tuple=True)
    return source, target, distance[source, target]


def drift(frames: list[Frame]) -> float:
    totals = [float(f.total) for f in frames]
    return max(abs(t - totals[0]) for t in totals) / abs(totals[0])


def main() -> int:
    _, cell = lattice()
    arms = {skin: argon_nve(skin) for skin in (0.0, 0.25, 0.5, 1.0)}

    gated_nl, gated = arms[0.5]
    assert len(gated) == N_STEPS, len(gated)
    for step, frame in enumerate(gated):
        source, target, reference = reference_pairs(frame.pos, cell, CUTOFF)
        rows = torch.full((N_ATOMS * N_ATOMS,), -1, dtype=torch.long)
        live = frame.edge_index
        rows[live[:, 0] * N_ATOMS + live[:, 1]] = torch.arange(live.shape[0])
        found = rows[source * N_ATOMS + target]
        missing = int((found < 0).sum())
        assert missing == 0, f"step {step}: {missing} pairs inside the cutoff are not listed"
        reconstructed = torch.linalg.norm(
            frame.pos[target] - frame.pos[source] + frame.shifts[found], dim=-1
        )
        torch.testing.assert_close(reconstructed, reference, atol=1e-9, rtol=0)
    print("  completeness: every in-cutoff pair listed at every step")

    every_eval = arms[0.0][1]
    for step, (a, b) in enumerate(zip(gated, every_eval, strict=True)):
        torch.testing.assert_close(a.total, b.total, atol=1e-10, rtol=0, msg=f"step {step}")
        torch.testing.assert_close(a.forces, b.forces, atol=1e-10, rtol=0, msg=f"step {step}")
    print("  same physics: skin=0.5 == skin=0 per step")

    assert gated_nl.ndanger == 0, gated_nl.ndanger
    assert gated_nl.rebuild_count > 0, gated_nl.rebuild_count
    print(f"  no dangerous builds: skin=0.5 rebuilds {gated_nl.rebuild_count}x, ndanger 0")

    counts = [arms[skin][0].rebuild_count for skin in (0.0, 0.25, 0.5, 1.0)]
    assert counts == sorted(counts, reverse=True), counts
    assert counts[-1] < counts[0], counts
    print(f"  rebuild counts over skin 0/0.25/0.5/1.0: {counts}")

    baseline = drift(arms[0.0][1])
    gated_drift = drift(arms[1.0][1])
    assert baseline > 0.0, "the no-skin baseline must have real discretisation drift"
    assert gated_drift <= 3.0 * baseline, (gated_drift, baseline)
    print(f"  energy drift: skin=1.0 {gated_drift:.3e} vs skin=0 {baseline:.3e}")

    print("md-neighborlist-skin-08-trajectory: OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 — standalone regression script
        print(f"FAIL: {exc}", file=sys.stderr)
        raise
