"""Public MD composition after the ForceField re-export.

Deleted adapter names are absent. Two-atom lj/cut energy is the analytic
V = 4ε[(σ/r)^12 − (σ/r)^6] at r=3.4, ε=0.15, σ=3.0 (2026-08-16).
"""

from __future__ import annotations

import math

import torch
from tensordict import TensorDict

import molix.md as md
from molix.md import MD, NeighborList, VelocityVerletIntegrator
from molpot.forcefield import ForceField as MolpotFF


def main() -> None:
    deleted = (
        "CallableForceField",
        "HarmonicForceField",
        "LennardJonesForceField",
        "LennardJonesCutForceField",
        "PotentialForceField",
        "PeriodicPotentialForceField",
    )
    still = [n for n in deleted if n in md.__all__]
    if still:
        raise AssertionError(f"deleted names still exported: {still}")
    r, eps, sig = 3.4, 0.15, 3.0
    expected = 4.0 * eps * ((sig / r) ** 12 - (sig / r) ** 6)
    ff = (
        MolpotFF("wire", units="metal")
        .def_pairstyle("lj/cut", cutoff=10.0)
        .with_params(
            atoms={
                "epsilon": torch.tensor([eps, eps], dtype=torch.float64),
                "sigma": torch.tensor([sig, sig], dtype=torch.float64),
            }
        )
    )
    pos = torch.tensor([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=torch.float64)
    cell = torch.eye(3, dtype=torch.float64) * 20.0
    batch = TensorDict(
        {
            "atoms": TensorDict(
                {
                    "pos": pos,
                    "vel": torch.zeros_like(pos),
                    "mass": torch.ones(2, dtype=torch.float64),
                    "batch": torch.zeros(2, dtype=torch.long),
                },
                batch_size=[2],
            ),
            "graphs": TensorDict({"cell": cell.unsqueeze(0)}, batch_size=[1]),
        },
        batch_size=[],
    )
    energy = float(ff.to_potentials().calc_energy(batch))
    if not math.isclose(energy, expected, rel_tol=0, abs_tol=1e-10):
        raise AssertionError(f"E={energy} expected {expected}")
    nl = NeighborList(cell=cell, cutoff=8.0, positions=pos, skin=0.0)
    engine = (
        MD(torch.float64)
        .set_forcefield(ff)
        .set_neighbors(nl)
        .set_integrator(VelocityVerletIntegrator(0.5))
    )
    state = engine.run(batch, 1)
    if not torch.isfinite(state.pos).all():
        raise AssertionError("md.run produced non-finite positions")
    print("md-forcefield-molpy-reexport-05-wire: goldens OK")


if __name__ == "__main__":
    main()
