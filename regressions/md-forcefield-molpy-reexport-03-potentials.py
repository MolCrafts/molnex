"""Public-API wrap LJ uses already-placed r=1 Å, not raw r=9 Å.

Golden: ε=1, σ=2, r=1 → E = 4*(2**12 - 2**6) = 16128 kcal/mol.
Formula: Jones 1924 / LAMMPS pair_lj. Date 2026-08-16. No third-party runtime.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from molpot.forcefield import ForceField


def main() -> None:
    ff = (
        ForceField()
        .def_pairstyle("lj/cut", cutoff=5.0)
        .with_params(atoms={"epsilon": torch.tensor([1.0, 1.0]), "sigma": torch.tensor([2.0, 2.0])})
    )
    pos = torch.tensor([[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
    batch = TensorDict(
        {
            "atoms": TensorDict(
                {
                    "pos": pos,
                    "batch": torch.zeros(2, dtype=torch.long),
                    "type": torch.zeros(2, dtype=torch.long),
                },
                batch_size=[2],
            ),
            "edges": TensorDict(
                {
                    "edge_index": torch.tensor([[0, 1]]),
                    "edge_diff": torch.tensor([[-1.0, 0.0, 0.0]]),
                    "edge_dist": torch.tensor([1.0]),
                },
                batch_size=[1],
            ),
            "graphs": TensorDict({"cell": (torch.eye(3) * 10.0).unsqueeze(0)}, batch_size=[1]),
        },
        batch_size=[],
    )
    energy = float(ff.to_potentials().calc_energy(batch))
    if abs(energy - 16128.0) / 16128.0 > 1e-10:
        raise AssertionError(f"E={energy} expected 16128 (r=1, not r=9)")
    print("md-forcefield-molpy-reexport-03-potentials: E = 16128 kcal/mol OK")


if __name__ == "__main__":
    main()
