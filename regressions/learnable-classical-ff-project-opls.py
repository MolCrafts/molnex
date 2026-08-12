"""Regression: ClassicalIMM.project Fourier → OPLS preserves energy/forces.

Hard-coded geometry; multi-term cosine with OPLS phases maps to k1..k4.
"""

from __future__ import annotations

import math
import sys

import torch
import torch.nn as nn
from tensordict import TensorDict


class _ConstBond(nn.Module):
    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        n = features.shape[0]
        return {
            "k": torch.full((n,), 2.0, dtype=features.dtype),
            "r0": torch.full((n,), 1.5, dtype=features.dtype),
        }


class _OplsPhasesProper(nn.Module):
    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        n = features.shape[0]
        device, dtype = features.device, features.dtype
        # k_m = K_m/2 for OPLS Ki=(1, 0.5, 0.25, 0.1)
        K = torch.tensor([1.0, 0.5, 0.25, 0.1], device=device, dtype=dtype)
        k = (0.5 * K).unsqueeze(0).expand(n, 4).contiguous()
        phase = torch.zeros(n, 4, device=device, dtype=dtype)
        phase[:, 1] = math.pi
        phase[:, 3] = math.pi
        return {
            "k": k,
            "phase": phase,
            "periodicity": torch.tensor([1, 2, 3, 4], device=device, dtype=torch.long),
            "idivf": torch.ones(n, device=device, dtype=dtype),
        }


class FakeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._p = nn.Parameter(torch.zeros(1))

    def forward(self, td: TensorDict) -> TensorDict:
        return td

    def embeddings(self, td: TensorDict):
        class _E:
            def interaction_dict(self_inner):
                return {
                    "bonds": torch.zeros(3, 2, dtype=torch.float64),
                    "propers": torch.zeros(1, 4, dtype=torch.float64),
                }

        return _E()


def main() -> int:
    from molpot.forcefield import ClassicalIMM, ForceField

    model = ClassicalIMM(
        FakeEncoder(),
        forcefield=ForceField.class_i(dihedral="fourier"),
        bond_head=_ConstBond(),
        proper_head=_OplsPhasesProper(),
    )
    pos = torch.tensor(
        [
            [1.0, 0.0, 0.2],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 1.0, 0.3],
        ],
        dtype=torch.float64,
    )
    batch = TensorDict(
        {
            "atoms": TensorDict({"pos": pos}, batch_size=[4]),
            "bonds": TensorDict(
                {
                    "atomi": torch.tensor([0, 1, 2], dtype=torch.long),
                    "atomj": torch.tensor([1, 2, 3], dtype=torch.long),
                },
                batch_size=[3],
            ),
            "propers": TensorDict(
                {
                    "atomi": torch.tensor([0], dtype=torch.long),
                    "atomj": torch.tensor([1], dtype=torch.long),
                    "atomk": torch.tensor([2], dtype=torch.long),
                    "atoml": torch.tensor([3], dtype=torch.long),
                },
                batch_size=[1],
            ),
        },
        batch_size=[],
    )

    src = model(batch)
    tgt = ForceField(name="OPLS").def_bondstyle("harmonic").def_dihedralstyle("opls")
    dst = model.project(tgt, source=src)
    e0 = src.to_potentials().calc_energy(batch)
    e1 = dst.to_potentials().calc_energy(batch)
    f0 = src.to_potentials().calc_forces(batch)
    f1 = dst.to_potentials().calc_forces(batch)
    assert torch.allclose(e0, e1, atol=1e-10)
    assert torch.allclose(f0, f1, atol=1e-8)
    print("project Fourier→OPLS E/F parity OK", float(e0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
