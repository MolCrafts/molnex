"""Public-API regression for Parameterizer (encoder → ForceField → E/F).

::

    ff_t = model(batch)
    E = ff_t.to_potentials().calc_energy(batch)

Hard-coded goldens only. Bond: k=2, r=1.5, r0=1 → E = 0.25 kcal/mol.
"""

from __future__ import annotations

import sys
from typing import Mapping

import torch
import torch.nn as nn
from tensordict import TensorDict


class _FakeEmbeddings:
    def __init__(self, features: Mapping[str, torch.Tensor]) -> None:
        self._features = dict(features)

    def interaction_dict(self) -> dict[str, torch.Tensor]:
        return dict(self._features)


class FakeEncoder(nn.Module):
    def __init__(self, features: Mapping[str, torch.Tensor]) -> None:
        super().__init__()
        self._features = dict(features)
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, td: TensorDict) -> TensorDict:
        return td

    def embeddings(self, td: TensorDict) -> _FakeEmbeddings:
        return _FakeEmbeddings(self._features)


class _ConstBond(nn.Module):
    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        n = features.shape[0]
        return {
            "k": torch.full((n,), 2.0, dtype=features.dtype),
            "r0": torch.full((n,), 1.0, dtype=features.dtype),
        }


def main() -> int:
    from molpot import ForceField, Parameterizer

    torch.manual_seed(0)
    features = {"bonds": torch.zeros(1, 4, dtype=torch.float64)}
    encoder = FakeEncoder(features)

    ff = ForceField(name="test", units="real").def_bondstyle("harmonic")
    model = Parameterizer(encoder, ff, bond_head=_ConstBond())
    assert not hasattr(model, "project")

    pos = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float64)
    batch = TensorDict(
        {
            "atoms": TensorDict(
                {
                    "pos": pos,
                    "Z": torch.tensor([1, 1]),
                    "batch": torch.zeros(2, dtype=torch.long),
                },
                batch_size=[2],
            ),
            "bonds": TensorDict(
                {
                    "atomi": torch.tensor([0], dtype=torch.long),
                    "atomj": torch.tensor([1], dtype=torch.long),
                },
                batch_size=[1],
            ),
        },
        batch_size=[],
    )

    filled = model(batch)
    assert isinstance(filled, ForceField)
    E = filled.to_potentials().calc_energy(batch)
    assert abs(float(E) - 0.25) < 1e-10, E

    forces = filled.to_potentials().calc_forces(batch)
    assert forces.shape == (2, 3)
    assert abs(float(forces[0, 0].detach()) - 1.0) < 1e-6
    assert abs(float(forces[1, 0].detach()) + 1.0) < 1e-6

    assert "bond:harmonic" in ff.to_potentials().get_schema()
    print("learnable-classical-ff-05 Parameterizer OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
