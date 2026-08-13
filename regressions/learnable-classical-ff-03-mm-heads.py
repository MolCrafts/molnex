"""Public-API regression for Class-I MM heads + ForceField.to_potentials."""

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
            "r0": torch.full((n,), 1.0, dtype=features.dtype),
        }


class FakeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, td: TensorDict) -> TensorDict:
        return td

    def embeddings(self, td: TensorDict):
        class _E:
            def interaction_dict(self_inner):
                return {"bonds": torch.zeros(1, 8, dtype=torch.float64)}

        return _E()


def main() -> int:
    from molpot import (
        AngleParamHead,
        BondParamHead,
        ClassicalIMM,
        ForceField,
        ProperTorsionParamHead,
    )

    torch.manual_seed(0)
    bond_head = BondParamHead(feature_dim=8, hidden_dim=16)
    angle_head = AngleParamHead(feature_dim=8, hidden_dim=16)
    proper_head = ProperTorsionParamHead(feature_dim=8, hidden_dim=16, n_terms=3)

    fb = bond_head(torch.randn(4, 8))
    assert fb["k"].shape == (4,) and (fb["k"] > 0).all()
    fa = angle_head(torch.randn(3, 8))
    assert (fa["theta0"] > 0).all() and (fa["theta0"] < math.pi).all()
    fp = proper_head(torch.randn(2, 8))
    assert fp["a"].shape == (2, 3) and fp["b"].shape == (2, 3)

    ff = ForceField(name="abs", units="real").def_bondstyle("harmonic")
    model = ClassicalIMM(FakeEncoder(), forcefield=ff, bond_head=_ConstBond())
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float64)
    batch = TensorDict(
        {
            "atoms": TensorDict({"pos": pos}, batch_size=[2]),
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
    E = model(batch).to_potentials().calc_energy(batch)
    assert abs(float(E) - 0.25) < 1e-10, E
    print("learnable-classical-ff-03 heads + ForceField OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
