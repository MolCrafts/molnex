"""Public-API chain regression for OPLSAAForceField f14 + harmonic energy.

Spec: ``oplsaa-parameterizer-04-cut``.

Hard-coded goldens only — no OpenMM / third-party oracle. Pins:

1. Default ``NonbondedScaling`` on :class:`~molpot.OPLSAAForceField`:
       ``scale_q_14 == scale_lj_14 == 0.5``
   (OPLS-AA; not AMBER ``scale_q_14 = 5/6``).
2. Bond-only analytic energy via filled OPLSAA path → ``to_potentials``:
       E = ½ k (r − r0)² with k=2, r0=1.0 Å, r=1.5 Å → E = 0.25 kcal/mol
   (``atol=1e-10``). No ``project()``.

Provenance
----------
    formula          : Class-I harmonic bond E = ½k(r−r0)²;
                       OPLS-AA f14^(q) = f14^(LJ) = 1/2
    geometry         : two H atoms on x̂, separation 1.5 Å
    citation         : Jorgensen, Maxwell & Tirado-Rives,
                       J. Am. Chem. Soc. 118, 11225–11236 (1996)
                       DOI 10.1021/ja9621760
    capture command  : PYTHONPATH=src python regressions/oplsaa-parameterizer.py
    date             : 2026-08-20
    tolerance        : energy atol 1e-10; f14 exact equality
"""

from __future__ import annotations

import math
import sys

import torch
import torch.nn as nn
from tensordict import TensorDict

from molpot.forcefield import ForceField, OPLSAAForceField, Parameterizer

_GOLDEN_E = 0.25  # kcal/mol; ½·2·(0.5)²
_ENERGY_ATOL = 1e-10
_F14 = 0.5


class _FakeEncoder(nn.Module):
    def __init__(self, features: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self._features = features

    def forward(self, td: TensorDict) -> TensorDict:
        return td

    def embeddings(self, td: TensorDict) -> dict[str, torch.Tensor]:
        del td
        return dict(self._features)


class _ConstantBondHead(nn.Module):
    def __init__(self, k: float = 2.0, r0: float = 1.0) -> None:
        super().__init__()
        self.k = k
        self.r0 = r0

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        n = features.shape[0]
        return {
            "k": torch.full((n,), self.k, dtype=features.dtype, device=features.device),
            "r0": torch.full((n,), self.r0, dtype=features.dtype, device=features.device),
        }


def main() -> int:
    ff = OPLSAAForceField()
    scaling = ff.nonbonded_scaling
    if scaling.scale_q_14 != _F14 or scaling.scale_lj_14 != _F14:
        raise AssertionError(
            f"OPLS-AA f14 must be {_F14}; got q={scaling.scale_q_14}, lj={scaling.scale_lj_14}"
        )

    # Bond-only template for the analytic pin (same harmonic form as OPLSAA bonds).
    bond_ff = ForceField().def_bondstyle("harmonic")
    model = Parameterizer(
        _FakeEncoder({"bonds": torch.zeros(1, 1, dtype=torch.float64)}),
        bond_ff,
        bond_head=_ConstantBondHead(k=2.0, r0=1.0),
    )
    if hasattr(model, "project"):
        raise AssertionError("Parameterizer must not expose project()")

    pos = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float64)
    batch = TensorDict(
        {
            "atoms": TensorDict(
                {
                    "pos": pos,
                    "Z": torch.tensor([1, 1], dtype=torch.long),
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
    energy = float(model(batch).to_potentials().calc_energy(batch))
    if not math.isclose(energy, _GOLDEN_E, abs_tol=_ENERGY_ATOL):
        raise AssertionError(f"bond E={energy}, want {_GOLDEN_E}")

    print(f"oplsaa-parameterizer: f14={_F14}/{_F14}, harmonic E={_GOLDEN_E} kcal/mol OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
